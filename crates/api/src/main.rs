use std::{net::SocketAddr, path::Path, sync::Arc, time::Duration};

use anyhow::Context;

use armor_api::{
    audit, broker_state,
    config::{ArmorMode, AuthMode, RateLimitMode, Settings},
    custom_rules,
    heartbeat::Heartbeat,
    middleware::rate_limit::RateLimiter,
    otel, profiles,
    profiles::ProfileResolver,
    retention::RetentionTask,
    routes,
    state::AppState,
    sync::{LiveResolver, SyncTask},
    telemetry::TelemetryEmitter,
};
use armor_core::policy::schema::PolicyConfig;
use armor_inference_client::{
    breaker::{BreakerConfig, CircuitBreaker},
    cache::CachingTransport,
    http::{HttpConfig, HttpTransport},
    transport::InferenceTransport,
};
use armor_storage::{
    policy_store::PgPolicyStore,
    vault::{EnvKeyProvider, Vault},
};

/// The data-plane wiring (`ArmorMode::Standalone`/`Edge`): boot policy +
/// profiles from disk, hot-swappable behind a `LiveResolver`, plus
/// auth/rate-limit config and the background sync task. Not run at all in
/// `ArmorMode::ControlPlane` — that mode mounts no `/api/*` routes, so
/// there's nothing here for it to protect.
struct DataPlane {
    profiles: LiveResolver,
    api_keys: Option<Arc<Vec<[u8; 32]>>>,
    rate_limiter: Option<Arc<RateLimiter>>,
    sync_task: SyncTask,
    detector_count: usize,
}

/// Reads and hardens `settings.policy_path` (`config/policies.yaml` by
/// default) — the one policy every deployment always has, file-based or
/// not. Factored out of `load_data_plane` so `wire_database`'s
/// seed-on-first-boot path (below) can load the same default policy
/// without needing a whole `DataPlane` (which also builds a full file-based
/// `ProfileResolver` that DB wiring immediately overrides).
fn load_default_policy(settings: &Settings) -> anyhow::Result<PolicyConfig> {
    let policy_yaml = std::fs::read_to_string(&settings.policy_path)
        .with_context(|| format!("reading policy file {}", settings.policy_path))?;
    let mut policy = armor_core::policy::loader::load(&policy_yaml)
        .with_context(|| format!("loading policy file {}", settings.policy_path))?;

    custom_rules::apply(&mut policy, Path::new(&settings.custom_rules_dir))
        .with_context(|| format!("applying custom rules from {}", settings.custom_rules_dir))?;

    // Fail the deploy now, with a clear message, rather than have a bad
    // customer regex surface later as a per-request fail-open/fail-closed
    // via the orchestrator's panic-recovery path.
    for check in &policy.checks {
        if check.category == "custom_regex" && check.enabled {
            armor_core::detectors::custom_regex::validate(&check.options)
                .map_err(|e| anyhow::anyhow!("invalid custom_regex options: {e}"))?;
        }
    }

    Ok(policy)
}

async fn load_data_plane(settings: &Settings) -> anyhow::Result<DataPlane> {
    let policy = load_default_policy(settings)?;
    tracing::info!(policy_id = %policy.id, checks = policy.checks.len(), "loaded default profile");

    // Named profiles + the application_id -> profile_id mapping — both
    // optional (`profiles.rs`'s module doc). Unconfigured deployments get
    // `ProfileResolver::single(default)`, i.e. every request still runs
    // `policy` exactly as it always has.
    let profiles = profiles::load(
        Arc::new(policy),
        Path::new(&settings.profiles_dir),
        Path::new(&settings.applications_path),
        Path::new(&settings.custom_rules_dir),
    )
    .context("loading profiles")?;

    let api_keys = match settings.auth_mode {
        AuthMode::None => None,
        AuthMode::ApiKey => {
            use sha2::{Digest, Sha256};
            let keys = settings
                .api_keys
                .iter()
                .map(|k| {
                    let mut hasher = Sha256::new();
                    hasher.update(k.as_bytes());
                    hasher.finalize().into()
                })
                .collect::<Vec<[u8; 32]>>();
            Some(Arc::new(keys))
        }
    };
    tracing::info!(auth_mode = ?settings.auth_mode, "auth configured");

    let rate_limiter = match settings.rate_limit_mode {
        RateLimitMode::None => None,
        RateLimitMode::Fixed => Some(Arc::new(RateLimiter::in_process(
            settings.rate_limit_rps,
            settings.rate_limit_burst,
            settings.trusted_proxies.clone(),
        ))),
        RateLimitMode::Redis => Some(Arc::new(
            RateLimiter::redis(
                &settings.redis.url,
                settings.rate_limit_rps,
                settings.rate_limit_burst,
                settings.redis.key_prefix.clone(),
                settings.trusted_proxies.clone(),
            )
            .await
            .context("connecting to Redis (ARMOR_REDIS_URL) for rate limiting")?,
        )),
    };
    tracing::info!(
        rate_limit_mode = ?settings.rate_limit_mode,
        rps = settings.rate_limit_rps,
        burst = settings.rate_limit_burst,
        trusted_proxies = settings.trusted_proxies.len(),
        redis_url = %settings.redis.url,
        "rate limiting configured"
    );

    // The default profile's check count — the meaningful "how many
    // detectors are configured" number when no named profiles exist yet;
    // once they do, this still just describes the default/fallback.
    let detector_count = profiles.resolve(None).checks.len();

    // Wrap the boot-time resolver in an atomic pointer so the background sync
    // task can hot-swap rules without blocking in-flight requests.
    let live_profiles = LiveResolver::new(profiles);
    let sync_task = SyncTask::spawn(
        live_profiles.clone(),
        settings.sync.clone(),
        settings.custom_rules_dir.clone(),
    );
    tracing::info!(
        sync_enabled = settings.sync.enabled,
        sync_url = %settings.sync.url,
        sync_interval_secs = settings.sync.interval_secs,
        "rule sync configured"
    );

    Ok(DataPlane {
        profiles: live_profiles,
        api_keys,
        rate_limiter,
        sync_task,
        detector_count,
    })
}

/// `ArmorMode::ControlPlane` mounts no `/api/*` routes (`routes::router`),
/// so `AppState.profiles` is never resolved against a real request — this
/// is an inert placeholder, not a policy any traffic actually runs under.
fn control_plane_stub_resolver() -> LiveResolver {
    use armor_core::policy::schema::{ExecutionMode, FailMode, NormalizeConfig, PolicyConfig};

    let stub = Arc::new(PolicyConfig {
        id: "control-plane-stub".to_string(),
        execution_mode: ExecutionMode::Parallel,
        fail_mode: FailMode::FailOpen,
        normalize: NormalizeConfig::default(),
        checks: Vec::new(),
    });
    LiveResolver::new(ProfileResolver::single(stub))
}

/// Builds the reversible-anonymization vault, or `None` when this
/// deployment hasn't asked for one.
///
/// Both halves are required and neither implies the other: without
/// `DATABASE_URL` there is nowhere to put an entry, and without
/// `ARMOR_VAULT_KEY` there is nothing to encrypt it with. Turning the key on
/// still stores nothing until a policy sets `on_fail: redact` on some check
/// — see `redaction.rs`.
///
/// A key that's set but malformed fails the boot rather than degrading to no
/// vault. The degraded mode is silent and looks identical to working
/// (redaction still happens, placeholders just aren't recoverable), so an
/// operator would discover the typo the day they needed to deanonymize
/// something — long after the values it should have kept were discarded.
fn wire_vault(
    settings: &Settings,
    db: Option<&Arc<PgPolicyStore>>,
) -> anyhow::Result<Option<Arc<Vault>>> {
    if settings.vault_key.trim().is_empty() {
        return Ok(None);
    }
    let Some(db) = db else {
        tracing::warn!(
            "ARMOR_VAULT_KEY is set but DATABASE_URL is not; the reversible-anonymization \
             vault is Postgres-backed, so redaction stays redact-and-discard"
        );
        return Ok(None);
    };

    let keys = EnvKeyProvider::from_base64(&settings.vault_key)
        .context("ARMOR_VAULT_KEY (expected standard base64 of exactly 32 bytes)")?;
    tracing::info!(
        entry_ttl_seconds = ?settings.vault_ttl_seconds,
        session_ttl_seconds = ?settings.session_ttl_seconds,
        "reversible-anonymization vault enabled; checks configured `on_fail: redact` will \
         store recoverable PII"
    );
    Ok(Some(Arc::new(
        Vault::new(db.pool().clone(), Box::new(keys)).with_ttl_seconds(settings.vault_ttl_seconds),
    )))
}

/// The optional `armor-inference` sidecar hop. Empty `ARMOR_INFERENCE_URL`
/// ⇒ `None`: `ml::escalate` is a no-op and every request runs the
/// deterministic path bit-for-bit as before.
/// When enabled, the stack is `HttpTransport` → breaker → result cache, so
/// repeated text never pays a forward pass and a sick sidecar degrades into
/// fallbacks instead of hanging the request.
///
/// A *set* URL that fails `HttpTransport::connect`'s preflight (unresolvable
/// host, blocked SSRF target, malformed URL — `net_guard::EndpointError`) is
/// reported as `Err` rather than retried, but the caller only logs it: this
/// process still comes up rules-only, the same as an empty URL, rather than
/// exiting. The one-time DNS snapshot this failed on would never have
/// self-healed on a bare restart anyway, and the control-plane API's own
/// `GET /api/v1/hardware` and `/api/v1/models` (`control_plane.rs`) probe the
/// sidecar independently on every call — an operator sees "unreachable"
/// there and in the sidebar badge instead of the process refusing to start.
async fn wire_inference(settings: &Settings) -> Option<Arc<dyn InferenceTransport>> {
    let cfg = &settings.inference;
    if cfg.url.trim().is_empty() {
        return None;
    }

    let breaker = Arc::new(CircuitBreaker::new(BreakerConfig {
        failure_threshold: 5,
        cooldown: Duration::from_secs(10),
    }));
    let http = match HttpTransport::connect(
        &cfg.url,
        HttpConfig {
            timeout: Duration::from_millis(cfg.timeout_ms),
            max_retries: 1,
            retry_backoff: Duration::from_millis(10),
            auth_token: cfg.auth_token.clone(),
        },
        Some(breaker),
    )
    .await
    {
        Ok(http) => http,
        Err(e) => {
            tracing::warn!(
                url = %cfg.url,
                error = %e,
                "ARMOR_INFERENCE_URL failed validation — starting rules-only; \
                 the control-plane UI will report the sidecar as unreachable \
                 until this is fixed and the process is restarted"
            );
            return None;
        }
    };

    tracing::info!(
        url = %cfg.url,
        timeout_ms = cfg.timeout_ms,
        budget_ms = cfg.budget_ms,
        cache_size = cfg.cache_size,
        "inference sidecar enabled"
    );

    Some(Arc::new(CachingTransport::new(
        Arc::new(http),
        cfg.cache_size,
    )))
}

/// Connects to the control-plane Postgres database, seeds it from the file
/// default policy on an empty first boot, then makes it authoritative:
/// loads every profile/application row and atomically swaps them into
/// `live_resolver`, overriding whatever `load_data_plane` (or
/// `control_plane_stub_resolver`) put there. Only called when
/// `settings.database_url` is non-empty and `settings.mode != Edge`
/// (`main`, below) — `/ui`'s Edge-mode gap means there's no CRUD surface to
/// back there either. Fails the boot on any error: a `DATABASE_URL` that's
/// set but broken should stop the deploy, not silently fall back to file
/// profiles the operator no longer expects to be in effect (README.md's
/// "DB is authoritative once configured").
async fn wire_database(
    settings: &Settings,
    live_resolver: &LiveResolver,
) -> anyhow::Result<Arc<PgPolicyStore>> {
    let store = PgPolicyStore::connect(&settings.database_url)
        .await
        .context("connecting to control-plane database (DATABASE_URL)")?;

    if store
        .is_empty()
        .await
        .context("checking control-plane database state")?
    {
        let default_policy = load_default_policy(settings)
            .context("loading default policy to seed the control-plane database")?;
        let policy_id = default_policy.id.clone();
        store
            .seed_default(&default_policy)
            .await
            .context("seeding control-plane database from the default policy")?;
        tracing::info!(
            policy_id = %policy_id,
            "seeded control-plane database from the default policy (first boot, profiles table was empty)"
        );
    }

    let (db_profiles, db_applications) = store
        .load_all_policies()
        .await
        .context("loading profiles from the control-plane database")?;
    let profile_count = db_profiles.len();
    let pin_rows = armor_storage::inference_pins::list_all(store.pool())
        .await
        .context("loading inference pins from the control-plane database")?;
    let pins = profiles::pins_from_rows(pin_rows);
    let resolver = profiles::resolver_from_policies(
        db_profiles,
        db_applications,
        Path::new(&settings.custom_rules_dir),
        &pins,
    )
    .context("building profile resolver from the control-plane database")?;
    live_resolver.swap(resolver);
    tracing::info!(
        profile_count,
        "control-plane database is now authoritative for profiles/applications"
    );

    Ok(Arc::new(store))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Force every embedded ruleset to parse/compile now: a malformed
    // rules.yaml shipped in the binary should fail the deploy at boot, not
    // surface as a panic on the first request that happens to exercise that
    // detector. Independent of settings/policy, so it runs before either —
    // and before `otel::init`, so this can only log to stderr, not the
    // configured tracing backend.
    if let Err(errors) = armor_core::detectors::validate_all_rules() {
        for error in &errors {
            eprintln!("rule bank failed to compile: {error}");
        }
        anyhow::bail!(
            "{} detector rule bank(s) failed to compile — refusing to start",
            errors.len()
        );
    }

    let settings = Settings::from_env();
    // Must run before any `tracing::*!` call — sets up fmt plus whichever
    // OTLP signal layers `settings.otel` turned on.
    let otel_guard = otel::init(&settings.otel)?;

    tracing::info!(mode = ?settings.mode, "armor mode configured");

    let broker_state = broker_state::track_process_start(Path::new(&settings.state_dir));
    tracing::info!(
        broker_id = %broker_state.broker_id,
        run_count = broker_state.run_count,
        "broker state recorded"
    );

    // `ArmorMode::ControlPlane` has no `/api/*` routes to protect, so it
    // skips policy/profiles/auth/rate-limit/sync entirely rather than
    // loading (and periodically re-syncing) rules nothing will ever
    // evaluate against.
    let (profiles, api_keys, rate_limiter, sync_task, detector_count) = match settings.mode {
        ArmorMode::ControlPlane => {
            tracing::info!(
                "control_plane mode: no /api/* routes mounted, skipping policy/profiles/sync wiring"
            );
            (control_plane_stub_resolver(), None, None, None, 0)
        }
        ArmorMode::Standalone | ArmorMode::Edge => {
            let data_plane = load_data_plane(&settings).await?;
            (
                data_plane.profiles,
                data_plane.api_keys,
                data_plane.rate_limiter,
                Some(data_plane.sync_task),
                data_plane.detector_count,
            )
        }
    };

    // Neither `/ui` nor its `/api/v1` control-plane CRUD surface is mounted
    // in Edge mode at all (`routes::router`) — thin/stateless deployments
    // stay Postgres-free — so there's nothing to back there either; skip
    // connecting even if `DATABASE_URL` happens to be set.
    let db = if !settings.database_url.trim().is_empty() && settings.mode != ArmorMode::Edge {
        Some(wire_database(&settings, &profiles).await?)
    } else {
        None
    };

    let vault = wire_vault(&settings, db.as_ref())?;

    // Purges rows past their `ARMOR_SESSION_TTL_SECONDS`/`ARMOR_VAULT_TTL_SECONDS`
    // expiry (state.rs's `session_ttl_seconds` doc comment) — a no-op when
    // `db` is `None`, since there's no `sessions`/`vault_entries` table to
    // sweep.
    let retention_task = RetentionTask::spawn(db.clone(), vault.clone());

    let inference = wire_inference(&settings).await;
    let inference_budget_ms = settings.inference.budget_ms;

    let telemetry = Arc::new(TelemetryEmitter::new(
        settings.telemetry.enabled,
        settings.telemetry.endpoint.clone(),
        settings.telemetry.api_key.clone(),
    ));
    let telemetry_handle = telemetry.clone().spawn();

    let heartbeat = Arc::new(Heartbeat::new(
        settings.heartbeat.enabled,
        settings.heartbeat.endpoint.clone(),
        broker_state.broker_id.clone(),
        detector_count,
    ));
    let heartbeat_handle = heartbeat.clone().spawn();
    // Fires once, only on the broker's actual first startup (never again,
    // even across restarts) — best-effort, must never block startup.
    if broker_state.run_count == 1 {
        heartbeat.ping_on_install().await;
    }

    let audit_sink: Arc<dyn audit::AuditSink> =
        audit::build_audit_sink(&settings.audit, db.clone()).into();
    tracing::info!(
        audit_sink_mode = ?settings.audit.mode,
        spool_path = %settings.audit.spool_path,
        db_backed = db.is_some(),
        "audit sink configured"
    );

    let state = AppState {
        profiles,
        api_keys,
        rate_limiter,
        telemetry: telemetry.clone(),
        audit_sink,
        heartbeat: heartbeat.clone(),
        db,
        custom_rules_dir: Arc::from(settings.custom_rules_dir.as_str()),
        session_ttl_seconds: settings.session_ttl_seconds,
        vault,
        inference,
        inference_budget_ms,
        inference_url: if settings.inference.url.trim().is_empty() {
            None
        } else {
            Some(settings.inference.url.clone())
        },
        inference_auth_token: settings.inference.auth_token.clone(),
        inference_token_file: settings.inference.token_file.as_str().into(),
    };
    let app = routes::router(state, &settings);

    tracing::info!(addr = %settings.bind_addr, mode = ?settings.mode, "starting armor-api");
    let listener = tokio::net::TcpListener::bind(&settings.bind_addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    // Stop the background emitters now that the server has stopped
    // accepting connections — telemetry does a final flush; heartbeat just
    // cancels its sleep loop. Neither is durable (that's the audit spool's
    // job), so this is best-effort, same as everything else about them.
    telemetry.stop(telemetry_handle).await;
    heartbeat.stop(heartbeat_handle).await;
    if let Some(sync_task) = sync_task {
        sync_task.stop().await;
    }
    retention_task.stop().await;

    // Flush any buffered OTLP batches now that the server has stopped
    // accepting connections — skipping this drops up to one export
    // interval's worth of spans/logs/metrics on every clean shutdown.
    otel_guard.shutdown();

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
