//! Env-based settings, all resolved once at boot into a single `Settings`.

use std::env;

use ipnet::IpNet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// No auth check — every request to `/api/v1/aidr/scan` and
    /// `/integrations/*/v1/aidr/scan` is accepted. Default.
    None,
    /// Require a valid key via `Authorization: Bearer <key>` or `X-API-Key`.
    ApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitMode {
    /// No rate limiting. Default.
    None,
    /// Per-client-IP token bucket, in-process (single-instance; see
    /// `middleware::rate_limit`'s header for why this doesn't scale past
    /// one replica and what the Redis-backed mode below does about it).
    Fixed,
    /// Same per-client-IP token bucket, but the bucket state lives in Redis
    /// (`ARMOR_REDIS_URL`) behind an atomic Lua script instead of an
    /// in-process `LruCache` — every replica behind a load balancer shares
    /// one limit per client instead of each enforcing its own. See
    /// `middleware::redis_rate_limit`.
    Redis,
}

/// Set via `ARMOR_ENV` (default `development`). Gates the checks that are
/// only worth enforcing once there's a real deployment behind this binary:
/// HSTS, and requiring `https://` origins in `ARMOR_ALLOWED_ORIGINS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Environment {
    Development,
    Production,
}

/// Set via `ARMOR_MODE` (default `standalone`) — which routes this process
/// mounts. One binary, three shapes, so an OSS user never has to stand up a
/// Data Plane container, a Control Plane container, and Postgres just to
/// try the tool: they run one binary in `standalone` mode and get both.
///
/// `routes::router` gates mounting on this; `main.rs` gates which of the
/// policy/profiles/sync/auth/rate-limit wiring even needs to run — a
/// `control_plane` process has no data plane to protect, so none of that
/// setup happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmorMode {
    /// Data Plane (`/api/*`) + Control Plane UI (`/ui/*`) in the same
    /// process. Default — the friction-free single-binary OSS experience.
    Standalone,
    /// Data Plane only: `/api/*` + `/healthz`/`/readyz`. No UI, no Postgres
    /// — meant for thin, stateless deployments (edge/serverless) that pull
    /// their rules from `ARMOR_SYNC_URL` (`sync.rs`).
    Edge,
    /// Control Plane only: `/ui/*` + `/healthz`/`/readyz`. No `/api/*` data
    /// plane routes, no policy/profiles/sync wiring. `/ui/*` currently
    /// responds `501` — the management UI and Postgres-backed policy store
    /// (`crates/storage`) are still stubs; this mode exists so the
    /// deployment topology and routing seam are already in place when they
    /// land.
    ControlPlane,
}

/// Which OTLP signals this instance exports, resolved independently per
/// signal — see `Settings::from_env`'s doc comment on the resolution rule.
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub service_name: String,
    pub traces_enabled: bool,
    pub metrics_enabled: bool,
    pub logs_enabled: bool,
}

/// Decision-audit backend (`audit.rs`). "Spool" is a durable local
/// JSON-lines file — the default, since it's local-only and never phones
/// home. "Noop" is an explicit opt-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditSinkMode {
    Spool,
    Noop,
}

/// Batched, metadata-only evaluation events shipped to a control plane
/// (`telemetry.rs`). Off by default — this product never phones home
/// unless explicitly configured.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub api_key: String,
}

/// Anonymous daily install ping (`heartbeat.rs`). Off by default.
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    pub enabled: bool,
    pub endpoint: String,
}

/// Redis connection for the `RateLimitMode::Redis` backend
/// (`middleware::redis_rate_limit`). Only consulted when
/// `rate_limit_mode == Redis` — an empty `url` in any other mode is inert.
#[derive(Debug, Clone)]
pub struct RedisConfig {
    /// e.g. `redis://redis:6379`. Empty means unconfigured; `Settings::from_env`
    /// panics at boot if `rate_limit_mode == Redis` and this is empty, same
    /// fail-fast posture as the `ARMOR_AUTH_MODE=api_key` check below.
    pub url: String,
    /// Prefix on every rate-limit key this process writes
    /// (`ARMOR_REDIS_KEY_PREFIX`, default `armor:ratelimit:`) — lets one
    /// Redis instance be shared with other data safely, and lets the
    /// bucket-refill Lua script's TTL sweep only its own keys.
    pub key_prefix: String,
}

/// Background rule / profile sync from a Control Plane or local reload.
/// Disabled when `sync_url` is empty — the embedded rules are used as-is.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Full URL of the control plane sync endpoint,
    /// e.g. `http://localhost:9000/v1/internal/sync`.
    pub url: String,
    /// Bearer token sent as `Authorization: Bearer <token>` to the sync
    /// endpoint. May be empty when the endpoint is on a trusted network.
    pub token: String,
    /// How often the background task polls. Defaults to 60 s.
    pub interval_secs: u64,
    pub enabled: bool,
}

/// Where and how the per-request decision log (`audit.rs`) is written.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub mode: AuditSinkMode,
    pub spool_path: String,
    pub max_size_bytes: u64,
}

/// The optional `armor-inference` sidecar hop. An empty `url`
/// means the tier is off: `state.inference` is `None`, `ml::escalate` is a
/// no-op, and every request runs the deterministic path exactly as it always
/// has — the "no strategy ⇒ byte-identical" property extends to "no URL ⇒
/// byte-identical".
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Base URL of the sidecar, e.g. `http://inference:9000`
    /// (`ARMOR_INFERENCE_URL`). Empty ⇒ tier disabled.
    pub url: String,
    /// Deadline for **one** inference call, retries included
    /// (`ARMOR_INFERENCE_TIMEOUT_MS`). Default 120 ms — ~2× the estimate
    /// for an INT8 256-token forward pass.
    pub timeout_ms: u64,
    /// Whole escalation-pass budget (`ARMOR_INFERENCE_BUDGET_MS`), applied on
    /// top of each call's own deadline — a slow sidecar degrades the request,
    /// never hangs it. Default 250 ms.
    pub budget_ms: u64,
    /// Client-side result-cache entries (`ARMOR_INFERENCE_CACHE_SIZE`),
    /// keyed on the exact scored text. Default 4096.
    pub cache_size: usize,
    /// Sent as `Authorization: Bearer` when the sidecar requires it
    /// (`ARMOR_INFERENCE_AUTH_TOKEN`). Explicit and always wins — see
    /// `token_file` below for the fallback when this is unset.
    pub auth_token: Option<String>,
    /// Fallback path (`ARMOR_INFERENCE_TOKEN_FILE`) `control_plane.rs`'s
    /// `resolve_inference_token` reads when `auth_token` above is unset —
    /// the sidecar (`inference/src/armor_inference/main.py`'s `lifespan`)
    /// writes its own auto-generated mutation token to the same path on its
    /// side of a volume `docker-compose.yml` mounts into both containers,
    /// so `POST /api/v1/models/install` works with zero shared config
    /// regardless of whether the stack was started via `make ml-up` or
    /// plain `docker compose --profile ml up`. A bare, non-compose run has
    /// nothing mounted here, so the read below just fails and this is a
    /// no-op — same as before this existed.
    pub token_file: String,
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub mode: ArmorMode,
    pub bind_addr: String,
    pub policy_path: String,
    /// Directory of per-category custom-rule YAML files, folded into every
    /// loaded policy's (default profile's, and every named profile's)
    /// check `options` at startup (`custom_rules::apply`). A missing
    /// directory just means the feature is unused — unlike `policy_path`,
    /// this has no hard requirement to exist.
    pub custom_rules_dir: String,
    /// Directory of named-profile policy YAML files (same schema as
    /// `policy_path`, one file per profile, each with its own top-level
    /// `id`) — see `profiles.rs`. A missing directory means no named
    /// profiles exist; every request resolves to the default profile, i.e.
    /// the original single-policy behavior unchanged.
    pub profiles_dir: String,
    /// Path to the `application_id -> profile_id` mapping file — see
    /// `profiles.rs`. A missing file means no application is mapped to a
    /// named profile, same fallback-to-default effect as an empty
    /// `profiles_dir`.
    pub applications_path: String,
    pub environment: Environment,
    pub auth_mode: AuthMode,
    pub api_keys: Vec<String>,
    pub rate_limit_mode: RateLimitMode,
    pub rate_limit_rps: u32,
    pub rate_limit_burst: u32,
    /// Upstream proxies/load balancers allowed to hand the rate limiter a
    /// client IP via `X-Forwarded-For` — CIDR blocks (e.g. `10.0.0.0/8`).
    /// Empty (default) means nothing is trusted and the raw TCP peer
    /// address is always used. Strict opt-in allowlist, never inferred:
    /// without an entry matching the actual peer, a direct caller can't
    /// spoof this header to dodge its own bucket (see
    /// `middleware::rate_limit`).
    pub trusted_proxies: Vec<IpNet>,
    /// Redis connection for `rate_limit_mode == Redis`. Empty `url` in any
    /// other mode.
    pub redis: RedisConfig,
    /// CORS is off (no `Access-Control-*` headers at all) when empty —
    /// browsers already can't read cross-origin responses without them, so
    /// this is a strict opt-in list, never a wildcard (rejected below).
    pub cors_allowed_origins: Vec<String>,
    pub otel: ObservabilityConfig,
    /// Home for `broker_state.rs`'s `state.json` and, by default, the
    /// audit spool — override with `ARMOR_STATE_DIR` (e.g. in read-only or
    /// multi-tenant deployments where `$HOME` isn't writable/appropriate).
    pub state_dir: String,
    pub telemetry: TelemetryConfig,
    pub heartbeat: HeartbeatConfig,
    pub audit: AuditConfig,
    pub max_body_bytes: usize,
    pub sync: SyncConfig,
    /// The optional `armor-inference` sidecar hop. `url` empty
    /// (the default) means the tier is off and the request path is unchanged.
    pub inference: InferenceConfig,
    /// Postgres connection string for the control-plane DB (profiles,
    /// applications, evaluation logs — `armor_storage::policy_store`).
    /// Empty (default) means the feature is off: `/ui` stays a stub, and
    /// profiles/applications resolve purely from the file-based
    /// `profiles_dir`/`applications_path` exactly as they always have. Not
    /// consulted at all when `mode == ArmorMode::Edge` (no `/ui` there
    /// either). See `main.rs`'s DB wiring and `README.md`'s "Profiles &
    /// applications" section.
    pub database_url: String,
    /// `ARMOR_UI_ENABLED` (default `true`) — whether the browser management
    /// UI (`/ui`, static HTML/JS/CSS) is mounted. The UI is a quick-testing
    /// convenience layer, not the only way in: its control-plane CRUD API
    /// (profiles, logs, stats, models, hardware — `control_plane.rs`) is
    /// mounted at `/api/v1` and stays reachable regardless of this flag, as
    /// long as `database_url` is set and `mode != ArmorMode::Edge`. Set to
    /// `false` to run headless (automation/API-only) without exposing the
    /// browser UI at all.
    pub ui_enabled: bool,
    /// Retention for durable session rows, in seconds
    /// (`ARMOR_SESSION_TTL_SECONDS`). `0` (the default) means no expiry:
    /// sessions persist until erased deliberately. Set a real value when
    /// the vault is in use — session rows cascade to `vault_entries`, so
    /// this is also the retention window on stored PII
    /// (`armor_storage::sessions::purge_expired`).
    pub session_ttl_seconds: Option<i64>,
    /// Base64 of 32 bytes (`ARMOR_VAULT_KEY`), enabling the
    /// reversible-anonymization vault (`armor_storage::vault`).
    /// Empty (default) means redaction stays redact-and-discard: spans are
    /// masked in `redacted_text` and the originals are gone.
    ///
    /// Setting this is **not** on its own enough to store anything. A span
    /// is only vaulted when the policy asked for it — a check configured
    /// `on_fail: redact` — so a deployment that turns the key on without
    /// changing its policy stores nothing. Generate one with
    /// `armor_storage::vault::EnvKeyProvider::generate_base64`, and read
    /// that module's threat model before turning it on: the vault holds
    /// recoverable PII.
    ///
    /// Ignored (with a warning at startup) when `database_url` is empty —
    /// the vault is Postgres-backed and has nowhere to put anything.
    pub vault_key: String,
    /// Retention for individual vault entries, in seconds
    /// (`ARMOR_VAULT_TTL_SECONDS`). `0` (the default) means entries live
    /// until their session expires or is erased. Independent of
    /// `session_ttl_seconds`, and usually shorter: it's how long a
    /// placeholder stays *resolvable*, which is a narrower question than
    /// how long the conversation itself stays alive.
    pub vault_ttl_seconds: Option<i64>,
}

impl Settings {
    pub fn from_env() -> Self {
        let mode = match env::var("ARMOR_MODE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "edge" => ArmorMode::Edge,
            "control_plane" | "controlplane" | "control-plane" => ArmorMode::ControlPlane,
            _ => ArmorMode::Standalone,
        };

        let auth_mode = match env::var("ARMOR_AUTH_MODE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "api_key" | "apikey" => AuthMode::ApiKey,
            _ => AuthMode::None,
        };

        let api_keys: Vec<String> = env::var("ARMOR_API_KEYS")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        if auth_mode == AuthMode::ApiKey && api_keys.is_empty() {
            panic!(
                "ARMOR_AUTH_MODE=api_key requires at least one key in ARMOR_API_KEYS (comma-separated)"
            );
        }

        let rate_limit_mode = match env::var("ARMOR_RATE_LIMIT_MODE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "fixed" | "enabled" | "token_bucket" => RateLimitMode::Fixed,
            "redis" => RateLimitMode::Redis,
            _ => RateLimitMode::None,
        };

        let redis = RedisConfig {
            url: env::var("ARMOR_REDIS_URL").unwrap_or_default(),
            key_prefix: {
                let configured = env::var("ARMOR_REDIS_KEY_PREFIX").unwrap_or_default();
                if configured.trim().is_empty() {
                    "armor:ratelimit:".to_string()
                } else {
                    configured
                }
            },
        };

        if rate_limit_mode == RateLimitMode::Redis && redis.url.trim().is_empty() {
            panic!(
                "ARMOR_RATE_LIMIT_MODE=redis requires ARMOR_REDIS_URL (e.g. redis://redis:6379)"
            );
        }

        // Same fail-fast posture as the panics elsewhere in this file: a
        // typo'd CIDR should refuse to start, not silently trust nothing
        // (or, worse, fail to parse and get swallowed).
        let trusted_proxies: Vec<IpNet> = env::var("ARMOR_TRUSTED_PROXIES")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse().unwrap_or_else(|_| {
                    panic!(
                        "ARMOR_TRUSTED_PROXIES entry {s:?} is not a valid CIDR (e.g. 10.0.0.0/8)"
                    )
                })
            })
            .collect();

        let environment = match env::var("ARMOR_ENV")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "production" | "prod" => Environment::Production,
            _ => Environment::Development,
        };

        let cors_allowed_origins: Vec<String> = env::var("ARMOR_ALLOWED_ORIGINS")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        // Fail fast at startup rather than silently misbehaving at request
        // time — same posture as the ARMOR_AUTH_MODE panic above.
        for origin in &cors_allowed_origins {
            if origin == "*" {
                panic!(
                    "ARMOR_ALLOWED_ORIGINS may not contain '*' — list explicit origins (browsers reject wildcard + credentials anyway)"
                );
            }
            let is_local =
                origin.starts_with("http://localhost") || origin.starts_with("http://127.0.0.1");
            if environment == Environment::Production
                && !origin.starts_with("https://")
                && !is_local
            {
                panic!(
                    "ARMOR_ALLOWED_ORIGINS entry '{origin}' must use https:// in production (ARMOR_ENV=production)"
                );
            }
        }

        let sync_url = env::var("ARMOR_SYNC_URL").unwrap_or_default();
        let sync_enabled = !sync_url.trim().is_empty();

        // Same fail-fast posture as the CORS check above: this endpoint's
        // response hot-swaps every enabled/disabled check for every profile
        // (`sync.rs`), so a plain-http sync URL in production is a MITM's
        // path to silently controlling the guardrails, not just a
        // cosmetic gap.
        if sync_enabled && environment == Environment::Production {
            let is_local = sync_url.starts_with("http://localhost")
                || sync_url.starts_with("http://127.0.0.1");
            if !sync_url.starts_with("https://") && !is_local {
                panic!(
                    "ARMOR_SYNC_URL '{sync_url}' must use https:// in production (ARMOR_ENV=production) — this endpoint controls which guardrails are enabled"
                );
            }
        }

        let state_dir = env::var("ARMOR_STATE_DIR").unwrap_or_else(|_| default_state_dir());

        let audit_mode = match env::var("ARMOR_AUDIT_SINK_MODE")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "noop" | "none" | "off" => AuditSinkMode::Noop,
            _ => AuditSinkMode::Spool,
        };
        let audit_spool_path = {
            let configured = env::var("ARMOR_AUDIT_SPOOL_PATH").unwrap_or_default();
            if configured.trim().is_empty() {
                format!("{state_dir}/audit.spool")
            } else {
                configured
            }
        };

        Self {
            mode,
            bind_addr: env::var("ARMOR_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string()),
            policy_path: env::var("ARMOR_POLICY_PATH")
                .unwrap_or_else(|_| "config/policies.yaml".to_string()),
            custom_rules_dir: env::var("ARMOR_CUSTOM_RULES_DIR")
                .unwrap_or_else(|_| "config/custom_rules".to_string()),
            profiles_dir: env::var("ARMOR_PROFILES_DIR")
                .unwrap_or_else(|_| "config/profiles".to_string()),
            applications_path: env::var("ARMOR_APPLICATIONS_PATH")
                .unwrap_or_else(|_| "config/applications.yaml".to_string()),
            environment,
            auth_mode,
            api_keys,
            rate_limit_mode,
            rate_limit_rps: parse_env_or("ARMOR_RATE_LIMIT_RPS", 10),
            rate_limit_burst: parse_env_or("ARMOR_RATE_LIMIT_BURST", 20),
            trusted_proxies,
            redis,
            cors_allowed_origins,
            otel: ObservabilityConfig {
                service_name: env::var("OTEL_SERVICE_NAME")
                    .unwrap_or_else(|_| "armor-api".to_string()),
                traces_enabled: otlp_signal_enabled("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
                metrics_enabled: otlp_signal_enabled("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"),
                logs_enabled: otlp_signal_enabled("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"),
            },
            state_dir,
            telemetry: TelemetryConfig {
                // Opt-in must be affirmative: only an explicit "true" turns
                // telemetry on. Actual effect is further gated on the
                // endpoint being set — see `telemetry::TelemetryEmitter::new`.
                enabled: parse_env_bool("ARMOR_TELEMETRY_ENABLED"),
                endpoint: env::var("ARMOR_TELEMETRY_URL").unwrap_or_default(),
                api_key: env::var("ARMOR_TELEMETRY_API_KEY").unwrap_or_default(),
            },
            heartbeat: HeartbeatConfig {
                enabled: parse_env_bool("ARMOR_HEARTBEAT_ENABLED"),
                endpoint: env::var("ARMOR_HEARTBEAT_URL").unwrap_or_default(),
            },
            audit: AuditConfig {
                mode: audit_mode,
                spool_path: audit_spool_path,
                max_size_bytes: parse_env_or_u64("ARMOR_AUDIT_MAX_SIZE_BYTES", 100 * 1024 * 1024),
            },
            max_body_bytes: parse_env_or("ARMOR_MAX_BODY_BYTES", 2 * 1024 * 1024) as usize,
            sync: SyncConfig {
                enabled: sync_enabled,
                url: sync_url,
                token: env::var("ARMOR_SYNC_TOKEN").unwrap_or_default(),
                interval_secs: parse_env_or_u64("ARMOR_SYNC_INTERVAL_SECS", 60),
            },
            inference: InferenceConfig {
                url: env::var("ARMOR_INFERENCE_URL").unwrap_or_default(),
                timeout_ms: parse_env_or_u64("ARMOR_INFERENCE_TIMEOUT_MS", 120),
                budget_ms: parse_env_or_u64("ARMOR_INFERENCE_BUDGET_MS", 250),
                cache_size: parse_env_or("ARMOR_INFERENCE_CACHE_SIZE", 4096) as usize,
                auth_token: match env::var("ARMOR_INFERENCE_AUTH_TOKEN") {
                    Ok(token) if !token.trim().is_empty() => Some(token),
                    _ => None,
                },
                token_file: {
                    let configured = env::var("ARMOR_INFERENCE_TOKEN_FILE").unwrap_or_default();
                    if configured.trim().is_empty() {
                        "/var/run/armor/inference-token".to_string()
                    } else {
                        configured
                    }
                },
            },
            database_url: env::var("DATABASE_URL").unwrap_or_default(),
            ui_enabled: parse_env_bool_or("ARMOR_UI_ENABLED", true),
            session_ttl_seconds: match parse_env_or_u64("ARMOR_SESSION_TTL_SECONDS", 0) {
                0 => None,
                secs => Some(secs as i64),
            },
            vault_key: env::var("ARMOR_VAULT_KEY").unwrap_or_default(),
            vault_ttl_seconds: match parse_env_or_u64("ARMOR_VAULT_TTL_SECONDS", 0) {
                0 => None,
                secs => Some(secs as i64),
            },
        }
    }
}

// Unset means "use the default" — fine. Set-but-unparseable ("1O" for "10")
// is a different failure mode: silently falling back would mean a typo'd
// rate limit or body-size cap quietly runs at the default instead of the
// operator's intended value, with nothing in the logs to say so. Same
// fail-fast posture as the `ARMOR_AUTH_MODE`/CORS panics above — refuse to
// start rather than misbehave.
fn parse_env_or(key: &str, default: u32) -> u32 {
    match env::var(key) {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            panic!("{key}={raw:?} is not a valid non-negative integer (default is {default})")
        }),
        Err(_) => default,
    }
}

fn parse_env_or_u64(key: &str, default: u64) -> u64 {
    match env::var(key) {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            panic!("{key}={raw:?} is not a valid non-negative integer (default is {default})")
        }),
        Err(_) => default,
    }
}

/// Only explicit truthy values opt in — empty strings, typos, or "off" all
/// stay disabled. Matches this file's existing `ARMOR_ENV`/`ARMOR_AUTH_MODE`
/// case-insensitive-match convention.
fn parse_env_bool(key: &str) -> bool {
    matches!(
        env::var(key).unwrap_or_default().to_lowercase().as_str(),
        "true" | "1" | "yes" | "on"
    )
}

/// Same truthy/falsy vocabulary as [`parse_env_bool`], but for flags that
/// default to *on* (e.g. `ARMOR_UI_ENABLED`): unset keeps `default`, and only
/// an explicit falsy value ("false"/"0"/"no"/"off") flips it.
fn parse_env_bool_or(key: &str, default: bool) -> bool {
    match env::var(key) {
        Err(_) => default,
        Ok(raw) => match raw.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" => false,
            _ => default,
        },
    }
}

/// `$HOME/.armor`, falling back to `.armor` (relative to the working
/// directory) when `$HOME` isn't set — e.g. some minimal container images.
fn default_state_dir() -> String {
    match env::var("HOME") {
        Ok(home) if !home.trim().is_empty() => format!("{home}/.armor"),
        _ => ".armor".to_string(),
    }
}

/// A signal is enabled iff an OTLP destination was actually configured for
/// it — its own `OTEL_EXPORTER_OTLP_<SIGNAL>_ENDPOINT`, or the generic
/// `OTEL_EXPORTER_OTLP_ENDPOINT` fallback. This mirrors the exact
/// resolution order `opentelemetry-otlp`'s exporter builders use internally
/// (signal-specific env var, then generic), so "is this signal on" and
/// "where does it send data" never disagree — and each of traces/metrics/logs
/// turns on independently, with no separate `ARMOR_OTEL_*_ENABLED` flag to
/// keep in sync. Endpoint/protocol/headers/compression themselves are read
/// directly by the exporter builders in `otel.rs`, not duplicated here.
fn otlp_signal_enabled(signal_endpoint_var: &str) -> bool {
    let set = |var: &str| env::var(var).map(|v| !v.trim().is_empty()).unwrap_or(false);
    set(signal_endpoint_var) || set("OTEL_EXPORTER_OTLP_ENDPOINT")
}
