//! Shared application state. Holds a `LiveResolver` — an atomically
//! hot-swappable wrapper around `ProfileResolver` — so control-plane rule
//! syncs (and `/api/v1` profile/application writes — `control_plane.rs`)
//! never stall in-flight requests (`sync.rs`). Optional auth/rate-limit
//! configuration is wired up in `middleware`. The optional `armor-inference`
//! sidecar hop hangs off `inference` — `None` means the tier is
//! off and every request runs the deterministic path exactly as it always
//! has.

use std::sync::Arc;

use armor_inference_client::transport::InferenceTransport;
use armor_storage::{policy_store::PgPolicyStore, vault::Vault};

use crate::{
    audit::AuditSink, heartbeat::Heartbeat, middleware::rate_limit::RateLimiter,
    sync::LiveResolver, telemetry::TelemetryEmitter,
};

#[derive(Clone)]
pub struct AppState {
    /// Atomically hot-swappable profile resolver — see `sync::LiveResolver`.
    pub profiles: LiveResolver,
    /// `None` when `ARMOR_AUTH_MODE` is `none` (default) — auth middleware
    /// passes every request through. `Some` holds the valid key set as SHA-256 digests.
    pub api_keys: Option<Arc<Vec<[u8; 32]>>>,
    /// `None` when `ARMOR_RATE_LIMIT_MODE` is `none` (default) — rate-limit
    /// middleware passes every request through.
    pub rate_limiter: Option<Arc<RateLimiter>>,
    /// Batched control-plane telemetry — off unless `ARMOR_TELEMETRY_ENABLED`.
    pub telemetry: Arc<TelemetryEmitter>,
    /// Per-request decision log (`audit.rs`) — local durable spool by
    /// default, plus a Postgres sink fanned in when `db` is `Some`.
    pub audit_sink: Arc<dyn AuditSink>,
    /// Anonymous daily install ping — off unless `ARMOR_HEARTBEAT_ENABLED`.
    pub heartbeat: Arc<Heartbeat>,
    /// `Some` when `DATABASE_URL` is set (and `mode != Edge`) — backs the
    /// `/ui` management UI and its `/api/v1` CRUD routes (`control_plane.rs`).
    /// `None` means `/ui` stays the `ui_stub` 501 (when the UI is enabled)
    /// and the CRUD routes aren't mounted at all, same as before this
    /// feature existed.
    pub db: Option<Arc<PgPolicyStore>>,
    /// Needed to re-harden DB-sourced profiles the same way file-based ones
    /// are hardened at boot (`profiles::harden`) — used by
    /// `control_plane.rs`'s post-mutation resolver rebuild.
    pub custom_rules_dir: Arc<str>,
    /// Retention for durable session rows (`ARMOR_SESSION_TTL_SECONDS`).
    /// `None` — the default — means sessions never expire on their own and
    /// must be erased deliberately; see `session_state`. When set, expired
    /// rows are swept by `retention::RetentionTask`, which also covers
    /// `vault_entries` via `ON DELETE CASCADE` (and the vault's own,
    /// possibly-shorter `ARMOR_VAULT_TTL_SECONDS`).
    pub session_ttl_seconds: Option<i64>,
    /// Reversible-anonymization vault — `Some` only when both
    /// `DATABASE_URL` and `ARMOR_VAULT_KEY` are set. `None` means redaction
    /// stays redact-and-discard, which is the default and what every
    /// deployment did before this existed. See `redaction.rs` for when a
    /// span actually reaches it.
    pub vault: Option<Arc<Vault>>,
    /// The `armor-inference` sidecar hop — `Some` only when
    /// `ARMOR_INFERENCE_URL` is set (`main::wire_inference`). `None` means
    /// `ml::escalate` returns immediately and the request path is unchanged:
    /// the same "feature is `None`-able end to end" posture as `db` and
    /// `vault`. Holds the transport with the circuit breaker and client-side
    /// cache already wrapped in; the breaker outlives every request, which is
    /// the only arrangement in which "five failures in a row" is a statement
    /// about the endpoint rather than about one request.
    pub inference: Option<Arc<dyn InferenceTransport>>,
    /// Whole escalation-pass budget, from `ARMOR_INFERENCE_BUDGET_MS` —
    /// applied on top of each call's own deadline (`ml::escalate`).
    pub inference_budget_ms: u64,
    /// Raw sidecar URL for control-plane proxy calls (install, models UI).
    /// `None` when the tier is off.
    pub inference_url: Option<String>,
    /// Bearer token for the sidecar, if configured (`ARMOR_INFERENCE_AUTH_TOKEN`).
    /// Prefer `resolve_inference_token` over reading this directly — it also
    /// falls back to `inference_token_file`.
    pub inference_auth_token: Option<String>,
    /// Fallback path for the sidecar's own auto-generated mutation token —
    /// see `config::InferenceConfig::token_file`'s doc comment for the full
    /// story. Always populated (defaults to `/var/run/armor/inference-token`
    /// like the sidecar side does), even though it usually resolves nothing
    /// on a bare, non-compose run.
    pub inference_token_file: Arc<str>,
}

impl AppState {
    /// The token to present to the sidecar for install/reload (and, if the
    /// sidecar has its own `ARMOR_INFERENCE_AUTH_TOKEN` set, everything else
    /// under `/v1`) — an explicit `inference_auth_token` always wins; only
    /// absent that does this try `inference_token_file`, which the sidecar
    /// writes only when *it* has no configured token either (`main.py`'s
    /// `lifespan`). Re-read on every call rather than cached at boot: the
    /// sidecar mints a fresh token each time it restarts, so this lets
    /// armor-core notice without needing a restart of its own.
    pub(crate) async fn resolve_inference_token(&self) -> Option<String> {
        if let Some(token) = &self.inference_auth_token {
            return Some(token.clone());
        }
        let contents = tokio::fs::read_to_string(&*self.inference_token_file)
            .await
            .ok()?;
        let token = contents.trim();
        (!token.is_empty()).then(|| token.to_string())
    }
}
