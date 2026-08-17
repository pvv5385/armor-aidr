//! Hot-reload of profiles and rules via an atomic pointer swap.
//!
//! `LiveResolver` wraps `ProfileResolver` inside an `ArcSwap` so the entire
//! resolver (default profile + all named profiles) can be replaced atomically
//! with zero blocking on the hot evaluation path.
//!
//! # Bootstrap + Sync Model
//! On startup, `main.rs` builds a `LiveResolver` from the profiles and rules
//! already embedded in / loaded at boot time. Requests immediately start
//! being served from these embedded rules — there is no external dependency
//! at boot.
//!
//! If `ARMOR_SYNC_URL` is configured, a background Tokio task (`SyncTask`)
//! polls that endpoint every `ARMOR_SYNC_INTERVAL_SECS` seconds. When the
//! control plane returns a new policy payload, `SyncTask` compiles the new
//! profiles on a blocking thread (keeping regex compilation off the async
//! executor) and then calls `LiveResolver::swap`, which does a single
//! `ArcSwap::store`. The *next* request that calls `LiveResolver::load()`
//! picks up the new profiles without any lock contention.
//!
//! When `ARMOR_SYNC_URL` is empty, the `SyncTask` is a no-op and the binary
//! behaves exactly as it always did — embedded rules, zero external
//! dependencies.
//!
//! # Sync Endpoint Contract
//! `GET <ARMOR_SYNC_URL>` must return `Content-Type: application/json` with
//! a body shaped as:
//! ```json
//! {
//!   "default_policy": { /* PolicyConfig YAML serialised as JSON */ },
//!   "profiles":       [ { /* PolicyConfig */ }, ... ],
//!   "applications":   [ { "application_id": "...", "profile_id": "..." }, ... ]
//! }
//! ```
//! The JSON field names and value shapes mirror the existing YAML schema
//! (`armor_core::policy::schema::PolicyConfig`) so the control plane can
//! simply re-encode the same data it already persists in Postgres.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
    time::Duration,
};

use arc_swap::ArcSwap;
use serde::Deserialize;
use tokio::task::JoinHandle;

use armor_core::policy::schema::PolicyConfig;

use crate::{config::SyncConfig, profiles::ProfileResolver};

// ── Sync payload shape ────────────────────────────────────────────────────────

/// Body returned by the control plane `GET /v1/internal/sync` endpoint.
#[derive(Debug, Deserialize)]
pub struct SyncPayload {
    pub default_policy: PolicyConfig,
    #[serde(default)]
    pub profiles: Vec<PolicyConfig>,
    #[serde(default)]
    pub applications: Vec<ApplicationMapping>,
    #[serde(default)]
    pub pins: Vec<InferencePinMapping>,
}

#[derive(Debug, Deserialize)]
pub struct ApplicationMapping {
    pub application_id: String,
    pub profile_id: String,
}

#[derive(Debug, Deserialize)]
pub struct InferencePinMapping {
    pub task: String,
    pub model_id: String,
    pub revision: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub threshold: Option<f64>,
}

// ── LiveResolver ─────────────────────────────────────────────────────────────

/// A `ProfileResolver` that can be swapped atomically while the server is
/// serving requests. Cloning this type is cheap — it only clones the inner
/// `Arc`.
#[derive(Clone)]
pub struct LiveResolver {
    inner: Arc<ArcSwap<ProfileResolver>>,
}

impl LiveResolver {
    /// Wraps an initial `ProfileResolver` (built from the embedded/boot-time
    /// rules) in the atomic pointer.
    pub fn new(initial: ProfileResolver) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(initial)),
        }
    }

    /// Returns a cheap-to-clone snapshot of the current `ProfileResolver`.
    /// The snapshot is reference-counted; it stays alive until the last clone
    /// is dropped, even if `swap` replaces it between now and then. This
    /// means individual requests always see a *consistent* resolver for their
    /// entire lifetime.
    pub fn load(&self) -> arc_swap::Guard<Arc<ProfileResolver>> {
        self.inner.load()
    }

    /// Replaces the resolver with a newly compiled one. Callers already
    /// holding a snapshot (`load()`) are unaffected — they finish their
    /// request against the old rules; only the *next* `load()` sees the new
    /// ones.
    pub fn swap(&self, next: ProfileResolver) {
        self.inner.store(Arc::new(next));
    }
}

// ── SyncTask ──────────────────────────────────────────────────────────────────

/// Polls the control plane on a timer and hot-swaps the `LiveResolver` when
/// the rules change. Returned by `SyncTask::spawn`; call `stop()` during
/// graceful shutdown.
pub struct SyncTask {
    handle: JoinHandle<()>,
}

impl SyncTask {
    /// Spawns the background poll loop. Returns immediately (does not block).
    /// When `config.enabled` is false, the spawned task exits immediately and
    /// costs nothing.
    pub fn spawn(resolver: LiveResolver, config: SyncConfig, custom_rules_dir: String) -> Self {
        let handle = tokio::spawn(async move {
            if !config.enabled {
                tracing::debug!("rule sync disabled (ARMOR_SYNC_URL not set)");
                return;
            }

            tracing::info!(
                url = %config.url,
                interval_secs = config.interval_secs,
                "rule sync task started"
            );

            let client = reqwest::Client::new();
            let interval = Duration::from_secs(config.interval_secs);

            loop {
                tokio::time::sleep(interval).await;

                match fetch_and_build(&client, &config.url, &config.token, &custom_rules_dir).await
                {
                    Ok(new_resolver) => {
                        log_policy_diff(&resolver.load(), &new_resolver);
                        resolver.swap(new_resolver);
                        tracing::info!(url = %config.url, "rules hot-swapped from control plane");
                    }
                    Err(e) => {
                        // Non-fatal: keep running with the rules we have.
                        tracing::warn!(error = %e, "rule sync fetch failed, retaining current rules");
                    }
                }
            }
        });

        Self { handle }
    }

    /// Cancels the poll loop. Non-blocking.
    pub async fn stop(self) {
        self.handle.abort();
        let _ = self.handle.await;
    }
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Maps each distinct profile's `id` to its set of enabled check
/// categories — the unit `log_policy_diff` compares old vs. new on.
fn enabled_checks_by_profile(resolver: &ProfileResolver) -> HashMap<String, HashSet<String>> {
    resolver
        .all_policies()
        .into_iter()
        .map(|policy| {
            let enabled = policy
                .checks
                .iter()
                .filter(|check| check.enabled)
                .map(|check| check.category.clone())
                .collect();
            (policy.id.clone(), enabled)
        })
        .collect()
}

/// Logs what a swap actually changes: an `info` summary on every sync, plus
/// a loud `warn` naming any check that was enabled before this swap and is
/// disabled (or whose whole profile is gone) after it. The sync endpoint
/// controls which guardrails run at all (module doc above) — a swap that
/// silently turns checks off is exactly the failure mode worth surfacing on
/// every single sync, not just the ones that later cause visible harm.
fn log_policy_diff(old: &ProfileResolver, new: &ProfileResolver) {
    let old_profiles = enabled_checks_by_profile(old);
    let new_profiles = enabled_checks_by_profile(new);

    for (profile_id, old_enabled) in &old_profiles {
        let mut newly_disabled: Vec<&str> = match new_profiles.get(profile_id) {
            Some(new_enabled) => old_enabled
                .difference(new_enabled)
                .map(String::as_str)
                .collect(),
            // The whole profile is gone — every check it had is, in effect,
            // now disabled.
            None => old_enabled.iter().map(String::as_str).collect(),
        };
        if !newly_disabled.is_empty() {
            newly_disabled.sort_unstable();
            tracing::warn!(
                profile_id = %profile_id,
                disabled_checks = ?newly_disabled,
                count = newly_disabled.len(),
                "rule sync: this swap disables checks previously enabled for this profile"
            );
        }
    }

    let old_ids: HashSet<&String> = old_profiles.keys().collect();
    let new_ids: HashSet<&String> = new_profiles.keys().collect();
    let mut removed_profiles: Vec<&str> =
        old_ids.difference(&new_ids).map(|id| id.as_str()).collect();
    if !removed_profiles.is_empty() {
        removed_profiles.sort_unstable();
        tracing::warn!(
            removed_profiles = ?removed_profiles,
            "rule sync: this swap removes profiles present before it"
        );
    }

    tracing::info!(
        profile_count = new_profiles.len(),
        total_enabled_checks = new_profiles.values().map(HashSet::len).sum::<usize>(),
        "rule sync: policy diff summary"
    );
}

async fn fetch_and_build(
    client: &reqwest::Client,
    url: &str,
    token: &str,
    custom_rules_dir: &str,
) -> anyhow::Result<ProfileResolver> {
    let mut req = client.get(url).timeout(Duration::from_secs(10));
    if !token.is_empty() {
        req = req.bearer_auth(token);
    }

    let payload: SyncPayload = req
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("sync GET failed: {e}"))?
        .error_for_status()
        .map_err(|e| anyhow::anyhow!("sync endpoint error: {e}"))?
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("sync payload deserialize failed: {e}"))?;

    // Compile regex-heavy rules off the async runtime.
    let custom_rules_dir = custom_rules_dir.to_string();
    tokio::task::spawn_blocking(move || build_resolver(payload, &custom_rules_dir))
        .await
        .map_err(|e| anyhow::anyhow!("rule compilation panicked: {e}"))?
}

/// Builds a fresh `ProfileResolver` from the control-plane payload.
/// Runs on the blocking thread pool (called via `spawn_blocking`).
fn build_resolver(payload: SyncPayload, custom_rules_dir: &str) -> anyhow::Result<ProfileResolver> {
    use std::collections::HashMap;

    let custom_rules_path = Path::new(custom_rules_dir);

    // Model overrides pushed alongside the policies — applied to every
    // policy's `LocalMl` backends below (`profiles::apply_pins`), the same
    // step `control_plane.rs`'s DB-backed reload does. Without this, an edge
    // instance's sync poll would happily deserialize `payload.pins` and then
    // throw it away, leaving pins that read as configured but do nothing.
    let pins: HashMap<String, crate::profiles::PinOverride> = payload
        .pins
        .into_iter()
        .map(|pin| {
            (
                pin.task,
                crate::profiles::PinOverride {
                    model_id: pin.model_id,
                    revision: pin.revision,
                },
            )
        })
        .collect();

    let mut default = crate::profiles::harden(payload.default_policy, custom_rules_path)?;
    crate::profiles::apply_pins(&mut default, &pins);
    let default = Arc::new(default);

    let mut by_profile_id: HashMap<String, Arc<PolicyConfig>> = HashMap::new();
    by_profile_id.insert(default.id.clone(), default.clone());

    for profile in payload.profiles {
        let mut hardened = crate::profiles::harden(profile, custom_rules_path)?;
        crate::profiles::apply_pins(&mut hardened, &pins);
        by_profile_id.insert(hardened.id.clone(), Arc::new(hardened));
    }

    let mut by_application_id: HashMap<String, Arc<PolicyConfig>> = HashMap::new();
    for mapping in payload.applications {
        let policy = by_profile_id
            .get(&mapping.profile_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "sync payload: application_id {:?} references unknown profile_id {:?}",
                    mapping.application_id,
                    mapping.profile_id,
                )
            })?
            .clone();
        by_application_id.insert(mapping.application_id, policy);
    }

    Ok(ProfileResolver::from_parts(default, by_application_id))
}
