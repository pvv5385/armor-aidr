//! Bridges the durable session store (`armor_storage::sessions`) to the
//! two detectors that need cross-request state, `abuse` and
//! `unbounded_consumption`.
//!
//! # Why the API layer does this and not the detectors
//!
//! `armor-core` is synchronous and I/O-free by hard architectural rule, so
//! a detector cannot query Postgres. Rather than relax that,
//! the lookup is inverted: this module fetches the session's durable
//! counters *before* the sweep and injects them into the relevant
//! `CheckConfig.options`, so the detectors stay pure functions that read a
//! number somebody else looked up. Each detector falls back to its existing
//! in-process counter when no value is injected, so behavior with no
//! database configured is byte-identical to before.
//!
//! # The hot-path exception, stated explicitly
//!
//! The data plane's golden rule: never query a database during request
//! evaluation. This module is a deliberate, narrow exception to it, and
//! the narrowness is what makes it acceptable:
//!
//! - It runs **only** when the resolved profile has `abuse` or
//!   `unbounded_consumption` enabled. Both ship `enabled: false`, so the
//!   default deployment does no database work per request and pays nothing.
//! - Those two checks are *definitionally* cross-request state. Unlike a
//!   rule or a profile — which the golden rule is really about, and which
//!   are cached in memory via `sync::LiveResolver` — a session counter cannot
//!   be preloaded, because it depends on requests that other replicas are
//!   handling right now. The alternative isn't a faster correct answer, it
//!   is the per-replica under-count these checks have today.
//! - A store failure never fails the request: it degrades to the
//!   in-process counter and logs, because a rate limiter that 500s is worse
//!   than one that is temporarily too permissive.
//!
//! Any future check that wants durable state should come through here and
//! extend [`REQUIRES_SESSION_STATE`], so the exception stays enumerable
//! rather than becoming a habit.

use std::sync::Arc;

use armor_core::policy::schema::PolicyConfig;
use armor_storage::sessions::{self, Touch};

use crate::state::AppState;

/// The only checks for which a database round trip is permitted on the scan
/// path. Keep this list short and keep the reason in the module doc.
const REQUIRES_SESSION_STATE: [&str; 2] = ["abuse", "unbounded_consumption"];

/// `abuse`'s window width, read from its own check config so the durable
/// window and the detector's comparison always describe the same window.
const DEFAULT_WINDOW_SECONDS: f64 = 60.0;

/// Whether this policy has any check that needs cross-request state.
///
/// Cheap enough to run per request (a scan over a handful of check configs)
/// and the gate that keeps the database out of the default hot path.
pub fn needs_session_state(policy: &PolicyConfig) -> bool {
    policy
        .checks
        .iter()
        .any(|c| c.enabled && REQUIRES_SESSION_STATE.contains(&c.category.as_str()))
}

/// Record this request against the session and return a policy whose
/// session-stateful checks carry the resulting durable counters.
///
/// Returns the original `policy` untouched when there is no database, when
/// no session-stateful check is enabled, or when the store errors — every
/// path leaves the caller with a usable policy, never an error.
///
/// The clone is deliberate and bounded: `PolicyConfig` is behind an `Arc`
/// shared by every in-flight request, so injecting per-request options
/// requires a private copy. It happens only under the gate above.
pub async fn apply(
    state: &AppState,
    policy: Arc<PolicyConfig>,
    session_id: &str,
    text: &str,
) -> Arc<PolicyConfig> {
    let Some(db) = state.db.as_ref() else {
        return policy;
    };
    if !needs_session_state(&policy) {
        return policy;
    }

    let window_seconds = policy
        .checks
        .iter()
        .find(|c| c.category == "abuse" && c.enabled)
        .and_then(|c| c.options.opt_f64("window_seconds"))
        .unwrap_or(DEFAULT_WINDOW_SECONDS);

    // Same estimate `unbounded_consumption` uses when the caller supplies
    // no token count, so the durable total and the in-process fallback
    // never disagree about what a token is.
    let estimated_tokens = policy
        .checks
        .iter()
        .find(|c| c.category == "unbounded_consumption" && c.enabled)
        .and_then(|c| c.options.opt_f64("estimated_tokens"))
        .unwrap_or((text.len() as f64 / 4.0).round()) as i64;

    let counters = match sessions::touch(
        db.pool(),
        Touch {
            session_id,
            estimated_tokens,
            window_seconds,
            ttl_seconds: state.session_ttl_seconds,
            now: None,
        },
    )
    .await
    {
        Ok(counters) => counters,
        Err(e) => {
            // Deliberately not an error response: see the module doc on
            // degrading to the in-process counter rather than failing the
            // request.
            tracing::warn!(
                error = %e,
                session_id = %session_id,
                "session store unavailable; falling back to in-process counters \
                 for this request (per-replica, so limits are too permissive)"
            );
            return policy;
        }
    };

    let mut injected = (*policy).clone();
    for check in &mut injected.checks {
        if !check.enabled {
            continue;
        }
        match check.category.as_str() {
            "abuse" => {
                check
                    .options
                    .set_f64("session_window_count", counters.window_request_count as f64);
            }
            "unbounded_consumption" => {
                check
                    .options
                    .set_f64("session_request_count", counters.request_count as f64);
                check
                    .options
                    .set_f64("session_total_tokens", counters.total_tokens as f64);
            }
            _ => {}
        }
    }
    Arc::new(injected)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Built from YAML rather than struct literals so these tests exercise
    /// the same deserialization path a real `config/policies.yaml` takes,
    /// including every `#[serde(default)]` on `CheckConfig`.
    fn policy_with(checks: &[(&str, bool)]) -> PolicyConfig {
        let mut yaml = String::from("id: test\nchecks:\n");
        for (category, enabled) in checks {
            yaml.push_str(&format!(
                "  - category: {category}\n    enabled: {enabled}\n"
            ));
        }
        serde_yaml::from_str(&yaml).expect("test policy parses")
    }

    #[test]
    fn a_policy_without_session_checks_needs_no_database() {
        let policy = policy_with(&[("pii", true), ("secrets", true)]);
        assert!(!needs_session_state(&policy));
    }

    #[test]
    fn the_shipped_default_of_disabled_needs_no_database() {
        // Both checks ship `enabled: false` in config/policies.yaml — the
        // reason the default deployment pays nothing for this feature.
        let policy = policy_with(&[("abuse", false), ("unbounded_consumption", false)]);
        assert!(!needs_session_state(&policy));
    }

    #[test]
    fn an_enabled_session_check_needs_the_database() {
        assert!(needs_session_state(&policy_with(&[
            ("pii", true),
            ("abuse", true)
        ])));
        assert!(needs_session_state(&policy_with(&[(
            "unbounded_consumption",
            true
        )])));
    }
}
