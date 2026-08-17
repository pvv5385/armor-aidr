//! Unbounded-consumption check — per-session token/cost/loop-depth budget
//! enforcement.
//!
//! Totals come from one of two places, in priority order — the same split
//! `abuse` uses, and that module's doc explains the reasoning at length:
//!
//! 1. **`options.session_request_count` / `options.session_total_tokens`**
//!    — durable lifetime totals, already including this request, injected
//!    by `armor-api` from `armor_storage::sessions::touch`. Shared across
//!    replicas, so this is the correct source whenever a session store is
//!    configured.
//! 2. **A process-global map** keyed by `options.session_id`, when no
//!    durable totals were supplied. Correct for a single instance, and
//!    per-replica (i.e. too permissive) behind a load balancer.
//!
//! Only with source 2 is this detector not a pure function of its input;
//! with source 1 it is, since somebody else did the accumulating.
//!
//! `loop_depth` is checked directly against `options.loop_depth` (the
//! caller-reported current depth of an agent loop) with no session state
//! needed — a single request can already be too deep.
//!
//! All three budgets (`max_tokens_per_session`, `max_requests_per_session`,
//! `max_loop_depth`) default to unlimited: this check is inert until a
//! deployment configures real numbers, same shipped-inert convention as
//! `keyword_blocklist`/`custom_regex`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

struct SessionUsage {
    total_tokens: f64,
    request_count: u64,
    last_seen: f64,
}

/// Hard cap on how many distinct `session_id`s the in-process fallback table
/// tracks at once — the same bound `abuse`'s window table carries and for
/// the same reason: `session_id` is caller-controlled and this table has no
/// external TTL sweep, so an attacker minting a fresh id per request would
/// otherwise grow it without bound for the life of the process. Unlike
/// `abuse`'s rolling window there is no natural "this entry expired" rule
/// here — a budget total has no rollover — so eviction is pure LRU by
/// `last_seen` rather than prune-then-evict.
const MAX_TRACKED_SESSIONS: usize = 100_000;

fn usage_table() -> &'static Mutex<HashMap<String, SessionUsage>> {
    static TABLE: OnceLock<Mutex<HashMap<String, SessionUsage>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Removes the single least-recently-touched entry, making room for one
/// more insert.
fn evict_oldest(table: &mut HashMap<String, SessionUsage>) {
    if let Some(oldest_key) = table
        .iter()
        .min_by(|a, b| a.1.last_seen.total_cmp(&b.1.last_seen))
        .map(|(k, _)| k.clone())
    {
        table.remove(&oldest_key);
    }
}

fn real_now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let session_id = options.str_option("session_id").unwrap_or("").to_string();
    // ~4 chars/token is a common rough estimate for English text; a real
    // deployment should pass the caller's own token count via
    // `estimated_tokens` rather than relying on this fallback.
    let estimated_tokens = options.f64_option("estimated_tokens", text.len() as f64 / 4.0);
    let loop_depth = options.f64_option("loop_depth", 0.0);
    let max_tokens_per_session = options.f64_option("max_tokens_per_session", f64::INFINITY);
    let max_requests_per_session = options.f64_option("max_requests_per_session", f64::INFINITY);
    let max_loop_depth = options.f64_option("max_loop_depth", f64::INFINITY);
    let now = options.f64_option("now", real_now_seconds());

    let mut hits: Vec<RuleHit> = Vec::new();
    let span = (0, text.len());

    if loop_depth > max_loop_depth {
        hits.push(RuleHit {
            rule_id: "unbounded-consumption-loop-depth-exceeded".to_string(),
            span,
            severity: Severity::High,
        });
    }

    // Durable totals win when supplied; they already include this request,
    // so they are compared as-is rather than accumulated again. Each is
    // taken independently — a deployment that tracks tokens durably but
    // not requests still gets the in-process count for the other.
    let durable_tokens = options.opt_f64("session_total_tokens");
    let durable_requests = options.opt_f64("session_request_count");

    let (total_tokens, request_count) = if durable_tokens.is_some() && durable_requests.is_some() {
        (durable_tokens, durable_requests)
    } else if !session_id.is_empty() {
        let mut table = usage_table().lock().expect("usage table poisoned");
        if !table.contains_key(&session_id) && table.len() >= MAX_TRACKED_SESSIONS {
            evict_oldest(&mut table);
        }
        let entry = table.entry(session_id).or_insert(SessionUsage {
            total_tokens: 0.0,
            request_count: 0,
            last_seen: now,
        });
        entry.total_tokens += estimated_tokens;
        entry.request_count += 1;
        entry.last_seen = now;
        (
            Some(durable_tokens.unwrap_or(entry.total_tokens)),
            Some(durable_requests.unwrap_or(entry.request_count as f64)),
        )
    } else {
        (durable_tokens, durable_requests)
    };

    if total_tokens.is_some_and(|t| t > max_tokens_per_session) {
        hits.push(RuleHit {
            rule_id: "unbounded-consumption-token-budget-exceeded".to_string(),
            span,
            severity: Severity::High,
        });
    }
    if request_count.is_some_and(|c| c > max_requests_per_session) {
        hits.push(RuleHit {
            rule_id: "unbounded-consumption-request-budget-exceeded".to_string(),
            span,
            severity: Severity::High,
        });
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "unbounded_consumption".to_string(),
        action,
        severity: Severity::High,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(session_id: &str, pairs: &[(&str, f64)]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str("session_id", session_id);
        for (k, v) in pairs {
            o.set_f64(k, *v);
        }
        o
    }

    #[test]
    fn unlimited_defaults_never_deny() {
        let result = evaluate("hello world", &opts("session-unlimited", &[]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn evict_oldest_removes_the_single_least_recently_seen_and_nothing_else() {
        let mut table = HashMap::new();
        table.insert(
            "oldest".to_string(),
            SessionUsage {
                total_tokens: 1.0,
                request_count: 1,
                last_seen: 1.0,
            },
        );
        table.insert(
            "middle".to_string(),
            SessionUsage {
                total_tokens: 1.0,
                request_count: 1,
                last_seen: 5.0,
            },
        );
        table.insert(
            "newest".to_string(),
            SessionUsage {
                total_tokens: 1.0,
                request_count: 1,
                last_seen: 9.0,
            },
        );
        evict_oldest(&mut table);
        assert_eq!(table.len(), 2);
        assert!(!table.contains_key("oldest"));
        assert!(table.contains_key("middle"));
        assert!(table.contains_key("newest"));
    }

    #[test]
    fn the_in_process_table_never_grows_past_the_cap() {
        // A session-id-per-request attacker (the caller-controlled
        // `X-Armor-Session-Id` header) must not be able to grow this table
        // without bound: each new distinct id beyond the cap evicts the
        // least-recently-touched entry rather than growing the table.
        let mut o = CheckOptions::default();
        o.set_f64("max_tokens_per_session", f64::INFINITY);
        for i in 0..(MAX_TRACKED_SESSIONS + 5) {
            let mut o = o.clone();
            o.set_str("session_id", &format!("session-{i}"));
            o.set_f64("now", i as f64);
            evaluate("x", &o);
        }
        let table = usage_table().lock().unwrap();
        assert!(table.len() <= MAX_TRACKED_SESSIONS);
    }

    #[test]
    fn injected_durable_totals_are_used_as_is() {
        // Already include this request — compared directly, not accumulated.
        let mut o = opts("session-durable", &[("max_tokens_per_session", 1000.0)]);
        o.set_f64("session_total_tokens", 1000.0);
        o.set_f64("session_request_count", 1.0);
        assert_eq!(evaluate("at the limit", &o).action, CheckAction::Log);

        o.set_f64("session_total_tokens", 1001.0);
        let over = evaluate("over it", &o);
        assert_eq!(over.action, CheckAction::Deny);
        assert!(over
            .hits
            .iter()
            .any(|h| h.rule_id == "unbounded-consumption-token-budget-exceeded"));
    }

    #[test]
    fn injected_totals_override_the_in_process_table() {
        // One request seen locally, forty across the fleet: the shared
        // number has to win or the budget is per-replica.
        let mut o = opts(
            "session-durable-override",
            &[("max_requests_per_session", 10.0)],
        );
        assert_eq!(evaluate("local", &o).action, CheckAction::Log);

        o.set_f64("session_request_count", 40.0);
        o.set_f64("session_total_tokens", 0.0);
        assert_eq!(evaluate("shared", &o).action, CheckAction::Deny);
    }

    #[test]
    fn a_partially_injected_total_still_uses_the_local_count_for_the_other() {
        // Tokens durable, requests not: the request budget must still be
        // enforced from the in-process counter rather than silently
        // dropping to "unlimited".
        let mut o = opts(
            "session-partial",
            &[
                ("max_requests_per_session", 1.0),
                ("max_tokens_per_session", 1e9),
            ],
        );
        o.set_f64("session_total_tokens", 5.0);
        assert_eq!(evaluate("first", &o).action, CheckAction::Log);
        assert_eq!(evaluate("second", &o).action, CheckAction::Deny);
    }

    #[test]
    fn injected_totals_work_without_a_session_id() {
        let mut o = CheckOptions::default();
        o.set_f64("max_tokens_per_session", 10.0);
        o.set_f64("session_total_tokens", 11.0);
        o.set_f64("session_request_count", 1.0);
        assert_eq!(evaluate("no session id", &o).action, CheckAction::Deny);
    }

    #[test]
    fn loop_depth_over_budget_denies() {
        let mut o = opts("session-loop", &[("max_loop_depth", 5.0)]);
        o.set_f64("loop_depth", 6.0);
        let result = evaluate("step", &o);
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "unbounded-consumption-loop-depth-exceeded"));
    }

    #[test]
    fn token_budget_accumulates_across_calls_for_same_session() {
        let session = "session-tokens-accumulate";
        let opts_call = opts(
            session,
            &[
                ("estimated_tokens", 60.0),
                ("max_tokens_per_session", 100.0),
            ],
        );
        let first = evaluate("first call", &opts_call);
        assert_eq!(first.action, CheckAction::Log);
        let second = evaluate("second call", &opts_call);
        assert_eq!(second.action, CheckAction::Deny);
        assert!(second
            .hits
            .iter()
            .any(|h| h.rule_id == "unbounded-consumption-token-budget-exceeded"));
    }

    #[test]
    fn request_budget_accumulates_across_calls_for_same_session() {
        let session = "session-requests-accumulate";
        let opts_call = opts(session, &[("max_requests_per_session", 2.0)]);
        assert_eq!(evaluate("one", &opts_call).action, CheckAction::Log);
        assert_eq!(evaluate("two", &opts_call).action, CheckAction::Log);
        assert_eq!(evaluate("three", &opts_call).action, CheckAction::Deny);
    }

    #[test]
    fn distinct_sessions_do_not_share_budget() {
        let opts_a = opts(
            "session-a-isolated",
            &[
                ("estimated_tokens", 90.0),
                ("max_tokens_per_session", 100.0),
            ],
        );
        let opts_b = opts(
            "session-b-isolated",
            &[
                ("estimated_tokens", 90.0),
                ("max_tokens_per_session", 100.0),
            ],
        );
        assert_eq!(evaluate("x", &opts_a).action, CheckAction::Log);
        assert_eq!(evaluate("y", &opts_b).action, CheckAction::Log);
    }

    #[test]
    fn empty_session_id_skips_session_tracking() {
        let mut o = CheckOptions::default();
        o.set_f64("estimated_tokens", 1_000_000.0);
        o.set_f64("max_tokens_per_session", 1.0);
        let result = evaluate("no session id set", &o);
        assert_eq!(result.action, CheckAction::Log);
    }
}
