//! Abuse / rate-limiting check — per-session request-rate limiting using a
//! fixed time window. Distinct from [`super::unbounded_consumption`]: that
//! detector tracks a cumulative budget for the *lifetime* of a session (no
//! time dimension — once you're over budget you stay over budget); this one
//! tracks *requests per rolling time window* (how many calls landed in the
//! last `window_seconds`), which is what "abuse"/rate-limiting means here —
//! a caller hammering the API in a burst trips this even if their lifetime
//! token/request budget is nowhere near its cap.
//!
//! # Where the count comes from
//!
//! Two sources, in priority order:
//!
//! 1. **`options.session_window_count`** — the durable count for the
//!    current window, already incremented for this request, injected by
//!    `armor-api` from `armor_storage::sessions::touch`. This is the
//!    correct source whenever a session store is configured, because it is
//!    shared across replicas.
//! 2. **In-process counters** keyed by `options.session_id`, when no
//!    durable count was supplied. Correct for a single instance and
//!    *silently wrong* behind a load balancer — each process only counts
//!    the requests it personally saw, so a caller spreading a burst across
//!    N replicas gets N times the budget. That is the whole reason for
//!    source 1.
//!
//! The comparison itself is identical either way: same fixed window, same
//! `max_requests_per_window` threshold. Only the counter's storage moves —
//! same mechanism regardless of which source is in play. `armor-core` stays
//! synchronous and I/O-free, so the detector never reaches for the database
//! itself; it reads a number `armor-api` looked up before the sweep.
//!
//! The current window's wall-clock time is read via `options.now` if the
//! caller supplies it (an injectable clock — seconds, any monotonically
//! increasing origin), falling back to the real `SystemTime` clock
//! otherwise. This is what makes the detector's window-rollover behavior
//! testable without a real `sleep()` in the test suite.
//!
//! `max_requests_per_window` defaults to unlimited: this check is inert
//! until a deployment configures a real number, same shipped-inert
//! convention as `keyword_blocklist`/`custom_regex`/`unbounded_consumption`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

struct WindowState {
    window_start: f64,
    count: u64,
}

/// Hard cap on how many distinct `session_id`s the in-process fallback table
/// tracks at once. `session_id` is caller-controlled (the
/// `X-Armor-Session-Id` header, `state.rs`) and this table has no external
/// TTL sweep the way the durable `sessions` table does (`retention.rs`) — an
/// attacker minting a fresh id per request would otherwise grow this
/// `HashMap` without bound for the life of the process. See [`make_room`].
const MAX_TRACKED_SESSIONS: usize = 100_000;

fn window_table() -> &'static Mutex<HashMap<String, WindowState>> {
    static TABLE: OnceLock<Mutex<HashMap<String, WindowState>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drops every entry whose window has already rolled over — by the same
/// rule that would reset it in place if it were the one being touched
/// (`now - window_start >= window_seconds`) — which is pure garbage by the
/// time anything else looks at it.
fn prune_expired(table: &mut HashMap<String, WindowState>, now: f64, window_seconds: f64) {
    table.retain(|_, state| now - state.window_start < window_seconds);
}

/// Removes the single oldest-`window_start` entry. The fallback once
/// [`prune_expired`] alone isn't enough — this many *concurrently active*
/// windows, not just accumulated garbage — so a legitimate burst of new
/// sessions can never grow the table past the cap either.
fn evict_oldest(table: &mut HashMap<String, WindowState>) {
    if let Some(oldest_key) = table
        .iter()
        .min_by(|a, b| a.1.window_start.total_cmp(&b.1.window_start))
        .map(|(k, _)| k.clone())
    {
        table.remove(&oldest_key);
    }
}

/// Bounds `table` before a new key is inserted into it.
fn make_room(table: &mut HashMap<String, WindowState>, now: f64, window_seconds: f64) {
    if table.len() < MAX_TRACKED_SESSIONS {
        return;
    }
    prune_expired(table, now, window_seconds);
    if table.len() >= MAX_TRACKED_SESSIONS {
        evict_oldest(table);
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
    let max_requests_per_window = options.f64_option("max_requests_per_window", f64::INFINITY);
    let window_seconds = options.f64_option("window_seconds", 60.0);
    let now = options.f64_option("now", real_now_seconds());

    let mut hits: Vec<RuleHit> = Vec::new();

    // Durable count wins when `armor-api` supplied one; it already counts
    // this request, so it is used as-is rather than incremented again.
    let count = match options.opt_f64("session_window_count") {
        Some(durable) => Some(durable),
        None if !session_id.is_empty() && window_seconds > 0.0 => {
            let mut table = window_table().lock().expect("window table poisoned");
            if !table.contains_key(&session_id) {
                make_room(&mut table, now, window_seconds);
            }
            let entry = table.entry(session_id).or_insert(WindowState {
                window_start: now,
                count: 0,
            });

            if now - entry.window_start >= window_seconds {
                entry.window_start = now;
                entry.count = 0;
            }
            entry.count += 1;
            Some(entry.count as f64)
        }
        None => None,
    };

    if count.is_some_and(|c| c > max_requests_per_window) {
        hits.push(RuleHit {
            rule_id: "abuse-rate-limit-exceeded".to_string(),
            span: (0, text.len()),
            severity: Severity::High,
        });
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "abuse".to_string(),
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
    fn unlimited_default_never_denies() {
        let result = evaluate("hello", &opts("session-abuse-unlimited", &[("now", 0.0)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn burst_within_window_over_limit_denies() {
        let session = "session-abuse-burst";
        let opts_call = opts(
            session,
            &[
                ("max_requests_per_window", 2.0),
                ("window_seconds", 60.0),
                ("now", 100.0),
            ],
        );
        assert_eq!(evaluate("one", &opts_call).action, CheckAction::Log);
        assert_eq!(evaluate("two", &opts_call).action, CheckAction::Log);
        let third = evaluate("three", &opts_call);
        assert_eq!(third.action, CheckAction::Deny);
        assert!(third
            .hits
            .iter()
            .any(|h| h.rule_id == "abuse-rate-limit-exceeded"));
    }

    #[test]
    fn window_rollover_resets_the_count() {
        let session = "session-abuse-rollover";
        let mut o = opts(
            session,
            &[
                ("max_requests_per_window", 1.0),
                ("window_seconds", 10.0),
                ("now", 0.0),
            ],
        );
        assert_eq!(evaluate("first", &o).action, CheckAction::Log);
        let second = evaluate("second", &o);
        assert_eq!(second.action, CheckAction::Deny);

        // Past the window boundary — count resets, so this call is allowed.
        o.set_f64("now", 11.0);
        let third = evaluate("third", &o);
        assert_eq!(third.action, CheckAction::Log);
    }

    #[test]
    fn distinct_sessions_do_not_share_a_window() {
        let opts_a = opts(
            "session-abuse-a",
            &[("max_requests_per_window", 1.0), ("now", 5.0)],
        );
        let opts_b = opts(
            "session-abuse-b",
            &[("max_requests_per_window", 1.0), ("now", 5.0)],
        );
        assert_eq!(evaluate("x", &opts_a).action, CheckAction::Log);
        assert_eq!(evaluate("y", &opts_b).action, CheckAction::Log);
    }

    #[test]
    fn empty_session_id_skips_tracking() {
        let mut o = CheckOptions::default();
        o.set_f64("max_requests_per_window", 0.0);
        o.set_f64("now", 0.0);
        let result = evaluate("no session id set", &o);
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn an_injected_durable_count_is_used_as_is() {
        // The durable count already includes this request, so the detector
        // must compare it directly rather than adding one to it.
        let mut o = opts("session-abuse-durable", &[("max_requests_per_window", 5.0)]);
        o.set_f64("session_window_count", 5.0);
        assert_eq!(evaluate("at the limit", &o).action, CheckAction::Log);

        o.set_f64("session_window_count", 6.0);
        assert_eq!(evaluate("over it", &o).action, CheckAction::Deny);
    }

    #[test]
    fn an_injected_count_overrides_the_in_process_table() {
        // Two replicas each saw one request; the shared store says six.
        // The store has to win, or the limit is per-replica.
        let session = "session-abuse-override";
        let mut o = opts(session, &[("max_requests_per_window", 3.0), ("now", 0.0)]);
        assert_eq!(evaluate("local one", &o).action, CheckAction::Log);

        o.set_f64("session_window_count", 6.0);
        assert_eq!(evaluate("shared six", &o).action, CheckAction::Deny);
    }

    #[test]
    fn an_injected_count_works_without_a_session_id() {
        // The session id is what the *in-process* table keys on; when the
        // count arrives pre-computed there is nothing left to key.
        let mut o = CheckOptions::default();
        o.set_f64("max_requests_per_window", 2.0);
        o.set_f64("session_window_count", 3.0);
        assert_eq!(evaluate("no session id", &o).action, CheckAction::Deny);
    }

    #[test]
    fn prune_expired_drops_only_rolled_over_windows() {
        let mut table = HashMap::new();
        table.insert(
            "stale".to_string(),
            WindowState {
                window_start: 0.0,
                count: 5,
            },
        );
        table.insert(
            "live".to_string(),
            WindowState {
                window_start: 95.0,
                count: 1,
            },
        );
        // now=100, window_seconds=60: "stale" rolled over 40s ago, "live"
        // started 5s ago and is still well within its window.
        prune_expired(&mut table, 100.0, 60.0);
        assert!(!table.contains_key("stale"));
        assert!(table.contains_key("live"));
    }

    #[test]
    fn evict_oldest_removes_the_single_oldest_window_and_nothing_else() {
        let mut table = HashMap::new();
        table.insert(
            "oldest".to_string(),
            WindowState {
                window_start: 1.0,
                count: 1,
            },
        );
        table.insert(
            "middle".to_string(),
            WindowState {
                window_start: 5.0,
                count: 1,
            },
        );
        table.insert(
            "newest".to_string(),
            WindowState {
                window_start: 9.0,
                count: 1,
            },
        );
        evict_oldest(&mut table);
        assert_eq!(table.len(), 2);
        assert!(!table.contains_key("oldest"));
        assert!(table.contains_key("middle"));
        assert!(table.contains_key("newest"));
    }

    #[test]
    fn make_room_is_a_no_op_below_capacity() {
        // The common case: nowhere near the cap, so no O(n) scan on every
        // new session — this is what keeps the fast path fast.
        let mut table = HashMap::new();
        table.insert(
            "only".to_string(),
            WindowState {
                window_start: 0.0,
                count: 1,
            },
        );
        make_room(&mut table, 1_000_000.0, 1.0);
        assert_eq!(table.len(), 1, "far below MAX_TRACKED_SESSIONS, untouched");
    }

    #[test]
    fn zero_window_seconds_skips_tracking() {
        let result = evaluate(
            "instant window",
            &opts(
                "session-abuse-zero-window",
                &[("max_requests_per_window", 0.0), ("window_seconds", 0.0)],
            ),
        );
        assert_eq!(result.action, CheckAction::Log);
    }
}
