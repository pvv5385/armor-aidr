//! Runs the configured checks for a request, ported from
//! `app/guardrails/engine.py`. `armor-core` stays synchronous by design — no
//! async runtime, no I/O — so it can be embedded anywhere without pulling in
//! a runtime.
//!
//! These detectors are pure, sub-millisecond, CPU-bound functions with no
//! I/O — every pattern-matching detector runs on `regex`, whose linear-time
//! guarantee precludes the catastrophic-backtracking hang a per-check OS
//! thread + timeout would exist to defend against. So there's no thread
//! spun up per check and no thread-per-check timeout: checks run in place,
//! panics are still caught with `catch_unwind` (a buggy detector shouldn't
//! be able to take the request down), and `timed_out` is a post-hoc flag —
//! elapsed vs. the configured budget — rather than a cancellation.
//!
//! - **Parallel** (default): every enabled check runs on rayon's global
//!   thread pool (sized once to the CPU count and reused across requests,
//!   not spawned per request) via `par_iter`.
//! - **Sequential**: checks run one at a time in cheapest-first order
//!   (ascending [`crate::detectors::default_order`] rank — a fixed
//!   backend-owned ranking, not configurable), short-circuiting on the
//!   first deny, and checking the whole-run deadline between checks so a
//!   budget that's already exhausted skips the rest instead of running
//!   them for nothing.
//!
//! Both modes run under a single whole-run wall-clock budget on top of each
//! check's own per-check timeout — mirrors `engine.py`'s
//! `guardrail_timeout_seconds` wrapping `check_timeout_seconds`. Since
//! nothing here is preemptible, the budget is enforced by checking elapsed
//! time against the deadline (between checks, and once more after the
//! batch) rather than by a supervisor thread racing the work.

use std::panic;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::detectors::CheckFn;
use crate::engine::decision::{compose, CheckOutcome, Decision};
use crate::engine::escalation;
use crate::engine::normalize::{build_views, NormalizeOptions, Views};
use crate::engine::redact::build_redacted_text;
use crate::models::{CheckAction, Severity, Verdict};
use crate::policy::schema::{CheckConfig, ExecutionMode, FailMode, PolicyConfig};

pub const DEFAULT_CHECK_TIMEOUT: Duration = Duration::from_secs(2);
pub const DEFAULT_GUARDRAIL_TIMEOUT: Duration = Duration::from_secs(5);

/// Looks up the detector function for a check's category. Production
/// callers pass [`crate::detectors::get_check`]; tests pass their own to
/// inject fake/slow checks without a global mutable registry.
pub type RegistryFn = fn(&str) -> Option<CheckFn>;

#[derive(Clone)]
struct ViewSweepResult {
    passed: bool,
    hits: Vec<crate::models::RuleHit>,
    view: String,
    severity: Severity,
    confidence: Option<f32>,
}

fn run_view_sweep(
    check_fn: CheckFn,
    views: &Views,
    options: &crate::policy::schema::CheckOptions,
    stateful: bool,
) -> ViewSweepResult {
    if stateful {
        // A stateful detector's `evaluate()` mutates process-global state
        // keyed by `options.session_id` (`detectors::is_stateful`'s doc
        // comment) — sweeping every view would invoke it, and its side
        // effect, once per view instead of once per request. Score `raw`
        // alone; there is nothing view-specific for a rate/budget counter to
        // catch by decoding a normalized view anyway.
        let raw_text = views.get("raw").unwrap_or_default();
        let result = check_fn(raw_text, options);
        return ViewSweepResult {
            passed: result.passed(),
            hits: result.hits,
            view: "raw".to_string(),
            severity: result.severity,
            confidence: result.confidence,
        };
    }

    let mut raw_result: Option<ViewSweepResult> = None;
    for (view_name, view_text) in views.iter() {
        let result = check_fn(view_text, options);
        let sweep = ViewSweepResult {
            passed: result.passed(),
            hits: result.hits,
            view: view_name.to_string(),
            severity: result.severity,
            confidence: result.confidence,
        };
        if view_name == "raw" {
            raw_result = Some(sweep.clone());
        }
        if !sweep.passed {
            return sweep;
        }
    }
    raw_result.expect("build_views always includes \"raw\"")
}

/// Runs one check to completion in place, mirroring `_run_one`. There's no
/// thread to cancel on a slow check, so `timeout` is checked against the
/// elapsed wall-clock time after the fact rather than enforced by racing a
/// spawned worker against a channel `recv_timeout`.
fn run_one(
    category: String,
    check_fn: Option<CheckFn>,
    views: &Views,
    config: &CheckConfig,
    timeout: Duration,
) -> CheckOutcome {
    // The raw input, unconditionally present in `views` — the `view_text`
    // for every path here that never actually ran a detector against a
    // (possibly different) view, so a caller can't be handed a `view_text`
    // that doesn't match `view`.
    let raw_text = || views.get("raw").unwrap_or_default().to_string();

    let Some(check_fn) = check_fn else {
        return CheckOutcome {
            category,
            passed: config.fail_mode != FailMode::FailClosed,
            action: config.on_fail,
            severity: Severity::Low,
            confidence: None,
            hits: Vec::new(),
            view: "raw".to_string(),
            view_text: raw_text(),
            error: Some("unknown check category".to_string()),
            timed_out: false,
            latency_ms: 0.0,
            mode: config.mode,
            ..Default::default()
        };
    };

    let stateful = crate::detectors::is_stateful(&category);
    let start = Instant::now();
    let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_view_sweep(check_fn, views, &config.options, stateful)
    }));
    let elapsed = start.elapsed();

    match outcome {
        Ok(sweep) if elapsed <= timeout => {
            // The text `sweep.hits[].span` are offsets into — the view that
            // actually fired (or "raw", when nothing did), not necessarily
            // `raw_text()`. See `CheckOutcome::view_text`'s doc comment.
            let view_text = views.get(&sweep.view).unwrap_or_default().to_string();
            CheckOutcome {
                category,
                passed: sweep.passed,
                action: config.on_fail,
                severity: sweep.severity,
                confidence: sweep.confidence,
                // Stamped here, on the one path where a detector actually
                // ran. The three paths below leave it at the default 0 —
                // but they also set `timed_out`/`error`, which
                // `escalation::plan` checks first, so a 0 there is never
                // read as "the rules found nothing".
                risk_score: escalation::risk_score(&sweep.hits),
                hits: sweep.hits,
                view: sweep.view,
                view_text,
                error: None,
                timed_out: false,
                latency_ms: elapsed.as_secs_f64() * 1000.0,
                mode: config.mode,
                ..Default::default()
            }
        }
        // Finished, but past its own budget. These checks are pure/sub-ms
        // so this only fires on a genuine regression; there's no worker to
        // cancel, so the best we can do is flag it and apply fail_mode.
        Ok(_) => CheckOutcome {
            category,
            passed: config.fail_mode != FailMode::FailClosed,
            action: config.on_fail,
            severity: Severity::Low,
            confidence: None,
            hits: Vec::new(),
            view: "raw".to_string(),
            view_text: raw_text(),
            error: None,
            timed_out: true,
            latency_ms: elapsed.as_secs_f64() * 1000.0,
            mode: config.mode,
            ..Default::default()
        },
        Err(panic_payload) => {
            let message = panic_payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "check panicked".to_string());
            CheckOutcome {
                category,
                passed: config.fail_mode != FailMode::FailClosed,
                action: config.on_fail,
                severity: Severity::Low,
                confidence: None,
                hits: Vec::new(),
                view: "raw".to_string(),
                view_text: raw_text(),
                error: Some(message),
                timed_out: false,
                latency_ms: elapsed.as_secs_f64() * 1000.0,
                mode: config.mode,
                ..Default::default()
            }
        }
    }
}

fn is_blocking_deny(outcome: &CheckOutcome) -> bool {
    outcome.mode == crate::models::EnforcementMode::Block
        && !outcome.passed
        && outcome.action == CheckAction::Deny
}

/// Runs every enabled check on rayon's global thread pool. That pool is
/// sized once (to the CPU count) and reused for every request — unlike
/// `std::thread::spawn`, `par_iter` doesn't create a new OS thread per
/// check. Checks are sub-millisecond, so (unlike `run_sequential`) this
/// doesn't bother short-circuiting on the first blocking deny — by the time
/// one arrives the rest have typically already finished.
fn run_parallel(
    enabled: &[CheckConfig],
    views: &Views,
    check_timeout: Duration,
    registry: RegistryFn,
) -> Vec<CheckOutcome> {
    enabled
        .par_iter()
        .map(|config| {
            let check_fn = registry(&config.category);
            run_one(
                config.category.clone(),
                check_fn,
                views,
                config,
                check_timeout,
            )
        })
        .collect()
}

fn run_sequential(
    enabled: &[CheckConfig],
    views: &Views,
    check_timeout: Duration,
    registry: RegistryFn,
    deadline: Instant,
) -> Vec<CheckOutcome> {
    let mut ordered: Vec<&CheckConfig> = enabled.iter().collect();
    ordered.sort_by_key(|c| crate::detectors::default_order(&c.category));

    let mut results = Vec::new();
    for config in ordered {
        if Instant::now() >= deadline {
            break; // whole-run budget already exhausted — no point running the rest
        }
        let check_fn = registry(&config.category);
        let outcome = run_one(
            config.category.clone(),
            check_fn,
            views,
            config,
            check_timeout,
        );
        let deny = is_blocking_deny(&outcome);
        results.push(outcome);
        if deny {
            break; // cheap check denied — never run the remaining, more expensive checks
        }
    }
    results
}

/// Runs every enabled check in `policy` against `text` and returns the
/// aggregate [`Decision`], mirroring `run_checks`. `registry` resolves a
/// check category to its detector function — production callers pass
/// [`crate::detectors::get_check`].
pub fn run_checks(policy: &PolicyConfig, text: &str, registry: RegistryFn) -> Decision {
    run_checks_with_budget(
        policy,
        text,
        registry,
        DEFAULT_CHECK_TIMEOUT,
        DEFAULT_GUARDRAIL_TIMEOUT,
    )
}

pub fn run_checks_with_budget(
    policy: &PolicyConfig,
    text: &str,
    registry: RegistryFn,
    check_timeout: Duration,
    guardrail_timeout: Duration,
) -> Decision {
    match run_deterministic(policy, text, registry, check_timeout, guardrail_timeout) {
        Ok((_views, outcomes)) => compose_with_redaction(text, outcomes),
        Err(exceeded) => exceeded.into_decision(text),
    }
}

/// The whole-run budget elapsed before the sweep finished. Carries the
/// policy's `fail_mode` so the caller — which may be `armor-api` mid-way
/// through an escalation pass, not just [`run_checks_with_budget`] — resolves
/// it the same way.
#[derive(Debug, Clone, Copy)]
pub struct BudgetExceeded {
    pub fail_mode: FailMode,
}

impl BudgetExceeded {
    /// The empty-outcomes decision a blown budget produces. `redacted_text`
    /// is the input unchanged: no check completed, so nothing is known to be
    /// worth masking.
    pub fn into_decision(self, text: &str) -> Decision {
        let verdict = if self.fail_mode == FailMode::FailClosed {
            Verdict::Block
        } else {
            Verdict::Allow
        };
        Decision {
            verdict,
            outcomes: Vec::new(),
            redacted_text: text.to_string(),
        }
    }
}

/// Runs the deterministic sweep and stops — no redaction, no composition.
///
/// This is the first half of the old `run_checks_with_budget`, split out so
/// `armor-api` can slot an async escalation pass between the two halves.
/// The [`Views`] come back with the outcomes because an ML layer
/// needs to score a *view*, and rebuilding them would be both wasteful and a
/// chance for the two passes to disagree about what was scanned.
///
/// Semantics are unchanged from the single-function version: same budget
/// enforcement, same fail-mode resolution, same ordering.
pub fn run_deterministic(
    policy: &PolicyConfig,
    text: &str,
    registry: RegistryFn,
    check_timeout: Duration,
    guardrail_timeout: Duration,
) -> Result<(Views, Vec<CheckOutcome>), BudgetExceeded> {
    let normalize_opts: NormalizeOptions = policy.normalize.into();

    let enabled: Vec<CheckConfig> = policy
        .checks
        .iter()
        .filter(|c| c.enabled)
        .cloned()
        .collect();
    if enabled.is_empty() {
        return Ok((build_views(text, normalize_opts), Vec::new()));
    }

    let views = build_views(text, normalize_opts);
    let execution_mode = policy.execution_mode;

    let start = Instant::now();
    let deadline = start + guardrail_timeout;
    let outcomes = if execution_mode == ExecutionMode::Sequential {
        run_sequential(&enabled, &views, check_timeout, registry, deadline)
    } else {
        run_parallel(&enabled, &views, check_timeout, registry)
    };

    // Nothing here is preemptible, so the whole-run budget is enforced by
    // checking elapsed time once the batch is done rather than by racing a
    // supervisor thread against the work.
    if start.elapsed() > guardrail_timeout {
        return Err(BudgetExceeded {
            fail_mode: policy.fail_mode,
        });
    }

    Ok((views, outcomes))
}

/// Builds `redacted_text` from the final outcomes and composes the verdict —
/// the second half of the split.
///
/// **This must run after any escalation pass, not before.** An NER layer's
/// unstructured-PII hits are merged onto the outcomes by
/// [`crate::engine::escalation::merge`], and they only make it into
/// `redacted_text` if redaction happens once those hits are already present.
/// That ordering constraint is the concrete reason redaction moved out of the
/// sweep and to the end of the run.
pub fn compose_with_redaction(text: &str, outcomes: Vec<CheckOutcome>) -> Decision {
    let redacted_text = build_redacted_text(text, &outcomes);
    compose(outcomes, redacted_text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    use crate::models::{DetectorResult, EnforcementMode, Severity};
    use crate::policy::schema::{CheckOptions, NormalizeConfig};

    const VALID_VISA: &str = "4242 4242 4242 4242";
    const INVALID_LUHN: &str = "1234 5678 9012 3456";

    fn pci_options() -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_bool("credit_card", true);
        o.set_bool("luhn_required", true);
        o
    }

    fn pci_check(overrides: impl FnOnce(&mut CheckConfig)) -> CheckConfig {
        let mut c = CheckConfig {
            category: "pci".to_string(),
            enabled: true,
            options: pci_options(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        overrides(&mut c);
        c
    }

    fn policy(checks: Vec<CheckConfig>) -> PolicyConfig {
        PolicyConfig {
            id: "gr_test".to_string(),
            execution_mode: ExecutionMode::Parallel,
            fail_mode: FailMode::FailOpen,
            normalize: NormalizeConfig::default(),
            checks,
        }
    }

    fn registry(category: &str) -> Option<CheckFn> {
        crate::detectors::get_check(category)
    }

    #[test]
    fn luhn_valid_card_denies() {
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&policy(vec![pci_check(|_| {})]), &text, registry);
        assert_eq!(decision.verdict, Verdict::Block);
        assert_eq!(decision.outcomes[0].category, "pci");
        assert!(!decision.outcomes[0].passed);
        assert_eq!(decision.outcomes[0].hits.len(), 1);
    }

    #[test]
    fn luhn_valid_card_is_masked_in_redacted_text() {
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&policy(vec![pci_check(|_| {})]), &text, registry);
        assert_eq!(decision.redacted_text, "my card is <PCI:PCI_CREDIT_CARD:1>");
    }

    #[test]
    fn luhn_invalid_card_allows() {
        let text = format!("my card is {INVALID_LUHN}");
        let decision = run_checks(&policy(vec![pci_check(|_| {})]), &text, registry);
        assert_eq!(decision.verdict, Verdict::Allow);
        assert!(decision.outcomes[0].passed);
        assert_eq!(decision.redacted_text, text);
    }

    #[test]
    fn empty_policy_redacted_text_is_the_input_unchanged() {
        let decision = run_checks(&policy(vec![]), VALID_VISA, registry);
        assert_eq!(decision.redacted_text, VALID_VISA);
    }

    #[test]
    fn credit_card_option_disabled_allows() {
        let check = pci_check(|c| c.options = CheckOptions::default());
        let decision = run_checks(&policy(vec![check]), VALID_VISA, registry);
        assert_eq!(decision.verdict, Verdict::Allow);
    }

    #[test]
    fn disabled_check_is_skipped() {
        let check = pci_check(|c| c.enabled = false);
        let decision = run_checks(&policy(vec![check]), VALID_VISA, registry);
        assert_eq!(decision.verdict, Verdict::Allow);
        assert!(decision.outcomes.is_empty());
    }

    // There's no thread to preempt anymore (see the module doc comment), so
    // `timed_out` is now detected post-hoc by comparing elapsed time to the
    // configured budget after the check actually finishes — these tests
    // really do wait out the sleep below, so it's kept short.
    const SLOW_CHECK_SLEEP: Duration = Duration::from_millis(30);

    fn slow_check(_text: &str, _options: &CheckOptions) -> DetectorResult {
        thread::sleep(SLOW_CHECK_SLEEP);
        DetectorResult {
            detector_id: "_slow_test".to_string(),
            action: CheckAction::Deny,
            severity: Severity::High,
            hits: Vec::new(),
            confidence: None,
        }
    }

    fn registry_with_slow(category: &str) -> Option<CheckFn> {
        match category {
            "_slow_test" => Some(slow_check),
            other => crate::detectors::get_check(other),
        }
    }

    #[test]
    fn timeout_fails_open_by_default() {
        let check = CheckConfig {
            category: "_slow_test".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        let decision = run_checks_with_budget(
            &policy(vec![check]),
            "text",
            registry_with_slow,
            SLOW_CHECK_SLEEP / 2,
            DEFAULT_GUARDRAIL_TIMEOUT,
        );
        assert_eq!(decision.verdict, Verdict::Allow);
        assert!(decision.outcomes[0].timed_out);
    }

    #[test]
    fn timeout_fails_closed_when_configured() {
        let check = CheckConfig {
            category: "_slow_test".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailClosed,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        let decision = run_checks_with_budget(
            &policy(vec![check]),
            "text",
            registry_with_slow,
            SLOW_CHECK_SLEEP / 2,
            DEFAULT_GUARDRAIL_TIMEOUT,
        );
        assert_eq!(decision.verdict, Verdict::Block);
        assert!(decision.outcomes[0].timed_out);
    }

    #[test]
    fn sequential_mode_short_circuits_before_expensive_check() {
        static EXPENSIVE_RAN: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);

        fn expensive_check(_text: &str, _options: &CheckOptions) -> DetectorResult {
            EXPENSIVE_RAN.store(true, std::sync::atomic::Ordering::SeqCst);
            DetectorResult {
                detector_id: "_expensive_test".to_string(),
                action: CheckAction::Deny,
                severity: Severity::Low,
                hits: Vec::new(),
                confidence: None,
            }
        }
        fn registry_with_expensive(category: &str) -> Option<CheckFn> {
            match category {
                "_expensive_test" => Some(expensive_check),
                other => crate::detectors::get_check(other),
            }
        }

        let cheap = pci_check(|_| {});
        let expensive = CheckConfig {
            category: "_expensive_test".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        let mut p = policy(vec![expensive, cheap]); // deliberately out of order: expensive first
        p.execution_mode = ExecutionMode::Sequential;

        let decision = run_checks(&p, VALID_VISA, registry_with_expensive);
        assert_eq!(decision.verdict, Verdict::Block);
        assert_eq!(decision.outcomes.len(), 1);
        assert_eq!(decision.outcomes[0].category, "pci");
        assert!(!EXPENSIVE_RAN.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn sequential_mode_runs_all_checks_when_none_deny() {
        let a = pci_check(|c| {
            c.options = CheckOptions::default();
        });
        let b = pci_check(|c| {
            c.options = CheckOptions::default();
        });
        let mut p = policy(vec![a, b]);
        p.execution_mode = ExecutionMode::Sequential;
        let decision = run_checks(&p, VALID_VISA, registry);
        assert_eq!(decision.verdict, Verdict::Allow);
        assert_eq!(decision.outcomes.len(), 2);
    }

    #[test]
    fn guardrail_wall_clock_budget_fails_open_by_default() {
        let check = CheckConfig {
            category: "_slow_test".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        let mut p = policy(vec![check]);
        p.fail_mode = FailMode::FailOpen;
        let decision = run_checks_with_budget(
            &p,
            "text",
            registry_with_slow,
            Duration::from_secs(1),
            SLOW_CHECK_SLEEP / 2,
        );
        assert_eq!(decision.verdict, Verdict::Allow);
        assert!(decision.outcomes.is_empty());
    }

    #[test]
    fn guardrail_wall_clock_budget_fails_closed_when_configured() {
        let check = CheckConfig {
            category: "_slow_test".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        let mut p = policy(vec![check]);
        p.fail_mode = FailMode::FailClosed;
        let decision = run_checks_with_budget(
            &p,
            "text",
            registry_with_slow,
            Duration::from_secs(1),
            SLOW_CHECK_SLEEP / 2,
        );
        assert_eq!(decision.verdict, Verdict::Block);
        assert!(decision.outcomes.is_empty());
    }

    #[test]
    fn monitor_mode_check_never_denies_but_result_is_still_visible() {
        let check = pci_check(|c| c.mode = EnforcementMode::Monitor);
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&policy(vec![check]), &text, registry);
        assert_eq!(decision.verdict, Verdict::Allow);
        assert_eq!(decision.outcomes.len(), 1);
        assert!(!decision.outcomes[0].passed);
        assert_eq!(decision.outcomes[0].mode, EnforcementMode::Monitor);
    }

    #[test]
    fn block_mode_is_the_default_when_unspecified() {
        let check = pci_check(|_| {});
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&policy(vec![check]), &text, registry);
        assert_eq!(decision.verdict, Verdict::Block);
        assert_eq!(decision.outcomes[0].mode, EnforcementMode::Block);
    }

    #[test]
    fn block_check_denies_even_alongside_a_monitor_check() {
        let block_check = pci_check(|c| {
            c.mode = EnforcementMode::Block;
        });
        let mut monitor_options = CheckOptions::default();
        monitor_options.set_bool("credit_card", false); // word_count-equivalent stand-in: never denies
        let monitor_check = CheckConfig {
            category: "pci".to_string(),
            enabled: true,
            options: monitor_options,
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Monitor,
            ..Default::default()
        };
        let mut p = policy(vec![block_check, monitor_check]);
        p.execution_mode = ExecutionMode::Sequential;
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&p, &text, registry);
        assert_eq!(decision.verdict, Verdict::Block);
        // Sequential short-circuit: the block check (first in the vector,
        // same category as the monitor check) denied first, so the monitor
        // check never ran at all.
        assert_eq!(decision.outcomes.len(), 1);
    }

    #[test]
    fn normalize_rot13_view_catches_encoded_phrase() {
        // No case_match detector exists yet — prompt_injection's ruleset
        // stands in: rot13 of a "DAN" jailbreak phrase should surface on the
        // "rot13" view.
        let mut check = CheckConfig {
            category: "prompt_injection".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        check.options.set_bool("pattern_match", true);
        let mut p = policy(vec![check]);
        p.normalize = NormalizeConfig {
            rot13: true,
            ..Default::default()
        };

        // rot13("do anything now") == "qb nalguvat abj"
        let decision = run_checks(&p, "plan: qb nalguvat abj", registry);
        assert_eq!(decision.verdict, Verdict::Block);
        assert_eq!(decision.outcomes[0].view, "rot13");
    }

    #[test]
    fn a_stateful_abuse_check_increments_its_counter_once_per_request_even_with_multiple_views() {
        // Regression guard: `run_view_sweep` used to invoke every check once
        // per normalized view. That's right for a stateless pattern check
        // (catches a rot13'd jailbreak) but wrong for a stateful one like
        // `abuse`, whose `evaluate()` bumps a process-global counter as a
        // side effect — sweeping N views bumped it N times for one request.
        let mut options = CheckOptions::default();
        options.set_str("session_id", "session-orchestrator-stateful-once");
        options.set_f64("max_requests_per_window", 1.0);
        options.set_f64("window_seconds", 60.0);
        options.set_f64("now", 100.0);

        let check = CheckConfig {
            category: "abuse".to_string(),
            enabled: true,
            options,
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        let mut p = policy(vec![check]);
        // Two views to sweep: "raw" and "rot13".
        p.normalize = NormalizeConfig {
            rot13: true,
            ..Default::default()
        };

        let first = run_checks(&p, "plan: qb nalguvat abj", registry);
        assert!(
            first.outcomes[0].passed,
            "one request must count as one hit, not one per normalized view"
        );

        let second = run_checks(&p, "plan: qb nalguvat abj", registry);
        assert!(
            !second.outcomes[0].passed,
            "the second request should now be over the limit of 1"
        );
    }

    #[test]
    fn normalize_off_by_default_does_not_decode() {
        let mut check = CheckConfig {
            category: "prompt_injection".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        check.options.set_bool("pattern_match", true);
        let decision = run_checks(&policy(vec![check]), "plan: qb nalguvat abj", registry);
        assert_eq!(decision.verdict, Verdict::Allow);
        assert_eq!(decision.outcomes[0].view, "raw");
    }

    #[test]
    fn severity_and_confidence_survive_from_the_detector_result() {
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&policy(vec![pci_check(|_| {})]), &text, registry);
        assert_eq!(decision.outcomes[0].severity, Severity::High);
    }

    #[test]
    fn view_text_matches_the_raw_input_when_no_normalization_fired() {
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&policy(vec![pci_check(|_| {})]), &text, registry);
        assert_eq!(decision.outcomes[0].view, "raw");
        assert_eq!(decision.outcomes[0].view_text, text);
    }

    #[test]
    fn view_text_matches_the_decoded_view_when_a_normalized_view_fires() {
        let mut check = CheckConfig {
            category: "prompt_injection".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        check.options.set_bool("pattern_match", true);
        let mut p = policy(vec![check]);
        p.normalize = NormalizeConfig {
            rot13: true,
            ..Default::default()
        };

        let decision = run_checks(&p, "plan: qb nalguvat abj", registry);
        assert_eq!(decision.outcomes[0].view, "rot13");
        assert_eq!(decision.outcomes[0].view_text, "cyna: do anything now");
    }

    // ---- the run_deterministic / compose_with_redaction split ----

    #[test]
    fn the_split_seams_compose_to_the_same_decision_as_the_wrapper() {
        let text = format!("my card is {VALID_VISA}");
        let p = policy(vec![pci_check(|_| {})]);

        let whole = run_checks(&p, &text, registry);
        let (_views, outcomes) = run_deterministic(
            &p,
            &text,
            registry,
            DEFAULT_CHECK_TIMEOUT,
            DEFAULT_GUARDRAIL_TIMEOUT,
        )
        .expect("budget is generous");
        let split = compose_with_redaction(&text, outcomes);

        assert_eq!(whole.verdict, split.verdict);
        assert_eq!(whole.redacted_text, split.redacted_text);
        assert_eq!(whole.outcomes.len(), split.outcomes.len());
    }

    #[test]
    fn run_deterministic_returns_the_views_it_actually_scanned() {
        let mut p = policy(vec![pci_check(|_| {})]);
        p.normalize = NormalizeConfig {
            rot13: true,
            ..Default::default()
        };
        let (views, _outcomes) = run_deterministic(
            &p,
            "hello",
            registry,
            DEFAULT_CHECK_TIMEOUT,
            DEFAULT_GUARDRAIL_TIMEOUT,
        )
        .unwrap();
        // An escalating layer scores one of these rather than rebuilding
        // them, so the two passes can't disagree about what was scanned.
        assert_eq!(views.get("raw"), Some("hello"));
        assert!(views.get("rot13").is_some());
    }

    #[test]
    fn a_blown_budget_surfaces_as_an_error_rather_than_a_decision() {
        let check = CheckConfig {
            category: "_slow_test".to_string(),
            fail_mode: FailMode::FailClosed,
            ..Default::default()
        };
        let mut p = policy(vec![check]);
        p.fail_mode = FailMode::FailClosed;

        let err = run_deterministic(
            &p,
            "text",
            registry_with_slow,
            Duration::from_secs(1),
            SLOW_CHECK_SLEEP / 2,
        )
        .expect_err("the whole-run budget should be exceeded");

        assert_eq!(err.fail_mode, FailMode::FailClosed);
        assert_eq!(err.into_decision("text").verdict, Verdict::Block);
    }

    #[test]
    fn redaction_sees_hits_added_after_the_sweep() {
        // The reason redaction moved to the end: an escalated layer's hits
        // are merged onto the outcome *after* run_deterministic returns, and
        // they still have to reach `redacted_text`. Stands in for a real
        // escalated layer (e.g. the NER runner) without needing one here.
        let text = "contact alice at the office";
        let p = policy(vec![]);
        let (_views, mut outcomes) = run_deterministic(
            &p,
            text,
            registry,
            DEFAULT_CHECK_TIMEOUT,
            DEFAULT_GUARDRAIL_TIMEOUT,
        )
        .unwrap();

        outcomes.push(CheckOutcome {
            category: "pii".to_string(),
            passed: false,
            view: "raw".to_string(),
            view_text: text.to_string(),
            hits: vec![crate::models::RuleHit {
                rule_id: "NER_PERSON".to_string(),
                span: (8, 13),
                severity: Severity::Medium,
            }],
            ..Default::default()
        });

        let decision = compose_with_redaction(text, outcomes);
        assert_eq!(
            decision.redacted_text,
            "contact <PII:NER_PERSON:1> at the office"
        );
    }

    #[test]
    fn risk_score_is_stamped_on_the_outcome_from_the_hits() {
        let text = format!("my card is {VALID_VISA}");
        let decision = run_checks(&policy(vec![pci_check(|_| {})]), &text, registry);
        let outcome = &decision.outcomes[0];
        assert_eq!(
            outcome.risk_score,
            crate::engine::escalation::risk_score(&outcome.hits)
        );
        assert!(outcome.risk_score > 0);
    }

    #[test]
    fn a_clean_check_scores_zero_and_stays_on_the_deterministic_layer() {
        let decision = run_checks(&policy(vec![pci_check(|_| {})]), INVALID_LUHN, registry);
        assert_eq!(decision.outcomes[0].risk_score, 0);
        assert_eq!(
            decision.outcomes[0].execution_layer,
            crate::policy::schema::ExecutionLayer::LocalDeterministic
        );
        assert!(decision.outcomes[0].layers.is_empty());
    }

    #[test]
    fn unknown_category_still_reports_the_raw_text() {
        let check = CheckConfig {
            category: "_no_such_check".to_string(),
            enabled: true,
            options: CheckOptions::default(),
            on_fail: CheckAction::Deny,
            fail_mode: FailMode::FailOpen,
            mode: EnforcementMode::Block,
            ..Default::default()
        };
        let decision = run_checks(&policy(vec![check]), "hello world", registry);
        assert_eq!(decision.outcomes[0].view, "raw");
        assert_eq!(decision.outcomes[0].view_text, "hello world");
        assert_eq!(decision.outcomes[0].severity, Severity::Low);
    }
}
