//! Escalation: deciding whether a stronger layer is worth running for a
//! check, and folding its answer back in when it does.
//!
//! Everything here is pure — no I/O, no async, no clock. `armor-api` owns
//! the actual calls; this module only decides *what* to ask for
//! and *what the answer means*. That split is what lets the whole escalation
//! contract be tested with no network and no sidecar.
//!
//! # The routing signal
//!
//! [`risk_score`] collapses a check's rule hits into `0..=100`. It is an
//! **ordinal routing signal, not a calibrated probability** — it decides
//! whether to spend a forward pass, and nothing else. Calibration lives in
//! the classifier's `calibrated_score`, which the scorecard gate actually
//! measures against a benchmark suite.
//!
//! Escalation fires when the score lands *inside* the configured band, which
//! reads backwards until you see what the band is for:
//!
//! - **score 0** — the rules found nothing. This is exactly where a
//!   classifier earns its keep, so it escalates.
//! - **mid band** — the rules found something ambiguous. Escalate.
//! - **above the band** — the rules are already confident. A forward pass
//!   buys nothing, and under the asymmetry rule in [`merge`] an ML layer
//!   cannot overturn a `Block`-mode deny anyway.

use std::collections::HashSet;

use crate::engine::decision::{CheckOutcome, LayerOutcome};
use crate::models::{CheckAction, EnforcementMode, RuleHit, Severity};
use crate::policy::schema::{
    Backend, CheckConfig, EscalateWhen, ExecutionLayer, Fallback, PolicyConfig,
};

/// Severity weights. Unlike the reference implementation this derives from,
/// `Critical` is in the table **explicitly**: that table stops at `High`, and
/// a lookup-with-default-0 would score the most severe hits at zero, ranking
/// them below the escalation band — the exact inversion of what the band is
/// for.
///
/// `Critical` sits at 80 rather than at the band edge (70) deliberately. At
/// 70 the most consequential severity class would route on whether the band
/// predicate is inclusive; at 80 it lands clearly outside the default
/// `[0, 70]`, and since `High` plus the maximum corroboration bonus tops out
/// at 60, the default band reduces to a clean "max severity is below
/// critical".
fn weight(severity: Severity) -> u8 {
    match severity {
        Severity::Low => 10,
        Severity::Medium => 30,
        Severity::High => 50,
        Severity::Critical => 80,
    }
}

/// The number of *distinct rules* beyond the first that a corroboration
/// bonus is paid for, and the points per rule. Capped so the bonus can never
/// carry a check across a severity tier: volume adjusts ordering within a
/// tier, it does not promote one.
const MAX_BONUS_RULES: usize = 5;
const BONUS_PER_RULE: u8 = 2;

/// A check's deterministic routing score, `0..=100`.
///
/// The score is **max severity, plus a small saturating bonus for distinct
/// corroborating rules**. It is deliberately not a sum over hits. Summing
/// makes the score a function of hit *count* as much as severity, which
/// inverts the band: eight low-severity fuzzy matches would total 80 and
/// escape the `[0, 70]` band, while a single critical hit would stay inside
/// it. That suppresses escalation on exactly the noisy-but-benign payloads
/// where an ML layer is most useful — under [`merge`]'s asymmetry rule the
/// only thing a model may do to a deterministic result is *downgrade* a
/// `warn`, so the chatty-rule-bank case is the one that most needs the
/// second opinion.
///
/// The bonus counts distinct `rule_id`s rather than hits for the same
/// reason: one chatty rule matching forty times is one signal, not forty.
/// Corroboration across distinct rules is weighted only lightly, because
/// rules within a bank are correlated — several fuzzy patterns firing on the
/// same phrase is not several independent observations.
pub fn risk_score(hits: &[RuleHit]) -> u8 {
    let Some(base) = hits.iter().map(|h| weight(h.severity)).max() else {
        return 0;
    };
    let distinct = hits
        .iter()
        .map(|h| &h.rule_id)
        .collect::<HashSet<_>>()
        .len();
    let bonus = BONUS_PER_RULE * distinct.saturating_sub(1).min(MAX_BONUS_RULES) as u8;
    base.saturating_add(bonus).min(100)
}

/// One check's request to run a specific layer against a specific backend.
#[derive(Debug, Clone)]
pub struct EscalationRequest {
    /// Index into the `outcomes` slice `plan` was given — the caller applies
    /// results positionally, so this must stay in sync with that slice.
    pub outcome_index: usize,
    pub category: String,
    pub layer: ExecutionLayer,
    pub backend: Backend,
    pub additive: bool,
    pub fallback: Fallback,
    /// The whole-strategy budget, if the policy set one.
    pub timeout_ms: Option<u64>,
}

/// What a layer actually returned, in `armor-core`'s vocabulary. The
/// `armor-inference-client` types are mapped into this by the caller, which
/// is what keeps this crate free of a dependency on the transport.
#[derive(Debug, Clone)]
pub struct MlOutcome {
    /// `false` means the model believes the content is a problem.
    pub passed: bool,
    /// What the model believes should happen. Only consulted when
    /// `!passed`; `merge` still filters it through the check's policy.
    pub action: CheckAction,
    pub severity: Severity,
    pub confidence: Option<f32>,
    pub risk_score: u8,
    pub hits: Vec<RuleHit>,
    pub model_version: Option<String>,
    /// Whether the model's decision was `BLOCK` as opposed to `WARN`. A
    /// `WARN` is pinned to [`EnforcementMode::Warn`] regardless of the
    /// check's configured mode.
    pub wants_block: bool,
    pub latency_ms: f64,
}

/// Whether `w` permits escalating out of `from`.
///
/// `risk` is the deterministic score when leaving the deterministic layer;
/// `confidence` is the previous layer's confidence when leaving an ML layer.
/// Both terms are ANDed, and a `None` predicate is "no opinion", so an
/// `EscalateWhen` with neither field set always escalates.
pub fn should_escalate(
    w: Option<&EscalateWhen>,
    from: ExecutionLayer,
    risk: Option<u8>,
    confidence: Option<f32>,
) -> bool {
    let Some(w) = w else {
        return true; // a strategy with no predicate always escalates
    };

    if let Some((low, high)) = w.deterministic_score_between {
        // Only meaningful leaving the deterministic layer — once a model has
        // spoken, its confidence is the better signal and the rules' score
        // is stale.
        if from == ExecutionLayer::LocalDeterministic {
            let score = risk.unwrap_or(0);
            // Inclusive on both ends. A band written `[0, 70]` is read by
            // operators as "up to and including 70".
            if score < low || score > high {
                return false;
            }
        }
    }

    if let Some(threshold) = w.ml_confidence_below {
        if from != ExecutionLayer::LocalDeterministic {
            // A missing confidence escalates: an abstention should fail
            // toward the stronger layer rather than quietly resolve at the
            // weaker one.
            match confidence {
                Some(c) if c >= threshold => return false,
                _ => {}
            }
        }
    }

    true
}

/// Which checks want to escalate, to which layer, against which backend.
///
/// Returns at most one request per outcome — the *first* layer in the
/// strategy's `order` that is both escalatable-to and has a configured
/// backend. Multi-hop chains (classifier, then judge) are driven by the
/// caller re-planning after applying the first round, so that each hop's
/// predicate sees the previous hop's real confidence.
pub fn plan(policy: &PolicyConfig, outcomes: &[CheckOutcome]) -> Vec<EscalationRequest> {
    let mut requests = Vec::new();

    for (index, outcome) in outcomes.iter().enumerate() {
        let Some(config) = find_check(policy, outcomes, index) else {
            continue;
        };
        let Some(strategy) = &config.strategy else {
            continue; // no strategy ⇒ deterministic is terminal
        };

        // A check that timed out or blew up produced no evidence, and its
        // `fail_mode` has already resolved it. Escalating would turn a
        // detector regression into inference load, and would read the
        // resulting `risk_score: 0` as "the rules found nothing" when in
        // truth the rules never ran.
        if outcome.timed_out || outcome.error.is_some() {
            continue;
        }

        let from = outcome.execution_layer;
        let already_ran: HashSet<ExecutionLayer> = outcome.layers.iter().map(|l| l.layer).collect();

        let next = strategy
            .order
            .iter()
            .copied()
            .filter(|l| *l != ExecutionLayer::LocalDeterministic)
            .find(|l| !already_ran.contains(l));

        let Some(layer) = next else {
            continue; // every configured layer has already run
        };

        if !should_escalate(
            strategy.escalate_when.as_ref(),
            from,
            Some(outcome.risk_score),
            outcome.confidence,
        ) {
            continue;
        }

        let Some(backend) = config.backends.get(&layer) else {
            continue; // layer named in `order` with nothing configured to serve it
        };

        requests.push(EscalationRequest {
            outcome_index: index,
            category: outcome.category.clone(),
            layer,
            backend: backend.clone(),
            additive: strategy.additive,
            fallback: strategy.fallback,
            timeout_ms: strategy.timeout_ms,
        });
    }

    requests
}

/// Matches an outcome back to its config. Categories can repeat within a
/// policy (the orchestrator tests rely on it), so this pairs by position
/// among the enabled checks of that category rather than by category alone:
/// `outcomes[index]` is the Nth outcome of its category in `outcomes`
/// (`run_parallel`/`run_sequential` both preserve, for same-category
/// entries, the relative order they appear in `policy.checks` — the latter
/// via a stable sort), so it is paired with the Nth enabled check of that
/// category in `policy.checks`.
fn find_check<'a>(
    policy: &'a PolicyConfig,
    outcomes: &[CheckOutcome],
    index: usize,
) -> Option<&'a CheckConfig> {
    let category = &outcomes[index].category;
    let occurrence = outcomes[..=index]
        .iter()
        .filter(|o| &o.category == category)
        .count();
    policy
        .checks
        .iter()
        .filter(|c| c.enabled && &c.category == category)
        .nth(occurrence - 1)
}

/// Folds one layer's result into `outcome`, appending a [`LayerOutcome`]
/// either way and re-flagging which layer is selected.
///
/// # The asymmetry rule
///
/// An ML layer may **downgrade** a failing `warn`-mode deterministic result —
/// that is the false-positive reduction the chatty rule banks exist to be
/// rescued from — but may **never overturn a `Block`-mode deterministic
/// deny**. A model does not get to unblock what the rules blocked with full
/// enforcement authority. This is the complement of the layer's other
/// constraint — it never *raises* enforcement authority either: it can move
/// a result down, never up, and never out of a block.
pub fn merge(outcome: &mut CheckOutcome, layer: ExecutionLayer, result: MlOutcome, additive: bool) {
    // Captured once, on the first hop (`layers` is still empty and
    // `execution_layer`/`mode`/`action` still hold the deterministic
    // layer's own answer) — every field this reads from `outcome` gets
    // overwritten below, so a second hop must consult the sticky flag
    // rather than recomputing from now-stale-or-wrong current state.
    if outcome.layers.is_empty() {
        outcome.deterministic_block_deny = outcome.execution_layer
            == ExecutionLayer::LocalDeterministic
            && outcome.mode == EnforcementMode::Block
            && !outcome.passed
            && outcome.action == CheckAction::Deny;
    }

    // The model may not overturn a block-mode deny. It still gets recorded:
    // an operator needs to see that the classifier disagreed, and the
    // scorecard gate needs the disagreement rate.
    let may_select = !(outcome.deterministic_block_deny && result.passed);

    outcome.layers.push(LayerOutcome {
        layer,
        passed: result.passed,
        severity: result.severity,
        confidence: result.confidence,
        risk_score: result.risk_score,
        model_version: result.model_version.clone(),
        error: None,
        latency_ms: result.latency_ms,
        selected: may_select,
    });

    if !may_select {
        return;
    }

    for previous in outcome.layers.iter_mut().rev().skip(1) {
        previous.selected = false;
    }

    if additive {
        // The NER layer adds unstructured findings; it must never erase the
        // regex layer's, because those drive redaction. So hits accumulate
        // and a failure latches — an additive layer can only ever make the
        // outcome worse.
        outcome.hits.extend(result.hits);
        outcome.passed = outcome.passed && result.passed;
        outcome.severity = outcome.severity.max(result.severity);
    } else {
        outcome.hits = result.hits;
        outcome.passed = result.passed;
        outcome.severity = result.severity;
    }

    outcome.confidence = result.confidence;
    outcome.execution_layer = layer;
    outcome.model_version = result.model_version;
    outcome.risk_score = result.risk_score;

    if !outcome.passed {
        // A model's WARN is pinned to warn-mode regardless of how the check
        // is configured; a BLOCK carries the check's configured mode, which
        // is the most authority the policy ever granted it.
        if !result.wants_block {
            outcome.mode = EnforcementMode::Warn;
        }
        if result.action == CheckAction::Redact {
            outcome.action = CheckAction::Redact;
        }
    }
}

/// What a failed backend means for this check.
///
/// Records the failure as a non-selected [`LayerOutcome`] and stamps
/// `fallback_path` so the audit trail can tell "the model said allow" apart
/// from "the model never answered".
pub fn apply_fallback(
    outcome: &mut CheckOutcome,
    fallback: Fallback,
    layer: ExecutionLayer,
    reason: &str,
) {
    outcome.layers.push(LayerOutcome {
        layer,
        passed: true,
        severity: Severity::Low,
        confidence: None,
        risk_score: 0,
        model_version: None,
        error: Some(reason.to_string()),
        latency_ms: 0.0,
        selected: false,
    });

    let path = match fallback {
        Fallback::FallbackToDeterministic => "fallback_to_deterministic",
        Fallback::FailOpen => "fail_open",
        Fallback::FailClosed => "fail_closed",
    };
    outcome.fallback_path = Some(format!("{path}:{}", layer_name(layer)));

    match fallback {
        // Keep the deterministic answer untouched — a sidecar outage
        // degrades detection quality, not availability.
        Fallback::FallbackToDeterministic => {}
        Fallback::FailOpen => {
            outcome.passed = true;
            outcome.hits.clear();
        }
        Fallback::FailClosed => {
            outcome.passed = false;
            outcome.action = CheckAction::Deny;
        }
    }
}

fn layer_name(layer: ExecutionLayer) -> &'static str {
    match layer {
        ExecutionLayer::LocalDeterministic => "local_deterministic",
        ExecutionLayer::LocalMl => "local_ml",
        ExecutionLayer::LocalEmbedding => "local_embedding",
        ExecutionLayer::LocalLlm => "local_llm",
        ExecutionLayer::RemoteLlm => "remote_llm",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::schema::{FailMode, Strategy};

    fn hit(rule_id: &str, severity: Severity) -> RuleHit {
        RuleHit {
            rule_id: rule_id.to_string(),
            span: (0, 1),
            severity,
        }
    }

    // ---- risk_score ------------------------------------------------------

    #[test]
    fn no_hits_scores_zero_so_a_clean_sweep_escalates() {
        assert_eq!(risk_score(&[]), 0);
    }

    #[test]
    fn score_is_driven_by_max_severity_not_hit_count() {
        assert_eq!(risk_score(&[hit("a", Severity::Low)]), 10);
        assert_eq!(risk_score(&[hit("a", Severity::Medium)]), 30);
        assert_eq!(risk_score(&[hit("a", Severity::High)]), 50);
        assert_eq!(risk_score(&[hit("a", Severity::Critical)]), 80);
    }

    #[test]
    fn many_low_hits_stay_inside_the_default_band() {
        // The regression this formula exists to prevent: under a plain
        // severity sum these eight would total 80 and fall outside [0,70],
        // silently suppressing the escalation that would have retired them
        // as false positives.
        let hits: Vec<RuleHit> = (0..8)
            .map(|i| hit(&format!("rule_{i}"), Severity::Low))
            .collect();
        let score = risk_score(&hits);
        assert_eq!(score, 20);
        assert!(score <= 70, "chatty low-severity hits must still escalate");
    }

    #[test]
    fn one_chatty_rule_matching_repeatedly_is_one_signal() {
        let hits: Vec<RuleHit> = (0..40).map(|_| hit("same_rule", Severity::Low)).collect();
        // 10 base + 2 for... nothing: there is only one distinct rule.
        assert_eq!(risk_score(&hits), 10);
    }

    #[test]
    fn corroboration_bonus_saturates_and_never_crosses_a_tier() {
        let hits: Vec<RuleHit> = (0..50)
            .map(|i| hit(&format!("rule_{i}"), Severity::High))
            .collect();
        // 50 base + capped 10 bonus — still below the 70 band edge, so a
        // High-severity check always remains escalatable.
        assert_eq!(risk_score(&hits), 60);
    }

    #[test]
    fn critical_is_weighted_explicitly_and_lands_outside_the_band() {
        let score = risk_score(&[hit("a", Severity::Critical)]);
        assert_eq!(score, 80);
        assert!(score > 70, "critical must not sit on the band boundary");
    }

    #[test]
    fn max_severity_wins_over_a_crowd_of_weaker_hits() {
        let mut hits: Vec<RuleHit> = (0..5)
            .map(|i| hit(&format!("low_{i}"), Severity::Low))
            .collect();
        hits.push(hit("crit", Severity::Critical));
        // 6 distinct rules ⇒ 5 extra ⇒ the bonus saturates at 10.
        assert_eq!(risk_score(&hits), 80 + 10);
    }

    #[test]
    fn score_is_capped_at_100() {
        let hits: Vec<RuleHit> = (0..20)
            .map(|i| hit(&format!("rule_{i}"), Severity::Critical))
            .collect();
        assert!(risk_score(&hits) <= 100);
    }

    // ---- should_escalate -------------------------------------------------

    fn band(low: u8, high: u8) -> EscalateWhen {
        EscalateWhen {
            deterministic_score_between: Some((low, high)),
            ml_confidence_below: None,
        }
    }

    #[test]
    fn no_predicate_always_escalates() {
        assert!(should_escalate(
            None,
            ExecutionLayer::LocalDeterministic,
            Some(95),
            None
        ));
    }

    #[test]
    fn band_is_inclusive_on_both_ends() {
        let w = band(20, 70);
        let from = ExecutionLayer::LocalDeterministic;
        assert!(should_escalate(Some(&w), from, Some(20), None));
        assert!(should_escalate(Some(&w), from, Some(70), None));
        assert!(!should_escalate(Some(&w), from, Some(19), None));
        assert!(!should_escalate(Some(&w), from, Some(71), None));
    }

    #[test]
    fn score_above_the_band_does_not_escalate() {
        // "The rules are already confident" — don't spend a forward pass.
        let w = band(0, 70);
        let score = risk_score(&[hit("a", Severity::Critical)]);
        assert!(!should_escalate(
            Some(&w),
            ExecutionLayer::LocalDeterministic,
            Some(score),
            None
        ));
    }

    #[test]
    fn a_clean_sweep_escalates_under_the_default_band() {
        let w = band(0, 70);
        assert!(should_escalate(
            Some(&w),
            ExecutionLayer::LocalDeterministic,
            Some(risk_score(&[])),
            None
        ));
    }

    #[test]
    fn missing_confidence_escalates_toward_the_stronger_layer() {
        let w = EscalateWhen {
            deterministic_score_between: None,
            ml_confidence_below: Some(0.8),
        };
        assert!(should_escalate(
            Some(&w),
            ExecutionLayer::LocalMl,
            None,
            None
        ));
    }

    #[test]
    fn confident_ml_layer_does_not_escalate_to_the_judge() {
        let w = EscalateWhen {
            deterministic_score_between: None,
            ml_confidence_below: Some(0.8),
        };
        assert!(!should_escalate(
            Some(&w),
            ExecutionLayer::LocalMl,
            None,
            Some(0.95)
        ));
        assert!(should_escalate(
            Some(&w),
            ExecutionLayer::LocalMl,
            None,
            Some(0.5)
        ));
    }

    #[test]
    fn deterministic_band_is_not_applied_when_leaving_an_ml_layer() {
        // The rules' score is stale once a model has spoken.
        let w = band(0, 70);
        assert!(should_escalate(
            Some(&w),
            ExecutionLayer::LocalMl,
            Some(95),
            None
        ));
    }

    // ---- plan ------------------------------------------------------------

    fn ml_backend(task: &str) -> Backend {
        Backend {
            task: task.to_string(),
            endpoint_url: None,
            model_id: None,
            revision: None,
            threshold: None,
            timeout_ms: None,
            params: None,
        }
    }

    fn check_with_strategy(category: &str, escalate_when: Option<EscalateWhen>) -> CheckConfig {
        let mut backends = std::collections::HashMap::new();
        backends.insert(ExecutionLayer::LocalMl, ml_backend(category));
        CheckConfig {
            category: category.to_string(),
            strategy: Some(Strategy {
                order: vec![ExecutionLayer::LocalDeterministic, ExecutionLayer::LocalMl],
                escalate_when,
                fallback: Fallback::FallbackToDeterministic,
                additive: false,
                timeout_ms: None,
            }),
            backends,
            ..Default::default()
        }
    }

    fn policy_of(checks: Vec<CheckConfig>) -> PolicyConfig {
        PolicyConfig {
            id: "gr_test".to_string(),
            execution_mode: Default::default(),
            fail_mode: FailMode::FailOpen,
            normalize: Default::default(),
            checks,
        }
    }

    /// Models what the orchestrator actually produces: `severity` is the
    /// detector's own, which for these fixtures is the max over its hits.
    fn outcome(category: &str, hits: Vec<RuleHit>) -> CheckOutcome {
        CheckOutcome {
            category: category.to_string(),
            passed: hits.is_empty(),
            risk_score: risk_score(&hits),
            severity: hits
                .iter()
                .map(|h| h.severity)
                .max()
                .unwrap_or(Severity::Low),
            hits,
            ..Default::default()
        }
    }

    #[test]
    fn check_without_a_strategy_never_escalates() {
        let policy = policy_of(vec![CheckConfig {
            category: "pci".to_string(),
            ..Default::default()
        }]);
        let outcomes = vec![outcome("pci", vec![])];
        assert!(plan(&policy, &outcomes).is_empty());
    }

    #[test]
    fn clean_check_with_a_strategy_escalates_to_the_ml_layer() {
        let policy = policy_of(vec![check_with_strategy(
            "prompt_injection",
            Some(band(0, 70)),
        )]);
        let outcomes = vec![outcome("prompt_injection", vec![])];
        let requests = plan(&policy, &outcomes);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].layer, ExecutionLayer::LocalMl);
        assert_eq!(requests[0].backend.task, "prompt_injection");
        assert_eq!(requests[0].outcome_index, 0);
    }

    #[test]
    fn critical_deterministic_hit_does_not_escalate() {
        let policy = policy_of(vec![check_with_strategy(
            "prompt_injection",
            Some(band(0, 70)),
        )]);
        let outcomes = vec![outcome(
            "prompt_injection",
            vec![hit("pi_exact", Severity::Critical)],
        )];
        assert!(plan(&policy, &outcomes).is_empty());
    }

    #[test]
    fn noisy_low_severity_check_still_escalates() {
        let policy = policy_of(vec![check_with_strategy(
            "prompt_injection",
            Some(band(0, 70)),
        )]);
        let hits: Vec<RuleHit> = (0..8)
            .map(|i| hit(&format!("fuzzy_{i}"), Severity::Low))
            .collect();
        let outcomes = vec![outcome("prompt_injection", hits)];
        assert_eq!(plan(&policy, &outcomes).len(), 1);
    }

    #[test]
    fn timed_out_check_does_not_escalate() {
        let policy = policy_of(vec![check_with_strategy("prompt_injection", None)]);
        let mut o = outcome("prompt_injection", vec![]);
        o.timed_out = true;
        assert!(plan(&policy, &[o]).is_empty());
    }

    #[test]
    fn errored_check_does_not_escalate() {
        let policy = policy_of(vec![check_with_strategy("prompt_injection", None)]);
        let mut o = outcome("prompt_injection", vec![]);
        o.error = Some("unknown check category".to_string());
        assert!(plan(&policy, &[o]).is_empty());
    }

    #[test]
    fn layer_named_in_order_with_no_backend_is_skipped() {
        let mut check = check_with_strategy("prompt_injection", None);
        check.backends.clear();
        let policy = policy_of(vec![check]);
        assert!(plan(&policy, &[outcome("prompt_injection", vec![])]).is_empty());
    }

    #[test]
    fn a_layer_that_already_ran_is_not_requested_again() {
        let policy = policy_of(vec![check_with_strategy("prompt_injection", None)]);
        let mut o = outcome("prompt_injection", vec![]);
        o.layers.push(LayerOutcome {
            layer: ExecutionLayer::LocalMl,
            passed: true,
            severity: Severity::Low,
            confidence: Some(0.9),
            risk_score: 0,
            model_version: None,
            error: None,
            latency_ms: 1.0,
            selected: true,
        });
        assert!(plan(&policy, &[o]).is_empty());
    }

    // ---- merge -----------------------------------------------------------

    fn ml(passed: bool, severity: Severity, confidence: Option<f32>) -> MlOutcome {
        MlOutcome {
            passed,
            action: CheckAction::Deny,
            severity,
            confidence,
            risk_score: if passed { 5 } else { 90 },
            hits: Vec::new(),
            model_version: Some("m@v1".to_string()),
            wants_block: !passed,
            latency_ms: 4.0,
        }
    }

    #[test]
    fn ml_layer_may_downgrade_a_failing_warn_mode_result() {
        let mut o = outcome("prompt_injection", vec![hit("fuzzy", Severity::Low)]);
        o.mode = EnforcementMode::Warn;
        o.passed = false;

        merge(
            &mut o,
            ExecutionLayer::LocalMl,
            ml(true, Severity::Low, Some(0.97)),
            false,
        );

        assert!(o.passed, "a warn-mode false positive must be retirable");
        assert_eq!(o.execution_layer, ExecutionLayer::LocalMl);
        assert_eq!(o.model_version.as_deref(), Some("m@v1"));
        assert!(o.layers[0].selected);
    }

    #[test]
    fn ml_layer_may_never_overturn_a_block_mode_deny() {
        let mut o = outcome("pci", vec![hit("pci_card", Severity::High)]);
        o.mode = EnforcementMode::Block;
        o.passed = false;
        o.action = CheckAction::Deny;

        merge(
            &mut o,
            ExecutionLayer::LocalMl,
            ml(true, Severity::Low, Some(0.99)),
            false,
        );

        assert!(!o.passed, "the block-mode deny must stand");
        assert_eq!(o.execution_layer, ExecutionLayer::LocalDeterministic);
        assert_eq!(o.mode, EnforcementMode::Block);
        // Recorded but not selected — the disagreement is still auditable.
        assert_eq!(o.layers.len(), 1);
        assert!(!o.layers[0].selected);
    }

    #[test]
    fn ml_layer_may_still_escalate_a_passing_check_to_a_failure() {
        let mut o = outcome("prompt_injection", vec![]);
        merge(
            &mut o,
            ExecutionLayer::LocalMl,
            ml(false, Severity::High, Some(0.93)),
            false,
        );
        assert!(!o.passed);
        assert_eq!(o.severity, Severity::High);
        assert_eq!(o.mode, EnforcementMode::Block);
    }

    #[test]
    fn a_model_warn_is_pinned_to_warn_mode() {
        let mut o = outcome("prompt_injection", vec![]);
        let mut result = ml(false, Severity::Medium, Some(0.6));
        result.wants_block = false;
        merge(&mut o, ExecutionLayer::LocalMl, result, false);
        assert!(!o.passed);
        assert_eq!(
            o.mode,
            EnforcementMode::Warn,
            "a model may not claim block authority the policy granted for rules"
        );
    }

    #[test]
    fn additive_merge_never_erases_the_regex_layers_hits() {
        let mut o = outcome("pii", vec![hit("pii_email", Severity::Medium)]);
        o.passed = false;
        o.mode = EnforcementMode::Warn;

        let mut result = ml(true, Severity::Low, Some(0.9));
        result.hits = vec![hit("ner_person", Severity::Low)];
        merge(&mut o, ExecutionLayer::LocalMl, result, true);

        assert_eq!(o.hits.len(), 2, "NER adds, it never replaces");
        assert!(!o.passed, "an additive layer can only make it worse");
        assert_eq!(o.severity, Severity::Medium);
    }

    #[test]
    fn replace_merge_swaps_the_hits_out() {
        let mut o = outcome("prompt_injection", vec![hit("fuzzy", Severity::Low)]);
        o.mode = EnforcementMode::Warn;
        o.passed = false;
        let mut result = ml(false, Severity::High, Some(0.9));
        result.hits = vec![hit("ml_pi", Severity::High)];
        merge(&mut o, ExecutionLayer::LocalMl, result, false);
        assert_eq!(o.hits.len(), 1);
        assert_eq!(o.hits[0].rule_id, "ml_pi");
    }

    #[test]
    fn a_second_layer_deselects_the_first() {
        let mut o = outcome("prompt_injection", vec![]);
        merge(
            &mut o,
            ExecutionLayer::LocalMl,
            ml(true, Severity::Low, Some(0.4)),
            false,
        );
        merge(
            &mut o,
            ExecutionLayer::LocalLlm,
            ml(false, Severity::High, Some(0.9)),
            false,
        );
        assert_eq!(o.layers.len(), 2);
        assert!(!o.layers[0].selected);
        assert!(o.layers[1].selected);
        assert_eq!(o.execution_layer, ExecutionLayer::LocalLlm);
    }

    // ---- apply_fallback --------------------------------------------------

    #[test]
    fn fallback_to_deterministic_keeps_the_rules_answer() {
        let mut o = outcome("prompt_injection", vec![hit("fuzzy", Severity::Low)]);
        o.passed = false;
        apply_fallback(
            &mut o,
            Fallback::FallbackToDeterministic,
            ExecutionLayer::LocalMl,
            "escalation_budget_exceeded",
        );
        assert!(!o.passed);
        assert_eq!(o.hits.len(), 1);
        assert_eq!(
            o.fallback_path.as_deref(),
            Some("fallback_to_deterministic:local_ml")
        );
        assert_eq!(o.execution_layer, ExecutionLayer::LocalDeterministic);
        assert!(!o.layers[0].selected);
        assert_eq!(
            o.layers[0].error.as_deref(),
            Some("escalation_budget_exceeded")
        );
    }

    #[test]
    fn fail_open_fallback_clears_the_failure() {
        let mut o = outcome("prompt_injection", vec![hit("fuzzy", Severity::Low)]);
        o.passed = false;
        apply_fallback(
            &mut o,
            Fallback::FailOpen,
            ExecutionLayer::LocalMl,
            "backend_unavailable",
        );
        assert!(o.passed);
        assert!(o.hits.is_empty());
        assert_eq!(o.fallback_path.as_deref(), Some("fail_open:local_ml"));
    }

    #[test]
    fn fail_closed_fallback_denies() {
        let mut o = outcome("prompt_injection", vec![]);
        apply_fallback(
            &mut o,
            Fallback::FailClosed,
            ExecutionLayer::LocalLlm,
            "circuit_open",
        );
        assert!(!o.passed);
        assert_eq!(o.action, CheckAction::Deny);
        assert_eq!(o.fallback_path.as_deref(), Some("fail_closed:local_llm"));
    }
}
