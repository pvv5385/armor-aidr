//! Composes per-check outcomes into the engine's final [`Decision`], ported
//! from `app/guardrails/types.py`'s `CheckResult`/`EngineResult` plus the
//! composition rule embedded in `engine.py`'s `_run_parallel`/`_run_sequential`
//! (AND semantics: any enabled `block`-mode check denying flips the run to
//! `BLOCK`). `warn`-mode is this port's one addition beyond the Python
//! source (whose `CheckSpec.mode` is only `block`/`monitor`): a failing
//! `warn`-mode check surfaces as `Verdict::Warn` instead of being
//! indistinguishable from a silent `monitor`. `Verdict::Redact` is the
//! second: a `block`-mode check configured `on_fail: redact` (or escalated
//! to it by a model layer — `escalation::merge`) now composes to it, which
//! is what makes the variant reachable through a policy at all.
//!
//! `redacted_text` (see [`crate::engine::redact`]) is deliberately *not*
//! gated on the verdict: it's computed unconditionally from every check's
//! hits, so a caller doing redact-and-continue never has to branch on which
//! policy fired to find it. `Verdict::Redact` says the caller is *obliged*
//! to use it; every other verdict still offers it.

use serde::{Deserialize, Serialize};

use crate::models::{CheckAction, EnforcementMode, RuleHit, Severity, Verdict};
use crate::policy::schema::ExecutionLayer;

/// One check's result after the orchestrator has run it against every
/// normalized view and applied policy (mode, on_fail) — mirrors Python's
/// `CheckResult`, stamped centrally rather than by the check function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckOutcome {
    pub category: String,
    pub passed: bool,
    pub action: CheckAction,
    /// The detector's own severity for this run — carried through from
    /// [`crate::models::DetectorResult::severity`] rather than dropped, so a
    /// caller sees more than a flat block/allow verdict. Defaults to `Low`
    /// for the no-detector-ran paths (unknown category, timeout, panic).
    pub severity: Severity,
    /// The detector's confidence in this run, when it reports one — also
    /// carried through from `DetectorResult` rather than dropped.
    pub confidence: Option<f32>,
    pub hits: Vec<RuleHit>,
    pub view: String,
    /// The exact text `hits[].span` are byte offsets into. Equal to the
    /// original input when `view == "raw"`; on any other view (e.g.
    /// `base64`, `homoglyph`) this is the *transformed* text for that view,
    /// not the original — spans are meaningless against anything else. See
    /// `crate::engine::redact`'s module doc for why only `"raw"`-view hits
    /// drive `redacted_text`.
    pub view_text: String,
    pub error: Option<String>,
    pub timed_out: bool,
    pub latency_ms: f64,
    pub mode: EnforcementMode,
    /// The deterministic layer's ordinal routing signal for this check — see
    /// [`crate::engine::escalation::risk_score`]. It exists to decide
    /// whether a stronger layer is worth running; it is **not** a calibrated
    /// probability, is never surfaced as "risk" in the public API, and
    /// nothing thresholds a business decision on it.
    #[serde(default)]
    pub risk_score: u8,
    /// Every layer that ran for this check, in attempt order, with exactly
    /// one flagged [`LayerOutcome::selected`]. Empty on the deterministic-only
    /// path, which is every path today.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<LayerOutcome>,
    /// Which layer produced the scalar fields above (`passed`, `action`,
    /// `severity`, `hits`).
    #[serde(default)]
    pub execution_layer: ExecutionLayer,
    /// `"model_id@revision"` of the selected layer, when a model produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// Set when a backend failed and the configured [`Fallback`] resolved
    /// the check instead, e.g. `"fallback_to_deterministic:local_ml"`. The
    /// audit trail needs to distinguish "the model said allow" from "the
    /// model never answered and we kept the rules' answer".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_path: Option<String>,
    /// Whether the *original* deterministic layer's answer, before any
    /// escalation, was a block-mode deny. Set once, on the first call to
    /// [`crate::engine::escalation::merge`], and sticky from then on —
    /// `mode`, `action`, and `execution_layer` are all overwritten by every
    /// subsequent hop, so the asymmetry rule (no layer may ever overturn a
    /// block-mode deterministic deny) has to be checked against this rather
    /// than against those mutable fields, or it only holds for the first hop
    /// of a multi-hop escalation chain.
    #[serde(default)]
    pub deterministic_block_deny: bool,
}

/// One layer's attempt at a check. Recorded whether or not it won, so an
/// operator can see that the classifier ran and abstained rather than
/// inferring it from a missing field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerOutcome {
    pub layer: ExecutionLayer,
    pub passed: bool,
    pub severity: Severity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub risk_score: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// Why this layer produced nothing usable, when it didn't.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub latency_ms: f64,
    /// Whether this is the layer whose result was promoted onto the
    /// `CheckOutcome`'s scalar fields.
    pub selected: bool,
}

/// Mirrors the `serde` defaults field-for-field. Construction sites in the
/// orchestrator spread this so that adding an optional field here doesn't
/// have to touch each of the four no-detector-ran paths.
impl Default for CheckOutcome {
    fn default() -> Self {
        Self {
            category: String::new(),
            passed: true,
            action: CheckAction::Deny,
            severity: Severity::Low,
            confidence: None,
            hits: Vec::new(),
            view: "raw".to_string(),
            view_text: String::new(),
            error: None,
            timed_out: false,
            latency_ms: 0.0,
            mode: EnforcementMode::Block,
            risk_score: 0,
            layers: Vec::new(),
            execution_layer: ExecutionLayer::LocalDeterministic,
            model_version: None,
            fallback_path: None,
            deterministic_block_deny: false,
        }
    }
}

/// The engine's final, composed decision for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub verdict: Verdict,
    pub outcomes: Vec<CheckOutcome>,
    /// The scanned text with every `"raw"`-view hit masked behind a
    /// `<CATEGORY:RULE_ID:n>` placeholder — see [`crate::engine::redact`].
    /// Always present, even when nothing fired (in which case it equals the
    /// input text unchanged), so callers doing redact-and-continue never
    /// have to branch on the verdict to find it.
    pub redacted_text: String,
}

/// AND semantics, with [`EnforcementMode`] deciding how much authority a
/// failing check's [`CheckAction`] carries and the action deciding what that
/// authority is used for:
///
/// | mode      | `deny`  | `redact` | `flag`/`log` |
/// |-----------|---------|----------|--------------|
/// | `block`   | `BLOCK` | `REDACT` | —            |
/// | `warn`    | `WARN`  | `WARN`   | —            |
/// | `monitor` | —       | —        | —            |
///
/// Precedence is `BLOCK` > `REDACT` > `WARN` > `ALLOW`. `REDACT` sits above
/// `WARN` because it is the one verdict that obliges the caller to *do*
/// something — send `redacted_text` rather than the original. Degrading it
/// to an advisory `WARN` because some other check also warned would pass the
/// unredacted text through, which is the opposite of what the policy asked
/// for.
///
/// A `monitor`-mode check can fail (visible in `outcomes`) without ever
/// flipping the verdict — "log it, pass the request." `flag`/`log` actions
/// are likewise non-enforcing at every mode; nothing produces them yet
/// (the `ask` degradation policy is still unbuilt), and when something
/// does they belong in `outcomes`, not in the verdict.
pub fn compose(outcomes: Vec<CheckOutcome>, redacted_text: String) -> Decision {
    let failed = |mode: EnforcementMode, action: CheckAction| {
        outcomes
            .iter()
            .any(|o| o.mode == mode && !o.passed && o.action == action)
    };
    let enforcing = |mode: EnforcementMode| {
        failed(mode, CheckAction::Deny) || failed(mode, CheckAction::Redact)
    };
    let verdict = if failed(EnforcementMode::Block, CheckAction::Deny) {
        Verdict::Block
    } else if failed(EnforcementMode::Block, CheckAction::Redact) {
        Verdict::Redact
    } else if enforcing(EnforcementMode::Warn) {
        Verdict::Warn
    } else {
        Verdict::Allow
    };
    Decision {
        verdict,
        outcomes,
        redacted_text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failing(mode: EnforcementMode, action: CheckAction) -> CheckOutcome {
        CheckOutcome {
            category: "pii".to_string(),
            passed: false,
            action,
            mode,
            ..Default::default()
        }
    }

    fn verdict_of(outcomes: Vec<CheckOutcome>) -> Verdict {
        compose(outcomes, String::new()).verdict
    }

    #[test]
    fn a_clean_run_allows() {
        assert_eq!(verdict_of(Vec::new()), Verdict::Allow);
        assert_eq!(
            verdict_of(vec![CheckOutcome {
                passed: true,
                action: CheckAction::Deny,
                ..Default::default()
            }]),
            Verdict::Allow
        );
    }

    #[test]
    fn the_mode_by_action_table_holds() {
        use CheckAction::{Deny, Flag, Log, Redact};
        use EnforcementMode::{Block, Monitor, Warn};
        let cases = [
            (Block, Deny, Verdict::Block),
            (Block, Redact, Verdict::Redact),
            (Block, Flag, Verdict::Allow),
            (Block, Log, Verdict::Allow),
            (Warn, Deny, Verdict::Warn),
            (Warn, Redact, Verdict::Warn),
            (Warn, Flag, Verdict::Allow),
            (Monitor, Deny, Verdict::Allow),
            (Monitor, Redact, Verdict::Allow),
        ];
        for (mode, action, expected) in cases {
            assert_eq!(
                verdict_of(vec![failing(mode, action)]),
                expected,
                "{mode:?} + {action:?}"
            );
        }
    }

    #[test]
    fn a_denial_outranks_a_redaction() {
        assert_eq!(
            verdict_of(vec![
                failing(EnforcementMode::Block, CheckAction::Redact),
                failing(EnforcementMode::Block, CheckAction::Deny),
            ]),
            Verdict::Block
        );
    }

    #[test]
    fn a_redaction_outranks_a_warning() {
        // The caller is obliged to substitute `redacted_text`; downgrading
        // to WARN because something else warned would ship the original.
        assert_eq!(
            verdict_of(vec![
                failing(EnforcementMode::Warn, CheckAction::Deny),
                failing(EnforcementMode::Block, CheckAction::Redact),
            ]),
            Verdict::Redact
        );
    }
}
