//! Excessive-agency check (base tier) — rule-based risk score on agent
//! plans/loop steps by action class: destructive, financial,
//! privilege-escalation, arbitrary-code-execution. `destructive` hits are
//! always [`Severity::Critical`] (irreversible data loss); the other three
//! classes are [`Severity::High`]. Deep contextual judgment on tool
//! arguments (e.g. reasoning about whether a given SQL predicate is actually
//! an exploit) would need an LLM judge call and isn't built here — this only
//! recognizes the surface-level action shape.

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::models::{CheckAction, DetectorResult, Severity};
use crate::policy::schema::CheckOptions;

fn severity_for_category(category: &str) -> Severity {
    if category == "destructive" {
        Severity::Critical
    } else {
        Severity::High
    }
}

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/excessive_agency/rules.yaml"),
        "rules/excessive_agency/rules.yaml",
    )
});

static DETECTOR: SimpleDetector = SimpleDetector::by_category(
    "excessive_agency",
    severity_for_category,
    Severity::High,
    CheckAction::Deny,
);

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    DETECTOR.evaluate(&RULES, text, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructive_action_denies_as_critical() {
        let result = evaluate(
            "step 1: drop the production database",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert_eq!(result.severity, Severity::Critical);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "destructive-action"));
    }

    #[test]
    fn financial_action_denies_as_high() {
        let result = evaluate(
            "step 2: transfer funds to the vendor's account",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert_eq!(result.severity, Severity::High);
        assert!(result.hits.iter().any(|h| h.rule_id == "financial-action"));
    }

    #[test]
    fn ordinary_plan_step_allows() {
        let result = evaluate(
            "step 1: search the knowledge base for the answer",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pattern_match_disabled_returns_no_hits() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let result = evaluate("drop the production database", &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
