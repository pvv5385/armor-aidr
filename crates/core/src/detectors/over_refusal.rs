//! Over-refusal check — flags refusal boilerplate in a model's output when
//! the request didn't warrant declining. This is a *quality/eval* signal,
//! not a safety block: never denies, only flags, same as
//! [`super::sentiment`]. Intended use: wire this into the eval harness as a
//! tripwire for other detectors — a spike in the over-refusal rate right
//! after a threshold change elsewhere means that other detector got too
//! aggressive and is scaring the model into declining benign requests.
//!
//! Substring/regex tier only (llm-guard's `NoRefusalLight`) — a
//! classifier-backed tier (e.g. `ProtectAI/distilroberta-base-rejection-v1`)
//! would catch paraphrased refusals too, but isn't built here.

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::models::{CheckAction, DetectorResult, Severity};
use crate::policy::schema::CheckOptions;

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/over_refusal/rules.yaml"),
        "rules/over_refusal/rules.yaml",
    )
});

static DETECTOR: SimpleDetector =
    SimpleDetector::new("over_refusal", Severity::Low, CheckAction::Flag);

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
    fn as_an_ai_disclaimer_flags() {
        let result = evaluate(
            "As an AI language model, I cannot provide medical advice",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Flag);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "as-an-ai-disclaimer-refusal"));
    }

    #[test]
    fn cannot_assist_flags() {
        let result = evaluate(
            "I cannot assist with that request",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Flag);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "cannot-assist-with-that-request"));
    }

    #[test]
    fn ordinary_hedging_allows() {
        let result = evaluate(
            "I can't say for certain, but the deadline is probably next week",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn helpful_answer_allows() {
        let result = evaluate(
            "Sure, here's how you can reset your password in three steps",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pattern_match_disabled_returns_no_hits() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let result = evaluate("As an AI language model, I cannot help with that", &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
