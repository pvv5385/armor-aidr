//! Sentiment/tone check — narrow phrase-level patterns for condescending,
//! dismissive, and sarcastic output, per `rules/sentiment/rules.yaml`'s
//! header on why this is a small curated bank rather than a general
//! polarity lexicon (that belongs behind a real classifier, not a
//! rules-based detector like this one).

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::models::{CheckAction, DetectorResult, Severity};
use crate::policy::schema::CheckOptions;

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/sentiment/rules.yaml"),
        "rules/sentiment/rules.yaml",
    )
});

static DETECTOR: SimpleDetector =
    SimpleDetector::new("sentiment", Severity::Low, CheckAction::Flag);

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
    fn condescending_phrase_flags() {
        let result = evaluate(
            "Let me explain this slowly for you since you clearly can't follow",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Flag);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "condescending-explanation"));
    }

    #[test]
    fn sarcastic_praise_flags() {
        let result = evaluate(
            "Oh great, another brilliant idea from the team",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Flag);
        assert!(result.hits.iter().any(|h| h.rule_id == "sarcastic-praise"));
    }

    #[test]
    fn neutral_explanation_allows() {
        let result = evaluate(
            "Let me explain how pagination works in this API",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pattern_match_disabled_returns_no_hits() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let result = evaluate("that's not my problem", &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
