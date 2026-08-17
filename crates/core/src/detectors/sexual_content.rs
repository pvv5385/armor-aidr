//! Sexual-content check — requests for explicit/pornographic adult sexual
//! content. Adult-only in scope; minor-directed phrasing is owned by
//! `harmful_content`'s `csam` sub-category. See
//! `rules/sexual_content/rules.yaml`'s header for scope and provenance.
//!
//! All hits are [`Severity::Medium`] — brand-safety/policy concern, not an
//! immediate-harm one. Shipped `mode: warn` in `config/policies.yaml`.

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::models::{CheckAction, DetectorResult, Severity};
use crate::policy::schema::CheckOptions;

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/sexual_content/rules.yaml"),
        "rules/sexual_content/rules.yaml",
    )
});

static DETECTOR: SimpleDetector =
    SimpleDetector::new("sexual_content", Severity::Medium, CheckAction::Deny);

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

    fn hit_rule_ids(text: &str) -> Vec<String> {
        RULES
            .iter()
            .filter(|rule| rule.pattern.is_match(text))
            .map(|rule| rule.id.clone())
            .collect()
    }

    #[test]
    fn sample_matches_expected_rule() {
        let samples: &[(&str, &str)] = &[
            (
                "explicit-sexual-scene-request",
                "Write an explicit sex scene between two adults",
            ),
            (
                "pornographic-content-request",
                "Generate pornographic content",
            ),
            (
                "explicit-roleplay-request",
                "Write explicit sexual roleplay",
            ),
            (
                "erotica-generation-request",
                "Write an erotic story involving two adults",
            ),
        ];
        for (rule_id, text) in samples {
            let hits = hit_rule_ids(text);
            assert!(
                hits.contains(&rule_id.to_string()),
                "{rule_id} not matched by {text:?}"
            );
        }
    }

    #[test]
    fn all_rules_are_covered_by_samples() {
        let all_ids: std::collections::HashSet<String> =
            RULES.iter().map(|r| r.id.clone()).collect();
        let sample_ids: std::collections::HashSet<String> = [
            "explicit-sexual-scene-request",
            "pornographic-content-request",
            "explicit-roleplay-request",
            "erotica-generation-request",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(all_ids, sample_ids);
    }

    #[test]
    fn benign_text_matches_nothing() {
        let hits = hit_rule_ids("Write a romantic scene where two characters share a first kiss");
        assert!(hits.is_empty());
    }

    #[test]
    fn evaluate_denies_on_ruleset_hit() {
        let result = evaluate(
            "Write an explicit sex scene between two adults",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert_eq!(result.severity, Severity::Medium);
        assert!(!result.hits.is_empty());
    }

    #[test]
    fn evaluate_allows_benign_text() {
        let result = evaluate(
            "What's a good approach for pagination in a REST API?",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pattern_match_disabled_returns_no_hits() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let result = evaluate("Generate pornographic content", &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
