//! Compliance check — mandatory-disclaimer / AI-disclosure enforcement.
//! Fires only when *both* halves are true: the text matches an
//! advice-trigger pattern for a regulated vertical
//! (`rules/compliance/rules.yaml` — medical/financial/legal) *and* none of
//! the deployment-supplied `options.disclaimer_phrases` appear anywhere in
//! the text. Neither half alone is a violation — an advice-shaped sentence
//! next to the right disclaimer is compliant; a disclaimer with no advice
//! nearby is just boilerplate.
//!
//! Shipped `enabled: false` in `config/policies.yaml`: the required
//! disclaimer text is deployment-specific (a healthcare customer's wording
//! differs from a fintech customer's), so this needs `disclaimer_phrases`
//! configured before it does anything useful — same convention as
//! `custom_regex`/`keyword_blocklist`.

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::models::{CheckAction, DetectorResult, Severity};
use crate::policy::schema::CheckOptions;

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/compliance/rules.yaml"),
        "rules/compliance/rules.yaml",
    )
});

static DETECTOR: SimpleDetector =
    SimpleDetector::new("compliance", Severity::Medium, CheckAction::Deny);

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

fn disclaimer_present(text: &str, options: &CheckOptions) -> bool {
    let disclaimer_phrases = options.str_list_option("disclaimer_phrases");
    let lower_text = text.to_lowercase();
    disclaimer_phrases
        .iter()
        .any(|d| lower_text.contains(&d.to_lowercase()))
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    DETECTOR.evaluate_with(
        &RULES,
        text,
        options,
        |text, options| {
            rule_loader::pattern_match_enabled(options) && !disclaimer_present(text, options)
        },
        |_, _| true,
        |_, _| Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(disclaimers: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("disclaimer_phrases", disclaimers);
        o
    }

    #[test]
    fn advice_without_disclaimer_denies() {
        let result = evaluate(
            "You should take this medication twice daily",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "medical-advice-trigger"));
    }

    #[test]
    fn advice_with_configured_disclaimer_allows() {
        let text = "You should take this medication twice daily. This is not a substitute for professional medical advice.";
        let result = evaluate(
            text,
            &opts(&["not a substitute for professional medical advice"]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn financial_advice_without_disclaimer_denies() {
        let result = evaluate(
            "You should invest in index funds for retirement",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "financial-advice-trigger"));
    }

    #[test]
    fn no_advice_trigger_allows() {
        let result = evaluate(
            "Index funds are a common way to diversify a portfolio",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pattern_match_disabled_returns_no_hits() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let result = evaluate("You should take this medication twice daily", &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
