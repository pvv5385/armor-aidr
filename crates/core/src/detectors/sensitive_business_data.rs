//! Sensitive-business-data check — dictionary/pattern bank for M&A,
//! pricing, roadmap, customer-list, and financials leakage. Phrase-level
//! patterns (an adjective qualifying the noun) rather than bare nouns,
//! which are everyday business vocabulary on their own — see
//! `rules/sensitive_business_data/rules.yaml`'s header.
//!
//! `options.allowlist` exempts specific phrases a deployment has decided
//! are non-sensitive in its own context (e.g. a company whose product *is*
//! an M&A due-diligence tool).

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::models::{CheckAction, DetectorResult, Severity};
use crate::policy::schema::CheckOptions;

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/sensitive_business_data/rules.yaml"),
        "rules/sensitive_business_data/rules.yaml",
    )
});

static DETECTOR: SimpleDetector = SimpleDetector::new(
    "sensitive_business_data",
    Severity::Medium,
    CheckAction::Deny,
);

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let allowlist = options.str_list_option("allowlist");

    DETECTOR.evaluate_with(
        &RULES,
        text,
        options,
        |_, _| true,
        |_, matched: &str| !allowlist.iter().any(|a| matched.eq_ignore_ascii_case(a)),
        |_, _| Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mna_leak_denies() {
        let result = evaluate(
            "we're in confidential merger talks with a competitor",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "mna-leak"));
    }

    #[test]
    fn pricing_leak_denies() {
        let result = evaluate(
            "attached is our internal pricing sheet for enterprise deals",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "pricing-leak"));
    }

    #[test]
    fn benign_pricing_mention_allows() {
        let result = evaluate(
            "our pricing page lists three public tiers",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn allowlisted_phrase_is_exempt() {
        let mut o = CheckOptions::default();
        o.set_str_list("allowlist", &["confidential merger"]);
        let result = evaluate("this is about a confidential merger", &o);
        assert_eq!(result.action, CheckAction::Log);
    }
}
