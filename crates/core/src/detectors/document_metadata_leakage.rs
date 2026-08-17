//! Document-metadata-leakage check — tracked-changes/comment markup,
//! revision-history markers, internal file paths, and hidden zero-width/
//! bidi text left over from a source document that shouldn't have survived
//! into text handed to an external caller. The first three are a regex
//! pattern bank (`rules/document_metadata_leakage/rules.yaml`); the last is
//! a direct character-category scan reusing
//! [`crate::engine::normalize::is_invisible`] (the same set the
//! normalization layer strips as a view), since "does this string contain
//! an invisible character" isn't naturally a single regex.

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::engine::normalize::is_invisible;
use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/document_metadata_leakage/rules.yaml"),
        "rules/document_metadata_leakage/rules.yaml",
    )
});

static DETECTOR: SimpleDetector = SimpleDetector::new_max_severity(
    "document_metadata_leakage",
    Severity::Medium,
    CheckAction::Deny,
);

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

fn invisible_text_hit(text: &str) -> Option<RuleHit> {
    let mut char_indices = text.char_indices().filter(|(_, c)| is_invisible(*c));
    let (start, first) = char_indices.next()?;
    let end = char_indices
        .next_back()
        .map_or(start + first.len_utf8(), |(i, c)| i + c.len_utf8());
    Some(RuleHit {
        rule_id: "hidden-invisible-text".to_string(),
        span: (start, end),
        severity: Severity::High,
    })
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    DETECTOR.evaluate_with(
        &RULES,
        text,
        options,
        |_, o| rule_loader::pattern_match_enabled(o),
        |_, _| true,
        |text, options| -> Vec<RuleHit> {
            if options.bool_option("invisible_text", true) {
                invisible_text_hit(text).into_iter().collect()
            } else {
                Vec::new()
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracked_changes_markup_denies() {
        let result = evaluate(
            "The text reads <w:ins>added</w:ins> here",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "tracked-changes-markup"));
    }

    #[test]
    fn windows_path_denies() {
        let result = evaluate(
            r"see C:\Users\jsmith\Documents\report.docx for details",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "internal-windows-path"));
    }

    #[test]
    fn unix_home_path_denies() {
        let result = evaluate(
            "the file lives at /home/jsmith/notes/private.txt on the server",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "internal-unix-home-path"));
    }

    #[test]
    fn zero_width_space_denies() {
        let text = "hello\u{200b}world";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "hidden-invisible-text"));
    }

    #[test]
    fn benign_text_allows() {
        let result = evaluate(
            "our public pricing page has three tiers",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn invisible_check_disabled_by_option() {
        let mut o = CheckOptions::default();
        o.set_bool("invisible_text", false);
        let text = "hello\u{200b}world";
        let result = evaluate(text, &o);
        assert_eq!(result.action, CheckAction::Log);
    }
}
