//! Keyword/topic blocklist — customer-supplied list of banned words or
//! phrases (`options.keywords`), matched literally rather than against a
//! fixed ruleset. Unlike `secrets`/`prompt_injection`/`pii`, there is no
//! `rules/*.yaml` for this detector: the pattern set *is* per-policy
//! configuration, not a shared asset, so it stays in `config/policies.yaml`.
//!
//! Deliberately plain substring matching, not regex — keywords are supplied
//! by whoever owns the policy, not a maintainer who's verified the pattern
//! is ReDoS-safe, so compiling them as regexes per request would be both a
//! footgun and unnecessary compile cost for what is almost always literal
//! phrase matching.

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_word_boundary(haystack: &str, start: usize, end: usize) -> bool {
    let before_ok = haystack[..start]
        .chars()
        .next_back()
        .map(|c| !is_word_char(c))
        .unwrap_or(true);
    let after_ok = haystack[end..]
        .chars()
        .next()
        .map(|c| !is_word_char(c))
        .unwrap_or(true);
    before_ok && after_ok
}

fn find_hits(haystack: &str, needle: &str, whole_word: bool) -> Vec<(usize, usize)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    let mut search_from = 0usize;
    while let Some(pos) = haystack[search_from..].find(needle) {
        let start = search_from + pos;
        let end = start + needle.len();
        if !whole_word || is_word_boundary(haystack, start, end) {
            hits.push((start, end));
        }
        search_from = start + needle.len().max(1);
    }
    hits
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let keywords = options.str_list_option("keywords");
    let case_sensitive = options.bool_option("case_sensitive", false);
    let whole_word = options.bool_option("whole_word", true);

    let mut hits: Vec<RuleHit> = Vec::new();
    if !keywords.is_empty() && !text.trim().is_empty() {
        let folded_text;
        let haystack: &str = if case_sensitive {
            text
        } else {
            folded_text = text.to_lowercase();
            &folded_text
        };

        for keyword in &keywords {
            let folded_keyword;
            let needle: &str = if case_sensitive {
                keyword
            } else {
                folded_keyword = keyword.to_lowercase();
                &folded_keyword
            };
            for (start, end) in find_hits(haystack, needle, whole_word) {
                hits.push(RuleHit {
                    rule_id: format!("keyword:{keyword}"),
                    span: (start, end),
                    severity: Severity::Medium,
                });
            }
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "keyword_blocklist".to_string(),
        action,
        severity: Severity::Medium,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(keywords: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("keywords", keywords);
        o
    }

    #[test]
    fn matching_keyword_denies() {
        let result = evaluate("please discuss our merger plans", &opts(&["merger"]));
        assert_eq!(result.action, CheckAction::Deny);
        assert_eq!(result.hits.len(), 1);
    }

    #[test]
    fn match_is_case_insensitive_by_default() {
        let result = evaluate("MERGER talks are ongoing", &opts(&["merger"]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn case_sensitive_option_respected() {
        let mut o = opts(&["merger"]);
        o.set_bool("case_sensitive", true);
        let result = evaluate("MERGER talks are ongoing", &o);
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn whole_word_default_skips_substring_match() {
        let result = evaluate("this document is classified", &opts(&["class"]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn whole_word_disabled_matches_substring() {
        let mut o = opts(&["class"]);
        o.set_bool("whole_word", false);
        let result = evaluate("this document is classified", &o);
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn empty_keyword_list_allows() {
        let result = evaluate("anything at all", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn benign_text_without_keywords_allows() {
        let result = evaluate("nothing sensitive here", &opts(&["merger", "layoff"]));
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
