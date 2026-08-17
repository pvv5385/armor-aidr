//! Competitor check — customer-supplied list of competitor names/products
//! (`options.competitors`), matched literally rather than against a fixed
//! ruleset. Same reasoning and shape as `keyword_blocklist`: which names
//! count as "a competitor" is per-deployment configuration, not a shared,
//! maintainer-curated asset, so there is no `rules/*.yaml` for this
//! detector.

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

fn find_hits(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    let mut search_from = 0usize;
    while let Some(pos) = haystack[search_from..].find(needle) {
        let start = search_from + pos;
        let end = start + needle.len();
        if is_word_boundary(haystack, start, end) {
            hits.push((start, end));
        }
        search_from = start + needle.len().max(1);
    }
    hits
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let competitors = options.str_list_option("competitors");

    let mut hits: Vec<RuleHit> = Vec::new();
    if !competitors.is_empty() && !text.trim().is_empty() {
        let lower_text = text.to_lowercase();
        for name in &competitors {
            let lower_name = name.to_lowercase();
            for (start, end) in find_hits(&lower_text, &lower_name) {
                hits.push(RuleHit {
                    rule_id: format!("competitor:{name}"),
                    span: (start, end),
                    severity: Severity::Low,
                });
            }
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Flag
    };
    DetectorResult {
        detector_id: "competitor".to_string(),
        action,
        severity: Severity::Low,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(competitors: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("competitors", competitors);
        o
    }

    #[test]
    fn mentioned_competitor_flags() {
        let result = evaluate(
            "you should really consider AcmeCorp for this",
            &opts(&["AcmeCorp"]),
        );
        assert_eq!(result.action, CheckAction::Flag);
        assert_eq!(result.hits.len(), 1);
    }

    #[test]
    fn match_is_case_insensitive() {
        let result = evaluate("ACMECORP has a better plan", &opts(&["AcmeCorp"]));
        assert_eq!(result.action, CheckAction::Flag);
    }

    #[test]
    fn whole_word_match_only() {
        let result = evaluate("acmecorporate events are unrelated", &opts(&["AcmeCorp"]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn empty_competitor_list_allows() {
        let result = evaluate("anything at all", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_mention_allows() {
        let result = evaluate("nothing relevant here", &opts(&["AcmeCorp", "Initech"]));
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
