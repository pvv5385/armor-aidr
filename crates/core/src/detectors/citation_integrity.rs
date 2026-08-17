//! Citation-integrity check — validates that citations in a response (`[1]`
//! style numbered references and `[source: Name]` style named references)
//! are actually grounded in `options.sources`, the caller-supplied list of
//! sources that were available for this response. A numbered reference is
//! valid if it's in range of the supplied source list (`[1]` through
//! `[N]` for `N` sources); a named reference is valid if it substring-
//! matches (case-insensitively, either direction) one of the supplied
//! source strings. No sources supplied means every citation is
//! unverifiable by construction.
//!
//! Deliberately narrower than real grounding verification (does the *cited
//! claim* actually appear in that source) — this only checks that the
//! citation *points somewhere real*. Claim-level grounding is
//! [`super::hallucination`]'s job.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

static NUMBERED_REF: Lazy<Regex> = Lazy::new(|| Regex::new(r"\[(-?\d+)\]").unwrap());
static NAMED_REF: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\[source:\s*([^\]]*)\]").unwrap());

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let sources = options.str_list_option("sources");

    let mut hits: Vec<RuleHit> = Vec::new();

    for m in NUMBERED_REF.captures_iter(text) {
        let full = m.get(0).unwrap();
        let n: usize = m[1].parse().unwrap_or(0);
        if n == 0 || n > sources.len() {
            hits.push(RuleHit {
                rule_id: "citation-unverifiable-reference".to_string(),
                span: (full.start(), full.end()),
                severity: Severity::Medium,
            });
        }
    }

    for m in NAMED_REF.captures_iter(text) {
        let full = m.get(0).unwrap();
        let name = m[1].trim().to_lowercase();
        let grounded = !name.is_empty()
            && sources.iter().any(|s| {
                let s_lower = s.to_lowercase();
                s_lower.contains(&name) || name.contains(&s_lower)
            });
        if !grounded {
            hits.push(RuleHit {
                rule_id: "citation-unverifiable-reference".to_string(),
                span: (full.start(), full.end()),
                severity: Severity::Medium,
            });
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "citation_integrity".to_string(),
        action,
        severity: Severity::Medium,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(sources: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("sources", sources);
        o
    }

    #[test]
    fn in_range_numbered_reference_allows() {
        let result = evaluate(
            "Rust is memory-safe [1] and fast [2]",
            &opts(&["Source A", "Source B"]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn out_of_range_numbered_reference_denies() {
        let result = evaluate("As shown in [3]", &opts(&["Source A", "Source B"]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "citation-unverifiable-reference"));
    }

    #[test]
    fn grounded_named_reference_allows() {
        let result = evaluate(
            "[source: Wikipedia] says this is correct",
            &opts(&["Wikipedia - Rust programming language"]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn ungrounded_named_reference_denies() {
        let result = evaluate(
            "[source: FakeSite] says this is correct",
            &opts(&["Wikipedia"]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn no_citations_present_allows() {
        let result = evaluate("Rust is a systems programming language", &opts(&[]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_sources_supplied_denies_any_numbered_citation() {
        let result = evaluate("As shown in [1]", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }
}
