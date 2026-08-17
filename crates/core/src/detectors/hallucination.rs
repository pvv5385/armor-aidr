//! Hallucination check (base tier) — token-overlap grounding against
//! `options.sources`: what fraction of the response's content words also
//! appear somewhere in the supplied source text. Low overlap is a cheap,
//! coarse signal that the response is talking about something the sources
//! don't actually cover. NLI-based entailment and an LLM judge are heavier,
//! costlier escalation tiers not built here — this is deliberately the
//! weakest, zero-cost tier: no sources supplied means nothing to check
//! against, so the check stays quiet rather than flagging every unsourced
//! response.

use std::collections::HashSet;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

/// High-frequency function words excluded from the overlap calculation so
/// they don't inflate the ratio independent of actual content grounding.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "at", "by", "for", "with", "and", "or", "but", "as", "it", "this", "that", "these", "those",
    "its", "their", "his", "her", "them", "they", "he", "she", "we", "you", "i",
];

fn content_tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 2 && !STOPWORDS.contains(&w.as_str()))
        .collect()
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let sources = options.str_list_option("sources");
    let overlap_threshold = options.f64_option("overlap_threshold", 0.3);
    let min_tokens = options.f64_option("min_tokens", 5.0) as usize;

    let mut hits: Vec<RuleHit> = Vec::new();

    if !sources.is_empty() {
        let response_tokens = content_tokens(text);
        if response_tokens.len() >= min_tokens {
            let source_tokens: HashSet<String> =
                sources.iter().flat_map(|s| content_tokens(s)).collect();
            let overlap = response_tokens.intersection(&source_tokens).count();
            let ratio = overlap as f64 / response_tokens.len() as f64;
            if ratio < overlap_threshold {
                hits.push(RuleHit {
                    rule_id: "hallucination-low-grounding-overlap".to_string(),
                    span: (0, text.len()),
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
        detector_id: "hallucination".to_string(),
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
    fn well_grounded_response_allows() {
        let result = evaluate(
            "Rust is a systems programming language focused on safety and performance",
            &opts(&["Rust is a systems programming language focused on memory safety, speed, and performance"]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn ungrounded_response_denies() {
        let result = evaluate(
            "The moon landing was faked by Hollywood studios in a secret bunker",
            &opts(&[
                "Rust is a systems programming language focused on memory safety and performance",
            ]),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "hallucination-low-grounding-overlap"));
    }

    #[test]
    fn no_sources_supplied_allows() {
        let result = evaluate(
            "The moon landing was faked by Hollywood studios in a secret bunker",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn short_response_below_min_tokens_allows() {
        let result = evaluate(
            "Yes, that's correct",
            &opts(&["completely unrelated source text"]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }
}
