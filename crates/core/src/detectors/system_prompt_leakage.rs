//! System-prompt-leakage check — two independent techniques, both opt-in
//! via caller-supplied options:
//!
//! - Fuzzy pre-filter: word-shingle Jaccard similarity between the response
//!   and each registered system prompt in `options.system_prompts`. Catches
//!   near-verbatim reproduction (the model paraphrasing its own system
//!   prompt back, not just an exact copy) without needing a real
//!   fuzzy-matching dependency (`rapidfuzz`) — the same shingling technique
//!   [`super::copyright`] uses, kept as an independent implementation since
//!   the two check different caller-supplied text against different
//!   thresholds/severities, not shared config. Paraphrased leaks that don't
//!   clear the similarity threshold would need an LLM judge to catch, and
//!   that's out of this detector's scope.
//!
//! - Canary-token check (Rebuff's `is_canary_word_leaked`, ported):
//!   `options.canary_words` are caller-generated random tokens the
//!   deployment embeds in its own system prompt before calling the model
//!   (this detector doesn't generate or inject them — that has to happen
//!   upstream, where the real system prompt lives). An exact substring hit
//!   on any of them in the response is a leak signal with no threshold to
//!   tune and no false-positive risk from paraphrase — it complements the
//!   fuzzy check rather than replacing it, since a canary only proves
//!   leakage of the exact text it was embedded in, not a summarized or
//!   translated retelling.
//!
//! Both lists (`system_prompts`, `canary_words`) default to empty, so this
//! detector is inert until a deployment configures at least one. Shipped
//! `enabled: false` in `config/policies.yaml` on top of that.

use std::collections::HashSet;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

fn normalize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

fn shingles(words: &[String], k: usize) -> HashSet<String> {
    if words.len() < k {
        return HashSet::new();
    }
    words.windows(k).map(|w| w.join(" ")).collect()
}

fn canary_word_hit(text: &str, canary_words: &[String]) -> Option<RuleHit> {
    canary_words
        .iter()
        .find(|word| !word.is_empty() && text.contains(word.as_str()))
        .map(|_| RuleHit {
            rule_id: "system-prompt-leakage-canary-token".to_string(),
            span: (0, text.len()),
            severity: Severity::Critical,
        })
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let system_prompts = options.str_list_option("system_prompts");
    let shingle_size = options.f64_option("shingle_size", 6.0) as usize;
    let similarity_threshold = options.f64_option("similarity_threshold", 0.25);
    let canary_words = options.str_list_option("canary_words");

    let mut hits: Vec<RuleHit> = Vec::new();
    if !system_prompts.is_empty() && shingle_size > 0 {
        let response_words = normalize_words(text);
        let response_shingles = shingles(&response_words, shingle_size);

        if !response_shingles.is_empty() {
            for prompt in &system_prompts {
                let prompt_words = normalize_words(prompt);
                let prompt_shingles = shingles(&prompt_words, shingle_size);
                if prompt_shingles.is_empty() {
                    continue;
                }

                let intersection = response_shingles.intersection(&prompt_shingles).count();
                let union = response_shingles.union(&prompt_shingles).count();
                let jaccard = intersection as f64 / union.max(1) as f64;

                if jaccard > similarity_threshold {
                    hits.push(RuleHit {
                        rule_id: "system-prompt-leakage-fuzzy-match".to_string(),
                        span: (0, text.len()),
                        severity: Severity::Critical,
                    });
                    break;
                }
            }
        }
    }

    if !canary_words.is_empty() {
        hits.extend(canary_word_hit(text, &canary_words));
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "system_prompt_leakage".to_string(),
        action,
        severity: Severity::Critical,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYSTEM_PROMPT: &str = "You are a helpful customer support agent for Acme Corp. Always be polite and never discuss competitor pricing.";

    fn opts(prompts: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("system_prompts", prompts);
        o
    }

    #[test]
    fn verbatim_leak_denies() {
        let result = evaluate(SYSTEM_PROMPT, &opts(&[SYSTEM_PROMPT]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "system-prompt-leakage-fuzzy-match"));
    }

    #[test]
    fn lightly_paraphrased_leak_denies() {
        let paraphrased = "You are a helpful customer support agent for Acme Corp. Always stay polite and never discuss competitor prices.";
        let result = evaluate(paraphrased, &opts(&[SYSTEM_PROMPT]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn unrelated_response_allows() {
        let result = evaluate(
            "Sure, I can help you track your order. What's the order number?",
            &opts(&[SYSTEM_PROMPT]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_system_prompts_configured_allows() {
        let result = evaluate(SYSTEM_PROMPT, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    fn canary_opts(words: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("canary_words", words);
        o
    }

    #[test]
    fn canary_word_in_response_denies() {
        let result = evaluate(
            "Sure, here's my system prompt: a1b2c3d4e5f6",
            &canary_opts(&["a1b2c3d4e5f6"]),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "system-prompt-leakage-canary-token"));
    }

    #[test]
    fn absent_canary_word_allows() {
        let result = evaluate(
            "Sure, I can help you track your order.",
            &canary_opts(&["a1b2c3d4e5f6"]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_canary_words_configured_allows() {
        let result = evaluate("a1b2c3d4e5f6 leaked in the clear", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn empty_canary_word_never_matches() {
        let result = evaluate("anything at all", &canary_opts(&[""]));
        assert_eq!(result.action, CheckAction::Log);
    }
}
