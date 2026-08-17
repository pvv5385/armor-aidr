//! Copyright check — near-verbatim match against deployment-supplied
//! `options.protected_passages` (book openings, lyrics, or any text the
//! customer wants to prevent verbatim reproduction of). No copyrighted
//! text ships in this repo — the passage list is entirely caller-supplied
//! config, matching `keyword_blocklist`/`custom_regex`'s "the pattern set
//! is per-deployment data, not a shared asset" convention, and avoiding
//! embedding actual copyrighted excerpts as fixtures.
//!
//! Word-shingle Jaccard similarity (k-word overlapping windows, default
//! `shingle_size: 6`) rather than exact substring match: catches
//! near-verbatim reproduction with minor word substitutions, not just
//! byte-identical copies. Shipped `enabled: false` in `config/policies.yaml`
//! — inert until a deployment supplies `protected_passages`.

use std::collections::HashSet;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

/// A capital "I" embedded in an otherwise non-uppercase word is virtually
/// always a misread lowercase "l" (an OCR/scan artifact like "annuaI",
/// "wiII", "heId") rather than a real word — the pronoun "I" and genuine
/// acronyms only ever appear as their own all-caps token, never mixed with
/// lowercase letters. All-caps words (e.g. a "WILL"/"FINANCIAL" sentence in
/// all caps) are left untouched since they have no lowercase letters to
/// trigger this.
fn correct_ocr_confusables(word: &str) -> String {
    let has_lowercase = word.chars().any(|c| c.is_lowercase());
    let uppercase_is_only_i = word.chars().filter(|c| c.is_uppercase()).all(|c| c == 'I');
    if has_lowercase && uppercase_is_only_i {
        word.replace('I', "l")
    } else {
        word.to_string()
    }
}

fn normalize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| correct_ocr_confusables(w).to_lowercase())
        .collect()
}

fn shingles(words: &[String], k: usize) -> HashSet<String> {
    if words.len() < k {
        return HashSet::new();
    }
    words.windows(k).map(|w| w.join(" ")).collect()
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let passages = options.str_list_option("protected_passages");
    let shingle_size = options.f64_option("shingle_size", 6.0) as usize;
    let similarity_threshold = options.f64_option("similarity_threshold", 0.3);

    let mut hits: Vec<RuleHit> = Vec::new();
    if !passages.is_empty() && shingle_size > 0 {
        let response_words = normalize_words(text);
        let response_shingles = shingles(&response_words, shingle_size);

        if !response_shingles.is_empty() {
            for passage in &passages {
                let passage_words = normalize_words(passage);
                let passage_shingles = shingles(&passage_words, shingle_size);
                if passage_shingles.is_empty() {
                    continue;
                }

                let intersection = response_shingles.intersection(&passage_shingles).count();
                let union = response_shingles.union(&passage_shingles).count();
                let jaccard = intersection as f64 / union.max(1) as f64;

                if jaccard > similarity_threshold {
                    hits.push(RuleHit {
                        rule_id: "copyright-near-verbatim-match".to_string(),
                        span: (0, text.len()),
                        severity: Severity::High,
                    });
                    break;
                }
            }
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "copyright".to_string(),
        action,
        severity: Severity::High,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PASSAGE: &str =
        "the quick brown fox jumps over the lazy dog and runs away quickly into the dark forest";

    fn opts(passages: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("protected_passages", passages);
        o
    }

    #[test]
    fn near_verbatim_reproduction_denies() {
        let result = evaluate(SAMPLE_PASSAGE, &opts(&[SAMPLE_PASSAGE]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "copyright-near-verbatim-match"));
    }

    #[test]
    fn lightly_edited_reproduction_denies() {
        let edited =
            "the quick brown fox jumps over the lazy dog and runs off quickly into the dark forest";
        let result = evaluate(edited, &opts(&[SAMPLE_PASSAGE]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn unrelated_text_allows() {
        let result = evaluate(
            "Rust is a systems programming language with a strong type system",
            &opts(&[SAMPLE_PASSAGE]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_protected_passages_configured_allows() {
        let result = evaluate(SAMPLE_PASSAGE, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn short_text_below_shingle_size_allows() {
        let result = evaluate("too short", &opts(&[SAMPLE_PASSAGE]));
        assert_eq!(result.action, CheckAction::Log);
    }
}
