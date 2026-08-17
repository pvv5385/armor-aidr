//! Gibberish check — formula-based, not a pattern bank (there's no fixed
//! "gibberish" vocabulary to enumerate): Shannon entropy over the
//! alphanumeric content, a low-vowel-ratio scan for random consonant
//! strings, and the invisible/bidi-control character ratio (reusing
//! [`crate::engine::normalize::is_invisible`]). Each is independently
//! gated by its own option so a deployment can disable the noisier one
//! (entropy, which structurally can't distinguish "random gibberish" from
//! "legitimate high-entropy content" like a hash or a token) without
//! losing the other two.
//!
//! `min_text_length` (default 10, trimmed char count) gates the detector
//! as a whole, before any sub-check runs — distinct from `min_length`/
//! `min_token_length`, which gate the entropy/vowel-ratio sub-checks
//! individually.

use std::collections::HashMap;

use crate::engine::normalize::is_invisible;
use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

fn shannon_entropy(chars: &[char]) -> f64 {
    if chars.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<char, usize> = HashMap::new();
    for &c in chars {
        *counts.entry(c).or_insert(0) += 1;
    }
    let total = chars.len() as f64;
    counts
        .values()
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn vowel_ratio(word: &str) -> f64 {
    let letters: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return 1.0;
    }
    let vowels = letters
        .iter()
        .filter(|c| "aeiouAEIOU".contains(**c))
        .count();
    vowels as f64 / letters.len() as f64
}

/// A detector-wide floor below which the input isn't evaluated at all —
/// short strings (a single word, an ID) trip the per-check heuristics on
/// too little signal to be meaningful, distinct from `min_length`/
/// `min_token_length` which gate individual sub-checks.
fn below_min_text_length(text: &str, min_text_length: usize) -> bool {
    text.trim().chars().count() < min_text_length
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let min_text_length = options.f64_option("min_text_length", 10.0) as usize;
    if below_min_text_length(text, min_text_length) {
        return DetectorResult {
            detector_id: "gibberish".to_string(),
            action: CheckAction::Log,
            severity: Severity::Low,
            hits: Vec::new(),
            confidence: None,
        };
    }

    let check_entropy = options.bool_option("entropy", true);
    let entropy_threshold = options.f64_option("entropy_threshold", 4.6);
    let min_length = options.f64_option("min_length", 20.0) as usize;

    let check_vowel_ratio = options.bool_option("vowel_ratio", true);
    let vowel_ratio_threshold = options.f64_option("vowel_ratio_threshold", 0.15);
    let min_token_length = options.f64_option("min_token_length", 8.0) as usize;

    let check_invisible = options.bool_option("invisible_char_ratio", true);
    let invisible_ratio_threshold = options.f64_option("invisible_char_ratio_threshold", 0.1);

    let mut hits: Vec<RuleHit> = Vec::new();

    if check_entropy {
        let alnum: Vec<char> = text
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        if alnum.len() >= min_length && shannon_entropy(&alnum) > entropy_threshold {
            hits.push(RuleHit {
                rule_id: "gibberish-high-entropy".to_string(),
                span: (0, text.len()),
                severity: Severity::Medium,
            });
        }
    }

    if check_vowel_ratio {
        let mut offset = 0usize;
        for token in text.split_whitespace() {
            let token_start = text[offset..]
                .find(token)
                .map(|i| offset + i)
                .unwrap_or(offset);
            offset = token_start + token.len();

            let letters_only: String = token.chars().filter(|c| c.is_alphabetic()).collect();
            if letters_only.len() >= min_token_length
                && letters_only.len() == token.chars().count()
                && vowel_ratio(&letters_only) < vowel_ratio_threshold
            {
                hits.push(RuleHit {
                    rule_id: "gibberish-low-vowel-ratio".to_string(),
                    span: (token_start, token_start + token.len()),
                    severity: Severity::Low,
                });
            }
        }
    }

    if check_invisible {
        let total_chars = text.chars().count();
        let invisible_count = text.chars().filter(|c| is_invisible(*c)).count();
        if total_chars > 0
            && (invisible_count as f64 / total_chars as f64) > invisible_ratio_threshold
        {
            hits.push(RuleHit {
                rule_id: "gibberish-invisible-char-ratio".to_string(),
                span: (0, text.len()),
                severity: Severity::High,
            });
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Flag
    };
    let severity = hits
        .iter()
        .map(|h| h.severity)
        .max()
        .unwrap_or(Severity::Low);
    DetectorResult {
        detector_id: "gibberish".to_string(),
        action,
        severity,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_high_entropy_string_flags() {
        let result = evaluate("xQ7mZ2vK9pL4wR8nT1yB6jH3dF5sC0gA", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Flag);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "gibberish-high-entropy"));
    }

    #[test]
    fn ordinary_english_prose_allows() {
        let result = evaluate(
            "The quick brown fox jumps over the lazy dog near the river bank",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn random_consonant_string_flags_low_vowel_ratio() {
        let result = evaluate(
            "the file is named xkqzjwvbnmfghjklp",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Flag);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "gibberish-low-vowel-ratio"));
    }

    #[test]
    fn short_text_below_min_length_allows() {
        let result = evaluate("xQ7mZ2vK", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn high_invisible_char_ratio_flags() {
        // 7 chars total, under the default min_text_length of 10 — disable
        // the whole-detector gate so this exercises the sub-check itself.
        let text = "a\u{200b}\u{200b}\u{200b}\u{200b}\u{200b}b";
        let mut o = CheckOptions::default();
        o.set_f64("min_text_length", 0.0);
        let result = evaluate(text, &o);
        assert_eq!(result.action, CheckAction::Flag);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "gibberish-invisible-char-ratio"));
    }

    #[test]
    fn checks_individually_disabled_by_option() {
        let mut o = CheckOptions::default();
        o.set_bool("entropy", false);
        o.set_bool("vowel_ratio", false);
        o.set_bool("invisible_char_ratio", false);
        let result = evaluate("xQ7mZ2vK9pL4wR8nT1yB6jH3dF5sC0gA", &o);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn text_below_min_text_length_skips_every_subcheck() {
        // Would trip gibberish-invisible-char-ratio if evaluated, but at 7
        // trimmed chars it's under the default min_text_length of 10.
        let text = "a\u{200b}\u{200b}\u{200b}\u{200b}\u{200b}b";
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn min_text_length_is_configurable() {
        let mut o = CheckOptions::default();
        o.set_f64("min_text_length", 50.0);
        // Long enough to trip low-vowel-ratio under the default 10-char
        // floor, but shorter than a configured 50-char floor.
        let result = evaluate("the file is named xkqzjwvbnmfghjklp", &o);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn whitespace_only_text_is_below_min_text_length() {
        let result = evaluate("          ", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
