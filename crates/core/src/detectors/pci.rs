//! PCI check — credit-card detection, ported from `app/guardrails/checks/pci.py`.
//!
//! Bare digit-sequence regex flags order numbers, tracking IDs, and UUIDs —
//! Luhn cuts card false positives ~90%, so it's on by default
//! (`luhn_required`). `bank_account`/`crypto_wallet` are not implemented
//! (would need IBAN mod-97 / base58-bech32 checksums respectively).
//!
//! ## Porting note: no lookaround in Rust's `regex` crate
//!
//! The Python source uses `(?<!\d)(?:\d[ -]?){13,19}(?!\d)` — a negative
//! lookbehind/lookahead pair Rust's `regex` crate cannot express (it supports
//! no lookaround at all, not even fixed-width). This is ported as: find the
//! *maximal* contiguous run of `\d(?:[ -]?\d)*` (digits with single optional
//! space/hyphen separators), which by construction cannot be immediately
//! preceded or followed by another digit — greedy matching would have
//! consumed it otherwise. That reproduces the lookaround boundary check
//! exactly whenever the whole run's digit count is within `[13, 19]`.
//!
//! It's a deliberate, documented simplification when a run's total digit
//! count exceeds 19 (e.g. two numbers separated by a single stray space,
//! pushing the merged run over the limit): Python's regex can still recover
//! a valid embedded window there via quantifier backtracking against
//! interior separator boundaries; this port does not attempt that and skips
//! the whole run instead. That trades a narrow recall loss on pathological
//! adjacent-digit-blob inputs for never producing a false positive — an
//! acceptable tradeoff given there's no eval harness yet to measure the
//! real-world cost either way.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

static DIGIT_RUN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d(?:[ -]?\d)*").unwrap());

fn luhn_valid(digits: &str) -> bool {
    let mut total: u32 = 0;
    for (i, ch) in digits.chars().rev().enumerate() {
        let mut n = ch.to_digit(10).expect("digits-only string");
        if i % 2 == 1 {
            n *= 2;
            if n > 9 {
                n -= 9;
            }
        }
        total += n;
    }
    total.is_multiple_of(10)
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let credit_card_enabled = options.bool_option("credit_card", false);
    let luhn_required = options.bool_option("luhn_required", true);

    let mut hits: Vec<RuleHit> = Vec::new();
    if credit_card_enabled {
        for m in DIGIT_RUN_RE.find_iter(text) {
            let digits: String = m.as_str().chars().filter(char::is_ascii_digit).collect();
            if !(13..=19).contains(&digits.len()) {
                continue;
            }
            if luhn_required && !luhn_valid(&digits) {
                continue;
            }
            hits.push(RuleHit {
                rule_id: "pci-credit-card".to_string(),
                span: (m.start(), m.end()),
                severity: Severity::High,
            });
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "pci".to_string(),
        action,
        severity: Severity::High,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_VISA: &str = "4242 4242 4242 4242"; // passes Luhn
    const INVALID_LUHN: &str = "1234 5678 9012 3456"; // fails Luhn

    fn opts(pairs: &[(&str, bool)]) -> CheckOptions {
        let mut o = CheckOptions::default();
        for (k, v) in pairs {
            o.set_bool(k, *v);
        }
        o
    }

    #[test]
    fn luhn_valid_card_denies() {
        let text = format!("my card is {VALID_VISA}");
        let result = evaluate(
            &text,
            &opts(&[("credit_card", true), ("luhn_required", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert_eq!(result.hits.len(), 1);
    }

    #[test]
    fn luhn_invalid_card_allows() {
        let text = format!("my card is {INVALID_LUHN}");
        let result = evaluate(
            &text,
            &opts(&[("credit_card", true), ("luhn_required", true)]),
        );
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn credit_card_option_disabled_allows() {
        let result = evaluate(VALID_VISA, &opts(&[("credit_card", false)]));
        assert!(result.hits.is_empty());
    }

    #[test]
    fn digits_outside_length_range_are_ignored() {
        let result = evaluate("1234567890123456789012", &opts(&[("credit_card", true)])); // 23 digits, no separators
        assert!(result.hits.is_empty());
    }
}
