//! Numerical-consistency check (base tier) — flags explicit `number <op>
//! number = number` and `X% of Y is Z` claims where the stated result is
//! arithmetically wrong. Pure computation, no rules.yaml: this isn't a
//! pattern-bank problem, it's "parse the claim, do the arithmetic, compare."
//! Reasoning-heavy numeric claims (word problems, multi-step derivations)
//! would need an LLM judge and aren't handled here — this only catches the
//! narrow, syntactically explicit case.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

const TOLERANCE: f64 = 0.005;

static ARITHMETIC_CLAIM: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(-?\d+(?:\.\d+)?)\s*([+\-*x/])\s*(-?\d+(?:\.\d+)?)\s*(?:=|equals)\s*(-?\d+(?:\.\d+)?)",
    )
    .unwrap()
});

static PERCENTAGE_CLAIM: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\d+(?:\.\d+)?)%\s*of\s*(\d+(?:\.\d+)?)\s*(?:is|equals|=)\s*(\d+(?:\.\d+)?)")
        .unwrap()
});

fn apply_op(a: f64, op: &str, b: f64) -> Option<f64> {
    match op {
        "+" => Some(a + b),
        "-" => Some(a - b),
        "*" | "x" | "X" => Some(a * b),
        "/" if b != 0.0 => Some(a / b),
        _ => None,
    }
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let check_arithmetic = options.bool_option("arithmetic", true);
    let check_percentage = options.bool_option("percentage", true);

    let mut hits: Vec<RuleHit> = Vec::new();

    if check_arithmetic {
        for m in ARITHMETIC_CLAIM.captures_iter(text) {
            let full = m.get(0).unwrap();
            let (a, op, b, claimed) = (
                m[1].parse::<f64>().unwrap(),
                &m[2],
                m[3].parse::<f64>().unwrap(),
                m[4].parse::<f64>().unwrap(),
            );
            if let Some(expected) = apply_op(a, op, b) {
                if (expected - claimed).abs() > TOLERANCE {
                    hits.push(RuleHit {
                        rule_id: "numerical-inconsistency".to_string(),
                        span: (full.start(), full.end()),
                        severity: Severity::Medium,
                    });
                }
            }
        }
    }

    if check_percentage {
        for m in PERCENTAGE_CLAIM.captures_iter(text) {
            let full = m.get(0).unwrap();
            let pct = m[1].parse::<f64>().unwrap();
            let base = m[2].parse::<f64>().unwrap();
            let claimed = m[3].parse::<f64>().unwrap();
            let expected = pct / 100.0 * base;
            if (expected - claimed).abs() > TOLERANCE.max(expected.abs() * 0.01) {
                hits.push(RuleHit {
                    rule_id: "numerical-inconsistency".to_string(),
                    span: (full.start(), full.end()),
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
        detector_id: "numerical_consistency".to_string(),
        action,
        severity: Severity::Medium,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrong_addition_denies() {
        let result = evaluate("12 + 5 = 18", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "numerical-inconsistency"));
    }

    #[test]
    fn correct_addition_allows() {
        let result = evaluate("12 + 5 = 17", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn wrong_percentage_denies() {
        let result = evaluate("10% of 200 is 25", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn correct_percentage_allows() {
        let result = evaluate("10% of 200 is 20", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_numeric_claim_allows() {
        let result = evaluate(
            "Rust is a systems programming language",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn arithmetic_check_disabled_by_option() {
        let mut o = CheckOptions::default();
        o.set_bool("arithmetic", false);
        let result = evaluate("12 + 5 = 18", &o);
        assert_eq!(result.action, CheckAction::Log);
    }
}
