//! PII check — regex + checksum-validated detection (email, phone,
//! SSN/national-ID formats, IBAN, regional identifiers).
//!
//! This is new code, not a port: `app/guardrails/checks/pii.py` is
//! Presidio+spaCy-backed (a local NER/ML pipeline) — that's a heavier,
//! model-backed `armor-inference` tier, out of scope for this rules-only
//! detector. Patterns live in `rules/pii/rules.yaml`; see that file's
//! header for the attribution note on the regional-identifier coverage and
//! schema.
//!
//! Each rule can optionally name a `validator` (a check-digit/shape function
//! in [`super::checksums`] or this module) that must pass on the raw matched
//! text, and/or a `context_exclude` regex checked against the text
//! immediately preceding a match — both exist to keep bare-digit-shaped
//! patterns (SSNs, national IDs) from firing on dates, order numbers, and
//! other incidental lookalikes.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;

use super::checksums;
use super::rule_loader;
use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

const SOURCE: &str = "rules/pii/rules.yaml";

/// How many characters of preceding context a `context_exclude` regex is
/// checked against.
const CONTEXT_WINDOW: usize = 30;

#[derive(Debug, Deserialize)]
struct RawRule {
    id: String,
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    entity: String,
    option: String,
    regex: String,
    #[serde(default)]
    validator: Option<String>,
    #[serde(default)]
    context_exclude: Option<String>,
    #[serde(default)]
    context_exclude_after: Option<String>,
    #[serde(default)]
    severity: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
}

struct CompiledRule {
    id: String,
    option: String,
    pattern: Regex,
    validator: Option<fn(&str) -> bool>,
    context_exclude: Option<Regex>,
    context_exclude_after: Option<Regex>,
    severity: Severity,
}

fn parse_severity(s: Option<&str>) -> Severity {
    match s {
        Some("low") => Severity::Low,
        Some("high") => Severity::High,
        Some("critical") => Severity::Critical,
        _ => Severity::Medium,
    }
}

fn resolve_validator(name: &str) -> fn(&str) -> bool {
    match name {
        "luhn" => checksums::luhn,
        "verhoeff" => checksums::verhoeff,
        "cpf" => checksums::cpf,
        "cnpj" => checksums::cnpj,
        "es_nif" => checksums::es_nif,
        "uk_nhs" => checksums::uk_nhs,
        "nl_bsn" => checksums::nl_bsn,
        "iban_mod97" => checksums::iban_mod97,
        "au_tfn" => checksums::au_tfn,
        "au_abn" => checksums::au_abn,
        "au_acn" => checksums::au_acn,
        "id_ktp_date_sanity" => checksums::id_ktp_date_sanity,
        other => panic!("unknown pii validator in rules/pii/rules.yaml: {other}"),
    }
}

static RULES: Lazy<Vec<CompiledRule>> = Lazy::new(|| {
    rule_loader::parse_rules::<RawRule>(include_str!("../../rules/pii/rules.yaml"), SOURCE)
        .into_iter()
        .map(|r| {
            let pattern = rule_loader::compile_regex(&r.regex, &r.id, SOURCE, r.case_insensitive);
            let compile_ctx = |p: &str, field: &str| {
                rule_loader::compile_regex(p, &format!("{} {field}", r.id), SOURCE, true)
            };
            let context_exclude = r
                .context_exclude
                .as_deref()
                .map(|p| compile_ctx(p, "context_exclude"));
            let context_exclude_after = r
                .context_exclude_after
                .as_deref()
                .map(|p| compile_ctx(p, "context_exclude_after"));
            CompiledRule {
                id: r.id,
                option: r.option,
                pattern,
                validator: r.validator.as_deref().map(resolve_validator),
                context_exclude,
                context_exclude_after,
                severity: parse_severity(r.severity.as_deref()),
            }
        })
        .collect()
});

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

/// RFC 1918 + loopback + link-local ranges — private IPv4 addresses are
/// noisy, near-universal false positives for a PII leak check (internal
/// infra addresses in logs/configs, not a person's identifying data).
fn is_private_ipv4(ip: &str) -> bool {
    let parts: Vec<u8> = ip.split('.').filter_map(|p| p.parse().ok()).collect();
    let [a, b, ..] = parts.as_slice() else {
        return false;
    };
    matches!(a, 10 | 127)
        || (*a == 172 && (16..=31).contains(b))
        || (*a == 192 && *b == 168)
        || (*a == 169 && *b == 254)
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let skip_private_ips = options.bool_option("skip_private_ips", true);

    let mut hits: Vec<RuleHit> = Vec::new();
    if !text.trim().is_empty() {
        for rule in RULES
            .iter()
            .filter(|r| options.bool_option(&r.option, false))
        {
            for m in rule.pattern.find_iter(text) {
                let matched = m.as_str();

                if rule.id == "ipv4-address" && skip_private_ips && is_private_ipv4(matched) {
                    continue;
                }

                if let Some(validator) = rule.validator {
                    if !validator(matched) {
                        continue;
                    }
                }

                if let Some(exclude) = &rule.context_exclude {
                    let start = m.start().saturating_sub(CONTEXT_WINDOW);
                    if exclude.is_match(&text[start..m.start()]) {
                        continue;
                    }
                }

                if let Some(exclude_after) = &rule.context_exclude_after {
                    let end = (m.end() + CONTEXT_WINDOW).min(text.len());
                    if exclude_after.is_match(&text[m.end()..end]) {
                        continue;
                    }
                }

                hits.push(RuleHit {
                    rule_id: rule.id.clone(),
                    span: (m.start(), m.end()),
                    severity: rule.severity,
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
        detector_id: "pii".to_string(),
        action,
        severity: Severity::Medium,
        hits,
        confidence: None,
    }
}

/// One `Bool` spec per distinct `rules.yaml` `option`, labeled by the rule's
/// own description — the management UI's PII checkbox list. In sync with
/// `evaluate` automatically because both read the same YAML.
pub(crate) fn rule_option_specs() -> Vec<super::OptionSpec> {
    use std::collections::HashSet;

    use super::{OptionKind, OptionSpec};

    let raw: Vec<RawRule> =
        rule_loader::parse_rules(include_str!("../../rules/pii/rules.yaml"), SOURCE);
    let mut seen = HashSet::new();
    let mut specs: Vec<OptionSpec> = Vec::new();
    for rule in &raw {
        if seen.insert(rule.option.clone()) {
            specs.push(OptionSpec {
                key: rule.option.clone(),
                kind: OptionKind::Bool,
                label: rule.description.clone(),
                help: Some(format!("option key: {}", rule.option)),
                default: serde_json::json!(false),
            });
        }
    }
    specs.push(OptionSpec {
        key: "skip_private_ips".to_string(),
        kind: OptionKind::Bool,
        label: "Skip private IP addresses".to_string(),
        help: Some(
            "Private/loopback IPv4 addresses (RFC 1918 etc.) are noise for a leak \
             check and are skipped by default."
                .to_string(),
        ),
        default: serde_json::json!(true),
    });
    specs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, bool)]) -> CheckOptions {
        let mut o = CheckOptions::default();
        for (k, v) in pairs {
            o.set_bool(k, *v);
        }
        o
    }

    #[test]
    fn email_denies_when_enabled() {
        let result = evaluate(
            "contact me at jane.doe@example.com",
            &opts(&[("email", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn email_allowed_when_option_disabled() {
        let result = evaluate(
            "contact me at jane.doe@example.com",
            &opts(&[("email", false)]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn ssn_denies_when_enabled() {
        let result = evaluate("my ssn is 123-45-6789", &opts(&[("ssn", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn ssn_context_excluded_after_date_keyword() {
        let result = evaluate("expiration date: 123-45-6789", &opts(&[("ssn", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn phone_denies_when_enabled() {
        let result = evaluate("call me at (415) 555-0132", &opts(&[("phone", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn phone_context_excluded_after_order_keyword() {
        let result = evaluate("order id: 415-555-0132", &opts(&[("phone", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn no_entities_enabled_allows() {
        let result = evaluate("jane.doe@example.com 123-45-6789", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn only_enabled_entity_is_checked() {
        // SSN present but only email is enabled -> allowed.
        let result = evaluate("my ssn is 123-45-6789", &opts(&[("email", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn private_ipv4_skipped_by_default() {
        let result = evaluate("server at 192.168.1.1", &opts(&[("ip_address", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn public_ipv4_denies() {
        let result = evaluate("server at 203.0.113.5", &opts(&[("ip_address", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn private_ipv4_denies_when_skip_disabled() {
        let mut o = opts(&[("ip_address", true)]);
        o.set_bool("skip_private_ips", false);
        let result = evaluate("server at 192.168.1.1", &o);
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn invalid_aadhaar_shape_is_rejected_by_checksum() {
        let result = evaluate("aadhaar: 2000 0000 0000", &opts(&[("in_aadhaar", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn br_cpf_valid_denies() {
        let result = evaluate("cpf: 111.444.777-35", &opts(&[("br_cpf", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn br_cpf_invalid_checksum_allows() {
        let result = evaluate("cpf: 111.444.777-36", &opts(&[("br_cpf", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn in_upi_denies_but_full_email_does_not_trigger_upi_rule() {
        let result = evaluate("pay to alice@paytm", &opts(&[("in_upi", true)]));
        assert_eq!(result.action, CheckAction::Deny);

        let email_like = evaluate("email me at alice@example.com", &opts(&[("in_upi", true)]));
        assert_eq!(email_like.action, CheckAction::Log);
    }

    #[test]
    fn au_acn_valid_denies() {
        let result = evaluate("ACN: 004 085 616", &opts(&[("au_acn", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn compact_iban_denies_when_enabled() {
        let result = evaluate(
            "wire to DE89370400440532013000 please",
            &opts(&[("iban", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn grouped_iban_denies_when_enabled() {
        let result = evaluate(
            "wire to DE89 3704 0044 0532 0130 00 please",
            &opts(&[("iban", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn iban_followed_by_a_trailing_word_still_denies() {
        // Regression guard: the old regex's per-character optional separator
        // let the match run straight through the space into "BIC", producing
        // a corrupted span that `iban_mod97` (correctly) rejected — so the
        // real IBAN in front of it never registered a hit at all.
        let result = evaluate(
            "IBAN: DE89 3704 0044 0532 0130 00 BIC: DEUTDEFF",
            &opts(&[("iban", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "iban"));
    }

    #[test]
    fn iban_followed_by_a_trailing_number_still_denies() {
        let result = evaluate(
            "account GB29 NWBK 6016 1331 9268 19 amount 500",
            &opts(&[("iban", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn iban_with_bad_checksum_allows() {
        let result = evaluate(
            "wire to DE89 3704 0044 0532 0130 01 please",
            &opts(&[("iban", true)]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn bank_account_us_denies_a_bare_digit_run() {
        let result = evaluate(
            "account number: 12345678901",
            &opts(&[("bank_account", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn bank_account_us_denies_a_dash_grouped_number() {
        // Regression guard: real account numbers are routinely written
        // grouped ("1234-5678-9012") — the old [0-9]{8,17} only matched one
        // contiguous digit run and missed every one of those.
        let result = evaluate("acct no. 1234-5678-9012", &opts(&[("bank_account", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn bank_account_us_denies_a_space_grouped_number() {
        let result = evaluate(
            "bank account number 1234 5678 9012 3456",
            &opts(&[("bank_account", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn bank_account_us_requires_the_keyword() {
        let result = evaluate("invoice 12345678901", &opts(&[("bank_account", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn ethereum_address_denies_when_enabled() {
        let result = evaluate(
            "send to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            &opts(&[("ethereum_address", true)]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }
}
