//! Secrets check — per-provider regex detection, ported from
//! `app/guardrails/checks/secrets.py` +
//! `app/guardrails/rulesets/secrets/{rules,manager}.py`.
//!
//! Patterns and stopwords live as data in `rules/secrets/*.yaml` (embedded
//! at compile time via `include_str!`, parsed once) rather than as Rust
//! code — rulesets ship as data, not code, so a new pattern is a YAML edit,
//! not a Rust change.
//!
//! Matching runs line-by-line so the allowlist-stopword suppression is a
//! same-line check — the same granularity gitleaks' own allowlist regexes use.
//!
//! Each rule's `provider` field doubles as its `CheckOptions` toggle key, so
//! a caller enables e.g. `stripe` or `openai` directly — no separate
//! provider registry to keep in sync with the ruleset as it grows.
//!
//! Two provider-independent passes run alongside the regex sweep, both
//! gated by their own options (default off):
//! - `require_entropy` on a rule (e.g. the bare-32-hex-char Azure key
//!   pattern, which is otherwise indistinguishable from any other hex ID)
//!   additionally requires the match to clear a Shannon-entropy threshold.
//! - `detect_base64` decodes candidate base64 substrings and re-checks them
//!   for a known secret-prefix or an enabled provider pattern — catches a
//!   secret that's been base64-encoded to dodge the plaintext patterns.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;

use super::rule_loader;
use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

const SOURCE: &str = "rules/secrets/rules.yaml";

#[derive(Debug, Deserialize)]
struct RawRule {
    id: String,
    #[allow(dead_code)]
    description: String,
    provider: String,
    regex: String,
    #[serde(default)]
    require_entropy: bool,
    #[serde(default)]
    case_insensitive: bool,
}

struct CompiledRule {
    id: String,
    provider: String,
    pattern: Regex,
    require_entropy: bool,
}

static RULES: Lazy<Vec<CompiledRule>> = Lazy::new(|| {
    rule_loader::parse_rules::<RawRule>(include_str!("../../rules/secrets/rules.yaml"), SOURCE)
        .into_iter()
        .map(|r| {
            let pattern = rule_loader::compile_regex(&r.regex, &r.id, SOURCE, r.case_insensitive);
            CompiledRule {
                id: r.id,
                provider: r.provider,
                pattern,
                require_entropy: r.require_entropy,
            }
        })
        .collect()
});

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

static ALLOWLIST_STOPWORDS: Lazy<Vec<String>> = Lazy::new(|| {
    rule_loader::parse_rules(
        include_str!("../../rules/secrets/allowlist_stopwords.yaml"),
        "rules/secrets/allowlist_stopwords.yaml",
    )
});

fn is_allowlisted(line: &str) -> bool {
    ALLOWLIST_STOPWORDS
        .iter()
        .any(|stopword| line.contains(stopword.as_str()))
}

/// Shannon entropy in bits/char. Real secrets (random keys, base64 blobs)
/// run high; patterned lookalikes (repeated hex, sequential IDs) run low.
fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = std::collections::HashMap::new();
    for c in s.chars() {
        *freq.entry(c).or_insert(0u32) += 1;
    }
    let len = s.chars().count() as f64;
    -freq
        .values()
        .map(|&count| {
            let p = count as f64 / len;
            p * p.log2()
        })
        .sum::<f64>()
}

const ENTROPY_THRESHOLD: f64 = 3.5;

/// Candidate base64 segments worth attempting to decode: 24+ base64-alphabet
/// chars, optionally padded. Shorter matches are skipped — the false-positive
/// rate on short alnum tokens (IDs, hashes-of-nothing) is too high.
static B64_CANDIDATE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z0-9+/]{24,}={0,2}").unwrap());

/// Literal byte prefixes that, found inside a decoded base64 blob, are
/// distinctive enough on their own to flag without an entropy check.
const DECODED_KEY_PREFIXES: &[&[u8]] = &[
    b"sk-",
    b"sk-ant-",
    b"AIza",
    b"AKIA",
    b"ASIA",
    b"-----BEGIN",
    b"postgres://",
    b"mongodb://",
    b"mysql://",
];

fn decode_base64(candidate: &str) -> Option<Vec<u8>> {
    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, candidate).ok()
}

/// Scans `line` for base64-encoded secrets: a decodable candidate that
/// either contains a known key-prefix, or is high-entropy and decodes to
/// text an *enabled* provider pattern still matches.
fn detect_base64_secrets(
    line: &str,
    active_rules: &[&CompiledRule],
) -> Vec<(usize, usize, String)> {
    let mut found = Vec::new();
    for m in B64_CANDIDATE_RE.find_iter(line) {
        let candidate = m.as_str();
        let Some(decoded) = decode_base64(candidate) else {
            continue;
        };
        if decoded.len() < 8 {
            continue;
        }
        if DECODED_KEY_PREFIXES
            .iter()
            .any(|p| decoded.windows(p.len().min(decoded.len())).any(|w| w == *p))
        {
            found.push((m.start(), m.end(), "base64-known-prefix".to_string()));
            continue;
        }
        if decoded.len() >= 20 && shannon_entropy(candidate) >= ENTROPY_THRESHOLD {
            let decoded_text = String::from_utf8_lossy(&decoded);
            for rule in active_rules {
                if rule.pattern.is_match(&decoded_text) {
                    found.push((m.start(), m.end(), format!("{}-base64", rule.id)));
                    break;
                }
            }
        }
    }
    found
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let active_rules: Vec<&CompiledRule> = RULES
        .iter()
        .filter(|r| options.bool_option(&r.provider, false))
        .collect();
    let detect_base64 = options.bool_option("detect_base64", false);
    // Only relevant to the openai-key skip below: whether anthropic-key is
    // *also* active for this profile. If it isn't, skipping unconditionally
    // means an sk-ant- key matches neither rule and goes undetected — the
    // exact failure the skip must not cause.
    let anthropic_rule_active = active_rules.iter().any(|r| r.id == "anthropic-key");

    let mut hits: Vec<RuleHit> = Vec::new();
    if !active_rules.is_empty() {
        let mut offset = 0usize;
        for line in text.split('\n') {
            if !is_allowlisted(line) {
                for rule in &active_rules {
                    for caps in rule.pattern.captures_iter(line) {
                        let m = caps.get(0).expect("whole match always present");
                        // openai-key's `sk-[A-Za-z0-9_-]{20,}` shape also
                        // matches any Anthropic `sk-ant-...` key (the `regex`
                        // crate has no lookaround to exclude the `ant-`
                        // sub-prefix directly in the pattern) — skip here so
                        // an Anthropic key doesn't double-fire as an OpenAI
                        // key too when both providers are enabled. Only when
                        // anthropic-key is itself active, though: if it
                        // isn't, this key would otherwise match neither rule
                        // and vanish entirely, which is worse than the
                        // double-fire this skip exists to prevent.
                        if rule.id == "openai-key"
                            && anthropic_rule_active
                            && m.as_str()[3..].starts_with("ant-")
                        {
                            continue;
                        }
                        if rule.require_entropy {
                            // Entropy-check capture group 1 (the secret value
                            // itself) when the rule's regex has one — a
                            // keyword-anchored rule's whole match includes the
                            // low-entropy keyword/quotes/`=` scaffolding, which
                            // would dilute the score below threshold even for a
                            // genuinely high-entropy value. Rules with no
                            // capture group (e.g. the bare-hex Azure pattern,
                            // where match == value already) fall back to the
                            // whole match, preserving prior behavior exactly.
                            let entropy_target = caps.get(1).map_or(m.as_str(), |g| g.as_str());
                            if shannon_entropy(entropy_target) < ENTROPY_THRESHOLD {
                                continue;
                            }
                        }
                        hits.push(RuleHit {
                            rule_id: rule.id.clone(),
                            span: (offset + m.start(), offset + m.end()),
                            severity: Severity::Critical,
                        });
                    }
                }

                if detect_base64 {
                    for (start, end, rule_id) in detect_base64_secrets(line, &active_rules) {
                        hits.push(RuleHit {
                            rule_id,
                            span: (offset + start, offset + end),
                            severity: Severity::Critical,
                        });
                    }
                }
            }
            offset += line.len() + 1; // +1 for the '\n' split() strips
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "secrets".to_string(),
        action,
        severity: Severity::Critical,
        hits,
        confidence: None,
    }
}

/// One `Bool` spec per distinct `rules.yaml` `provider`, labeled by the
/// rule's own description — the management UI's secrets checkbox list. In
/// sync with `evaluate` automatically because both read the same YAML.
pub(crate) fn rule_option_specs() -> Vec<super::OptionSpec> {
    use std::collections::HashSet;

    use super::{OptionKind, OptionSpec};

    let raw: Vec<RawRule> =
        rule_loader::parse_rules(include_str!("../../rules/secrets/rules.yaml"), SOURCE);
    let mut seen = HashSet::new();
    let mut specs: Vec<OptionSpec> = Vec::new();
    for rule in &raw {
        if seen.insert(rule.provider.clone()) {
            specs.push(OptionSpec {
                key: rule.provider.clone(),
                kind: OptionKind::Bool,
                label: rule.description.clone(),
                help: Some(format!("option key: {}", rule.provider)),
                default: serde_json::json!(false),
            });
        }
    }
    specs.push(OptionSpec {
        key: "detect_base64".to_string(),
        kind: OptionKind::Bool,
        label: "Detect base64-encoded secrets".to_string(),
        help: Some(
            "Decodes candidate base64 substrings and re-checks them for a known \
             secret prefix or an enabled provider pattern."
                .to_string(),
        ),
        default: serde_json::json!(false),
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

    fn github_pat() -> String {
        format!("ghp_{}", "A".repeat(36))
    }

    fn aws_access_key_id() -> String {
        format!("AKIA{}", "B".repeat(16))
    }

    #[test]
    fn github_token_denies_when_provider_enabled() {
        let text = format!("leaked token: {}", github_pat());
        let result = evaluate(&text, &opts(&[("github", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert_eq!(result.hits.len(), 1);
    }

    #[test]
    fn aws_key_allowed_when_provider_disabled() {
        let text = format!("key id: {}", aws_access_key_id());
        let result = evaluate(&text, &opts(&[("github", true), ("aws", false)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn aws_key_denies_when_provider_enabled() {
        let text = format!("key id: {}", aws_access_key_id());
        let result = evaluate(&text, &opts(&[("aws", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn stopword_adjacent_match_is_suppressed() {
        let text = format!("placeholder token for docs: {}", github_pat());
        let result = evaluate(&text, &opts(&[("github", true)]));
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn no_providers_enabled_allows() {
        let text = format!("{} {}", github_pat(), aws_access_key_id());
        let result = evaluate(&text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn openai_key_denies_when_provider_enabled() {
        let text = format!("OPENAI_API_KEY={}", "sk-".to_string() + &"A".repeat(30));
        let result = evaluate(&text, &opts(&[("openai", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn anthropic_key_does_not_double_fire_as_openai_key() {
        let text = format!(
            "ANTHROPIC_API_KEY={}",
            "sk-ant-".to_string() + &"A".repeat(30)
        );
        let result = evaluate(&text, &opts(&[("openai", true), ("anthropic", true)]));
        let rule_ids: Vec<&str> = result.hits.iter().map(|h| h.rule_id.as_str()).collect();
        assert!(rule_ids.contains(&"anthropic-key"));
        assert!(!rule_ids.contains(&"openai-key"));
    }

    #[test]
    fn anthropic_key_is_still_caught_by_openai_rule_when_anthropic_disabled() {
        // Regression guard: the openai-key rule used to skip an sk-ant- match
        // unconditionally, so with only "openai" enabled (anthropic-key
        // inactive) the key matched neither rule and went undetected.
        let text = format!(
            "ANTHROPIC_API_KEY={}",
            "sk-ant-".to_string() + &"A".repeat(30)
        );
        let result = evaluate(&text, &opts(&[("openai", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "openai-key"));
    }

    #[test]
    fn anthropic_key_denies_when_provider_enabled() {
        let text = format!(
            "ANTHROPIC_API_KEY={}",
            "sk-ant-".to_string() + &"A".repeat(30)
        );
        let result = evaluate(&text, &opts(&[("anthropic", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn azure_bare_hex_key_requires_high_entropy() {
        // Repeated-digit hex is low entropy -- should NOT trip the
        // entropy-gated bare-hex rule even though it matches the shape.
        let low_entropy_hex = "a".repeat(32);
        let result = evaluate(&low_entropy_hex, &opts(&[("azure", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn azure_bare_hex_key_denies_when_high_entropy() {
        let high_entropy_hex = "3f9a2b7c1e4d6f8091a2b3c4d5e6f701";
        let result = evaluate(high_entropy_hex, &opts(&[("azure", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn postgres_connection_string_denies_when_provider_enabled() {
        let text = "DATABASE_URL=postgres://user:hunter2@db.internal:5432/prod";
        let result = evaluate(text, &opts(&[("postgres", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn base64_encoded_secret_detected_when_enabled() {
        // base64("sk-ant-" + 30 A's)
        use base64::Engine;
        let secret = format!("sk-ant-{}", "A".repeat(30));
        let encoded = base64::engine::general_purpose::STANDARD.encode(&secret);
        let text = format!("payload: {encoded}");
        let mut o = opts(&[("anthropic", true)]);
        o.set_bool("detect_base64", true);
        let result = evaluate(&text, &o);
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn slack_admin_token_denies_when_provider_enabled() {
        let text = format!("leaked: xoxa-{}", "A".repeat(45));
        let result = evaluate(&text, &opts(&[("slack", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "slack-admin-token"));
    }

    #[test]
    fn slack_refresh_token_denies_when_provider_enabled() {
        let text = format!("leaked: xoxr-{}", "A".repeat(45));
        let result = evaluate(&text, &opts(&[("slack", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "slack-refresh-token"));
    }

    #[test]
    fn twilio_api_key_denies_when_provider_enabled() {
        let text = format!("leaked: SK{}", "a".repeat(32));
        let result = evaluate(&text, &opts(&[("twilio", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "twilio-api-key"));
    }

    #[test]
    fn twilio_account_sid_still_denies_and_is_distinct_from_api_key() {
        let text = "leaked: ACaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let result = evaluate(text, &opts(&[("twilio", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "twilio-account-sid"));
        assert!(result.hits.iter().all(|h| h.rule_id != "twilio-api-key"));
    }

    #[test]
    fn custom_key_format_denies_when_high_entropy() {
        let text = r#"internal_secret_key = "Zk9q2Lp7vXw4Rt6Ym1Bn8Cd3Fs5Hj0Ae""#;
        let result = evaluate(text, &opts(&[("generic_secret", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "generic-high-entropy-secret"));
    }

    #[test]
    fn placeholder_value_allows_despite_matching_shape() {
        let text = r#"api_key = "your_api_key_here_please_replace""#;
        let result = evaluate(text, &opts(&[("generic_secret", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn low_entropy_repeated_password_allows() {
        let text = r#"password = "aaaaaaaaaaaaaaaaaaaaaaaa""#;
        let result = evaluate(text, &opts(&[("generic_secret", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn generic_secret_disabled_by_default() {
        let text = r#"internal_secret_key = "Zk9q2Lp7vXw4Rt6Ym1Bn8Cd3Fs5Hj0Ae""#;
        let result = evaluate(text, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn azure_bare_hex_entropy_check_unaffected_by_capture_group_change() {
        // Regression guard: azure-openai-key has no capture group, so the
        // entropy check must still fall back to the whole match.
        let high_entropy_hex = "3f9a2b7c1e4d6f8091a2b3c4d5e6f701";
        let result = evaluate(high_entropy_hex, &opts(&[("azure", true)]));
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn azure_sas_token_denies_when_provider_enabled() {
        let text = format!(
            "leaked: https://acct.blob.core.windows.net/container/file?sv=2021-08-06&ss=b&srt=co&sp=rl&se=2026-01-01&sig={}",
            "A".repeat(45)
        );
        let result = evaluate(&text, &opts(&[("azure", true)]));
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "azure-sas"));
    }

    #[test]
    fn base64_detection_off_by_default() {
        use base64::Engine;
        let secret = format!("sk-ant-{}", "A".repeat(30));
        let encoded = base64::engine::general_purpose::STANDARD.encode(&secret);
        let text = format!("payload: {encoded}");
        let result = evaluate(&text, &opts(&[("anthropic", true)]));
        assert_eq!(result.action, CheckAction::Log);
    }
}
