//! Detector implementations and the category → detector-fn registry, ported
//! from `app/guardrails/checks/` + `app/guardrails/checks/registry.py`.
//!
//! Each detector is `fn(text: &str, options: &CheckOptions) -> DetectorResult`
//! — synchronous, side-effect free. The orchestrator (`engine::orchestrator`)
//! owns scheduling, timeouts, and cancellation; detectors never see any of
//! that.

pub mod abuse;
pub mod checksums;
pub mod citation_integrity;
pub mod code_safety;
pub mod competitor;
pub mod compliance;
// Shared helper bank, not a standalone detector — no `get_check` entry of
// its own. Consumed via `super::context_bloat` by `retrieval_chunk_injection`.
mod context_bloat;
pub mod copyright;
pub mod custom_regex;
pub mod defamation;
pub mod document_metadata_leakage;
pub mod elections;
pub mod excessive_agency;
pub mod exfiltration;
pub mod gibberish;
pub mod hallucination;
pub mod harmful_content;
pub mod hate;
// Shared helper bank, not a standalone detector — no `get_check` entry of
// its own. Consumed via `super::injection_markers` by `mcp_manifest_scanner`,
// `tool_output_injection`, and `retrieval_chunk_injection`.
mod injection_markers;
pub mod jailbreak;
pub mod keyword_blocklist;
pub mod malicious_url;
pub mod mcp_manifest_scanner;
pub mod memory_write_poisoning;
pub mod numerical_consistency;
pub mod over_refusal;
pub mod pci;
pub mod pii;
pub mod prompt_injection;
pub mod retrieval_chunk_injection;
pub(crate) mod rule_loader;
pub mod secrets;
pub mod self_harm;
pub mod sensitive_business_data;
pub mod sentiment;
pub mod sex_crimes;
pub mod sexual_content;
pub mod structure_validation;
pub mod system_prompt_leakage;
pub mod tool_allowlist;
pub mod tool_output_injection;
pub mod unbounded_consumption;
pub mod web_sanitization;

use crate::models::DetectorResult;
use crate::policy::schema::CheckOptions;
use serde::{Deserialize, Serialize};

pub type CheckFn = fn(&str, &CheckOptions) -> DetectorResult;

/// How the management UI should render one `options` value — the friendly
/// counterpart to the free-form JSON the profile editor used to expose.
/// `armor-core` owns the schema because the detectors are the only place
/// that knows what each category actually reads; the API just forwards it,
/// so the UI never hardcodes detector knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionKind {
    Bool,
    Number,
    String,
    StringList,
}

/// One configurable option of a detector category. `default` mirrors the
/// value the detector uses when the key is absent (`options.bool_option(key,
/// default)` and friends), so the UI can pre-check inputs instead of leaving
/// them blank; `null` means "unset" (the detector's own fallback applies).
#[derive(Debug, Clone, Serialize)]
pub struct OptionSpec {
    pub key: String,
    pub kind: OptionKind,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
    pub default: serde_json::Value,
}

fn bool_spec(key: &str, label: &str, default: bool, help: Option<&str>) -> OptionSpec {
    OptionSpec {
        key: key.to_string(),
        kind: OptionKind::Bool,
        label: label.to_string(),
        help: help.map(str::to_string),
        default: serde_json::json!(default),
    }
}

fn num_spec(key: &str, label: &str, default: Option<f64>, help: Option<&str>) -> OptionSpec {
    OptionSpec {
        key: key.to_string(),
        kind: OptionKind::Number,
        label: label.to_string(),
        help: help.map(str::to_string),
        default: serde_json::to_value(default).unwrap_or(serde_json::Value::Null),
    }
}

fn str_spec(key: &str, label: &str, default: &str, help: Option<&str>) -> OptionSpec {
    OptionSpec {
        key: key.to_string(),
        kind: OptionKind::String,
        label: label.to_string(),
        help: help.map(str::to_string),
        default: serde_json::json!(default),
    }
}

fn list_spec(key: &str, label: &str, help: Option<&str>) -> OptionSpec {
    OptionSpec {
        key: key.to_string(),
        kind: OptionKind::StringList,
        label: label.to_string(),
        help: help.map(str::to_string),
        default: serde_json::Value::Array(Vec::new()),
    }
}

/// The options each detector category understands, for the management UI's
/// profile editor. Returns an empty list for detectors with no configurable
/// options (their defaults *are* the detector's behavior). Rule-driven
/// categories (`pii`, `secrets`) get one `Bool` spec per rule, derived from
/// the rules YAML's own `option`/`provider` + `description` fields, so the
/// editor's checkbox list can never drift from the actual ruleset.
pub fn option_schema(category: &str) -> Vec<OptionSpec> {
    match category {
        "pii" => pii::rule_option_specs(),
        "secrets" => secrets::rule_option_specs(),
        "pci" => vec![
            bool_spec("credit_card", "Credit card numbers", false, None),
            bool_spec(
                "luhn_required",
                "Require Luhn check-digit validation",
                true,
                Some("Only flag card-shaped numbers whose Luhn checksum passes."),
            ),
        ],
        "keyword_blocklist" => vec![
            list_spec("keywords", "Blocked keywords/phrases (one per line)", None),
            bool_spec("case_sensitive", "Case-sensitive matching", false, None),
            bool_spec(
                "whole_word",
                "Whole-word matches only",
                true,
                Some("Off matches substrings inside larger words."),
            ),
        ],
        "malicious_url" => vec![
            bool_spec("ip_literal_host", "IP-literal hosts", true, None),
            bool_spec("punycode", "Punycode/obfuscated hostnames", true, None),
            bool_spec("credentials_in_url", "Credentials in URL", true, None),
            bool_spec("shorteners", "URL shorteners", true, None),
            bool_spec(
                "homoglyph_typosquat",
                "Homoglyph / typosquatted domains",
                true,
                None,
            ),
            bool_spec("suspicious_tld", "Suspicious TLDs", true, None),
            bool_spec(
                "excessive_subdomains",
                "Excessive subdomain depth",
                true,
                None,
            ),
            bool_spec("data_uri", "data: URIs", true, None),
            num_spec(
                "max_subdomain_depth",
                "Max subdomain depth",
                Some(3.0),
                Some("Subdomain chains deeper than this are flagged."),
            ),
        ],
        "structure_validation" => vec![
            bool_spec(
                "require_valid_json",
                "Require valid JSON",
                false,
                Some("Non-JSON payloads are flagged when on."),
            ),
            num_spec(
                "max_bytes",
                "Max payload bytes",
                Some(262_144.0),
                Some("Payloads larger than this are flagged."),
            ),
            num_spec(
                "max_depth",
                "Max nesting depth",
                Some(16.0),
                Some("JSON/object nesting deeper than this is flagged."),
            ),
        ],
        "tool_allowlist" => vec![
            str_spec(
                "field",
                "Tool-name field",
                "tool",
                Some("Top-level JSON field that holds the tool name."),
            ),
            list_spec("allow", "Allowed tools (one per line)", None),
            list_spec("deny", "Denied tools (one per line)", None),
        ],
        "code_safety" => vec![bool_spec(
            "pattern_match",
            "Pattern-based scanning",
            true,
            Some("Off disables the built-in pattern bank."),
        )],
        "document_metadata_leakage" => vec![bool_spec(
            "invisible_text",
            "Detect invisible/zero-width text",
            true,
            None,
        )],
        "retrieval_chunk_injection" => vec![
            bool_spec("pattern_match", "Injection-marker scan", true, None),
            bool_spec("context_bloat", "Context-bloat scan", true, None),
            num_spec(
                "context_bloat_max_chars",
                "Bloat: max chars",
                Some(5000.0),
                None,
            ),
            num_spec(
                "context_bloat_min_chars",
                "Bloat: min chars",
                Some(50.0),
                None,
            ),
            num_spec(
                "context_bloat_min_entropy",
                "Bloat: min entropy",
                Some(3.5),
                None,
            ),
            num_spec(
                "context_bloat_max_repetition_ratio",
                "Bloat: max repetition ratio",
                Some(0.4),
                None,
            ),
            num_spec(
                "context_bloat_ngram_size",
                "Bloat: n-gram size",
                Some(3.0),
                None,
            ),
            num_spec(
                "context_bloat_max_run_ratio",
                "Bloat: max run ratio",
                Some(0.1),
                None,
            ),
        ],
        "tool_output_injection" => vec![bool_spec(
            "pattern_match",
            "Injection-marker scan",
            true,
            Some("Off disables the built-in pattern bank."),
        )],
        "gibberish" => vec![
            num_spec(
                "min_text_length",
                "Min text length to check",
                Some(10.0),
                None,
            ),
            bool_spec("entropy", "Entropy check", true, None),
            num_spec("entropy_threshold", "Entropy threshold", Some(4.6), None),
            num_spec(
                "min_length",
                "Min length for entropy check",
                Some(20.0),
                None,
            ),
            bool_spec("vowel_ratio", "Vowel-ratio check", true, None),
            num_spec(
                "vowel_ratio_threshold",
                "Vowel-ratio threshold",
                Some(0.15),
                None,
            ),
            num_spec("min_token_length", "Min token length", Some(8.0), None),
            bool_spec("invisible_char_ratio", "Invisible-char check", true, None),
            num_spec(
                "invisible_char_ratio_threshold",
                "Invisible-char ratio threshold",
                Some(0.1),
                None,
            ),
        ],
        "numerical_consistency" => vec![
            bool_spec("arithmetic", "Arithmetic consistency", true, None),
            bool_spec("percentage", "Percentage consistency", true, None),
        ],
        "hallucination" => vec![
            num_spec("overlap_threshold", "Overlap threshold", Some(0.3), None),
            num_spec("min_tokens", "Min tokens", Some(5.0), None),
        ],
        "system_prompt_leakage" => vec![
            num_spec("shingle_size", "Shingle size", Some(6.0), None),
            num_spec(
                "similarity_threshold",
                "Similarity threshold",
                Some(0.25),
                None,
            ),
        ],
        "copyright" => vec![
            num_spec("shingle_size", "Shingle size", Some(6.0), None),
            num_spec(
                "similarity_threshold",
                "Similarity threshold",
                Some(0.3),
                None,
            ),
        ],
        "unbounded_consumption" => vec![
            num_spec(
                "max_tokens_per_session",
                "Max tokens per session",
                None,
                Some("Leave blank for no limit."),
            ),
            num_spec(
                "max_requests_per_session",
                "Max requests per session",
                None,
                Some("Leave blank for no limit."),
            ),
            num_spec(
                "max_loop_depth",
                "Max loop depth",
                None,
                Some("Leave blank for no limit."),
            ),
        ],
        "abuse" => vec![
            num_spec(
                "max_requests_per_window",
                "Max requests per window",
                None,
                Some("Leave blank for no limit."),
            ),
            num_spec("window_seconds", "Window (seconds)", Some(60.0), None),
        ],
        _ => Vec::new(),
    }
}

/// Forces every rules.yaml-backed detector's `Lazy` rule set to compile now,
/// so a malformed rule file fails process startup instead of surfacing as a
/// panic on the first request that happens to exercise that detector — see
/// `rule_loader`'s module doc. Call this once, before accepting traffic.
///
/// Each detector's `Lazy` still panics internally on a bad rule (changing
/// ~25 detectors' `RULES` statics to a fallible type for a failure mode
/// that, today, can only happen at compile time — every `rules.yaml` is
/// `include_str!`-embedded, not loaded from disk at runtime — isn't
/// justified). What this function does instead: `catch_unwind` around every
/// detector's `warm()`
/// so one bad rule file doesn't abort the process before the rest have even
/// been checked, and collect every failure into one `Result` so a caller
/// deploying a change that broke three rule files gets all three
/// error messages instead of fixing them one crash-and-rebuild at a time.
pub fn validate_all_rules() -> Result<(), Vec<String>> {
    let detectors: &[(&str, fn())] = &[
        ("code_safety", code_safety::warm),
        ("compliance", compliance::warm),
        ("defamation", defamation::warm),
        ("document_metadata_leakage", document_metadata_leakage::warm),
        ("elections", elections::warm),
        ("excessive_agency", excessive_agency::warm),
        ("exfiltration", exfiltration::warm),
        ("harmful_content", harmful_content::warm),
        ("hate", hate::warm),
        ("injection_markers", injection_markers::warm),
        ("jailbreak", jailbreak::warm),
        ("malicious_url", malicious_url::warm),
        ("mcp_manifest_scanner", mcp_manifest_scanner::warm),
        ("memory_write_poisoning", memory_write_poisoning::warm),
        ("over_refusal", over_refusal::warm),
        ("pii", pii::warm),
        ("prompt_injection", prompt_injection::warm),
        ("secrets", secrets::warm),
        ("self_harm", self_harm::warm),
        ("sensitive_business_data", sensitive_business_data::warm),
        ("sentiment", sentiment::warm),
        ("sex_crimes", sex_crimes::warm),
        ("sexual_content", sexual_content::warm),
        ("web_sanitization", web_sanitization::warm),
    ];

    let errors: Vec<String> = detectors
        .iter()
        .filter_map(|&(name, warm)| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(warm))
                .err()
                .map(|payload| {
                    let message = payload
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "rule bank failed to compile".to_string());
                    format!("{name}: {message}")
                })
        })
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Looks up the detector function for a policy `category` string, mirroring
/// `registry.py::get_check`. `None` for an unknown category — the caller
/// decides how to fail (reject the policy at load time).
pub fn get_check(category: &str) -> Option<CheckFn> {
    match category {
        "abuse" => Some(abuse::evaluate),
        "pci" => Some(pci::evaluate),
        "secrets" => Some(secrets::evaluate),
        "prompt_injection" => Some(prompt_injection::evaluate),
        "jailbreak" => Some(jailbreak::evaluate),
        "harmful_content" => Some(harmful_content::evaluate),
        "pii" => Some(pii::evaluate),
        "keyword_blocklist" => Some(keyword_blocklist::evaluate),
        "custom_regex" => Some(custom_regex::evaluate),
        "malicious_url" => Some(malicious_url::evaluate),
        "structure_validation" => Some(structure_validation::evaluate),
        "tool_allowlist" => Some(tool_allowlist::evaluate),
        "exfiltration" => Some(exfiltration::evaluate),
        "sensitive_business_data" => Some(sensitive_business_data::evaluate),
        "code_safety" => Some(code_safety::evaluate),
        "document_metadata_leakage" => Some(document_metadata_leakage::evaluate),
        "web_sanitization" => Some(web_sanitization::evaluate),
        "competitor" => Some(competitor::evaluate),
        "retrieval_chunk_injection" => Some(retrieval_chunk_injection::evaluate),
        "tool_output_injection" => Some(tool_output_injection::evaluate),
        "mcp_manifest_scanner" => Some(mcp_manifest_scanner::evaluate),
        "memory_write_poisoning" => Some(memory_write_poisoning::evaluate),
        "excessive_agency" => Some(excessive_agency::evaluate),
        "sentiment" => Some(sentiment::evaluate),
        "gibberish" => Some(gibberish::evaluate),
        "compliance" => Some(compliance::evaluate),
        "citation_integrity" => Some(citation_integrity::evaluate),
        "numerical_consistency" => Some(numerical_consistency::evaluate),
        "hallucination" => Some(hallucination::evaluate),
        "unbounded_consumption" => Some(unbounded_consumption::evaluate),
        "copyright" => Some(copyright::evaluate),
        "system_prompt_leakage" => Some(system_prompt_leakage::evaluate),
        "over_refusal" => Some(over_refusal::evaluate),
        "hate" => Some(hate::evaluate),
        "self_harm" => Some(self_harm::evaluate),
        "sexual_content" => Some(sexual_content::evaluate),
        "sex_crimes" => Some(sex_crimes::evaluate),
        "defamation" => Some(defamation::evaluate),
        "elections" => Some(elections::evaluate),
        _ => None,
    }
}

/// Default cheapest-first execution order for a check's `category` — a
/// fixed, backend-owned ranking (not configurable) the orchestrator's
/// sequential mode sorts on, so cheap detectors can short-circuit a deny
/// before the expensive ones run. Cheapest tier first: literal/substring
/// list checks, then single-pass regex pattern banks, then structured
/// analysis (checksums, parsing, field extraction), then stateful detectors
/// that need session/DB lookups. Unknown categories sort last — they're
/// unvetted and shouldn't outrank the shipped detectors.
pub fn default_order(category: &str) -> u32 {
    const ORDER: &[(&str, u32)] = &[
        ("keyword_blocklist", 0),
        ("competitor", 1),
        ("custom_regex", 2),
        ("sentiment", 3),
        ("gibberish", 4),
        ("harmful_content", 5),
        ("over_refusal", 6),
        ("copyright", 7),
        ("compliance", 8),
        ("hate", 9),
        ("self_harm", 10),
        ("sexual_content", 11),
        ("sex_crimes", 12),
        ("defamation", 13),
        ("elections", 14),
        ("prompt_injection", 15),
        ("jailbreak", 16),
        ("exfiltration", 17),
        ("code_safety", 18),
        ("document_metadata_leakage", 19),
        ("web_sanitization", 20),
        ("tool_output_injection", 21),
        ("mcp_manifest_scanner", 22),
        ("memory_write_poisoning", 23),
        ("sensitive_business_data", 24),
        ("retrieval_chunk_injection", 25),
        ("excessive_agency", 26),
        ("pii", 27),
        ("pci", 28),
        ("secrets", 29),
        ("malicious_url", 30),
        ("structure_validation", 31),
        ("tool_allowlist", 32),
        ("numerical_consistency", 33),
        ("citation_integrity", 34),
        ("hallucination", 35),
        ("abuse", 36),
        ("unbounded_consumption", 37),
        ("system_prompt_leakage", 38),
    ];
    ORDER
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, rank)| *rank)
        .unwrap_or(u32::MAX)
}

/// Whether this category's `evaluate()` has a side effect beyond reading its
/// input: `abuse` and `unbounded_consumption` each mutate a process-global
/// counter keyed by `options.session_id` (or touch the durable one) every
/// time they run. The orchestrator sweeps every normalized view against a
/// category's detector (`orchestrator::run_view_sweep`) so a rot13'd
/// jailbreak still gets caught by a pattern check — but running a *stateful*
/// detector once per view multiplies its side effect once per view too,
/// inflating one request's rate-limit/budget usage by however many views
/// `NormalizeConfig` turned on. These run against the `raw` view only.
pub fn is_stateful(category: &str) -> bool {
    matches!(category, "abuse" | "unbounded_consumption")
}

/// Every category string [`get_check`] recognizes — used by the management
/// UI (`armor-api`'s control-plane API) to populate a detector picker
/// without hand-duplicating this list client-side. Kept in sync with
/// `get_check`'s match arms by the round-trip test below.
pub fn categories() -> &'static [&'static str] {
    &[
        "abuse",
        "pci",
        "secrets",
        "prompt_injection",
        "jailbreak",
        "harmful_content",
        "pii",
        "keyword_blocklist",
        "custom_regex",
        "malicious_url",
        "structure_validation",
        "tool_allowlist",
        "exfiltration",
        "sensitive_business_data",
        "code_safety",
        "document_metadata_leakage",
        "web_sanitization",
        "competitor",
        "retrieval_chunk_injection",
        "tool_output_injection",
        "mcp_manifest_scanner",
        "memory_write_poisoning",
        "excessive_agency",
        "sentiment",
        "gibberish",
        "compliance",
        "citation_integrity",
        "numerical_consistency",
        "hallucination",
        "unbounded_consumption",
        "copyright",
        "system_prompt_leakage",
        "over_refusal",
        "hate",
        "self_harm",
        "sexual_content",
        "sex_crimes",
        "defamation",
        "elections",
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn validate_all_rules_succeeds_on_the_shipped_rule_banks() {
        assert_eq!(validate_all_rules(), Ok(()));
    }

    #[test]
    fn every_category_resolves_to_a_real_detector() {
        for category in categories() {
            assert!(
                get_check(category).is_some(),
                "categories() lists {category:?} but get_check doesn't recognize it"
            );
        }
    }

    #[test]
    fn categories_has_no_duplicates() {
        let mut sorted = categories().to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), categories().len());
    }

    #[test]
    fn default_order_covers_every_category_with_a_distinct_rank() {
        let mut ranked: Vec<u32> = categories()
            .iter()
            .map(|c| {
                let rank = default_order(c);
                assert_ne!(
                    rank,
                    u32::MAX,
                    "default_order has no entry for {c:?} — add one so sequential mode sorts it"
                );
                rank
            })
            .collect();
        ranked.sort_unstable();
        ranked.dedup();
        assert_eq!(
            ranked.len(),
            categories().len(),
            "default_order ranks must be unique"
        );
    }

    #[test]
    fn default_order_puts_unknown_categories_last() {
        assert_eq!(default_order("_no_such_category"), u32::MAX);
    }

    #[test]
    fn option_schema_never_panics_for_any_category() {
        for category in categories() {
            let _ = option_schema(category);
        }
    }

    #[test]
    fn pii_schema_covers_contact_and_regional_options() {
        let keys: Vec<String> = option_schema("pii").iter().map(|s| s.key.clone()).collect();
        for expected in ["email", "phone", "ssn", "iban", "ip_address"] {
            assert!(
                keys.iter().any(|k| k == expected),
                "pii schema missing {expected}"
            );
        }
        assert!(option_schema("pii")
            .iter()
            .any(|s| s.key == "skip_private_ips"));
    }

    #[test]
    fn secrets_schema_derives_providers_from_the_rule_bank() {
        let keys: Vec<String> = option_schema("secrets")
            .iter()
            .map(|s| s.key.clone())
            .collect();
        for expected in ["aws", "github", "stripe", "detect_base64"] {
            assert!(
                keys.iter().any(|k| k == expected),
                "secrets schema missing {expected}"
            );
        }
    }

    #[test]
    fn schemas_use_expected_kinds() {
        let keyword = option_schema("keyword_blocklist");
        let by_key: HashMap<&str, OptionKind> =
            keyword.iter().map(|s| (s.key.as_str(), s.kind)).collect();
        assert_eq!(by_key["keywords"], OptionKind::StringList);
        assert_eq!(by_key["case_sensitive"], OptionKind::Bool);

        let structure = option_schema("structure_validation");
        assert!(structure.iter().any(|s| s.kind == OptionKind::Number));
    }
}
