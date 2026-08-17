//! Customer-supplied regex rules — the generalized counterpart to
//! `keyword_blocklist`'s literal-only matching, for deployments that need
//! actual regex patterns. Unlike the shipped rulesets
//! (secrets/prompt_injection/pii/jailbreak/harmful_content/malicious_url),
//! there is no `rules/*.yaml` for this detector: the pattern set is runtime,
//! per-deployment data — typically loaded from `ARMOR_CUSTOM_RULES_DIR` (see
//! `armor-api`'s `custom_rules` module) rather than `include_str!`'d at
//! compile time. It carries none of the false-positive vetting the shipped
//! rulesets get (`config/benchmarks/*` + the eval-harness tests), so the
//! sample config ships this `mode: warn` — promote to `block` only once
//! you've validated your own patterns.
//!
//! Patterns arrive as plain data via `options.patterns` (a list of
//! `{rule_id, pattern, severity, case_sensitive}` maps), not by
//! `armor-core` doing file I/O itself — `armor-core` never does I/O;
//! `armor-api` reads the custom-rules directory and folds it into policy
//! `options` before any of this runs.
//!
//! `CheckOptions` is cloned once per check run (`orchestrator::run_checks_with_budget`
//! clones each enabled `CheckConfig` off the policy), so a cache living
//! *inside* it would be rebuilt every single request —
//! useless. Instead each distinct rule set compiles exactly once, ever, in
//! a process-global cache keyed by the rule content itself (not the
//! `CheckOptions` instance): safe even when a policy has more than one
//! `custom_regex` check entry (e.g. one `block`-mode set, one `warn`-mode
//! set) with different patterns.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use once_cell::sync::Lazy;
use regex::RegexBuilder;
use serde::Deserialize;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

#[derive(Debug, Clone, Deserialize)]
struct RawRule {
    rule_id: String,
    pattern: String,
    #[serde(default = "default_severity")]
    severity: Severity,
    #[serde(default)]
    case_sensitive: bool,
}

fn default_severity() -> Severity {
    Severity::Medium
}

struct CompiledRule {
    rule_id: String,
    pattern: regex::Regex,
    severity: Severity,
}

static CACHE: Lazy<RwLock<HashMap<String, Arc<Vec<CompiledRule>>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

fn build_compiled(options: &CheckOptions) -> Result<Vec<CompiledRule>, String> {
    let raw = options.struct_list_option::<RawRule>("patterns")?;
    raw.into_iter()
        .map(|r| {
            RegexBuilder::new(&r.pattern)
                .case_insensitive(!r.case_sensitive)
                .build()
                .map(|pattern| CompiledRule {
                    rule_id: r.rule_id.clone(),
                    pattern,
                    severity: r.severity,
                })
                .map_err(|e| format!("rule {:?}: invalid pattern {:?}: {e}", r.rule_id, r.pattern))
        })
        .collect()
}

/// Content-addressed, not identity-addressed — `CheckOptions` has no
/// stable identity across the per-request `.clone()`, so the cache key has
/// to be derived from the rule data itself.
fn cache_key(options: &CheckOptions) -> String {
    format!("{:?}", options.struct_list_option::<RawRule>("patterns"))
}

fn compiled_rules(options: &CheckOptions) -> Arc<Vec<CompiledRule>> {
    let key = cache_key(options);
    if let Some(hit) = CACHE.read().expect("custom_regex cache poisoned").get(&key) {
        return hit.clone();
    }
    // Should already be unreachable in a real deployment — `validate` runs
    // at startup — but a detector must never crash the process on bad
    // config it wasn't asked to validate (e.g. a test building `options`
    // directly). The orchestrator wraps every check in
    // `catch_unwind` and applies the check's own `fail_mode` on panic, so
    // this degrades to a normal fail-open/fail-closed check error instead.
    let compiled = build_compiled(options)
        .unwrap_or_else(|e| panic!("custom_regex: invalid options.patterns: {e}"));
    let compiled = Arc::new(compiled);
    CACHE
        .write()
        .expect("custom_regex cache poisoned")
        .insert(key, compiled.clone());
    compiled
}

/// Called once at startup (`armor-api`, right after policy load) so a bad
/// customer regex fails the deploy immediately with a clear message,
/// instead of surfacing as a per-request fail-open/fail-closed at request
/// time via the panic-recovery path in `compiled_rules`.
pub fn validate(options: &CheckOptions) -> Result<(), String> {
    build_compiled(options).map(|_| ())
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let rules = compiled_rules(options);

    let mut hits: Vec<RuleHit> = Vec::new();
    for rule in rules.iter() {
        for m in rule.pattern.find_iter(text) {
            hits.push(RuleHit {
                rule_id: rule.rule_id.clone(),
                span: (m.start(), m.end()),
                severity: rule.severity,
            });
        }
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    let severity = hits
        .iter()
        .map(|h| h.severity)
        .max()
        .unwrap_or(Severity::Low);

    DetectorResult {
        detector_id: "custom_regex".to_string(),
        action,
        severity,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(patterns: &[(&str, &str)]) -> CheckOptions {
        let mut o = CheckOptions::default();
        let seq: Vec<serde_yaml::Value> = patterns
            .iter()
            .map(|(rule_id, pattern)| {
                let mut m = serde_yaml::Mapping::new();
                m.insert("rule_id".into(), (*rule_id).into());
                m.insert("pattern".into(), (*pattern).into());
                serde_yaml::Value::Mapping(m)
            })
            .collect();
        o.set_raw("patterns", serde_yaml::Value::Sequence(seq));
        o
    }

    #[test]
    fn matching_pattern_denies() {
        let result = evaluate(
            "employee id EMP-4471 was flagged",
            &opts(&[("employee-id", r"EMP-\d{4}")]),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].rule_id, "employee-id");
        assert_eq!(result.hits[0].span, (12, 20));
    }

    #[test]
    fn no_match_logs_only() {
        let result = evaluate(
            "nothing to see here",
            &opts(&[("employee-id", r"EMP-\d{4}")]),
        );
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn case_insensitive_by_default() {
        let result = evaluate(
            "secret codeword: MERGERFALCON",
            &opts(&[("codeword", "mergerfalcon")]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn multiple_rules_all_evaluated() {
        let result = evaluate(
            "id EMP-1234 and codeword FALCON",
            &opts(&[("employee-id", r"EMP-\d{4}"), ("codeword", "FALCON")]),
        );
        assert_eq!(result.hits.len(), 2);
    }

    #[test]
    fn validate_rejects_invalid_regex() {
        let err = validate(&opts(&[("bad", "(unclosed")])).unwrap_err();
        assert!(err.contains("bad"));
    }

    #[test]
    fn validate_accepts_empty_patterns() {
        assert!(validate(&CheckOptions::default()).is_ok());
    }

    #[test]
    fn distinct_rule_sets_do_not_collide_in_the_cache() {
        let a = evaluate("alpha-token", &opts(&[("a", "alpha-token")]));
        let b = evaluate("alpha-token", &opts(&[("b", "bravo-token")]));
        assert_eq!(a.action, CheckAction::Deny);
        assert_eq!(b.action, CheckAction::Log);
    }
}
