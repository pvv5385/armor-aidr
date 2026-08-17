//! Structure/schema validation — deterministic shape checks on a
//! (typically JSON) payload: oversized body, excessive nesting depth, and
//! optionally "must parse as JSON at all". Aimed at the `tool`/`agent-plan`/
//! `agent-step`/`memory-write` stages, where the payload is structured data
//! rather than free text a human wrote.
//!
//! This is a shape check, not a JSON-Schema validator — there is no
//! per-tool schema registry yet (that's `tool_allowlist` plus, later, a
//! real schema-per-tool feature). What it catches today: the abuse patterns
//! that don't need a schema to recognize — a payload deliberately bloated
//! or deeply nested to exhaust a downstream parser or hide content past a
//! reviewer's normal reading depth.

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

fn default_max_bytes() -> f64 {
    262_144.0 // 256 KiB
}

fn default_max_depth() -> f64 {
    16.0
}

fn json_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(items) => 1 + items.iter().map(json_depth).max().unwrap_or(0),
        serde_json::Value::Object(map) => 1 + map.values().map(json_depth).max().unwrap_or(0),
        _ => 0,
    }
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let require_valid_json = options.bool_option("require_valid_json", false);
    let max_bytes = options.f64_option("max_bytes", default_max_bytes()) as usize;
    let max_depth = options.f64_option("max_depth", default_max_depth()) as usize;

    let mut hits: Vec<RuleHit> = Vec::new();
    let span = (0, text.len());

    if text.len() > max_bytes {
        hits.push(RuleHit {
            rule_id: "structure-oversized-payload".to_string(),
            span,
            severity: Severity::Medium,
        });
    }

    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(value) => {
            if json_depth(&value) > max_depth {
                hits.push(RuleHit {
                    rule_id: "structure-excessive-nesting".to_string(),
                    span,
                    severity: Severity::Medium,
                });
            }
        }
        Err(_) if require_valid_json && !text.trim().is_empty() => {
            hits.push(RuleHit {
                rule_id: "structure-invalid-json".to_string(),
                span,
                severity: Severity::Medium,
            });
        }
        Err(_) => {}
    }

    let action = if hits.is_empty() {
        CheckAction::Log
    } else {
        CheckAction::Deny
    };
    DetectorResult {
        detector_id: "structure_validation".to_string(),
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
    fn valid_small_json_allows() {
        let result = evaluate(
            r#"{"tool":"search","args":{"q":"rust"}}"#,
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn plain_text_without_require_valid_json_allows() {
        let result = evaluate("just some ordinary free text", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn invalid_json_denies_when_required() {
        let mut o = CheckOptions::default();
        o.set_bool("require_valid_json", true);
        let result = evaluate("not json at all", &o);
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "structure-invalid-json"));
    }

    #[test]
    fn oversized_payload_denies() {
        let mut o = CheckOptions::default();
        o.set_f64("max_bytes", 10.0);
        let result = evaluate("this text is definitely longer than ten bytes", &o);
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "structure-oversized-payload"));
    }

    #[test]
    fn excessive_nesting_denies() {
        let mut o = CheckOptions::default();
        o.set_f64("max_depth", 2.0);
        let nested = r#"{"a":{"b":{"c":{"d":1}}}}"#; // depth 4
        let result = evaluate(nested, &o);
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "structure-excessive-nesting"));
    }

    #[test]
    fn shallow_nesting_within_default_allows() {
        let nested = r#"{"a":{"b":{"c":1}}}"#; // depth 3, well under default 16
        let result = evaluate(nested, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }
}
