//! Tool allow/deny list — for the `tool`/`agent-plan`/`agent-step` stages,
//! checks the invoked tool's name (a top-level string field in the JSON
//! payload, `options.field`, default `"tool"`) against a per-policy allow
//! and/or deny list.
//!
//! Payloads that aren't JSON, or that don't carry the configured field, are
//! not this detector's concern — `structure_validation` owns flagging a
//! malformed payload; this detector only judges a tool name it actually
//! found.

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

fn tool_name(text: &str, field: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    value.get(field)?.as_str().map(str::to_string)
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    let field = options.str_option("field").unwrap_or("tool");
    let allow = options.str_list_option("allow");
    let deny = options.str_list_option("deny");

    let mut hits: Vec<RuleHit> = Vec::new();
    if let Some(name) = tool_name(text, field) {
        let denied = deny.contains(&name);
        let not_allowlisted = !allow.is_empty() && !allow.contains(&name);
        if denied || not_allowlisted {
            hits.push(RuleHit {
                rule_id: format!("tool:{name}"),
                span: (0, text.len()),
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
        detector_id: "tool_allowlist".to_string(),
        action,
        severity: Severity::High,
        hits,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(allow: &[&str], deny: &[&str]) -> CheckOptions {
        let mut o = CheckOptions::default();
        o.set_str_list("allow", allow);
        o.set_str_list("deny", deny);
        o
    }

    #[test]
    fn denied_tool_denies() {
        let result = evaluate(
            r#"{"tool":"delete_database"}"#,
            &opts(&[], &["delete_database"]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn tool_outside_nonempty_allowlist_denies() {
        let result = evaluate(
            r#"{"tool":"send_email"}"#,
            &opts(&["search", "read_file"], &[]),
        );
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn tool_inside_allowlist_allows() {
        let result = evaluate(r#"{"tool":"search"}"#, &opts(&["search", "read_file"], &[]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn empty_allow_and_deny_allows_everything() {
        let result = evaluate(r#"{"tool":"anything"}"#, &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn default_field_ignores_payload_keyed_under_a_different_name() {
        // default field is "tool", so a payload keyed "name" isn't inspected at all.
        let result = evaluate(
            r#"{"name":"delete_database"}"#,
            &opts(&[], &["delete_database"]),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn custom_field_name_is_honored() {
        let mut o = opts(&[], &["delete_database"]);
        o.set_str("field", "name");
        let result = evaluate(r#"{"name":"delete_database"}"#, &o);
        assert_eq!(result.action, CheckAction::Deny);
    }

    #[test]
    fn non_json_payload_allows() {
        let result = evaluate("not json", &opts(&[], &["anything"]));
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn missing_field_allows() {
        let result = evaluate(r#"{"args":{}}"#, &opts(&[], &["search"]));
        assert_eq!(result.action, CheckAction::Log);
    }
}
