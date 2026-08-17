//! Web-sanitization check — browser-executable markup smuggled into prose
//! output that should have been plain text/Markdown: `<script>` tags,
//! inline event-handler attributes, `javascript:` URIs, `<iframe>` embeds,
//! and base64 `data:text/html` URIs.
//!
//! Two layers: the hand-written rule bank in
//! `rules/web_sanitization/rules.yaml` (fast, named, specific), plus a real
//! sanitizer second opinion below via `ammonia` (a maintained HTML
//! sanitizer — the same role Python's `bleach.clean()` plays) that catches
//! event-handler attributes and dangerous URL schemes the hand-written list
//! didn't anticipate (only 6 named handlers, only `href`/`src`).
//!
//! Deliberately NOT a raw before/after string diff on the whole input —
//! ammonia entity-escapes ordinary prose punctuation (`<`, `>`, `&`) and
//! re-quotes/reorders/adds-`rel`-to benign attributes/tags it keeps, so a
//! whole-text diff false-positives on things like "if x < 5", a bracketed
//! placeholder like `<YOUR_API_KEY>`, or code containing generics
//! (`fn foo<T>`). See `ammonia_stripped_hits` for the narrower check that
//! avoids this.

use once_cell::sync::Lazy;
use regex::Regex;

use super::rule_loader::{self, SimpleDetector, SimpleRule};
use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/web_sanitization/rules.yaml"),
        "rules/web_sanitization/rules.yaml",
    )
});

static DETECTOR: SimpleDetector =
    SimpleDetector::new("web_sanitization", Severity::High, CheckAction::Deny);

/// An HTML attribute assignment with a quoted value — feeds
/// `ammonia_stripped_hits` below, not the primary rule bank.
static ATTR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)\b([a-z][\w-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')"#).unwrap());

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
    Lazy::force(&ATTR_RE);
}

/// Runs `text` through `ammonia::clean` and flags any `on*=`-shaped
/// event-handler attribute or `javascript:`/`vbscript:`/`data:text/html`-
/// scheme attribute value that a real sanitizer removed.
///
/// Scoped to exactly these two attribute shapes, and requiring the *exact*
/// pre-image substring to have vanished from the cleaned output, is what
/// keeps this safe against false positives: neither shape ever appears in
/// ordinary prose (no `=` immediately after stray `<`/`>`/`&`), and a
/// kept-but-reformatted safe attribute (ammonia may re-quote or
/// HTML-escape a value it preserves, e.g. `title='hi & bye'` →
/// `title="hi &amp; bye"`) is never inspected because its name/value
/// doesn't match either pattern. Confirmed empirically that ammonia never
/// keeps-but-reformats a genuinely dangerous attribute — it's always
/// either preserved byte-for-byte or removed outright (often taking the
/// whole tag with it), so there's no false-negative gap from reformatting.
fn ammonia_stripped_hits(text: &str) -> Vec<RuleHit> {
    let mut hits = Vec::new();
    if !ATTR_RE.is_match(text) {
        return hits;
    }
    let cleaned = ammonia::clean(text);
    for cap in ATTR_RE.captures_iter(text) {
        let whole = cap.get(0).unwrap();
        if cleaned.contains(whole.as_str()) {
            continue; // kept verbatim -- not stripped, nothing to flag
        }
        let name = cap[1].to_ascii_lowercase();
        let value = cap.get(2).or(cap.get(3)).map(|m| m.as_str()).unwrap_or("");
        let value_lower = value.to_ascii_lowercase();
        if name.starts_with("on") && name.len() > 2 {
            hits.push(RuleHit {
                rule_id: "sanitizer-stripped-event-handler".to_string(),
                span: (whole.start(), whole.end()),
                severity: Severity::High,
            });
        } else if value_lower.starts_with("javascript:")
            || value_lower.starts_with("vbscript:")
            || value_lower.starts_with("data:text/html")
        {
            hits.push(RuleHit {
                rule_id: "sanitizer-stripped-dangerous-url".to_string(),
                span: (whole.start(), whole.end()),
                severity: Severity::High,
            });
        }
    }
    hits
}

pub fn evaluate(text: &str, options: &CheckOptions) -> DetectorResult {
    DETECTOR.evaluate_with(
        &RULES,
        text,
        options,
        |_, o| rule_loader::pattern_match_enabled(o),
        |_, _| true,
        |text, options| -> Vec<RuleHit> {
            if rule_loader::pattern_match_enabled(options) {
                ammonia_stripped_hits(text)
            } else {
                Vec::new()
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_tag_denies() {
        let result = evaluate(
            "here you go: <script>alert(1)</script>",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result.hits.iter().any(|h| h.rule_id == "script-tag"));
    }

    #[test]
    fn event_handler_denies() {
        let result = evaluate(
            r#"<img src=x onerror="alert(1)">"#,
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "event-handler-attribute"));
    }

    #[test]
    fn plain_markdown_allows() {
        let result = evaluate(
            "Check out [our docs](https://example.com/docs) for more",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn pattern_match_disabled_returns_no_hits() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let result = evaluate("<script>alert(1)</script>", &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn unnamed_event_handler_denies_via_sanitizer_diff() {
        // Not one of the 6 named handlers the regex rule covers.
        let result = evaluate(
            r#"<div onmouseenter="steal()">hover</div>"#,
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sanitizer-stripped-event-handler"));
    }

    #[test]
    fn javascript_uri_on_non_href_src_attribute_denies_via_sanitizer_diff() {
        let result = evaluate(
            r#"<video poster="javascript:alert(1)">v</video>"#,
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Deny);
        assert!(result
            .hits
            .iter()
            .any(|h| h.rule_id == "sanitizer-stripped-dangerous-url"));
    }

    #[test]
    fn bare_comparison_operators_allow() {
        let result = evaluate("if x < 5 and y > 10, print done", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn ampersand_in_prose_allows() {
        let result = evaluate("Tom & Jerry is a classic cartoon", &CheckOptions::default());
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn bracketed_placeholder_in_prose_allows() {
        let result = evaluate(
            "Use the <YOUR_API_KEY> placeholder here",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn code_with_generics_allows() {
        let result = evaluate(
            "```rust\nfn foo<T>(x: T) -> T { x }\n```",
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn benign_anchor_with_title_allows() {
        let result = evaluate(
            r#"<a href='https://example.com' title='hi & bye'>text</a>"#,
            &CheckOptions::default(),
        );
        assert_eq!(result.action, CheckAction::Log);
    }

    #[test]
    fn ammonia_pass_also_disabled_by_pattern_match_option() {
        let mut options = CheckOptions::default();
        options.set_bool("pattern_match", false);
        let result = evaluate(r#"<div onmouseenter="steal()">hover</div>"#, &options);
        assert_eq!(result.action, CheckAction::Log);
        assert!(result.hits.is_empty());
    }
}
