//! Turns [`crate::models::RuleHit`] spans into a span-masked rendering of the
//! scanned text — the piece that actually produces sanitized text for
//! `Verdict::Redact` / `CheckAction::Redact`. `redacted_text` is
//! computed unconditionally alongside every [`Decision`](crate::engine::decision::Decision),
//! independent of which action/verdict the run produced, so a caller doing
//! redact-and-continue never has to special-case the policy that fired.
//!
//! Only hits on the `"raw"` view are masked. Every other view
//! (`nfkc`, `homoglyph`, `base64`, ...) transforms the text before matching,
//! so a hit's span there is a byte offset into a *different* string than the
//! one passed in here — masking against it would silently corrupt the
//! output. Attacks that only surface on a normalized view still drive the
//! block/warn verdict as before via `outcomes`; they just aren't reflected
//! in `redacted_text` yet — mapping normalized-view offsets back to raw
//! byte spans is a separate, not-yet-built project.
//!
//! # Why this is split into plan / number / apply
//!
//! [`build_redacted_text`] is the whole pipeline, and is what
//! `armor-core` itself uses. But the *placeholders* don't have to come from
//! here: `armor-storage`'s reversible-anonymization vault mints
//! session-scoped placeholders that survive across requests and can be
//! deanonymized later, and it is async and Postgres-backed — neither of
//! which `armor-core` is allowed to be.
//!
//! So the pipeline is three separable steps and the middle one is
//! substitutable:
//!
//! 1. [`plan_redactions`] — pure, sync: which byte ranges get masked, with
//!    the labels and the owning check's [`CheckAction`]. No placeholder
//!    text yet.
//! 2. A placeholder per planned span. [`local_placeholders`] is the
//!    in-process, redact-and-discard numbering; `armor-api`'s `redaction`
//!    module swaps in vault-minted ones for the spans a policy actually
//!    asked to be reversible.
//! 3. [`apply_redactions`] — pure, sync: splice them into the text.

use std::collections::{HashMap, HashSet};

use crate::engine::decision::CheckOutcome;
use crate::models::CheckAction;

/// One byte range that will be masked, resolved out of the run's outcomes
/// but not yet assigned a placeholder.
///
/// Spans here are always offsets into the `"raw"` text passed to
/// [`plan_redactions`], already sorted by start and guaranteed
/// non-overlapping and on `char` boundaries — so a caller can slice with
/// them without re-validating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRedaction {
    pub start: usize,
    pub end: usize,
    /// The owning [`CheckOutcome`]'s category, e.g. `pii`.
    pub category: String,
    /// The [`RuleHit`](crate::models::RuleHit) that matched, e.g.
    /// `email-address`.
    pub rule_id: String,
    /// What the owning check's policy said to do. The only consumer is the
    /// vault bridge: `Redact` is a policy explicitly asking for reversible
    /// anonymization, and anything else is redact-and-discard, which must
    /// **not** put recoverable PII in the database just because a vault
    /// happens to be configured.
    pub action: CheckAction,
}

impl PlannedRedaction {
    /// `CATEGORY:RULE_ID`, the placeholder's inner text.
    pub fn label(&self) -> String {
        placeholder_label(&self.category, &self.rule_id)
    }

    /// The original text this span covers. Infallible for any `text` this
    /// plan was built from — the span is validated at plan time.
    pub fn value<'a>(&self, text: &'a str) -> &'a str {
        text.get(self.start..self.end).unwrap_or_default()
    }

    /// Whether the policy asked for this span to be *reversibly*
    /// anonymized rather than masked and discarded.
    pub fn reversible(&self) -> bool {
        self.action == CheckAction::Redact
    }
}

/// The `CATEGORY:RULE_ID` inside a `<CATEGORY:RULE_ID:n>` placeholder, e.g.
/// `PII:EMAIL_ADDRESS` — built from data every check already carries
/// (`CheckOutcome::category`, `RuleHit::rule_id`) rather than a new per-rule
/// taxonomy field.
///
/// Public because it is the *canonical* placeholder shape and there must be
/// exactly one of it: `armor_storage`'s vault mints placeholders for the
/// same spans out of the same two strings, and a second implementation of
/// this three-line normalization is a bug waiting to happen — the first one
/// (`EMAIL-ADDRESS` vs `EMAIL_ADDRESS`, from the hyphen rule below) made
/// vaulted placeholders unresolvable against the text they appeared in.
pub fn placeholder_label(category: &str, rule_id: &str) -> String {
    let category = category.to_uppercase();
    let rule_id = rule_id.to_uppercase().replace(['-', ' '], "_");
    format!("{category}:{rule_id}")
}

/// Resolves which byte ranges of `text` get masked, in the order they'll be
/// spliced.
///
/// Spans are sorted by start (ties broken by longest-first) and accepted
/// greedily; one that overlaps a span already accepted is dropped rather
/// than allowed to corrupt the output.
///
/// A span that isn't a usable slice of `text` — empty, inverted, past the
/// end, or landing mid-`char` — is dropped too. Every deterministic detector
/// derives its spans from a `regex` match on this exact string and so can't
/// produce one, but an escalated NER layer's spans arrive from a model on
/// the far side of a wire (`escalation::merge`), and a bad offset there
/// should cost a missed mask, not a panicked request.
pub fn plan_redactions(text: &str, outcomes: &[CheckOutcome]) -> Vec<PlannedRedaction> {
    let mut planned: Vec<PlannedRedaction> = outcomes
        .iter()
        // `abuse`/`unbounded_consumption` hits carry a `(0, text.len())`
        // span to flag the whole message as the thing that tripped a
        // rate-limit or token budget — there's no content there to mask.
        // Treating it as a redaction span anyway makes it the widest,
        // earliest-sorted span every time, so it wins `plan_redactions`'
        // greedy non-overlap pass and silently drops every real PII hit in
        // the same message.
        .filter(|o| o.view == "raw" && !crate::detectors::is_stateful(&o.category))
        .flat_map(|o| {
            o.hits.iter().map(|h| PlannedRedaction {
                start: h.span.0,
                end: h.span.1,
                category: o.category.clone(),
                rule_id: h.rule_id.clone(),
                action: o.action,
            })
        })
        .filter(|p| {
            p.start < p.end && text.is_char_boundary(p.start) && text.is_char_boundary(p.end)
        })
        .collect();
    planned.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));

    let mut cursor = 0usize;
    planned.retain(|p| {
        if p.start < cursor {
            return false; // overlaps a span already accepted
        }
        cursor = p.end;
        true
    });
    planned
}

/// The in-process placeholder for every planned span, positionally aligned
/// with `plan`.
///
/// Numbering is **per distinct value**, not per occurrence: the same value
/// mentioned twice gets the same placeholder both times — the vault's own
/// stability property, held here too, so a deployment's `redacted_text`
/// doesn't change shape when the vault is switched on — only whether the
/// placeholders survive the request.
pub fn local_placeholders(text: &str, plan: &[PlannedRedaction]) -> Vec<String> {
    local_placeholders_avoiding(text, plan, &HashSet::new())
}

/// [`local_placeholders`], but never minting a placeholder already present
/// in `taken`.
///
/// Exists for the mixed run: when a vault has already minted placeholders
/// for the reversible spans, the remaining spans are numbered from a
/// separate counter and could otherwise land on a string the vault handed
/// out for a *different* value. Ordinals are skipped rather than reused, so
/// one placeholder always means one value.
pub fn local_placeholders_avoiding(
    text: &str,
    plan: &[PlannedRedaction],
    taken: &HashSet<String>,
) -> Vec<String> {
    let mut by_value: HashMap<&str, String> = HashMap::new();
    let mut next_ordinal: HashMap<String, usize> = HashMap::new();

    plan.iter()
        .map(|p| {
            let value = p.value(text);
            if let Some(existing) = by_value.get(value) {
                return existing.clone();
            }
            let label = p.label();
            let placeholder = loop {
                let n = next_ordinal.entry(label.clone()).or_insert(0);
                *n += 1;
                let candidate = format!("<{label}:{n}>");
                if !taken.contains(&candidate) {
                    break candidate;
                }
            };
            by_value.insert(value, placeholder.clone());
            placeholder
        })
        .collect()
}

/// Splices `placeholders` into `text` at the planned spans, preserving every
/// other byte untouched.
///
/// `placeholders` is positionally aligned with `plan` — both
/// [`local_placeholders`] and the vault bridge build it by mapping over the
/// plan. A span with no corresponding placeholder is left unmasked rather
/// than shifting the rest.
pub fn apply_redactions(text: &str, plan: &[PlannedRedaction], placeholders: &[String]) -> String {
    if plan.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (span, placeholder) in plan.iter().zip(placeholders) {
        out.push_str(&text[cursor..span.start]);
        out.push_str(placeholder);
        cursor = span.end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Masks every `"raw"`-view hit across `outcomes` with a numbered
/// placeholder — the whole pipeline, with in-process numbering.
///
/// This is `armor-core`'s own redaction, and stays the behavior of any
/// deployment without a vault configured.
pub fn build_redacted_text(text: &str, outcomes: &[CheckOutcome]) -> String {
    let plan = plan_redactions(text, outcomes);
    let placeholders = local_placeholders(text, &plan);
    apply_redactions(text, &plan, &placeholders)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EnforcementMode, RuleHit, Severity};

    fn outcome(category: &str, view: &str, hits: Vec<RuleHit>) -> CheckOutcome {
        CheckOutcome {
            category: category.to_string(),
            passed: hits.is_empty(),
            action: CheckAction::Deny,
            severity: Severity::High,
            confidence: None,
            hits,
            view: view.to_string(),
            view_text: String::new(),
            error: None,
            timed_out: false,
            latency_ms: 0.0,
            mode: EnforcementMode::Block,
            ..Default::default()
        }
    }

    fn hit(rule_id: &str, span: (usize, usize)) -> RuleHit {
        RuleHit {
            rule_id: rule_id.to_string(),
            span,
            severity: Severity::High,
        }
    }

    #[test]
    fn no_hits_returns_text_unchanged() {
        let text = "hello world";
        assert_eq!(build_redacted_text(text, &[]), text);
    }

    #[test]
    fn masks_a_single_span_with_a_numbered_placeholder() {
        let text = "email me at a@b.com please";
        let outcomes = vec![outcome("pii", "raw", vec![hit("email-address", (12, 19))])];
        assert_eq!(
            build_redacted_text(text, &outcomes),
            "email me at <PII:EMAIL_ADDRESS:1> please"
        );
    }

    #[test]
    fn numbers_repeated_hits_of_the_same_label_independently() {
        let text = "a@b.com and c@d.com";
        let outcomes = vec![outcome(
            "pii",
            "raw",
            vec![hit("email-address", (0, 7)), hit("email-address", (12, 19))],
        )];
        assert_eq!(
            build_redacted_text(text, &outcomes),
            "<PII:EMAIL_ADDRESS:1> and <PII:EMAIL_ADDRESS:2>"
        );
    }

    #[test]
    fn the_same_value_twice_reuses_one_placeholder() {
        // The vault's stability property, held in-process so `redacted_text`
        // doesn't change shape when the vault is switched on.
        let text = "a@b.com and a@b.com";
        let outcomes = vec![outcome(
            "pii",
            "raw",
            vec![hit("email-address", (0, 7)), hit("email-address", (12, 19))],
        )];
        assert_eq!(
            build_redacted_text(text, &outcomes),
            "<PII:EMAIL_ADDRESS:1> and <PII:EMAIL_ADDRESS:1>"
        );
    }

    #[test]
    fn hits_from_non_raw_views_are_not_masked() {
        let text = "plain ascii text";
        let outcomes = vec![outcome("pii", "nfkc", vec![hit("email-address", (0, 5))])];
        assert_eq!(build_redacted_text(text, &outcomes), text);
    }

    #[test]
    fn overlapping_spans_keep_the_first_and_drop_the_rest() {
        let text = "0123456789";
        let outcomes = vec![outcome(
            "secrets",
            "raw",
            vec![
                hit("aws-access-key-id", (0, 5)),
                hit("generic-secret", (3, 8)),
            ],
        )];
        assert_eq!(
            build_redacted_text(text, &outcomes),
            "<SECRETS:AWS_ACCESS_KEY_ID:1>56789"
        );
    }

    // ---- the plan / number / apply split ----

    #[test]
    fn a_plan_carries_the_owning_checks_action() {
        let text = "email me at a@b.com please";
        let mut redacting = outcome("pii", "raw", vec![hit("email-address", (12, 19))]);
        redacting.action = CheckAction::Redact;
        let plan = plan_redactions(text, &[redacting]);

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].value(text), "a@b.com");
        assert_eq!(plan[0].label(), "PII:EMAIL_ADDRESS");
        assert!(plan[0].reversible());
    }

    #[test]
    fn a_deny_action_span_is_not_reversible() {
        let text = "key AKIAIOSFODNN7EXAMPLE here";
        let plan = plan_redactions(
            text,
            &[outcome("secrets", "raw", vec![hit("aws-key", (4, 24))])],
        );
        assert!(!plan[0].reversible(), "redact-and-discard, not vaultable");
    }

    #[test]
    fn a_caller_supplied_placeholder_is_spliced_verbatim() {
        // What the vault bridge does: same plan, different placeholders.
        let text = "email me at a@b.com please";
        let plan = plan_redactions(
            text,
            &[outcome("pii", "raw", vec![hit("email-address", (12, 19))])],
        );
        let vaulted = vec!["<PII:EMAIL_ADDRESS:7>".to_string()];
        assert_eq!(
            apply_redactions(text, &plan, &vaulted),
            "email me at <PII:EMAIL_ADDRESS:7> please"
        );
    }

    #[test]
    fn local_numbering_skips_ordinals_a_vault_already_took() {
        let text = "a@b.com and c@d.com";
        let plan = plan_redactions(
            text,
            &[outcome(
                "pii",
                "raw",
                vec![hit("email-address", (0, 7)), hit("email-address", (12, 19))],
            )],
        );
        let taken = HashSet::from(["<PII:EMAIL_ADDRESS:1>".to_string()]);
        assert_eq!(
            local_placeholders_avoiding(text, &plan, &taken),
            vec![
                "<PII:EMAIL_ADDRESS:2>".to_string(),
                "<PII:EMAIL_ADDRESS:3>".to_string()
            ]
        );
    }

    #[test]
    fn a_span_that_is_not_a_valid_slice_is_dropped_rather_than_panicking() {
        // "é" is two bytes, so (0, 1) lands mid-`char`; (0, 99) is past the
        // end. An NER layer on the far side of a wire can produce either.
        let text = "héllo";
        let outcomes = vec![outcome(
            "pii",
            "raw",
            vec![
                hit("ner-person", (1, 2)),
                hit("ner-person", (0, 99)),
                hit("ner-person", (5, 5)),
                hit("ner-person", (4, 3)),
            ],
        )];
        assert!(plan_redactions(text, &outcomes).is_empty());
        assert_eq!(build_redacted_text(text, &outcomes), text);
    }

    #[test]
    fn a_span_with_no_placeholder_is_left_unmasked() {
        let text = "a@b.com and c@d.com";
        let plan = plan_redactions(
            text,
            &[outcome(
                "pii",
                "raw",
                vec![hit("email-address", (0, 7)), hit("email-address", (12, 19))],
            )],
        );
        assert_eq!(
            apply_redactions(text, &plan, &["<PII:EMAIL_ADDRESS:1>".to_string()]),
            "<PII:EMAIL_ADDRESS:1> and c@d.com"
        );
    }
}
