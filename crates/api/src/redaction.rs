//! Bridges the reversible-anonymization vault (`armor_storage::vault`)
//! into `armor-core`'s redaction seam.
//!
//! # The seam, and why the bridge lives here
//!
//! `armor_core::engine::redact` resolves *which* byte ranges get masked
//! ([`plan_redactions`]) and *how* to splice replacements in
//! ([`apply_redactions`]) — both pure and synchronous, as everything in
//! `armor-core` must be. What it deliberately leaves open is where
//! the placeholder strings come from. Its own
//! [`local_placeholders`] numbers them per request and forgets them: mask
//! and discard. The vault mints them per *session*, keeps an encrypted copy
//! of the original, and can hand it back later.
//!
//! That's the whole job of this module — the middle step. It runs after the
//! deterministic sweep (`orchestrator::run_deterministic`) and before the
//! verdict is composed, which is exactly the ordering
//! `orchestrator::compose_with_redaction` documents as the reason redaction
//! moved to the end of the run.
//!
//! # What actually gets stored, and what doesn't
//!
//! Two gates, and both have to open:
//!
//! 1. **The deployment configured a vault** — `DATABASE_URL` *and*
//!    `ARMOR_VAULT_KEY` (`main::wire_vault`). Neither implies the other.
//! 2. **The policy asked for reversibility** — the span belongs to a check
//!    whose resolved action is [`CheckAction::Redact`], i.e. `on_fail:
//!    redact` in the profile (or a model layer escalating to it,
//!    `escalation::merge`).
//!
//! Gate 2 is the important one. Reversible anonymization means recoverable
//! PII in Postgres — the part of this design most likely to fail an
//! enterprise security review on its own. A `secrets`
//! check firing on an AWS key wants that key *gone*, not filed away
//! encrypted-but-recoverable — so `on_fail: deny` spans keep in-process
//! numbering and their originals are discarded, even with a vault sitting
//! right there. Turning the key on changes nothing until someone edits a
//! policy.
//!
//! # Cost, stated plainly
//!
//! This puts database round trips on the scan path — the same golden rule
//! and the same narrow exception `session_state` takes, for the same
//! reason: a session-scoped
//! placeholder is definitionally cross-request state and cannot be
//! preloaded. It is bounded by the number of *distinct values* redacted (one
//! `anonymize` per distinct value, deduplicated below, and a hit on an
//! already-vaulted value is a single indexed `SELECT`), and it is paid only
//! by requests that both matched a `redact` check and run under a policy
//! that asked for this.
//!
//! A vault failure never fails the request: it degrades to in-process
//! numbering and logs. The text is still redacted — what's lost is the
//! ability to reverse it, which is a capability degradation, not a
//! disclosure.

use std::collections::{HashMap, HashSet};

use armor_core::engine::{
    decision::{self, CheckOutcome, Decision},
    orchestrator,
    redact::{self, PlannedRedaction},
};
use armor_storage::{
    sessions,
    vault::{NewSecret, Vault, VaultError},
};

use crate::state::AppState;

/// Builds `redacted_text` and composes the verdict, vaulting the spans a
/// policy asked to be reversible.
///
/// The no-vault path is `orchestrator::compose_with_redaction` unchanged, so
/// a deployment without `ARMOR_VAULT_KEY` gets byte-identical output to
/// before this module existed.
pub async fn compose(
    state: &AppState,
    session_id: &str,
    text: &str,
    outcomes: Vec<CheckOutcome>,
) -> Decision {
    let Some(vault) = state.vault.as_ref() else {
        return orchestrator::compose_with_redaction(text, outcomes);
    };

    let plan = redact::plan_redactions(text, &outcomes);
    let placeholders = match mint(vault, state, session_id, text, &plan).await {
        Ok(placeholders) => placeholders,
        Err(e) => {
            // Deliberately not an error response, matching `session_state`'s
            // posture: the text below is still fully masked, so degrading
            // costs recoverability, not confidentiality.
            tracing::warn!(
                error = %e,
                session_id = %session_id,
                "vault unavailable; falling back to in-process placeholders for this \
                 request (redaction still applied, but it is not reversible)"
            );
            redact::local_placeholders(text, &plan)
        }
    };

    decision::compose(
        outcomes,
        redact::apply_redactions(text, &plan, &placeholders),
    )
}

/// A placeholder per planned span, with the reversible ones vault-minted.
///
/// An empty `plan` means no database work happens at all.
async fn mint(
    vault: &Vault,
    state: &AppState,
    session_id: &str,
    text: &str,
    plan: &[PlannedRedaction],
) -> Result<Vec<String>, VaultError> {
    if plan.is_empty() {
        return Ok(Vec::new());
    }
    let Some(db) = state.db.as_ref() else {
        // Structurally unreachable — `main::wire_vault` builds the vault out
        // of the database pool, so one never exists without the other. A
        // panic on the request path isn't worth the assertion.
        tracing::warn!("vault configured without a database; skipping reversible anonymization");
        return Ok(redact::local_placeholders(text, plan));
    };

    let has_reversible = plan.iter().any(PlannedRedaction::reversible);
    if has_reversible {
        // `vault_entries.session_id` is a foreign key. The scan path has
        // usually created the row already (`session_state::apply` ->
        // `sessions::touch`), but only when a session-stateful check is
        // enabled — which has nothing to do with whether this request
        // contains PII. `ensure` creates it without counting a second
        // request against the session's budgets.
        sessions::ensure(db.pool(), session_id, state.session_ttl_seconds).await?;
    }

    // One `anonymize` per distinct value rather than per span. The vault
    // would return the same placeholder either way (a stable mapping per
    // value), so this is purely about not paying a round trip to be told
    // something we already know.
    let mut vaulted: Vec<Option<String>> = Vec::with_capacity(plan.len());
    let mut by_value: HashMap<&str, String> = HashMap::new();

    for span in plan {
        if !span.reversible() {
            vaulted.push(None);
            continue;
        }
        let value = span.value(text);
        let placeholder = match by_value.get(value) {
            Some(existing) => existing.clone(),
            None => {
                let stored = vault
                    .anonymize(
                        session_id,
                        NewSecret {
                            value,
                            category: &span.category,
                            rule_id: &span.rule_id,
                        },
                    )
                    .await?;
                by_value.insert(value, stored.placeholder.clone());
                stored.placeholder
            }
        };
        vaulted.push(Some(placeholder));
    }

    // The discard-only spans are numbered locally, but must avoid every
    // placeholder this *session* has ever minted — not just the ones this
    // request happened to vault above. A local counter that only knew about
    // this request's own mints could still land on an ordinal an earlier
    // request in the same session already gave to a *different* value (its
    // reversible spans don't necessarily touch every ordinal this request's
    // discard spans would reach), silently making one placeholder mean two
    // people's PII. `list_placeholders` is the vault's own record of what it
    // has ever handed out for this session, so it's authoritative regardless
    // of which past request minted what.
    let discard: Vec<PlannedRedaction> = plan.iter().filter(|p| !p.reversible()).cloned().collect();
    let mut discard_placeholders = if discard.is_empty() {
        Vec::new()
    } else {
        let taken: HashSet<String> = vault
            .list_placeholders(session_id)
            .await?
            .into_iter()
            .map(|p| p.placeholder)
            .collect();
        redact::local_placeholders_avoiding(text, &discard, &taken)
    }
    .into_iter();

    // `discard_placeholders` has exactly one entry per `None` in `vaulted`,
    // both built from the same predicate over the same plan.
    Ok(vaulted
        .into_iter()
        .map(|v| {
            v.or_else(|| discard_placeholders.next())
                .unwrap_or_default()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use armor_core::models::{CheckAction, EnforcementMode, RuleHit, Severity};

    fn outcome(category: &str, action: CheckAction, hits: Vec<RuleHit>) -> CheckOutcome {
        CheckOutcome {
            category: category.to_string(),
            passed: hits.is_empty(),
            action,
            severity: Severity::High,
            hits,
            view: "raw".to_string(),
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

    /// The gate `mint` opens on. Mirrors its first line, so these tests
    /// assert the real predicate rather than a paraphrase of it.
    fn wants_vaulting(text: &str, outcomes: &[CheckOutcome]) -> bool {
        redact::plan_redactions(text, outcomes)
            .iter()
            .any(PlannedRedaction::reversible)
    }

    #[test]
    fn a_deny_only_run_vaults_nothing() {
        // A `secrets` hit wants the key gone, not filed away recoverable —
        // even with a vault configured and sitting right there.
        let outcomes = vec![outcome(
            "secrets",
            CheckAction::Deny,
            vec![hit("aws-key", (0, 4))],
        )];
        assert!(!wants_vaulting("AKIA rest", &outcomes));
    }

    #[test]
    fn a_redact_check_that_fired_vaults_its_spans() {
        let outcomes = vec![outcome(
            "pii",
            CheckAction::Redact,
            vec![hit("email-address", (0, 7))],
        )];
        assert!(wants_vaulting("a@b.com and more", &outcomes));
    }

    #[test]
    fn a_redact_check_that_found_nothing_vaults_nothing() {
        // No hits means no spans, so the database gate never opens — a
        // policy configured for reversibility costs nothing on clean traffic.
        let outcomes = vec![outcome("pii", CheckAction::Redact, Vec::new())];
        assert!(!wants_vaulting("nothing to see", &outcomes));
    }

    #[test]
    fn a_redact_hit_on_a_normalized_view_is_not_vaulted() {
        // Its span is an offset into the transformed text, so it can't be
        // masked (`redact`'s module doc) — and a value we can't locate in
        // the original is not one we can meaningfully store either.
        let mut on_nfkc = outcome(
            "pii",
            CheckAction::Redact,
            vec![hit("email-address", (0, 7))],
        );
        on_nfkc.view = "nfkc".to_string();
        assert!(!wants_vaulting("a@b.com and more", &[on_nfkc]));
    }
}
