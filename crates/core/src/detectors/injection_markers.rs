//! Shared injection-marker pattern bank — not a policy category of its own
//! (no entry in [`super::get_check`]), just the compiled ruleset from
//! `rules/injection_markers/rules.yaml` that
//! [`super::retrieval_chunk_injection`], [`super::tool_output_injection`],
//! and [`super::mcp_manifest_scanner`] each fold into their own
//! [`crate::models::DetectorResult`]. See that YAML file's header for why
//! this is shared rather than duplicated three times.

use once_cell::sync::Lazy;

use super::rule_loader::{self, SimpleRule};
use crate::models::{RuleHit, Severity};

static RULES: Lazy<Vec<SimpleRule>> = Lazy::new(|| {
    rule_loader::compile_simple_rules(
        include_str!("../../rules/injection_markers/rules.yaml"),
        "rules/injection_markers/rules.yaml",
    )
});

/// Forces regex compilation now rather than on the first request that
/// exercises this detector — see `detectors::validate_all`.
pub(crate) fn warm() {
    Lazy::force(&RULES);
}

/// Every injection-marker rule that matches `text`, at [`Severity::High`].
pub(crate) fn hits(text: &str) -> Vec<RuleHit> {
    let mut hits = Vec::new();
    for rule in RULES.iter() {
        for m in rule.pattern.find_iter(text) {
            hits.push(RuleHit {
                rule_id: rule.id.clone(),
                span: (m.start(), m.end()),
                severity: Severity::High,
            });
        }
    }
    hits
}
