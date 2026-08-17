//! Core types shared by every detector and the orchestration engine.
//!
//! The verdict model has three distinct axes that earlier design passes
//! collapsed into one vocabulary — kept separate here on purpose:
//!   - [`CheckAction`]: what a single detector believes should happen.
//!   - [`EnforcementMode`]: how much authority that detector's action carries
//!     right now (policy-controlled, e.g. downgraded by the scorecard gate).
//!   - [`Verdict`]: the single answer the engine returns to the caller.

use serde::{Deserialize, Serialize};

/// What a detector believes should happen to the content it inspected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckAction {
    Deny,
    Redact,
    Flag,
    Log,
}

/// How much authority a check's action carries, per policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnforcementMode {
    Block,
    Warn,
    Monitor,
}

/// The single verdict the engine returns to the caller for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    Allow,
    Warn,
    Redact,
    Block,
    Ask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// A single pattern/rule match within a detector's evaluation of one input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleHit {
    pub rule_id: String,
    pub span: (usize, usize),
    pub severity: Severity,
}

/// The output of one detector running against one normalized view of the input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorResult {
    pub detector_id: String,
    pub action: CheckAction,
    pub severity: Severity,
    pub hits: Vec<RuleHit>,
    pub confidence: Option<f32>,
}

impl DetectorResult {
    /// A detector "passes" (content is clean) when it isn't recommending `deny`.
    pub fn passed(&self) -> bool {
        self.action != CheckAction::Deny
    }
}
