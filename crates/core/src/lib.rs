//! armor-core: detectors, policy, and orchestration.
//!
//! Synchronous by design — plain functions and data, no `async`, no I/O, no
//! server. All async and I/O live in `armor-api`.

pub mod detectors;
pub mod engine;
mod homoglyphs;
pub mod models;
pub mod policy;

pub use engine::decision::{CheckOutcome, Decision};
pub use models::{CheckAction, DetectorResult, EnforcementMode, RuleHit, Severity, Verdict};
