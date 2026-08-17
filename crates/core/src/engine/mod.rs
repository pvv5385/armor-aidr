//! Orchestration: parallel check execution with fast-deny cancellation,
//! sequential cheap-to-expensive mode, per-check timeout plus whole-run
//! wall-clock budget.

pub mod decision;
pub mod escalation;
pub mod normalize;
pub mod orchestrator;
pub mod redact;
pub mod scorecard_gate;
