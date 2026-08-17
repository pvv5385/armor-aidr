//! The transport seam. `HttpTransport` (reqwest) is the shipped
//! implementation; a `GrpcTransport` (tonic + `tower::balance`) could swap
//! in behind this same trait if and when many-to-many load balancing is
//! actually needed.
//!
//! Failures are a typed enum, not string-matched: the caller has to branch
//! on them to pick a fallback path, and matching on message text is how
//! that goes wrong silently.

use async_trait::async_trait;

use crate::contract::{InferRequest, InferResult, ModelInfo};

#[derive(Debug, thiserror::Error)]
pub enum InferError {
    /// The per-call deadline elapsed. Distinct from `Unavailable` because it
    /// is the one failure that says nothing about the sidecar's health — a
    /// single slow call should not trip a circuit breaker on its own.
    #[error("inference call timed out after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },
    /// Could not reach the sidecar at all (connect refused, DNS, pool
    /// exhausted).
    #[error("inference backend unavailable: {0}")]
    Unavailable(String),
    /// Reached it; it declined. `429` from the saturation guard lands here.
    #[error("inference backend returned {status}: {message}")]
    Status { status: u16, message: String },
    /// The task name is not in the sidecar's registry, or its artifact
    /// failed verification and the task is marked `available: false`.
    #[error("unknown or unavailable task {0:?}")]
    UnknownTask(String),
    /// Reached it, it answered, the answer did not fit the contract. Note
    /// that out-of-range *values* do not land here — those clamp during
    /// deserialization (see `contract`); this is a shape mismatch.
    #[error("malformed inference response: {0}")]
    Malformed(String),
    /// The breaker is open, so the call was never made. Kept separate from
    /// `Unavailable` so `fallback_path` can record which one happened.
    #[error("inference circuit breaker open")]
    CircuitOpen,
}

impl InferError {
    /// Whether this failure should count toward tripping a circuit breaker.
    /// A timeout does not: it is the caller's deadline expiring, and one
    /// slow call is not evidence the pool is unhealthy. `CircuitOpen` does
    /// not either — the call never happened.
    pub fn is_breaker_signal(&self) -> bool {
        matches!(
            self,
            InferError::Unavailable(_) | InferError::Status { .. } | InferError::Malformed(_)
        )
    }
}

#[async_trait]
pub trait InferenceTransport: Send + Sync {
    /// Score one request against `task`. Implementations own their own
    /// per-call deadline; the whole-pass budget is layered on top of this by
    /// the caller, mirroring how `check_timeout` nests inside
    /// `guardrail_timeout` in the deterministic sweep.
    async fn infer(&self, task: &str, req: InferRequest<'_>) -> Result<InferResult, InferError>;

    /// What the pool can serve. Used at startup for the scorecard gate and
    /// by the control plane's models view.
    async fn models(&self) -> Result<Vec<ModelInfo>, InferError>;
}
