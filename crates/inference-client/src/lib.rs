//! Client for the `armor-inference` sidecar.
//!
//! What it holds: the types that go over the wire ([`contract`]), the
//! [`InferenceTransport`] trait and its typed errors, an HTTP implementation
//! ([`http`]), the address guard that implementation validates its endpoint
//! with ([`net_guard`]), a shareable [`CircuitBreaker`], and a
//! [`MockTransport`] for tests that want no network at all.
//!
//! `armor-api` calls this in the request path via `ml::escalate`
//! (`crates/api/src/ml.rs`), behind a `None`-able feature flag: no
//! `ARMOR_INFERENCE_URL` means the tier is off and this crate is never
//! reached.
//!
//! Deliberately free of an `armor-core` dependency. The sidecar speaks a
//! model's vocabulary (`MlDecision`, confidence, thresholds); Armor speaks
//! policy's (`CheckAction`, `EnforcementMode`, `Verdict`). Mapping between
//! them is policy-dependent and lives in `armor-core`'s
//! `engine::escalation`, which keeps this crate a pure description of the
//! remote service.

pub mod breaker;
pub mod cache;
pub mod contract;
pub mod http;
pub mod mock;
pub mod net_guard;
pub mod transport;

pub use breaker::{BreakerConfig, CircuitBreaker};
pub use cache::CachingTransport;
pub use contract::{InferRequest, InferResult, MlDecision, ModelInfo};
pub use http::{HttpConfig, HttpTransport};
pub use mock::MockTransport;
pub use net_guard::EndpointError;
pub use transport::{InferError, InferenceTransport};
