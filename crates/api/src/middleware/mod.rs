//! Cross-cutting `tower::Layer`s: auth, rate-limit, security headers,
//! OTel metrics. Auth (API key) and rate-limit (in-process token bucket)
//! are both off by default and gated behind `Settings`.
//!
//! The per-request decision log lives at the crate-root `audit.rs`, not
//! here — it isn't cross-cutting HTTP middleware, it's called explicitly
//! from `aidr::run_scan` once a decision exists to log.

pub mod auth;
pub mod otel_metrics;
pub mod rate_limit;
pub mod redis_rate_limit;
pub mod security_headers;
