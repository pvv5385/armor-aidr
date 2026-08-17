//! Library surface for `armor-api`. `main.rs` is a thin binary entrypoint
//! over this; splitting it out is what lets `tests/` (e.g.
//! `portkey_integration.rs`) exercise the real router/state construction
//! instead of re-implementing it.

pub mod aidr;
pub mod audit;
pub mod broker_state;
pub mod config;
pub mod control_plane;
pub mod custom_rules;
pub mod hardware;
pub mod heartbeat;
pub mod integrations;
pub mod middleware;
pub mod ml;
pub mod otel;
pub mod profiles;
pub mod redaction;
pub mod retention;
pub mod routes;
pub mod session_state;
pub mod state;
pub mod sync;
pub mod telemetry;
