//! Postgres-backed persistence: DB-backed tenant policy (`policy_store`),
//! the per-request decision log (`audit_events`), durable session counters
//! (`sessions`), and the reversible-anonymization vault (`vault`) — see
//! each module's doc comment.
//!
//! Two caveats that span the whole crate:
//!
//! - **Single-tenant only.** `sessions` and `vault` key on a
//!   caller-supplied `session_id` with no `tenant_id`, which is safe
//!   exactly as long as one deployment serves one tenant. `sessions`'
//!   module doc explains what has to change before that stops being true.
//! - **`vault` holds recoverable PII.** It is encrypted by this crate
//!   before it reaches the database, and nothing exposes decryption over
//!   HTTP — see `vault`'s module doc for the threat model.

pub mod audit_events;
pub mod inference_pins;
pub mod policy_store;
pub mod sessions;
pub mod vault;

#[cfg(test)]
mod test_support;
