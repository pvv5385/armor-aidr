//! Gateway integration adapters — API integration first (LiteLLM/Portkey
//! guardrail hooks, no traffic interception), one module per vendor, each
//! exposing its own `/integrations/<vendor>/v1/aidr/scan`
//! route that normalizes that vendor's request shape into the shared
//! `aidr::AidrScanRequest` and runs it through `aidr::run_scan` — the same
//! engine entrypoint `/api/v1/aidr/scan` uses. `litellm.rs` normalizes the
//! payload Armor's own LiteLLM plugin (`integrations/litellm/` at the repo
//! root — a Python custom guardrail that runs inside the LiteLLM proxy
//! process, not a Rust module) sends; `portkey.rs` normalizes Portkey's
//! "Bring Your Own Guardrail" webhook schema, a contract Portkey dictates,
//! not us.
//!
//! Each adapter documents its own BLOCK/REDACT/WARN/ASK capability limits
//! and the `Ask`/`Redact` degradation policy it follows (see its module
//! doc comment).

pub mod litellm;
pub mod portkey;
