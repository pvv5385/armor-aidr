//! LiteLLM gateway adapter — normalizes the LiteLLM proxy's own request
//! shape (an OpenAI-compatible `messages` array plus LiteLLM's ambient
//! metadata) into the shared `aidr::AidrScanRequest`, runs it through
//! `aidr::run_scan`, and responds with the standard `aidr::ScanResponse` JSON — same
//! response shape as `/api/v1/aidr/scan`, since (unlike Portkey) LiteLLM
//! doesn't dictate a response contract here: this is Armor's own plugin
//! (`integrations/litellm/armor_guardrail.py`) calling Armor's own
//! endpoint, not a third party's fixed webhook schema.
//!
//! Request shape (built by the plugin from whatever LiteLLM's hook handed
//! it):
//!
//! ```json
//! {
//!   "mode": "input",
//!   "messages": [{"role": "user", "content": "..."}],
//!   "metadata": {"application_id": "...", "...": "..."},
//!   "model": "gpt-4o",
//!   "litellm_session_id": "thread-42",
//!   "litellm_call_id": "9f3a2b1c-..."
//! }
//! ```
//!
//! `litellm_session_id`, when present, is used as the scan's session id
//! (the role `X-Armor-Session-Id` plays for direct callers) so per-session
//! detector state (e.g. `unbounded_consumption`) accumulates correctly
//! across turns of one LiteLLM conversation — falls back to the
//! `X-Armor-Session-Id` header, then a self-minted UUID, same precedence
//! `routes::resolve_session_id` uses everywhere else.
//!
//! `litellm_call_id` — LiteLLM's own id for this one gateway call — becomes
//! `metadata.request_id` (`ScanResponse.request_id`) unless the caller
//! already set `metadata.request_id` themselves, in which case the
//! explicit value wins. One call, one `request_id`; `litellm_session_id`
//! is the coarser, multi-call id.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::Deserialize;
use std::collections::HashMap;

use crate::{
    aidr::{AidrScanRequest, Message, Metadata},
    routes::resolve_session_id,
    state::AppState,
};

#[derive(Debug, Default, Deserialize)]
struct LiteLlmScanRequest {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    messages: Vec<Message>,
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    litellm_session_id: Option<String>,
    /// LiteLLM's own per-call id (`data["litellm_call_id"]` in the plugin's
    /// hook — see `integrations/litellm/armor_guardrail.py`), used as
    /// `metadata.request_id`'s fallback when the caller's own `metadata`
    /// didn't set one explicitly. Distinct from `litellm_session_id`, which
    /// spans a whole conversation rather than one gateway call.
    #[serde(default)]
    litellm_call_id: Option<String>,
}

pub fn router() -> Router<AppState> {
    Router::new().route("/integrations/litellm/v1/aidr/scan", post(scan))
}

async fn scan(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<LiteLlmScanRequest>,
) -> Result<Response, StatusCode> {
    let session_id = match payload.litellm_session_id.as_deref() {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => resolve_session_id(&headers)?,
    };

    let application_id = payload
        .metadata
        .get("application_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    // Explicit wins: a `request_id` the caller set on `metadata` themselves
    // takes priority over LiteLLM's own `litellm_call_id`.
    let request_id = payload
        .metadata
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or(payload.litellm_call_id);

    let request = AidrScanRequest {
        text: String::new(),
        messages: payload.messages,
        metadata: Metadata {
            mode: payload.mode,
            application_id,
            request_id,
            model: payload.model,
            provider: Some("litellm".to_string()),
            extra: payload.metadata,
            ..Default::default()
        },
        ..Default::default()
    };

    let outcome = crate::aidr::run_scan(&state, &session_id, request).await?;
    Ok(Json(crate::aidr::ScanResponse::from(&outcome)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_minimal_payload() {
        let payload: LiteLlmScanRequest = serde_json::from_str(
            r#"{"mode":"input","messages":[{"role":"user","content":"hello"}]}"#,
        )
        .unwrap();
        assert_eq!(payload.mode.as_deref(), Some("input"));
        assert_eq!(payload.messages.len(), 1);
        assert!(payload.litellm_session_id.is_none());
    }

    #[test]
    fn application_id_is_read_from_metadata() {
        let payload: LiteLlmScanRequest = serde_json::from_str(
            r#"{"messages":[],"metadata":{"application_id":"hr-agent-prod"}}"#,
        )
        .unwrap();
        assert_eq!(
            payload
                .metadata
                .get("application_id")
                .and_then(|v| v.as_str()),
            Some("hr-agent-prod")
        );
    }
}
