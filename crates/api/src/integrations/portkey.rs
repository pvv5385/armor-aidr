//! Portkey "Bring Your Own Guardrail" webhook adapter.
//!
//! Portkey calls this endpoint directly (unlike LiteLLM, where Armor is
//! called from a plugin running inside the gateway process) — the request
//! shape here is Portkey's contract, not ours, verified against
//! <https://portkey.ai/docs/integrations/guardrails/bring-your-own-guardrails>:
//!
//! ```json
//! {
//!   "request": {"json": {...}, "text": "...", "isStreamingRequest": false},
//!   "response": {"json": {...}, "text": "...", "statusCode": null},
//!   "provider": "openai",
//!   "requestType": "chatComplete",
//!   "metadata": {...},
//!   "eventType": "beforeRequestHook"
//! }
//! ```
//!
//! This route (`POST /integrations/portkey/v1/aidr/scan`) does the vendor
//! normalization: parse Portkey's schema above, build the shared
//! `aidr::AidrScanRequest` from it (`eventType: "beforeRequestHook"` reads
//! `request.text`, i.e. Armor's `input` mode; `"afterRequestHook"` reads
//! `response.text`, `output` mode; Portkey's own `metadata` object is
//! folded into ours, and its `request_id` key — if the caller set one —
//! promoted to `metadata.request_id`/`ScanResponse.request_id`; Portkey has
//! no dedicated call-id field of its own to fall back to, unlike LiteLLM's
//! `litellm_call_id`), then run it through `aidr::run_scan` — the same
//! engine entrypoint `/api/v1/aidr/scan` and every other adapter use.
//! Because scanning goes through the shared path, this call also gets an
//! audit-sink/telemetry record — see `integrations/portkey/README.md`'s
//! "Known gaps" for what's still not covered.
//!
//! Expected response: `{"verdict": bool, "transformedData": {...}}` —
//! `transformedData` is how Portkey supports REDACT-style rewrites, but
//! `PortkeyVerdictResponse` below has no such field, so this adapter never
//! populates it; a caught check can only flip `verdict` to `false`, never
//! rewrite the payload.
//!
//! Degradation policy: Portkey's `verdict` is binary, same constraint as
//! LiteLLM. `Verdict::Block` is the only outcome that flips it to `false`;
//! `Warn` and `Allow` both pass (`true`) since Portkey has no separate
//! "flag but allow" lane either.
//! `Verdict::Ask`/`Verdict::Redact` aren't reachable today — no shipped
//! detector emits them yet — but if that changes before this adapter does,
//! the same "only Block fails the check" rule applies to them too, same as
//! LiteLLM's degradation policy.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use armor_core::models::Verdict;

use crate::{
    aidr::{AidrScanRequest, Metadata},
    routes::resolve_session_id,
    state::AppState,
};

#[derive(Debug, Default, Deserialize)]
struct PortkeySide {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct PortkeyHookPayload {
    #[serde(default)]
    request: PortkeySide,
    #[serde(default)]
    response: PortkeySide,
    #[serde(rename = "eventType")]
    event_type: String,
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct PortkeyVerdictResponse {
    verdict: bool,
}

pub fn router() -> Router<AppState> {
    Router::new().route("/integrations/portkey/v1/aidr/scan", post(guardrail))
}

/// Unlike LiteLLM, Portkey's BYOG webhook payload has no dedicated per-call
/// id field (verified against its own docs — `request`, `response`,
/// `provider`, `requestType`, `metadata`, `eventType` are the whole
/// payload). The only place a caller's own correlation id can travel is
/// inside their own `metadata`, under the same `request_id` key the direct
/// `/api/v1/aidr/scan` schema uses. `None` here still reaches
/// `metadata.request_id` (and so `aidr::run_scan`'s validation) unset,
/// same as a direct caller who never sent one.
fn request_id_from_metadata(metadata: &HashMap<String, serde_json::Value>) -> Option<String> {
    metadata
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

async fn guardrail(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<PortkeyHookPayload>,
) -> Result<Json<PortkeyVerdictResponse>, StatusCode> {
    let (text, mode) = match payload.event_type.as_str() {
        "beforeRequestHook" => (payload.request.text, "input"),
        "afterRequestHook" => (payload.response.text, "output"),
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    let session_id = resolve_session_id(&headers)?;
    let request_id = request_id_from_metadata(&payload.metadata);
    let request = AidrScanRequest {
        text,
        messages: Vec::new(),
        metadata: Metadata {
            mode: Some(mode.to_string()),
            request_id,
            provider: Some("portkey".to_string()),
            extra: payload.metadata,
            ..Default::default()
        },
        ..Default::default()
    };
    let outcome = crate::aidr::run_scan(&state, &session_id, request).await?;

    let verdict = !matches!(outcome.decision.verdict, Verdict::Block);
    Ok(Json(PortkeyVerdictResponse { verdict }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_id_from_metadata_reads_the_request_id_key() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "request_id".to_string(),
            serde_json::Value::String("caller-trace-1".to_string()),
        );
        assert_eq!(
            request_id_from_metadata(&metadata),
            Some("caller-trace-1".to_string())
        );
    }

    #[test]
    fn request_id_from_metadata_is_none_when_absent() {
        assert_eq!(request_id_from_metadata(&HashMap::new()), None);
    }

    #[test]
    fn before_request_hook_reads_request_text() {
        let payload: PortkeyHookPayload = serde_json::from_str(
            r#"{"request":{"json":{},"text":"hello","isStreamingRequest":false},
                "response":{"json":{},"text":"","statusCode":null},
                "provider":"openai","requestType":"chatComplete","metadata":{},
                "eventType":"beforeRequestHook"}"#,
        )
        .unwrap();
        assert_eq!(payload.event_type, "beforeRequestHook");
        assert_eq!(payload.request.text, "hello");
    }

    #[test]
    fn after_request_hook_reads_response_text() {
        let payload: PortkeyHookPayload = serde_json::from_str(
            r#"{"request":{"json":{},"text":"","isStreamingRequest":false},
                "response":{"json":{},"text":"here you go","statusCode":200},
                "provider":"openai","requestType":"chatComplete","metadata":{},
                "eventType":"afterRequestHook"}"#,
        )
        .unwrap();
        assert_eq!(payload.event_type, "afterRequestHook");
        assert_eq!(payload.response.text, "here you go");
    }

    #[test]
    fn verdict_response_serializes_as_portkey_expects() {
        let body = serde_json::to_string(&PortkeyVerdictResponse { verdict: true }).unwrap();
        assert_eq!(body, r#"{"verdict":true}"#);
    }
}
