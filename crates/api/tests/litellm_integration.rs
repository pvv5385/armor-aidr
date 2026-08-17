//! End-to-end test of the LiteLLM adapter (`crates/api/src/integrations/litellm.rs`)
//! against the shipped default policy — exercises the real router, same
//! shape as `portkey_integration.rs`/`aidr_scan_integration.rs`.

use std::sync::Arc;

use armor_api::integrations;
use armor_api::state::AppState;
use armor_api::{audit::DiscardAuditSink, heartbeat::Heartbeat, telemetry::TelemetryEmitter};
use armor_core::policy::loader;
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

fn app_state() -> AppState {
    let yaml = include_str!("../../../config/policies.yaml");
    let policy = loader::load(yaml).expect("shipped default policy must load");
    AppState {
        profiles: armor_api::sync::LiveResolver::new(armor_api::profiles::ProfileResolver::single(
            Arc::new(policy),
        )),
        api_keys: None,
        rate_limiter: None,
        telemetry: Arc::new(TelemetryEmitter::new(false, String::new(), String::new())),
        audit_sink: Arc::new(DiscardAuditSink),
        heartbeat: Arc::new(Heartbeat::new(false, String::new(), String::new(), 0)),
        db: None,
        custom_rules_dir: Arc::from(""),
        session_ttl_seconds: None,
        vault: None,
        inference: None,
        inference_budget_ms: 250,
        inference_url: None,
        inference_auth_token: None,
        inference_token_file: "".into(),
    }
}

async fn call(payload: Value) -> (StatusCode, Value) {
    let router = integrations::litellm::router().with_state(app_state());
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/integrations/litellm/v1/aidr/scan")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    (status, body)
}

#[tokio::test]
async fn pre_call_messages_with_a_credit_card_blocks() {
    let payload = json!({
        "mode": "input",
        "messages": [{"role": "user", "content": "my card is 4242 4242 4242 4242"}],
        "metadata": {"application_id": "travel-assistant"},
        "litellm_session_id": "thread-42"
    });
    let (status, body) = call(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("BLOCK"));
}

#[tokio::test]
async fn post_call_synthesized_assistant_message_is_scanned() {
    let payload = json!({
        "mode": "output",
        "messages": [{"role": "assistant", "content": "here's a banana bread recipe"}]
    });
    let (status, body) = call(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("ALLOW"));
}

#[tokio::test]
async fn litellm_call_id_becomes_request_id() {
    let payload = json!({
        "messages": [{"role": "user", "content": "benign text"}],
        "litellm_call_id": "9f3a2b1c-litellm"
    });
    let (status, body) = call(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["request_id"], json!("9f3a2b1c-litellm"));
}

#[tokio::test]
async fn explicit_metadata_request_id_wins_over_litellm_call_id() {
    let payload = json!({
        "messages": [{"role": "user", "content": "benign text"}],
        "metadata": {"request_id": "caller-set-this"},
        "litellm_call_id": "9f3a2b1c-litellm"
    });
    let (status, body) = call(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["request_id"], json!("caller-set-this"));
}
