//! End-to-end test of the Portkey "Bring Your Own Guardrail" webhook
//! adapter (`crates/api/src/integrations/portkey.rs`) against the shipped
//! default policy, exercising the router the same way Portkey's real
//! webhook call would hit it.

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
    let router = integrations::portkey::router().with_state(app_state());
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/integrations/portkey/v1/aidr/scan")
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
async fn before_request_hook_with_a_credit_card_blocks() {
    let payload = json!({
        "request": {"json": {}, "text": "my card is 4242 4242 4242 4242", "isStreamingRequest": false},
        "response": {"json": {}, "text": "", "statusCode": null},
        "provider": "openai",
        "requestType": "chatComplete",
        "metadata": {},
        "eventType": "beforeRequestHook",
    });
    let (status, body) = call(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!(false));
}

#[tokio::test]
async fn before_request_hook_with_benign_text_passes() {
    let payload = json!({
        "request": {"json": {}, "text": "what's a good recipe for banana bread?", "isStreamingRequest": false},
        "response": {"json": {}, "text": "", "statusCode": null},
        "provider": "openai",
        "requestType": "chatComplete",
        "metadata": {},
        "eventType": "beforeRequestHook",
    });
    let (status, body) = call(payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!(true));
}

#[tokio::test]
async fn after_request_hook_checks_response_text_not_request_text() {
    let payload = json!({
        "request": {"json": {}, "text": "my card is 4242 4242 4242 4242", "isStreamingRequest": false},
        "response": {"json": {}, "text": "here's a banana bread recipe", "statusCode": 200},
        "provider": "openai",
        "requestType": "chatComplete",
        "metadata": {},
        "eventType": "afterRequestHook",
    });
    let (status, body) = call(payload).await;
    assert_eq!(status, StatusCode::OK);
    // The card number is in `request.text`, not `response.text` — an
    // afterRequestHook call must not accidentally check the wrong field.
    assert_eq!(body["verdict"], json!(true));
}

#[tokio::test]
async fn unknown_event_type_is_rejected() {
    let payload = json!({
        "request": {"json": {}, "text": "hello", "isStreamingRequest": false},
        "response": {"json": {}, "text": "", "statusCode": null},
        "provider": "openai",
        "requestType": "chatComplete",
        "metadata": {},
        "eventType": "somethingElse",
    });
    let (status, _) = call(payload).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
