//! End-to-end test of the unified `POST /api/v1/aidr/scan` endpoint
//! (`crates/api/src/routes.rs` + `crates/api/src/aidr.rs`) against the
//! shipped default policy — exercises the real router, same as
//! `portkey_integration.rs` does for the Portkey adapter.

use std::sync::Arc;

use armor_api::state::AppState;
use armor_api::{
    audit::DiscardAuditSink, config::Settings, heartbeat::Heartbeat, routes,
    telemetry::TelemetryEmitter,
};
use armor_core::policy::loader;
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

fn app() -> axum::Router {
    let yaml = include_str!("../../../config/policies.yaml");
    let policy = loader::load(yaml).expect("shipped default policy must load");
    let state = AppState {
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
    };
    routes::router(state, &Settings::from_env())
}

async fn call(payload: Value) -> (StatusCode, Value) {
    let response = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/aidr/scan")
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
async fn plain_text_field_is_scanned() {
    let (status, body) = call(json!({
        "text": "my card is 4242 4242 4242 4242",
        "metadata": {"mode": "input", "application_id": "travel-assistant"}
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("BLOCK"));
    assert_eq!(
        body["redacted_text"],
        json!("my card is <PCI:PCI_CREDIT_CARD:1>")
    );
}

#[tokio::test]
async fn benign_text_allows() {
    let (status, body) = call(json!({
        "text": "what's a good recipe for banana bread?",
        "metadata": {"mode": "input"}
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("ALLOW"));
}

#[tokio::test]
async fn messages_array_is_scanned() {
    let (status, body) = call(json!({
        "messages": [
            {"role": "user", "content": "Find me flights to London."}
        ],
        "metadata": {"mode": "input"}
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("ALLOW"));
}

#[tokio::test]
async fn missing_body_fields_default_to_an_empty_allow() {
    let (status, body) = call(json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("ALLOW"));
    assert_eq!(body["redacted_text"], json!(""));
}

#[tokio::test]
async fn scan_id_is_always_present_and_unique_per_request() {
    let (_, first) = call(json!({"text": "benign text"})).await;
    let (_, second) = call(json!({"text": "benign text"})).await;

    let first_id = first["scan_id"].as_str().expect("scan_id must be a string");
    let second_id = second["scan_id"]
        .as_str()
        .expect("scan_id must be a string");
    assert!(uuid::Uuid::parse_str(first_id).is_ok());
    assert_ne!(first_id, second_id);
}

#[tokio::test]
async fn request_id_is_absent_when_not_supplied() {
    let (status, body) = call(json!({"text": "benign text"})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.as_object().unwrap().contains_key("request_id"));
}

#[tokio::test]
async fn request_id_is_echoed_back_when_supplied() {
    let (status, body) = call(json!({
        "text": "benign text",
        "metadata": {"request_id": "litellm-call-9f"}
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["request_id"], json!("litellm-call-9f"));
}

#[tokio::test]
async fn empty_request_id_is_rejected() {
    let (status, _) = call(json!({
        "text": "benign text",
        "metadata": {"request_id": ""}
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn oversized_request_id_is_rejected() {
    let (status, _) = call(json!({
        "text": "benign text",
        "metadata": {"request_id": "x".repeat(129)}
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn non_visible_ascii_request_id_is_rejected() {
    let (status, _) = call(json!({
        "text": "benign text",
        "metadata": {"request_id": "has a space"}
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn request_id_at_the_length_limit_is_accepted() {
    let (status, body) = call(json!({
        "text": "benign text",
        "metadata": {"request_id": "x".repeat(128)}
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["request_id"], json!("x".repeat(128)));
}
