//! End-to-end test that `metadata.application_id` actually changes which
//! profile's checks run (`crates/api/src/profiles.rs`,
//! `crates/api/src/aidr.rs::run_scan`) — two applications mapped to two
//! different profiles get different verdicts for the identical text.

use std::sync::Arc;

use armor_api::config::Settings;
use armor_api::profiles::{self, ProfileResolver};
use armor_api::state::AppState;
use armor_api::sync::LiveResolver;
use armor_api::{
    audit::DiscardAuditSink, heartbeat::Heartbeat, routes, telemetry::TelemetryEmitter,
};
use armor_core::policy::loader;
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

/// `default` runs the shipped policy (PCI enabled, blocks); `no-pci` is a
/// named profile with every check disabled, mapped to `application_id:
/// "lenient-app"` — everything else (including no `application_id` at all)
/// keeps resolving to `default`.
fn resolver_with_two_profiles() -> ProfileResolver {
    let default_yaml = include_str!("../../../config/policies.yaml");
    let default = Arc::new(loader::load(default_yaml).expect("shipped default policy must load"));

    let dir = tempfile::tempdir().unwrap();
    let profiles_dir = dir.path().join("profiles");
    std::fs::create_dir_all(&profiles_dir).unwrap();
    std::fs::write(profiles_dir.join("no-pci.yaml"), "id: no-pci\nchecks: []\n").unwrap();

    let applications_path = dir.path().join("applications.yaml");
    std::fs::write(
        &applications_path,
        "applications:\n  - application_id: lenient-app\n    profile_id: no-pci\n",
    )
    .unwrap();

    profiles::load(
        default,
        &profiles_dir,
        &applications_path,
        &dir.path().join("no-such-custom-rules-dir"),
    )
    .unwrap()
}

fn app() -> axum::Router {
    let state = AppState {
        profiles: LiveResolver::new(resolver_with_two_profiles()),
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
async fn unmapped_application_id_gets_the_default_profile_and_blocks() {
    let (status, body) = call(json!({
        "text": "my card is 4242 4242 4242 4242",
        "metadata": {"application_id": "some-other-app"}
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("BLOCK"));
}

#[tokio::test]
async fn mapped_application_id_gets_its_own_profile_and_allows() {
    let (status, body) = call(json!({
        "text": "my card is 4242 4242 4242 4242",
        "metadata": {"application_id": "lenient-app"}
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("ALLOW"));
    assert_eq!(body["checks"], json!([]));
}

#[tokio::test]
async fn absent_application_id_gets_the_default_profile() {
    let (status, body) = call(json!({"text": "my card is 4242 4242 4242 4242"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["verdict"], json!("BLOCK"));
}
