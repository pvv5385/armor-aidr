//! End-to-end test that durable session counters actually change a verdict
//! across *separate* `AppState`s — the property the whole durable session
//! store exists for.
//!
//! Two `AppState`s sharing one database stand in for two replicas behind a
//! load balancer. With in-process counters only, each would count its own
//! requests and a caller could spend the budget twice; with the shared
//! store, the second replica sees the first one's traffic.
//!
//! Requires `ARMOR_TEST_DATABASE_URL` (same variable
//! `crates/storage`'s tests use); skips with a notice when it isn't set.

use std::sync::Arc;

use armor_api::state::AppState;
use armor_api::{
    audit::DiscardAuditSink, config::Settings, heartbeat::Heartbeat, routes,
    telemetry::TelemetryEmitter,
};
use armor_core::policy::schema::PolicyConfig;
use armor_storage::policy_store::PgPolicyStore;
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

/// `abuse` enabled with a 2-request budget, everything else off — so the
/// only thing this test can trip is the rate limit it is about.
const POLICY_YAML: &str = r#"
id: session-state-test
checks:
  - category: abuse
    enabled: true
    mode: block
    on_fail: deny
    options:
      max_requests_per_window: 2
      window_seconds: 3600
"#;

async fn db() -> Option<Arc<PgPolicyStore>> {
    let url = std::env::var("ARMOR_TEST_DATABASE_URL")
        .ok()
        .filter(|u| !u.trim().is_empty());
    let Some(url) = url else {
        eprintln!("SKIPPING session-state integration test: ARMOR_TEST_DATABASE_URL is not set.");
        return None;
    };
    Some(Arc::new(
        PgPolicyStore::connect(&url)
            .await
            .expect("connecting to ARMOR_TEST_DATABASE_URL"),
    ))
}

/// One "replica": its own `AppState` and its own in-process detector
/// counters, sharing only the database.
fn replica(db: Option<Arc<PgPolicyStore>>) -> axum::Router {
    let policy: PolicyConfig = serde_yaml::from_str(POLICY_YAML).expect("test policy parses");
    let state = AppState {
        profiles: armor_api::sync::LiveResolver::new(armor_api::profiles::ProfileResolver::single(
            Arc::new(policy),
        )),
        api_keys: None,
        rate_limiter: None,
        telemetry: Arc::new(TelemetryEmitter::new(false, String::new(), String::new())),
        audit_sink: Arc::new(DiscardAuditSink),
        heartbeat: Arc::new(Heartbeat::new(false, String::new(), String::new(), 0)),
        db,
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

async fn scan(app: axum::Router, session_id: &str) -> Value {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/aidr/scan")
                .header("content-type", "application/json")
                .header("x-armor-session-id", session_id)
                .body(Body::from(json!({"text": "hello"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn verdict(body: &Value) -> String {
    body.get("verdict")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_uppercase()
}

#[tokio::test]
async fn the_rate_budget_is_shared_across_replicas() {
    let Some(db) = db().await else {
        return;
    };
    let session = format!("shared-{}", uuid::Uuid::new_v4());

    // Requests 1 and 2 land on replica A and are within budget.
    assert_eq!(
        verdict(&scan(replica(Some(db.clone())), &session).await),
        "ALLOW"
    );
    assert_eq!(
        verdict(&scan(replica(Some(db.clone())), &session).await),
        "ALLOW"
    );

    // Request 3 lands on a *different* replica, whose in-process counter
    // for this session is zero. Without the shared store it would allow;
    // with it, the durable count is 3 and the budget is spent.
    assert_eq!(
        verdict(&scan(replica(Some(db.clone())), &session).await),
        "BLOCK"
    );
}

#[tokio::test]
async fn distinct_sessions_get_independent_budgets() {
    let Some(db) = db().await else {
        return;
    };
    let (a, b) = (
        format!("a-{}", uuid::Uuid::new_v4()),
        format!("b-{}", uuid::Uuid::new_v4()),
    );

    for _ in 0..3 {
        scan(replica(Some(db.clone())), &a).await;
    }
    // `a` is over budget by now; `b` must be unaffected.
    assert_eq!(verdict(&scan(replica(Some(db.clone())), &a).await), "BLOCK");
    assert_eq!(verdict(&scan(replica(Some(db.clone())), &b).await), "ALLOW");
}

#[tokio::test]
async fn without_a_database_each_replica_counts_only_its_own_traffic() {
    // The no-database behavior, pinned so the fallback path stays intact:
    // three requests, three fresh replicas, no shared store — every one is
    // that replica's first, so nothing is ever over budget.
    let session = format!("local-{}", uuid::Uuid::new_v4());
    for _ in 0..3 {
        assert_eq!(verdict(&scan(replica(None), &session).await), "ALLOW");
    }
}
