//! `/readyz` reflects dependency health; `/healthz` does not.
//!
//! The property under test is the one that makes the two probes worth having
//! separately: when the control-plane database is gone, this replica should
//! stop receiving traffic (`/readyz` → 503) but should NOT be restarted
//! (`/healthz` → 200), because restarting `armor-api` does not fix Postgres.
//!
//! The dead-dependency case uses a real pool that is then closed, rather than
//! a bad URL — `PgPolicyStore::connect` runs migrations and fails outright on
//! an unreachable database, so there is no way to hold a store whose pool was
//! never live. Closing a working pool reproduces the situation that actually
//! matters in production: the process came up fine and lost its database
//! later.
//!
//! Requires `ARMOR_TEST_DATABASE_URL` for the database-backed cases (same
//! variable `crates/storage`'s tests use); they skip with a notice when it is
//! not set. The no-database case needs no service and always runs.

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
use serde_json::Value;
use tower::ServiceExt;

/// Everything off — readiness has nothing to do with which detectors run, and
/// an empty check list keeps this test from depending on the shipped policy.
const POLICY_YAML: &str = r#"
id: readiness-test
checks: []
"#;

async fn db() -> Option<Arc<PgPolicyStore>> {
    let url = std::env::var("ARMOR_TEST_DATABASE_URL")
        .ok()
        .filter(|u| !u.trim().is_empty());
    let Some(url) = url else {
        eprintln!("SKIPPING readiness integration test: ARMOR_TEST_DATABASE_URL is not set.");
        return None;
    };
    Some(Arc::new(
        PgPolicyStore::connect(&url)
            .await
            .expect("connecting to ARMOR_TEST_DATABASE_URL"),
    ))
}

fn app(db: Option<Arc<PgPolicyStore>>) -> axum::Router {
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

async fn get(router: axum::Router, path: &str) -> (StatusCode, Value) {
    let response = router
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    (status, body)
}

/// No `DATABASE_URL` is a supported deployment (rules-only / `edge`), not a
/// degraded one — there is no dependency to be unready for.
#[tokio::test]
async fn ready_when_no_database_is_configured() {
    let (status, body) = get(app(None), "/readyz").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");
}

#[tokio::test]
async fn ready_when_the_database_answers() {
    let Some(db) = db().await else { return };
    let (status, body) = get(app(Some(db)), "/readyz").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");
}

/// The regression this endpoint exists to prevent: before it did a real
/// check, this returned 200 and a load balancer kept sending traffic to a
/// replica that could not serve a single control-plane request.
#[tokio::test]
async fn not_ready_once_the_database_is_gone() {
    let Some(db) = db().await else { return };

    // Ready first, so a 503 below is the closed pool and not a broken fixture.
    let (status, _) = get(app(Some(db.clone())), "/readyz").await;
    assert_eq!(status, StatusCode::OK);

    db.pool().close().await;

    let (status, body) = get(app(Some(db)), "/readyz").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["status"], "not_ready");
    assert_eq!(body["dependency"], "database");
}

/// Liveness must stay unconditional: a dead database is not a reason to
/// restart this process, and a probe that conflates the two turns a
/// dependency outage into a crash loop.
#[tokio::test]
async fn healthz_stays_ok_when_the_database_is_gone() {
    let Some(db) = db().await else { return };
    db.pool().close().await;

    let (status, _) = get(app(Some(db)), "/healthz").await;
    assert_eq!(status, StatusCode::OK);
}

/// The failure body must not leak the database host, user, or driver error —
/// `/readyz` is never auth-gated (`routes::router`), so anyone who can reach
/// the port can read this.
#[tokio::test]
async fn the_not_ready_body_does_not_leak_connection_details() {
    let Some(db) = db().await else { return };
    db.pool().close().await;

    let (_, body) = get(app(Some(db)), "/readyz").await;
    let rendered = body.to_string();
    assert!(
        !rendered.contains("postgres") && !rendered.contains("://"),
        "readiness failure body leaked connection details: {rendered}"
    );
    // Exactly the two keys, so a future field cannot smuggle detail in.
    let object = body.as_object().expect("body is a JSON object");
    assert_eq!(object.len(), 2, "unexpected fields in body: {rendered}");
}
