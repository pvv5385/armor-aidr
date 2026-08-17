//! End-to-end test that a policy asking for redaction actually produces
//! `Verdict::Redact` *and*, when a vault is configured, placeholders that
//! survive the request and can be reversed — reversible anonymization,
//! reachable through a policy for the first time.
//!
//! Three properties, and each is a thing that was previously untrue:
//!
//! 1. `on_fail: redact` composes to `REDACT` rather than being
//!    indistinguishable from a monitor-mode pass.
//! 2. The same value gets the same placeholder in a later request in the
//!    same session — the stability property that makes a placeholder usable
//!    as a reference across a conversation.
//! 3. `on_fail: deny` still discards. A vault sitting in `AppState` is not
//!    permission to store a secret nobody asked to keep.
//!
//! Requires `ARMOR_TEST_DATABASE_URL` (same variable `crates/storage`'s
//! tests use); skips with a notice when it isn't set.

use std::sync::Arc;

use armor_api::state::AppState;
use armor_api::{
    audit::DiscardAuditSink, config::Settings, heartbeat::Heartbeat, routes,
    telemetry::TelemetryEmitter,
};
use armor_core::policy::schema::PolicyConfig;
use armor_storage::policy_store::PgPolicyStore;
use armor_storage::vault::{EnvKeyProvider, Vault};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

/// `pii` set to redact rather than deny — the configuration that makes
/// `Verdict::Redact` reachable at all. Only `email` is on, so the only
/// thing these tests can trip is the rule they are about.
const REDACT_POLICY: &str = r#"
id: vault-redaction-test
checks:
  - category: pii
    enabled: true
    mode: block
    on_fail: redact
    options:
      email: true
"#;

/// The same check, left at the shipped `on_fail: deny`. Same detector, same
/// hit, same masked output — but nothing recoverable is stored.
const DENY_POLICY: &str = r#"
id: vault-deny-test
checks:
  - category: pii
    enabled: true
    mode: block
    on_fail: deny
    options:
      email: true
"#;

async fn db() -> Option<Arc<PgPolicyStore>> {
    let url = std::env::var("ARMOR_TEST_DATABASE_URL")
        .ok()
        .filter(|u| !u.trim().is_empty());
    let Some(url) = url else {
        eprintln!("SKIPPING vault-redaction integration test: ARMOR_TEST_DATABASE_URL is not set.");
        return None;
    };
    Some(Arc::new(
        PgPolicyStore::connect(&url)
            .await
            .expect("connecting to ARMOR_TEST_DATABASE_URL"),
    ))
}

/// A key shared by the router under test and the out-of-band [`Vault`] the
/// assertions read through — they have to agree, or the blind index and the
/// ciphertext are both unreadable.
fn key_base64() -> String {
    EnvKeyProvider::generate_base64()
}

fn vault_with(db: &Arc<PgPolicyStore>, key: &str) -> Arc<Vault> {
    Arc::new(Vault::new(
        db.pool().clone(),
        Box::new(EnvKeyProvider::from_base64(key).expect("test key parses")),
    ))
}

fn app(policy_yaml: &str, db: Arc<PgPolicyStore>, vault: Option<Arc<Vault>>) -> axum::Router {
    let policy: PolicyConfig = serde_yaml::from_str(policy_yaml).expect("test policy parses");
    let state = AppState {
        profiles: armor_api::sync::LiveResolver::new(armor_api::profiles::ProfileResolver::single(
            Arc::new(policy),
        )),
        api_keys: None,
        rate_limiter: None,
        telemetry: Arc::new(TelemetryEmitter::new(false, String::new(), String::new())),
        audit_sink: Arc::new(DiscardAuditSink),
        heartbeat: Arc::new(Heartbeat::new(false, String::new(), String::new(), 0)),
        db: Some(db),
        custom_rules_dir: Arc::from(""),
        session_ttl_seconds: None,
        vault,
        inference: None,
        inference_budget_ms: 250,
        inference_url: None,
        inference_auth_token: None,
        inference_token_file: "".into(),
    };
    routes::router(state, &Settings::from_env())
}

async fn scan(app: axum::Router, session_id: &str, text: &str) -> Value {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/aidr/scan")
                .header("content-type", "application/json")
                .header("x-armor-session-id", session_id)
                .body(Body::from(json!({ "text": text }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn field<'a>(body: &'a Value, key: &str) -> &'a str {
    body.get(key).and_then(Value::as_str).unwrap_or_default()
}

fn session() -> String {
    format!("vault-{}", uuid::Uuid::new_v4())
}

#[tokio::test]
async fn a_redact_policy_masks_the_value_and_returns_the_redact_verdict() {
    let Some(db) = db().await else {
        return;
    };
    let key = key_base64();
    let session = session();

    let body = scan(
        app(REDACT_POLICY, db.clone(), Some(vault_with(&db, &key))),
        &session,
        "email me at ada@example.com please",
    )
    .await;

    assert_eq!(field(&body, "verdict"), "REDACT");
    assert_eq!(
        field(&body, "redacted_text"),
        "email me at <PII:EMAIL_ADDRESS:1> please"
    );
    assert_eq!(body["checks"][0]["action_taken"], json!("redacted"));
}

#[tokio::test]
async fn a_vaulted_placeholder_reverses_to_the_original_value() {
    let Some(db) = db().await else {
        return;
    };
    let key = key_base64();
    let session = session();

    scan(
        app(REDACT_POLICY, db.clone(), Some(vault_with(&db, &key))),
        &session,
        "email me at ada@example.com please",
    )
    .await;

    // Out of band, the way a trusted downstream consumer would: there is
    // deliberately no HTTP surface for this (no RBAC/authorization model
    // exists yet to safely expose it over HTTP), so callers link the crate.
    let vault = vault_with(&db, &key);
    assert_eq!(
        vault
            .deanonymize(&session, "<PII:EMAIL_ADDRESS:1>")
            .await
            .unwrap(),
        Some("ada@example.com".to_string())
    );
}

#[tokio::test]
async fn the_same_value_keeps_its_placeholder_across_requests_in_a_session() {
    let Some(db) = db().await else {
        return;
    };
    let key = key_base64();
    let session = session();

    // Turn one mentions Ada. Turn two mentions Grace first, then Ada again —
    // so per-request numbering would call Ada `:1` in turn one and `:2` in
    // turn two, and the placeholder would be useless as a reference.
    let first = scan(
        app(REDACT_POLICY, db.clone(), Some(vault_with(&db, &key))),
        &session,
        "write to ada@example.com",
    )
    .await;
    assert_eq!(
        field(&first, "redacted_text"),
        "write to <PII:EMAIL_ADDRESS:1>"
    );

    let second = scan(
        app(REDACT_POLICY, db.clone(), Some(vault_with(&db, &key))),
        &session,
        "cc grace@example.com and ada@example.com",
    )
    .await;
    assert_eq!(
        field(&second, "redacted_text"),
        "cc <PII:EMAIL_ADDRESS:2> and <PII:EMAIL_ADDRESS:1>"
    );
}

#[tokio::test]
async fn a_discard_only_request_does_not_reuse_a_placeholder_the_session_s_vault_already_owns() {
    let Some(db) = db().await else {
        return;
    };
    let key = key_base64();
    let session = session();

    // Turn one: a `redact` policy vaults Ada's email as
    // `<PII:EMAIL_ADDRESS:1>`.
    let first = scan(
        app(REDACT_POLICY, db.clone(), Some(vault_with(&db, &key))),
        &session,
        "write to ada@example.com",
    )
    .await;
    assert_eq!(
        field(&first, "redacted_text"),
        "write to <PII:EMAIL_ADDRESS:1>"
    );

    // Turn two: a `deny` policy (discard-only, never touches the vault)
    // masks a *different* person's email in the same session. Numbered from
    // scratch with no knowledge of turn one, it would also want
    // `<PII:EMAIL_ADDRESS:1>` — which must not happen, since that string
    // already means Ada's address to anything that looks it up in the vault.
    let second = scan(
        app(DENY_POLICY, db.clone(), Some(vault_with(&db, &key))),
        &session,
        "and grace@example.com too",
    )
    .await;
    assert_ne!(
        field(&second, "redacted_text"),
        "and <PII:EMAIL_ADDRESS:1> too",
        "reused a placeholder the vault already owns for a different person's PII"
    );

    // The vault-owned placeholder still resolves to Ada, not Grace — turn
    // two's masking never touched the vault at all (`deny` discards).
    let vault = vault_with(&db, &key);
    assert_eq!(
        vault
            .deanonymize(&session, "<PII:EMAIL_ADDRESS:1>")
            .await
            .unwrap(),
        Some("ada@example.com".to_string())
    );
}

#[tokio::test]
async fn placeholders_do_not_leak_across_sessions() {
    let Some(db) = db().await else {
        return;
    };
    let key = key_base64();
    let (a, b) = (session(), session());

    for s in [&a, &b] {
        scan(
            app(REDACT_POLICY, db.clone(), Some(vault_with(&db, &key))),
            s,
            "write to ada@example.com",
        )
        .await;
    }

    // Session `b` never saw Grace, so `b`'s numbering is its own.
    let vault = vault_with(&db, &key);
    assert_eq!(
        vault
            .deanonymize(&a, "<PII:EMAIL_ADDRESS:1>")
            .await
            .unwrap(),
        Some("ada@example.com".to_string())
    );
    assert!(vault
        .deanonymize(&b, "<PII:EMAIL_ADDRESS:2>")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn a_deny_policy_masks_but_stores_nothing_even_with_a_vault_configured() {
    let Some(db) = db().await else {
        return;
    };
    let key = key_base64();
    let session = session();

    let body = scan(
        app(DENY_POLICY, db.clone(), Some(vault_with(&db, &key))),
        &session,
        "email me at ada@example.com please",
    )
    .await;

    // Identical masked output, different verdict — and, the point of this
    // test, nothing recoverable left behind.
    assert_eq!(field(&body, "verdict"), "BLOCK");
    assert_eq!(
        field(&body, "redacted_text"),
        "email me at <PII:EMAIL_ADDRESS:1> please"
    );
    assert!(vault_with(&db, &key)
        .list_placeholders(&session)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn without_a_vault_a_redact_policy_still_redacts_it_is_just_not_reversible() {
    let Some(db) = db().await else {
        return;
    };
    let session = session();

    // The default deployment: `ARMOR_VAULT_KEY` unset. Byte-identical
    // `redacted_text`, same verdict, no database rows.
    let body = scan(
        app(REDACT_POLICY, db.clone(), None),
        &session,
        "email me at ada@example.com please",
    )
    .await;

    assert_eq!(field(&body, "verdict"), "REDACT");
    assert_eq!(
        field(&body, "redacted_text"),
        "email me at <PII:EMAIL_ADDRESS:1> please"
    );
    assert!(vault_with(&db, &key_base64())
        .list_placeholders(&session)
        .await
        .unwrap()
        .is_empty());
}
