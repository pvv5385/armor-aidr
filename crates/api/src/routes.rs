//! HTTP surface: `POST /api/v1/aidr/scan` (see `crate::aidr` for the
//! request schema), plus `/healthz`/`/readyz`. Every gateway adapter under
//! `/integrations/*/v1/aidr/scan` (`crate::integrations`) normalizes its
//! own vendor payload into `aidr::AidrScanRequest` and runs through the
//! same `aidr::run_scan` this handler calls — one engine entrypoint, one
//! request schema, for both the direct API and every integration.
//!
//! The engine runs the same flat policy regardless of `metadata.mode` —
//! per-mode policy selection (and the gateway capability matrix / `ask`
//! degradation policy) is future work, not yet needed with one policy and
//! no tenants.

use std::time::Duration;

use axum::{
    extract::State,
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    middleware::{from_fn, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use tower_http::{
    cors::CorsLayer,
    trace::{DefaultMakeSpan, TraceLayer},
};
use tracing::Level;

use crate::{
    aidr::AidrScanRequest,
    config::{ArmorMode, AuthMode, RateLimitMode, Settings},
    control_plane, integrations, middleware,
    state::AppState,
};

const UI_INDEX_HTML: &str = include_str!("../ui/index.html");
const UI_APP_JS: &str = include_str!("../ui/app.js");
const UI_STYLE_CSS: &str = include_str!("../ui/style.css");

/// Ties every `/api/v1/aidr/scan` call for one conversation together.
/// Minted by the caller (the gateway/app owns the conversation boundary,
/// not armor), opaque, replayed on every call in that conversation. Always
/// echoed back on the response, self-minted when absent, so every request
/// belongs to *some* session.
const SESSION_ID_HEADER: &str = "x-armor-session-id";
/// Cheap abuse guard against the header being used as an arbitrary side
/// channel — not a structural/UUID check, since callers may reuse their own
/// conversation/thread ID format.
const MAX_SESSION_ID_LEN: usize = 128;

/// `/healthz`/`/readyz` are never auth- or rate-limit-gated (health probes
/// shouldn't need a key); `/api/v1/aidr/scan` and every
/// `/integrations/*/v1/aidr/scan` adapter get whichever of the two
/// `Settings` has turned on — both off (passthrough) by default.
///
/// Layer order (outermost first, i.e. last `.layer()` call wins): security
/// headers and CORS wrap *everything*, including health checks, since both
/// are response-shaping rather than access-control; rate-limit sits outside
/// auth so an over-budget caller is rejected before spending an auth check;
/// request tracing/metrics wrap the whole stack so every response — allowed
/// or rejected — is observed.
pub fn router(state: AppState, settings: &Settings) -> Router {
    let mut app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz));

    // `ArmorMode` (`config.rs`) gates which surface this process exposes —
    // "Single Binary, Multiple Modes": `standalone` mounts both, `edge`
    // drops the UI (thin data-plane deployments), `control_plane` drops the
    // data plane (it has no policy/profiles/sync wiring to serve requests
    // against — see `main.rs`).
    if settings.mode != ArmorMode::ControlPlane {
        let mut protected = Router::new()
            .route("/api/v1/aidr/scan", post(scan))
            .merge(integrations::portkey::router())
            .merge(integrations::litellm::router());

        if settings.auth_mode == AuthMode::ApiKey {
            protected = protected.layer(from_fn_with_state(
                state.clone(),
                middleware::auth::require_api_key,
            ));
        }

        if settings.rate_limit_mode != RateLimitMode::None {
            // Added last so it's the outermost layer, i.e. checked first —
            // reject over-budget callers before spending an auth check on them.
            protected = protected.layer(from_fn_with_state(
                state.clone(),
                middleware::rate_limit::enforce,
            ));
        }

        // CORS is scoped to just the data-plane API: it decides which
        // browser origins may *read* a response, and the only routes any
        // deployment would opt a cross-origin browser-hosted SDK/gateway
        // into are these — never the control plane below, which has no
        // notion of caller-supplied origins being trustworthy. Added last
        // (outermost) so preflight `OPTIONS` requests are answered by
        // tower_http's CorsLayer before they'd otherwise hit auth/rate-limit.
        if let Some(cors) = cors_layer(settings) {
            protected = protected.layer(cors);
        }

        app = app.merge(protected);
    }

    if settings.mode != ArmorMode::Edge {
        let mut control_plane_app = Router::new();

        // The control-plane CRUD API (profiles, applications, logs, stats,
        // models, hardware — `control_plane.rs`) lives at `/api/v1`
        // regardless of `settings.ui_enabled`: the browser UI is a
        // quick-testing convenience layer on top of it, not the only way
        // to reach it, so turning the UI off must not also take down
        // headless/automated management. Only needs `DATABASE_URL`
        // configured (`main.rs`'s DB wiring, `AppState.db`).
        if state.db.is_some() {
            control_plane_app = control_plane_app.nest("/api/v1", control_plane::router());
        }

        if settings.ui_enabled {
            control_plane_app = if state.db.is_some() {
                // Serve the real management UI (plain HTML/CSS/vanilla JS,
                // embedded via `include_str!`, no build step) — it talks to
                // the same `/api/v1` control-plane routes mounted above.
                control_plane_app
                    .route("/ui", get(ui_index))
                    .route("/ui/", get(ui_index))
                    .route("/ui/dashboard", get(ui_index))
                    .route("/ui/profiles", get(ui_index))
                    .route("/ui/applications", get(ui_index))
                    .route("/ui/inference", get(ui_index))
                    .route("/ui/test", get(ui_index))
                    .route("/ui/logs", get(ui_index))
                    .route("/ui/app.js", get(ui_app_js))
                    .route("/ui/style.css", get(ui_style_css))
            } else {
                // No `DATABASE_URL` — there's nothing for the UI to manage,
                // so `/ui` stays a stub. Mounted regardless of `db` so a
                // deployment can already point at `/ui` and get a clear,
                // typed "not yet" instead of a bare 404.
                control_plane_app
                    .route("/ui", get(ui_stub))
                    .route("/ui/*rest", get(ui_stub))
            };
        }

        // The control plane can rewrite every detector/policy the data
        // plane enforces, so it gets exactly the same auth/rate-limit
        // posture as `protected` above, not a weaker one: whichever of
        // `ARMOR_AUTH_MODE`/`ARMOR_RATE_LIMIT_MODE` is turned on must
        // actually cover `/ui` and its `/api/v1` CRUD routes, not just
        // `/api/v1/aidr/scan`.
        if settings.auth_mode == AuthMode::ApiKey {
            control_plane_app = control_plane_app.layer(from_fn_with_state(
                state.clone(),
                middleware::auth::require_api_key,
            ));
        }

        if settings.rate_limit_mode != RateLimitMode::None {
            control_plane_app = control_plane_app.layer(from_fn_with_state(
                state.clone(),
                middleware::rate_limit::enforce,
            ));
        }

        app = app.merge(control_plane_app);
    }

    let environment = settings.environment;

    app.with_state(state)
        .layer(from_fn(middleware::otel_metrics::record))
        // tower_http's request span defaults to DEBUG — under the default
        // `info` filter (`otel.rs`, no `RUST_LOG` set) that means the span
        // is never even created, so nothing reaches the traces pipeline.
        // INFO here matches the filter default without requiring every
        // deployment to turn on debug logging just to get request spans.
        .layer(TraceLayer::new_for_http().make_span_with(DefaultMakeSpan::new().level(Level::INFO)))
        .layer(axum::extract::DefaultBodyLimit::max(
            settings.max_body_bytes,
        ))
        .layer(from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| async move {
                middleware::security_headers::apply(environment, req, next).await
            },
        ))
}

/// `None` when `ARMOR_ALLOWED_ORIGINS` is empty — no CORS headers at all,
/// same as today (browsers can't read cross-origin responses without them
/// regardless; this just makes cross-origin browser access an explicit
/// opt-in rather than a silent 404 on the preflight). Only ever layered
/// onto the data-plane `protected` router — see `router()`.
fn cors_layer(settings: &Settings) -> Option<CorsLayer> {
    if settings.cors_allowed_origins.is_empty() {
        return None;
    }

    let origins: Vec<HeaderValue> = settings
        .cors_allowed_origins
        .iter()
        .filter_map(|o| HeaderValue::from_str(o).ok())
        .collect();

    Some(
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([
                header::CONTENT_TYPE,
                header::AUTHORIZATION,
                HeaderName::from_static("x-api-key"),
            ])
            .max_age(Duration::from_secs(600)),
    )
}

async fn healthz() -> StatusCode {
    StatusCode::OK
}

async fn readyz() -> StatusCode {
    StatusCode::OK
}

/// Placeholder for every `/ui/*` route when `DATABASE_URL` isn't set — the
/// management UI and its control-plane API (`control_plane.rs`, mounted at
/// `/api/v1`) need Postgres to have anything to manage. Mounted whenever
/// `ArmorMode` is `standalone` or `control_plane` and `settings.ui_enabled`
/// is true (`config.rs`), so a deployment can already point at `/ui` and
/// get a clear, typed "not yet" instead of a bare 404.
async fn ui_stub() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "control_plane_db_not_configured",
            "detail": "The management UI needs a Postgres-backed control plane to manage. Set DATABASE_URL (see README.md's \"Profiles & applications\" section) and restart to enable /ui.",
        })),
    )
}

async fn ui_index() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        UI_INDEX_HTML,
    )
}

async fn ui_app_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        UI_APP_JS,
    )
}

async fn ui_style_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        UI_STYLE_CSS,
    )
}

/// `armor-core` is synchronous by design, so the actual engine run happens
/// on a blocking-pool thread inside `aidr::run_scan` — this handler's job
/// is just to resolve the session id and bridge into and out of that
/// shared path.
async fn scan(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<AidrScanRequest>,
) -> Result<Response, StatusCode> {
    let session_id = resolve_session_id(&headers)?;
    let outcome = crate::aidr::run_scan(&state, &session_id, request).await?;

    let mut response = Json(crate::aidr::ScanResponse::from(&outcome)).into_response();
    response.headers_mut().insert(
        HeaderName::from_static(SESSION_ID_HEADER),
        // Infallible: `session_id` is either the caller's own header value
        // (already validated as a legal header value below) or a
        // self-minted UUID.
        HeaderValue::from_str(&session_id).expect("session_id validated as a legal header value"),
    );
    Ok(response)
}

/// Reads `X-Armor-Session-Id`, minting a UUID v4 when absent — see the
/// module-level doc comment above `SESSION_ID_HEADER`. Rejects (400) a
/// present-but-empty value or one past `MAX_SESSION_ID_LEN`, and a value
/// that isn't legal header content (not visible ASCII) — a cheap guard
/// against the header being used as a side channel, not a UUID/structural
/// check.
pub(crate) fn resolve_session_id(headers: &HeaderMap) -> Result<String, StatusCode> {
    let Some(value) = headers.get(SESSION_ID_HEADER) else {
        return Ok(uuid::Uuid::new_v4().to_string());
    };
    let value = value.to_str().map_err(|_| StatusCode::BAD_REQUEST)?;
    if value.is_empty() || value.len() > MAX_SESSION_ID_LEN {
        return Err(StatusCode::BAD_REQUEST);
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod session_id_tests {
    use super::*;

    #[test]
    fn absent_header_mints_a_fresh_uuid() {
        let id = resolve_session_id(&HeaderMap::new()).unwrap();
        assert!(uuid::Uuid::parse_str(&id).is_ok());
    }

    #[test]
    fn two_absent_header_calls_mint_different_ids() {
        let a = resolve_session_id(&HeaderMap::new()).unwrap();
        let b = resolve_session_id(&HeaderMap::new()).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn present_header_is_echoed_back_unchanged() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(SESSION_ID_HEADER),
            HeaderValue::from_static("caller-supplied-thread-42"),
        );
        assert_eq!(
            resolve_session_id(&headers).unwrap(),
            "caller-supplied-thread-42"
        );
    }

    #[test]
    fn empty_header_is_rejected() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(SESSION_ID_HEADER),
            HeaderValue::from_static(""),
        );
        assert_eq!(resolve_session_id(&headers), Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn oversized_header_is_rejected() {
        let mut headers = HeaderMap::new();
        let oversized = "a".repeat(MAX_SESSION_ID_LEN + 1);
        headers.insert(
            HeaderName::from_static(SESSION_ID_HEADER),
            HeaderValue::from_str(&oversized).unwrap(),
        );
        assert_eq!(resolve_session_id(&headers), Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn header_at_the_length_limit_is_accepted() {
        let mut headers = HeaderMap::new();
        let max_len = "a".repeat(MAX_SESSION_ID_LEN);
        headers.insert(
            HeaderName::from_static(SESSION_ID_HEADER),
            HeaderValue::from_str(&max_len).unwrap(),
        );
        assert_eq!(resolve_session_id(&headers).unwrap(), max_len);
    }

    #[test]
    fn non_visible_ascii_header_is_rejected() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(SESSION_ID_HEADER),
            HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
        );
        assert_eq!(resolve_session_id(&headers), Err(StatusCode::BAD_REQUEST));
    }
}
