//! `/api/v1/*` — CRUD for Postgres-backed profiles (with their detector
//! checks) and applications, plus the "simple logging" viewer
//! (`GET /api/v1/logs`). Reachable regardless of whether the browser UI
//! (`ARMOR_UI_ENABLED`) is on — the UI is just one caller of this API.
//! Only mounted when `AppState.db.is_some()` (`routes.rs`); every handler
//! here still defensively checks `require_db` so a future wiring change
//! can't turn a missing DB into a panic.
//!
//! Every mutating handler ends by calling [`reload_and_swap`], which
//! reloads every profile/application row from Postgres, hardens each
//! (`profiles::harden`, same treatment file/sync-sourced profiles get),
//! and atomically swaps the result into `AppState.profiles`
//! (`sync::LiveResolver`) — a write here takes effect on the very next
//! request, no restart, no waiting on a sync poll. This is the same
//! "compile off the async path, then one atomic swap" shape `sync.rs`
//! already uses for control-plane pushes; here a UI write is the trigger
//! instead of a timer.
//!
//! No auth layer on this router — matches this deployment's current
//! default posture (`ARMOR_AUTH_MODE=none`, `ARMOR_RATE_LIMIT_MODE=none`):
//! off until explicitly configured. Anyone who can reach this HTTP surface
//! can rewrite security policy; treat network exposure accordingly.

use std::{path::Path, sync::Arc};

use chrono::{DateTime, Utc};

use axum::{
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use armor_core::engine::scorecard_gate::{self, ScorecardThresholds};
use armor_core::policy::schema::{
    CheckConfig, ExecutionMode, FailMode, NormalizeConfig, PolicyConfig,
};
use armor_storage::{
    audit_events,
    policy_store::{
        ApplicationInput, ApplicationRow, PgPolicyStore, PolicyStoreError, ProfileInput,
        ProfileSummary, StoredProfile,
    },
};

use crate::{profiles, state::AppState};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/detector-categories", get(list_detector_categories))
        .route("/detector-options", get(list_detector_options))
        .route("/profiles", get(list_profiles).post(create_profile))
        .route(
            "/profiles/:id",
            get(get_profile).put(update_profile).delete(delete_profile),
        )
        .route(
            "/applications",
            get(list_applications).post(create_application),
        )
        .route(
            "/applications/:application_id",
            put(update_application).delete(delete_application),
        )
        .route("/logs", get(list_logs))
        .route("/stats", get(get_stats))
        .route("/inference-pins", get(list_pins).put(upsert_pin))
        .route("/inference-pins/:task", delete(delete_pin))
        .route("/models", get(list_models))
        .route("/models/catalog", get(models_catalog))
        .route("/hardware", get(get_hardware))
        .route("/models/install", post(install_model))
        .route("/models/install/:job_id", get(install_status))
}

// ── Errors ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    detail: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, detail: impl std::fmt::Display) -> Self {
        Self {
            status,
            code,
            detail: detail.to_string(),
        }
    }

    fn db_not_configured() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "db_not_configured",
            "DATABASE_URL is not set",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.code, "detail": self.detail })),
        )
            .into_response()
    }
}

impl From<PolicyStoreError> for ApiError {
    fn from(e: PolicyStoreError) -> Self {
        match e {
            PolicyStoreError::ProfileNotFound(id) => Self::new(
                StatusCode::NOT_FOUND,
                "profile_not_found",
                format!("profile {id:?} not found"),
            ),
            PolicyStoreError::ProfileInUse(id) => Self::new(
                StatusCode::CONFLICT,
                "profile_in_use",
                format!("profile {id:?} is still assigned to at least one application"),
            ),
            PolicyStoreError::ApplicationNotFound(id) => Self::new(
                StatusCode::NOT_FOUND,
                "application_not_found",
                format!("application {id:?} not found"),
            ),
            PolicyStoreError::ApplicationInvalidProfile(id) => Self::new(
                StatusCode::BAD_REQUEST,
                "unknown_profile_id",
                format!("profile_id {id:?} does not exist"),
            ),
            other => Self::new(StatusCode::INTERNAL_SERVER_ERROR, "database_error", other),
        }
    }
}

fn db_err(e: impl std::fmt::Display) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "database_error", e)
}

fn require_db(state: &AppState) -> Result<&Arc<PgPolicyStore>, ApiError> {
    state.db.as_ref().ok_or_else(ApiError::db_not_configured)
}

/// Reloads every profile/application from Postgres and atomically swaps
/// the rebuilt `ProfileResolver` into `state.profiles` — see this module's
/// doc comment. Called after every mutating handler below.
async fn reload_and_swap(state: &AppState) -> Result<(), ApiError> {
    let db = require_db(state)?;
    let (policies, applications) = db.load_all_policies().await?;
    let pin_rows = armor_storage::inference_pins::list_all(db.pool())
        .await
        .map_err(db_err)?;
    let pins = profiles::pins_from_rows(pin_rows);
    let resolver = profiles::resolver_from_policies(
        policies,
        applications,
        Path::new(&*state.custom_rules_dir),
        &pins,
    )
    .map_err(|e| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "resolver_rebuild_failed",
            e,
        )
    })?;
    state.profiles.swap(resolver);
    Ok(())
}

/// Validate any `custom_regex` check's patterns compile, before the policy
/// is ever committed to the DB — `resolver_from_policies`/`harden` also
/// validate this, but only on the post-commit reload, which is too late: a
/// bad regex would already be a persisted row that reload can never load
/// again (see `create_profile`/`update_profile`).
fn validate_custom_regex_checks(policy: &PolicyConfig) -> Result<(), ApiError> {
    for check in &policy.checks {
        if check.category == "custom_regex" && check.enabled {
            armor_core::detectors::custom_regex::validate(&check.options).map_err(|e| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_policy",
                    format!("invalid custom_regex options: {e}"),
                )
            })?;
        }
    }
    Ok(())
}

/// Validate that every model-backed check's scorecard metrics pass the gate.
/// Returns 422 with per-check details when any check fails.
fn validate_scorecards(policy: &PolicyConfig) -> Result<(), ApiError> {
    let thresholds = ScorecardThresholds::default();
    let mut failures: Vec<String> = Vec::new();

    for check in &policy.checks {
        if let Some(ref metrics) = check.scorecard {
            let verdict = scorecard_gate::evaluate(metrics, &thresholds);
            if !verdict.may_run() {
                failures.push(format!(
                    "{category}: scorecard gate FAIL (metrics below advisory thresholds)",
                    category = check.category,
                ));
            }
        }
    }

    if failures.is_empty() {
        return Ok(());
    }

    Err(ApiError::new(
        StatusCode::UNPROCESSABLE_ENTITY,
        "scorecard_gate_fail",
        format!(
            "scorecard gate rejected {} check(es): {}",
            failures.len(),
            failures.join("; "),
        ),
    ))
}

// ── Detector categories ─────────────────────────────────────────────────

async fn list_detector_categories() -> Json<&'static [&'static str]> {
    Json(armor_core::detectors::categories())
}

/// Per-category `options` schema, so the profile editor can render friendly
/// checkboxes/number inputs instead of a raw JSON blob. Owned by
/// `armor-core` (`detectors::option_schema`) — the same source the
/// detectors themselves read, so the editor can't drift from behavior.
#[derive(Debug, Serialize)]
struct DetectorOptionsResponse {
    category: &'static str,
    options: Vec<armor_core::detectors::OptionSpec>,
}

async fn list_detector_options() -> Json<Vec<DetectorOptionsResponse>> {
    let categories = armor_core::detectors::categories();
    Json(
        categories
            .iter()
            .map(|category| DetectorOptionsResponse {
                category,
                options: armor_core::detectors::option_schema(category),
            })
            .collect(),
    )
}

// ── Profiles ─────────────────────────────────────────────────────────────

/// Request body for `POST`/`PUT /api/v1/profiles[/:id]` — the "simple" end
/// of a policy editor: per-check `options` is a raw JSON object, not a
/// bespoke form per detector.
#[derive(Debug, Deserialize)]
struct ProfileRequest {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    execution_mode: ExecutionMode,
    #[serde(default)]
    fail_mode: FailMode,
    #[serde(default)]
    normalize: NormalizeConfig,
    #[serde(default)]
    checks: Vec<CheckConfig>,
}

#[derive(Debug, Serialize)]
struct ProfileResponse {
    id: String,
    description: Option<String>,
    execution_mode: ExecutionMode,
    fail_mode: FailMode,
    normalize: NormalizeConfig,
    checks: Vec<CheckConfig>,
    updated_at: String,
}

impl From<StoredProfile> for ProfileResponse {
    fn from(p: StoredProfile) -> Self {
        Self {
            id: p.policy.id,
            description: p.description,
            execution_mode: p.policy.execution_mode,
            fail_mode: p.policy.fail_mode,
            normalize: p.policy.normalize,
            checks: p.policy.checks,
            updated_at: p.updated_at.to_rfc3339(),
        }
    }
}

async fn list_profiles(
    State(state): State<AppState>,
) -> Result<Json<Vec<ProfileSummary>>, ApiError> {
    let db = require_db(&state)?;
    Ok(Json(db.list_profiles().await?))
}

async fn get_profile(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ProfileResponse>, ApiError> {
    let db = require_db(&state)?;
    let profile = db.get_profile(&id).await?.ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            "profile_not_found",
            format!("profile {id:?} not found"),
        )
    })?;
    Ok(Json(profile.into()))
}

fn require_nonempty_id(id: Option<String>) -> Result<String, ApiError> {
    id.filter(|s| !s.trim().is_empty())
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "missing_id", "id is required"))
}

async fn create_profile(
    State(state): State<AppState>,
    Json(body): Json<ProfileRequest>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;
    let id = require_nonempty_id(body.id)?;

    if db.get_profile(&id).await?.is_some() {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "profile_already_exists",
            format!("profile {id:?} already exists"),
        ));
    }

    let policy = PolicyConfig {
        id,
        execution_mode: body.execution_mode,
        fail_mode: body.fail_mode,
        normalize: body.normalize,
        checks: body.checks,
    };
    armor_core::policy::loader::validate(&policy)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "invalid_policy", e))?;
    validate_custom_regex_checks(&policy)?;
    validate_scorecards(&policy)?;

    db.upsert_profile(&ProfileInput {
        policy,
        description: body.description,
    })
    .await?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::CREATED)
}

async fn update_profile(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<ProfileRequest>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;

    if let Some(body_id) = &body.id {
        if body_id != &id {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "id_mismatch",
                "body id must match the path id, or be omitted",
            ));
        }
    }

    if db.get_profile(&id).await?.is_none() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "profile_not_found",
            format!("profile {id:?} not found"),
        ));
    }

    let policy = PolicyConfig {
        id,
        execution_mode: body.execution_mode,
        fail_mode: body.fail_mode,
        normalize: body.normalize,
        checks: body.checks,
    };
    armor_core::policy::loader::validate(&policy)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "invalid_policy", e))?;
    validate_custom_regex_checks(&policy)?;
    validate_scorecards(&policy)?;

    db.upsert_profile(&ProfileInput {
        policy,
        description: body.description,
    })
    .await?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::OK)
}

async fn delete_profile(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;
    if id == "default" {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "cannot_delete_default",
            "the \"default\" profile can't be deleted, it's the resolver's fallback",
        ));
    }
    db.delete_profile(&id).await?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

// ── Applications ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ApplicationRequest {
    #[serde(default)]
    application_id: Option<String>,
    profile_id: String,
    #[serde(default)]
    name: Option<String>,
}

async fn list_applications(
    State(state): State<AppState>,
) -> Result<Json<Vec<ApplicationRow>>, ApiError> {
    let db = require_db(&state)?;
    Ok(Json(db.list_applications().await?))
}

async fn create_application(
    State(state): State<AppState>,
    Json(body): Json<ApplicationRequest>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;
    let application_id = require_nonempty_id(body.application_id)?;

    if db.get_application(&application_id).await?.is_some() {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "application_already_exists",
            format!("application {application_id:?} already exists"),
        ));
    }

    db.upsert_application(&ApplicationInput {
        application_id,
        profile_id: body.profile_id,
        name: body.name,
    })
    .await?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::CREATED)
}

async fn update_application(
    State(state): State<AppState>,
    AxumPath(application_id): AxumPath<String>,
    Json(body): Json<ApplicationRequest>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;

    if let Some(body_id) = &body.application_id {
        if body_id != &application_id {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "id_mismatch",
                "body application_id must match the path id, or be omitted",
            ));
        }
    }

    if db.get_application(&application_id).await?.is_none() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "application_not_found",
            format!("application {application_id:?} not found"),
        ));
    }

    db.upsert_application(&ApplicationInput {
        application_id,
        profile_id: body.profile_id,
        name: body.name,
    })
    .await?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::OK)
}

async fn delete_application(
    State(state): State<AppState>,
    AxumPath(application_id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;
    db.delete_application(&application_id).await?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

// ── Logs ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct LogsQuery {
    limit: Option<i64>,
    /// Rows to skip before the page — the logs table pages `limit` at a time.
    offset: Option<i64>,
    application_id: Option<String>,
    scan_id: Option<String>,
    /// `YYYY-MM-DD`, inclusive — the start of this UTC day.
    from: Option<String>,
    /// `YYYY-MM-DD`, inclusive — internally turned into an exclusive
    /// `< the day after` bound, so the whole day is covered regardless of
    /// what time within it a row's `occurred_at` falls.
    to: Option<String>,
}

/// `GET /api/v1/logs` response: the page plus enough pagination metadata for
/// the UI to render page controls without a second count request.
#[derive(Debug, Serialize)]
struct LogsResponse {
    rows: Vec<audit_events::EvaluationLogRow>,
    total: i64,
    limit: i64,
    offset: i64,
}

/// Parses a `from`/`to` query param as a UTC day boundary — `400` (not a
/// panic or a silently-ignored filter) on a malformed date, since this
/// reaches the handler as untrusted user input from the logs page's date
/// pickers.
fn parse_day_boundary(raw: &str, field: &'static str) -> Result<DateTime<Utc>, ApiError> {
    let date = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_date",
            format!("{field} must be an ISO date (YYYY-MM-DD), got {raw:?}"),
        )
    })?;
    Ok(DateTime::<Utc>::from_naive_utc_and_offset(
        date.and_hms_opt(0, 0, 0).expect("midnight is always valid"),
        Utc,
    ))
}

async fn list_logs(
    State(state): State<AppState>,
    Query(query): Query<LogsQuery>,
) -> Result<Json<LogsResponse>, ApiError> {
    let db = require_db(&state)?;
    let limit = query.limit.unwrap_or(100).clamp(1, 1000);
    let offset = query.offset.unwrap_or(0).max(0);

    let from = query
        .from
        .as_deref()
        .map(|raw| parse_day_boundary(raw, "from"))
        .transpose()?;
    // Exclusive `< to + 1 day` so a `to` of e.g. 2026-08-02 includes every
    // row that occurred during 2026-08-02, not just its midnight instant.
    let to = query
        .to
        .as_deref()
        .map(|raw| parse_day_boundary(raw, "to"))
        .transpose()?
        .map(|start_of_day| start_of_day + chrono::Duration::days(1));

    let filters = audit_events::LogFilters {
        application_id: query.application_id.as_deref(),
        scan_id: query.scan_id.as_deref(),
        from,
        to,
    };
    let rows = audit_events::list_recent(db.pool(), limit, offset, &filters)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "database_error", e))?;
    let total = audit_events::count_recent(db.pool(), &filters)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "database_error", e))?;

    Ok(Json(LogsResponse {
        rows,
        total,
        limit,
        offset,
    }))
}

// ── Stats (Overview dashboard) ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct StatsQuery {
    /// Preset window for every windowed metric: `24h` (default), `7d`, `30d`.
    range: Option<String>,
}

/// Preset dashboard windows — returns the window length and the area chart's
/// bucket granularity (seconds) for that window. Fixed presets rather than
/// arbitrary from/to pairs keep the UI simple and the bucket sizes sensible
/// (hourly for 24h, 6-hourly for 7d, daily for 30d).
fn stats_window(raw: Option<&str>) -> Result<(chrono::Duration, i64), ApiError> {
    match raw.unwrap_or("24h") {
        "24h" => Ok((chrono::Duration::hours(24), 3600)),
        "7d" => Ok((chrono::Duration::days(7), 6 * 3600)),
        "30d" => Ok((chrono::Duration::days(30), 24 * 3600)),
        other => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_range",
            format!("range must be one of \"24h\", \"7d\", \"30d\", got {other:?}"),
        )),
    }
}

/// The window `GET /api/v1/stats` covered, plus its bucket granularity — lets
/// the UI draw a gap-free time axis from the series without re-deriving the
/// boundaries (and potentially disagreeing with the server).
#[derive(Debug, Serialize)]
struct StatsWindow {
    from: String,
    to: String,
    bucket_seconds: i64,
}

/// `GET /api/v1/stats?range=24h|7d|30d` — everything the Overview tab needs
/// in one round trip: profile/application counts (current state, not
/// windowed) plus request volume, latency, verdict split, the bucketed
/// per-verdict series, and the most-firing detectors for the selected
/// window, plus a handful of the most recent events for an activity feed.
/// Combining these avoids the UI firing several requests on every dashboard
/// load or range change.
#[derive(Debug, Serialize)]
struct StatsResponse {
    profile_count: usize,
    application_count: usize,
    requests: i64,
    avg_latency_ms: Option<f64>,
    verdict_counts: Vec<audit_events::VerdictCount>,
    series: Vec<audit_events::VerdictSeriesPoint>,
    top_detectors: Vec<audit_events::DetectorCount>,
    recent: Vec<audit_events::EvaluationLogRow>,
    window: StatsWindow,
}

async fn get_stats(
    State(state): State<AppState>,
    Query(query): Query<StatsQuery>,
) -> Result<Json<StatsResponse>, ApiError> {
    let db = require_db(&state)?;
    let now = Utc::now();
    let (window, bucket_seconds) = stats_window(query.range.as_deref())?;
    let since = now - window;

    let profiles = db.list_profiles().await?;
    let applications = db.list_applications().await?;

    fn db_err(e: impl std::fmt::Display) -> ApiError {
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "database_error", e)
    }
    let request_stats = audit_events::request_stats_since(db.pool(), since)
        .await
        .map_err(db_err)?;
    let verdict_counts = audit_events::verdict_counts_since(db.pool(), since)
        .await
        .map_err(db_err)?;
    let series = audit_events::verdict_series(db.pool(), since, now, bucket_seconds)
        .await
        .map_err(db_err)?;
    let top_detectors = audit_events::top_detectors(db.pool(), since, now, 8)
        .await
        .map_err(db_err)?;
    let recent = audit_events::list_recent(db.pool(), 8, 0, &audit_events::LogFilters::default())
        .await
        .map_err(db_err)?;

    Ok(Json(StatsResponse {
        profile_count: profiles.len(),
        application_count: applications.len(),
        requests: request_stats.total,
        avg_latency_ms: request_stats.avg_latency_ms,
        verdict_counts,
        series,
        top_detectors,
        recent,
        window: StatsWindow {
            from: since.to_rfc3339(),
            to: now.to_rfc3339(),
            bucket_seconds,
        },
    }))
}

// ── Inference pins ────────────────────────────────────────────────────

async fn list_pins(
    State(state): State<AppState>,
) -> Result<Json<Vec<armor_storage::inference_pins::InferencePin>>, ApiError> {
    let db = require_db(&state)?;
    let pins = armor_storage::inference_pins::list_all(db.pool())
        .await
        .map_err(db_err)?;
    Ok(Json(pins))
}

#[derive(Deserialize)]
struct PinRequest {
    task: String,
    model_id: String,
    revision: String,
    sha256: Option<String>,
    threshold: Option<f64>,
}

async fn upsert_pin(
    State(state): State<AppState>,
    Json(body): Json<PinRequest>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;
    let now = chrono::Utc::now();
    let pin = armor_storage::inference_pins::InferencePin {
        task: body.task,
        model_id: body.model_id,
        revision: body.revision,
        sha256: body.sha256,
        threshold: body.threshold,
        updated_at: now,
    };
    armor_storage::inference_pins::upsert(db.pool(), &pin)
        .await
        .map_err(db_err)?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::OK)
}

async fn delete_pin(
    State(state): State<AppState>,
    AxumPath(task): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let db = require_db(&state)?;
    armor_storage::inference_pins::delete(db.pool(), &task)
        .await
        .map_err(db_err)?;
    reload_and_swap(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

// ── Models UI ─────────────────────────────────────────────────────────

/// `GET /api/v1/models` — list available models from the sidecar, enriched
/// with catalog metadata (license, size, hardware) and stored scorecard.
/// Returns 501 when the inference tier is off.
async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let url = state.inference_url.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "inference_disabled",
            "inference tier is not configured",
        )
    })?;

    let client = reqwest::Client::new();
    let mut req = client.get(format!("{url}/v1/models"));
    if let Some(token) = state.resolve_inference_token().await {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| db_err(format!("sidecar request failed: {e}")))?;
    #[derive(Deserialize)]
    struct ModelsResponse {
        models: Vec<serde_json::Value>,
    }
    let parsed: ModelsResponse = resp
        .json()
        .await
        .map_err(|e| db_err(format!("sidecar response parse failed: {e}")))?;
    Ok(Json(parsed.models))
}

/// `GET /api/v1/models/catalog` — static catalog metadata (display name,
/// rationale, vetted shortlist) per task, distinct from `GET
/// /api/v1/models`'s live registry state. Returns 501 when the inference
/// tier is off.
async fn models_catalog(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let url = state.inference_url.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "inference_disabled",
            "inference tier is not configured",
        )
    })?;

    let client = reqwest::Client::new();
    let mut req = client.get(format!("{url}/v1/models/catalog"));
    if let Some(token) = state.resolve_inference_token().await {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| db_err(format!("sidecar request failed: {e}")))?;
    #[derive(Deserialize)]
    struct CatalogResponse {
        tasks: Vec<serde_json::Value>,
    }
    let parsed: CatalogResponse = resp
        .json()
        .await
        .map_err(|e| db_err(format!("sidecar response parse failed: {e}")))?;
    Ok(Json(parsed.tasks))
}

/// `GET /api/v1/hardware` — hardware inventory for both tiers, reported
/// separately because core and the inference sidecar are commonly deployed
/// on different hardware: `core` is this process's own host
/// ([`crate::hardware::local_hardware_info`]), `inference` is proxied from
/// the sidecar's own `GET /v1/hardware` (`armor_inference.hardware`). Always
/// 200 — a missing/unreachable sidecar shows up as
/// `inference.status != "ok"`, not as a request failure, since core's own
/// hardware is worth reporting even when the sidecar isn't.
async fn get_hardware(State(state): State<AppState>) -> Json<serde_json::Value> {
    let core = crate::hardware::local_hardware_info();

    let inference = match state.inference_url.as_ref() {
        None => serde_json::json!({
            "status": "not_configured",
            "detail": null,
            "hardware": null,
        }),
        Some(url) => {
            let client = reqwest::Client::new();
            let mut req = client.get(format!("{url}/v1/hardware"));
            if let Some(token) = state.resolve_inference_token().await {
                req = req.bearer_auth(token);
            }
            match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    match resp.json::<serde_json::Value>().await {
                        Ok(body) => {
                            serde_json::json!({ "status": "ok", "detail": null, "hardware": body })
                        }
                        Err(e) => serde_json::json!({
                            "status": "unreachable",
                            "detail": format!("sidecar response parse failed: {e}"),
                            "hardware": null,
                        }),
                    }
                }
                Ok(resp) => serde_json::json!({
                    "status": "unreachable",
                    "detail": format!("sidecar returned HTTP {}", resp.status()),
                    "hardware": null,
                }),
                Err(e) => serde_json::json!({
                    "status": "unreachable",
                    "detail": format!("sidecar request failed: {e}"),
                    "hardware": null,
                }),
            }
        }
    };

    Json(serde_json::json!({ "core": core, "inference": inference }))
}

#[derive(Deserialize)]
struct InstallRequest {
    task: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    revision: Option<String>,
}

/// Maps a reqwest status onto its axum equivalent — both are just a u16
/// underneath, but the crates don't share a type, so a sidecar response
/// can't be forwarded verbatim without this.
/// 401 is remapped to 502: it means armor-core's own `ARMOR_INFERENCE_AUTH_TOKEN`
/// didn't match what the sidecar expects (an operator/infra credential, per
/// `require_mutation_token` in `armor_inference.main`) — never the caller's
/// fault, and relaying it verbatim would collide with `/api/v1/*`'s own 401
/// contract (armor-ui-api-key auth, `middleware/auth.rs`). `app.js`'s `api()`
/// treats *any* 401 from `/api/v1/*` as "your UI login is invalid" and signs
/// the operator out; passing this one through as literal 401 was sending
/// them into a sign-out loop over a token mismatch they can't fix by
/// logging back in.
fn sidecar_status(status: reqwest::StatusCode) -> StatusCode {
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return StatusCode::BAD_GATEWAY;
    }
    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY)
}

/// The sidecar's error responses are `{"detail": "..."}` (FastAPI's default
/// shape). Used to turn a non-2xx sidecar response into an `ApiError` that
/// carries the sidecar's own status (403 disabled, 409 already installing,
/// 400 unknown task/model, ...) instead of masking it behind a 200 the UI
/// would otherwise mistake for success.
fn sidecar_error(status: reqwest::StatusCode, body: &serde_json::Value) -> ApiError {
    let detail = body
        .get("detail")
        .and_then(|d| d.as_str())
        .unwrap_or("sidecar request failed")
        .to_string();
    ApiError::new(sidecar_status(status), "sidecar_error", detail)
}

/// `POST /api/v1/models/install` — trigger a model install job on the sidecar.
/// Returns the job ID for polling via `GET /models/install/:job_id`.
async fn install_model(
    State(state): State<AppState>,
    Json(body): Json<InstallRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let url = state.inference_url.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "inference_disabled",
            "inference tier is not configured",
        )
    })?;

    let client = reqwest::Client::new();
    let mut req = client
        .post(format!("{url}/v1/models/install"))
        .json(&serde_json::json!({
            "task": body.task,
            "model_id": body.model_id,
            "revision": body.revision,
        }));
    if let Some(token) = state.resolve_inference_token().await {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| db_err(format!("sidecar install request failed: {e}")))?;
    let status = resp.status();
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| db_err(format!("sidecar install response parse failed: {e}")))?;
    if !status.is_success() {
        return Err(sidecar_error(status, &body));
    }
    Ok(Json(body))
}

/// `GET /api/v1/models/install/:job_id` — poll an install job's status.
async fn install_status(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let url = state.inference_url.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "inference_disabled",
            "inference tier is not configured",
        )
    })?;

    // The sidecar only ever mints `uuid::v4` job IDs (`install.py`). Reject
    // anything else here rather than splicing the raw, axum-percent-decoded
    // path segment into the outbound URL: axum matches routes on the raw,
    // still-encoded path, so a segment like `..%2f..%2fadmin` is captured
    // whole and only decoded to `../../admin` by this `Path` extractor,
    // letting a caller traverse the sidecar's internal API with our bearer
    // token attached.
    let job_id: uuid::Uuid = job_id.parse().map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_job_id",
            "job_id must be a UUID",
        )
    })?;

    let client = reqwest::Client::new();
    let mut req = client.get(format!("{url}/v1/models/install/{job_id}"));
    if let Some(token) = state.resolve_inference_token().await {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| db_err(format!("sidecar status request failed: {e}")))?;
    let status = resp.status();
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| db_err(format!("sidecar status response parse failed: {e}")))?;
    if !status.is_success() {
        return Err(sidecar_error(status, &body));
    }
    Ok(Json(body))
}

#[cfg(test)]
mod logs_query_tests {
    use super::*;

    #[test]
    fn valid_date_parses_to_utc_midnight() {
        let parsed = parse_day_boundary("2026-08-02", "from").unwrap();
        assert_eq!(parsed.to_rfc3339(), "2026-08-02T00:00:00+00:00");
    }

    #[test]
    fn malformed_date_is_rejected() {
        let err = parse_day_boundary("08/02/2026", "from").unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "invalid_date");
    }

    #[test]
    fn empty_string_is_rejected() {
        assert!(parse_day_boundary("", "to").is_err());
    }

    #[test]
    fn to_bound_becomes_exclusive_next_day() {
        let start_of_day = parse_day_boundary("2026-08-02", "to").unwrap();
        let exclusive_upper = start_of_day + chrono::Duration::days(1);
        assert_eq!(exclusive_upper.to_rfc3339(), "2026-08-03T00:00:00+00:00");
    }
}

#[cfg(test)]
mod stats_query_tests {
    use super::*;

    #[test]
    fn default_window_is_last_24h_hourly() {
        let (window, bucket) = stats_window(None).unwrap();
        assert_eq!(window, chrono::Duration::hours(24));
        assert_eq!(bucket, 3600);
    }

    #[test]
    fn seven_days_buckets_six_hourly() {
        let (window, bucket) = stats_window(Some("7d")).unwrap();
        assert_eq!(window, chrono::Duration::days(7));
        assert_eq!(bucket, 6 * 3600);
    }

    #[test]
    fn thirty_days_buckets_daily() {
        let (window, bucket) = stats_window(Some("30d")).unwrap();
        assert_eq!(window, chrono::Duration::days(30));
        assert_eq!(bucket, 24 * 3600);
    }

    #[test]
    fn unknown_range_is_rejected() {
        let err = stats_window(Some("10m")).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "invalid_range");
    }
}
