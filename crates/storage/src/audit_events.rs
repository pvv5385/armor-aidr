//! Per-request decision log: rule hits, per-check verdicts, latency, final
//! action — the Postgres-backed twin of `crates/api/src/audit.rs`'s
//! `JsonlSpoolAuditSink`, written by `crates/api/src/audit.rs::PgAuditSink`
//! when `DATABASE_URL` is set (both run together — see that module's doc
//! comment on `MultiAuditSink`). Also the source `GET /api/v1/logs` reads
//! from for the management UI's "simple logging" view.
//!
//! This module owns only DB primitives (row shape, insert, query) — the
//! domain type (`audit::EvaluationEvent`) and the `AuditSink` trait stay in
//! `armor-api`, which converts to/from [`NewEvaluationLog`]/[`EvaluationLogRow`].
//! Keeps the dependency direction `core` <- `storage` <- `api` intact
//! (`storage` never depends on `api`).

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::postgres::PgPool;
use uuid::Uuid;

/// One row as returned by [`insert`]'s caller and by [`list_recent`] — also
/// serialized directly as `GET /api/v1/logs`'s JSON response body.
#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct EvaluationLogRow {
    pub id: Uuid,
    /// Armor's own per-request id — always present. See the column comment
    /// on `evaluation_logs.scan_id` (`migrations/0001_control_plane.sql`).
    pub scan_id: String,
    pub session_id: String,
    /// Caller-supplied correlation id, when they sent one — `None`
    /// otherwise. See `evaluation_logs.client_request_id`'s column comment.
    pub client_request_id: Option<String>,
    pub application_id: Option<String>,
    pub profile_id: Option<String>,
    pub occurred_at: DateTime<Utc>,
    pub stage: String,
    pub verdict: String,
    pub checks: serde_json::Value,
    pub latency_ms: f64,
    /// Per-check layer trace as JSON, when escalation ran.
    pub layers: Option<serde_json::Value>,
    /// Model version of the selected layer, when an ML layer ran.
    pub model_version: Option<String>,
}

/// Insert input. `occurred_at_unix_ms` (not a `DateTime<Utc>`) so callers
/// don't need a `chrono` dependency of their own — `armor-api`'s
/// `EvaluationEvent::occurred_at_unix_ms` already carries this shape.
pub struct NewEvaluationLog<'a> {
    pub scan_id: &'a str,
    pub session_id: &'a str,
    pub client_request_id: Option<&'a str>,
    pub application_id: Option<&'a str>,
    pub profile_id: Option<&'a str>,
    pub occurred_at_unix_ms: i64,
    pub stage: &'a str,
    pub verdict: &'a str,
    pub checks: serde_json::Value,
    pub latency_ms: f64,
    pub layers: Option<serde_json::Value>,
    pub model_version: Option<&'a str>,
}

pub async fn insert(pool: &PgPool, entry: &NewEvaluationLog<'_>) -> Result<(), sqlx::Error> {
    let occurred_at =
        DateTime::<Utc>::from_timestamp_millis(entry.occurred_at_unix_ms).unwrap_or_else(Utc::now);

    sqlx::query(
        r#"
        INSERT INTO evaluation_logs
            (id, scan_id, session_id, client_request_id, application_id, profile_id, occurred_at, stage, verdict, checks, latency_ms, layers, model_version)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        "#,
    )
    .bind(Uuid::now_v7())
    .bind(entry.scan_id)
    .bind(entry.session_id)
    .bind(entry.client_request_id)
    .bind(entry.application_id)
    .bind(entry.profile_id)
    .bind(occurred_at)
    .bind(entry.stage)
    .bind(entry.verdict)
    .bind(&entry.checks)
    .bind(entry.latency_ms)
    .bind(&entry.layers)
    .bind(entry.model_version)
    .execute(pool)
    .await?;

    Ok(())
}

/// Optional filters for [`list_recent`] — every field is an exact match
/// (`application_id`/`scan_id`) or an inclusive lower/exclusive upper bound
/// (`from`/`to`) on `occurred_at`. All `None` means "no filtering, just the
/// most recent rows." Built with [`sqlx::QueryBuilder`] rather than a fixed
/// set of hand-written queries per combination — four independent optional
/// filters would otherwise mean 16 query strings to keep in sync.
#[derive(Debug, Default, Clone, Copy)]
pub struct LogFilters<'a> {
    pub application_id: Option<&'a str>,
    pub scan_id: Option<&'a str>,
    pub from: Option<DateTime<Utc>>,
    pub to: Option<DateTime<Utc>>,
}

/// One verdict's row count within a [`request_stats_since`]/[`verdict_counts_since`]
/// window — e.g. `{ verdict: "deny", count: 12 }`. Powers the Overview
/// dashboard's verdict-breakdown bars.
#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct VerdictCount {
    pub verdict: String,
    pub count: i64,
}

/// One detector's fired count within a window — `{ category: "pii", count: 9 }`.
/// Powers the Overview dashboard's "Top detectors" ranking.
#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct DetectorCount {
    pub category: String,
    pub count: i64,
}

/// The `limit` most-frequently-firing detectors over `[from, to)` — a check
/// counts when it failed (`passed = false`), i.e. it produced a signal. The
/// `checks` JSONB column is an array of `{category, passed, action, severity}`
/// (see `crates/api/src/audit.rs`'s `CheckSummary`).
pub async fn top_detectors(
    pool: &PgPool,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    limit: i64,
) -> Result<Vec<DetectorCount>, sqlx::Error> {
    sqlx::query_as::<_, DetectorCount>(
        "SELECT elem->>'category' AS category, COUNT(*) AS count \
         FROM evaluation_logs \
         CROSS JOIN LATERAL jsonb_array_elements(checks) AS elem \
         WHERE occurred_at >= $1 AND occurred_at < $2 \
           AND (elem->>'passed')::boolean = false \
         GROUP BY 1 \
         ORDER BY count DESC \
         LIMIT $3",
    )
    .bind(from)
    .bind(to)
    .bind(limit)
    .fetch_all(pool)
    .await
}

/// `GROUP BY verdict` over the window `[since, now)` — one row per distinct
/// verdict seen, no zero-filling for verdicts that didn't occur (the UI
/// treats "absent" and "zero" the same way).
pub async fn verdict_counts_since(
    pool: &PgPool,
    since: DateTime<Utc>,
) -> Result<Vec<VerdictCount>, sqlx::Error> {
    sqlx::query_as::<_, VerdictCount>(
        "SELECT verdict, COUNT(*) AS count FROM evaluation_logs \
         WHERE occurred_at >= $1 GROUP BY verdict ORDER BY count DESC",
    )
    .bind(since)
    .fetch_all(pool)
    .await
}

/// Aggregate request volume/latency over the window `[since, now)` — the
/// Overview dashboard's headline numbers.
#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct RequestStats {
    pub total: i64,
    pub avg_latency_ms: Option<f64>,
}

pub async fn request_stats_since(
    pool: &PgPool,
    since: DateTime<Utc>,
) -> Result<RequestStats, sqlx::Error> {
    sqlx::query_as::<_, RequestStats>(
        "SELECT COUNT(*) AS total, AVG(latency_ms) AS avg_latency_ms \
         FROM evaluation_logs WHERE occurred_at >= $1",
    )
    .bind(since)
    .fetch_one(pool)
    .await
}

/// One bucket's per-verdict count from [`verdict_series`] — e.g.
/// `{ bucket: 2026-08-02T14:00:00Z, verdict: "deny", count: 3 }`. Powers the
/// Overview dashboard's stacked area chart.
#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct VerdictSeriesPoint {
    /// Start of the bucket (UTC), aligned to the requested `bucket_seconds`.
    pub bucket: DateTime<Utc>,
    pub verdict: String,
    pub count: i64,
}

/// Per-verdict request counts bucketed over `[from, to)`. `bucket_seconds`
/// is any positive granularity (3600 = hourly, 86400 = daily), so the same
/// query serves every dashboard range without a fixed set of SQL strings.
/// Buckets are floored to the epoch so consecutive windows never straddle
/// boundaries differently, and no zero-filling is done — the caller aligns
/// a gap-free axis itself.
pub async fn verdict_series(
    pool: &PgPool,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    bucket_seconds: i64,
) -> Result<Vec<VerdictSeriesPoint>, sqlx::Error> {
    sqlx::query_as::<_, VerdictSeriesPoint>(
        "SELECT \
            to_timestamp((extract(epoch FROM occurred_at)::bigint / $2) * $2) AS bucket, \
            verdict, COUNT(*) AS count \
         FROM evaluation_logs \
         WHERE occurred_at >= $1 AND occurred_at < $3 \
         GROUP BY 1, verdict \
         ORDER BY 1, verdict",
    )
    .bind(from)
    .bind(bucket_seconds)
    .bind(to)
    .fetch_all(pool)
    .await
}

/// Applies the four optional [`LogFilters`] predicates to a `QueryBuilder`.
/// Shared by [`list_recent`] and [`count_recent`] so the filter set can't
/// drift between the row fetch and its count.
fn push_log_filters(query: &mut sqlx::QueryBuilder<sqlx::Postgres>, filters: &LogFilters<'_>) {
    if let Some(application_id) = filters.application_id {
        query
            .push(" AND application_id = ")
            .push_bind(application_id);
    }
    if let Some(scan_id) = filters.scan_id {
        query.push(" AND scan_id = ").push_bind(scan_id);
    }
    if let Some(from) = filters.from {
        query.push(" AND occurred_at >= ").push_bind(from);
    }
    if let Some(to) = filters.to {
        query.push(" AND occurred_at < ").push_bind(to);
    }
}

/// Total rows matching `filters` — the logs page's pagination uses this for
/// the page count (alongside [`list_recent`]'s `offset`).
pub async fn count_recent(pool: &PgPool, filters: &LogFilters<'_>) -> Result<i64, sqlx::Error> {
    let mut query = sqlx::QueryBuilder::<sqlx::Postgres>::new(
        "SELECT COUNT(*) FROM evaluation_logs WHERE 1 = 1",
    );
    push_log_filters(&mut query, filters);
    query.build_query_scalar::<i64>().fetch_one(pool).await
}

/// A window of the most recent `limit` events starting at `offset`, newest
/// first, narrowed by `filters`. `id` breaks `occurred_at` ties so paging is
/// deterministic (v7 UUIDs are time-ordered, so `id DESC` agrees with the
/// primary sort).
pub async fn list_recent(
    pool: &PgPool,
    limit: i64,
    offset: i64,
    filters: &LogFilters<'_>,
) -> Result<Vec<EvaluationLogRow>, sqlx::Error> {
    let mut query = sqlx::QueryBuilder::<sqlx::Postgres>::new(
        "SELECT id, scan_id, session_id, client_request_id, application_id, profile_id, occurred_at, stage, verdict, checks, latency_ms, layers, model_version \
         FROM evaluation_logs WHERE 1 = 1",
    );
    push_log_filters(&mut query, filters);
    query
        .push(" ORDER BY occurred_at DESC, id DESC LIMIT ")
        .push_bind(limit)
        .push(" OFFSET ")
        .push_bind(offset);

    query
        .build_query_as::<EvaluationLogRow>()
        .fetch_all(pool)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{test_pool, unique_id};

    /// Regression test for the bug this module shipped with: `list_recent`'s
    /// hand-written column list didn't include `layers`/`model_version`
    /// after migration 0005 added them to `EvaluationLogRow`, so
    /// `build_query_as::<EvaluationLogRow>()` failed every call with
    /// `ColumnNotFound` — `GET /api/v1/logs` 500'd unconditionally. A unit
    /// test on the struct/query shapes wouldn't have caught this; only
    /// exercising the query against a real schema does.
    #[tokio::test]
    async fn list_recent_selects_every_column_evaluation_log_row_needs() {
        let Some(pool) = test_pool().await else {
            return;
        };

        let scan_id = unique_id("scan");
        let session_id = unique_id("session");
        insert(
            &pool,
            &NewEvaluationLog {
                scan_id: &scan_id,
                session_id: &session_id,
                client_request_id: None,
                application_id: None,
                profile_id: None,
                occurred_at_unix_ms: Utc::now().timestamp_millis(),
                stage: "pre",
                verdict: "allow",
                checks: serde_json::json!([]),
                latency_ms: 1.5,
                layers: Some(serde_json::json!([{"name": "regex"}])),
                model_version: Some("v1"),
            },
        )
        .await
        .expect("insert");

        let rows = list_recent(
            &pool,
            10,
            0,
            &LogFilters {
                scan_id: Some(&scan_id),
                ..Default::default()
            },
        )
        .await
        .expect("list_recent must not fail with ColumnNotFound");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model_version.as_deref(), Some("v1"));
        assert!(rows[0].layers.is_some());
    }
}
