//! Per-task model pins for the inference tier — which model (and revision)
//! each task should use, pushed to edge instances through the existing
//! sync bundle (`sync.rs`). The sync payload gains a `pins` array so
//! `ARMOR_MODE=edge` instances receive them without a second channel.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;

/// One task's model pin, as returned by [`list_all`] and serialized in the
/// sync payload.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct InferencePin {
    pub task: String,
    pub model_id: String,
    pub revision: String,
    pub sha256: Option<String>,
    /// Per-task confidence threshold — below this, the model's verdict is
    /// advisory-only (a coarse alternative to the per-check scorecard gate).
    pub threshold: Option<f64>,
    pub updated_at: DateTime<Utc>,
}

/// Insert or update a pin. Upserts on `task` (primary key).
pub async fn upsert(pool: &PgPool, pin: &InferencePin) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO inference_pins (task, model_id, revision, sha256, threshold, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (task) DO UPDATE SET
            model_id = EXCLUDED.model_id,
            revision = EXCLUDED.revision,
            sha256 = EXCLUDED.sha256,
            threshold = EXCLUDED.threshold,
            updated_at = EXCLUDED.updated_at
        "#,
    )
    .bind(&pin.task)
    .bind(&pin.model_id)
    .bind(&pin.revision)
    .bind(&pin.sha256)
    .bind(pin.threshold)
    .bind(pin.updated_at)
    .execute(pool)
    .await?;

    Ok(())
}

/// Delete a pin by task.
pub async fn delete(pool: &PgPool, task: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM inference_pins WHERE task = $1")
        .bind(task)
        .execute(pool)
        .await?;
    Ok(())
}

/// List all pins — the sync bundle's source of truth.
pub async fn list_all(pool: &PgPool) -> Result<Vec<InferencePin>, sqlx::Error> {
    sqlx::query_as::<_, InferencePin>(
        "SELECT task, model_id, revision, sha256, threshold, updated_at FROM inference_pins ORDER BY task",
    )
    .fetch_all(pool)
    .await
}
