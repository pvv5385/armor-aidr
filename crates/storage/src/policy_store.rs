//! DB-backed, versioned, multi-tenant policy config — replaces the static
//! `config/policies.yaml` once there's more than one customer to serve.
//! Same `application_id -> profile` resolution shape
//! `crates/api/src/profiles.rs::ProfileResolver` already serves from disk
//! (worth reading first) — this module is the storage swap, not a
//! resolution-semantics redesign: [`PgPolicyStore::load_all_policies`]
//! returns exactly the `(Vec<PolicyConfig>, application_id -> profile_id
//! pairs)` shape `armor-api`'s boot/DB-reload code already knows how to turn
//! into a `ProfileResolver` (mirroring `sync.rs`'s `SyncPayload` handling).
//!
//! Enum-shaped columns (`execution_mode`, `fail_mode`, `on_fail`, `mode`)
//! are stored as JSONB, not TEXT, so they round-trip through
//! `serde_json::to_value`/`from_value` exactly like every other
//! `armor_core::policy::schema` value already does elsewhere in this
//! codebase (`sync.rs`'s `SyncPayload` deserializes the same
//! `PolicyConfig`/`CheckConfig` types from a JSON body) — no hand-written
//! enum<->string mapping to keep in sync as detector options evolve.
//!
//! Queries use the runtime `sqlx::query`/`query_as` API, not the
//! compile-time-checked `query!`/`query_as!` macros — those need a live DB
//! (or an `.sqlx` offline cache) at `cargo build` time, which this repo
//! doesn't assume every dev/CI environment has.

use std::collections::HashMap;

use armor_core::policy::schema::PolicyConfig;
use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::postgres::{PgPool, PgPoolOptions};
use uuid::Uuid;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("../../migrations");

#[derive(Debug, thiserror::Error)]
pub enum PolicyStoreError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("migration failed: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("serializing policy data: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("profile {0:?} not found")]
    ProfileNotFound(String),
    #[error("profile {0:?} is still assigned to at least one application")]
    ProfileInUse(String),
    #[error("application {0:?} not found")]
    ApplicationNotFound(String),
    #[error("application references unknown profile_id {0:?}")]
    ApplicationInvalidProfile(String),
}

fn is_fk_violation(e: &sqlx::Error) -> bool {
    matches!(e.as_database_error().and_then(|de| de.code()), Some(code) if code == "23503")
}

/// One row of `GET /api/v1/profiles` — cheap summary, not the full
/// check list (see [`StoredProfile`] for that).
#[derive(Debug, Clone, Serialize)]
pub struct ProfileSummary {
    pub id: String,
    pub description: Option<String>,
    pub check_count: i64,
    pub updated_at: DateTime<Utc>,
}

/// A profile's full editable shape, as returned by `GET
/// /api/v1/profiles/:id` and consumed (as [`ProfileInput`]) by the
/// create/update handlers.
pub struct StoredProfile {
    pub policy: PolicyConfig,
    pub description: Option<String>,
    pub updated_at: DateTime<Utc>,
}

pub struct ProfileInput {
    pub policy: PolicyConfig,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct ApplicationRow {
    pub application_id: String,
    pub profile_id: String,
    pub name: Option<String>,
    pub updated_at: DateTime<Utc>,
}

pub struct ApplicationInput {
    pub application_id: String,
    pub profile_id: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct ProfileRow {
    id: String,
    description: Option<String>,
    execution_mode: serde_json::Value,
    fail_mode: serde_json::Value,
    normalize: serde_json::Value,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct CheckRow {
    profile_id: String,
    category: String,
    enabled: bool,
    on_fail: serde_json::Value,
    mode: serde_json::Value,
    fail_mode: serde_json::Value,
    options: serde_json::Value,
    strategy: Option<serde_json::Value>,
    backends: serde_json::Value,
    scorecard: Option<serde_json::Value>,
}

/// Reassembles a DB-stored profile into the same `PolicyConfig` shape
/// `armor_core::policy::loader::load` produces from YAML — every consumer
/// downstream (hardening, the orchestrator) stays oblivious to where the
/// policy came from.
fn assemble_policy(
    row: &ProfileRow,
    checks: &[CheckRow],
) -> Result<PolicyConfig, PolicyStoreError> {
    let checks_json: Vec<serde_json::Value> = checks
        .iter()
        .map(|c| {
            let mut check = serde_json::Map::new();
            check.insert(
                "category".to_string(),
                serde_json::Value::String(c.category.clone()),
            );
            check.insert("enabled".to_string(), serde_json::Value::Bool(c.enabled));
            check.insert("options".to_string(), c.options.clone());
            check.insert("on_fail".to_string(), c.on_fail.clone());
            check.insert("fail_mode".to_string(), c.fail_mode.clone());
            check.insert("mode".to_string(), c.mode.clone());
            check.insert("backends".to_string(), c.backends.clone());
            if let Some(strategy) = &c.strategy {
                check.insert("strategy".to_string(), strategy.clone());
            }
            if let Some(scorecard) = &c.scorecard {
                check.insert("scorecard".to_string(), scorecard.clone());
            }
            serde_json::Value::Object(check)
        })
        .collect();

    let mut policy = serde_json::Map::new();
    policy.insert("id".to_string(), serde_json::Value::String(row.id.clone()));
    policy.insert("execution_mode".to_string(), row.execution_mode.clone());
    policy.insert("fail_mode".to_string(), row.fail_mode.clone());
    policy.insert("normalize".to_string(), row.normalize.clone());
    policy.insert("checks".to_string(), serde_json::Value::Array(checks_json));

    Ok(serde_json::from_value(serde_json::Value::Object(policy))?)
}

/// Run every migration in `migrations/` against an existing pool.
///
/// [`PgPolicyStore::connect`] does this as part of connecting; this is the
/// same thing for a caller that already has a pool and isn't building a
/// policy store — today, the database-backed tests in [`crate::sessions`]
/// and [`crate::vault`]. Idempotent, and `sqlx` takes an advisory lock for
/// the duration, so concurrent callers are safe.
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::migrate::MigrateError> {
    MIGRATOR.run(pool).await
}

pub struct PgPolicyStore {
    pool: PgPool,
}

impl PgPolicyStore {
    /// Connects and runs every migration in `migrations/` (idempotent —
    /// `sqlx::migrate!` tracks what's already applied). Fails fast on a bad
    /// `DATABASE_URL` or a migration error rather than starting the server
    /// against an unusable store.
    pub async fn connect(database_url: &str) -> Result<Self, PolicyStoreError> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    /// Exposed so `armor-api`'s audit sink (which owns the
    /// `EvaluationEvent`/`AuditSink` domain types — see this module's doc
    /// comment on dependency direction) can write to `evaluation_logs`
    /// through `armor_storage::audit_events` using the same pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    pub async fn is_empty(&self) -> Result<bool, PolicyStoreError> {
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM profiles")
            .fetch_one(&self.pool)
            .await?;
        Ok(count == 0)
    }

    /// First-boot convenience: seeds the store with the default policy
    /// already loaded from `config/policies.yaml` (or embedded fallback),
    /// so a fresh Postgres deployment doesn't start with zero profiles.
    pub async fn seed_default(&self, policy: &PolicyConfig) -> Result<(), PolicyStoreError> {
        self.upsert_profile(&ProfileInput {
            policy: policy.clone(),
            description: Some("Seeded from config/policies.yaml on first boot.".to_string()),
        })
        .await
    }

    pub async fn list_profiles(&self) -> Result<Vec<ProfileSummary>, PolicyStoreError> {
        let rows: Vec<(String, Option<String>, DateTime<Utc>, i64)> = sqlx::query_as(
            r#"
            SELECT p.id, p.description, p.updated_at, COUNT(c.id) AS check_count
            FROM profiles p
            LEFT JOIN checks c ON c.profile_id = p.id
            GROUP BY p.id, p.description, p.updated_at
            ORDER BY p.id
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |(id, description, updated_at, check_count)| ProfileSummary {
                    id,
                    description,
                    updated_at,
                    check_count,
                },
            )
            .collect())
    }

    pub async fn get_profile(&self, id: &str) -> Result<Option<StoredProfile>, PolicyStoreError> {
        let Some(profile_row): Option<ProfileRow> = sqlx::query_as(
            "SELECT id, description, execution_mode, fail_mode, normalize, updated_at \
             FROM profiles WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?
        else {
            return Ok(None);
        };

        let check_rows: Vec<CheckRow> = sqlx::query_as(
            "SELECT profile_id, category, enabled, on_fail, mode, fail_mode, options, \
                    strategy, backends, scorecard \
             FROM checks WHERE profile_id = $1 ORDER BY category",
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await?;

        let policy = assemble_policy(&profile_row, &check_rows)?;
        Ok(Some(StoredProfile {
            policy,
            description: profile_row.description,
            updated_at: profile_row.updated_at,
        }))
    }

    /// Creates or fully replaces a profile and its checks in one
    /// transaction (`DELETE`-then-`INSERT` on `checks` — profiles are
    /// small, edited rarely, and this avoids reconciling a diff). Used both
    /// by `POST`/`PUT /api/v1/profiles` and by [`Self::seed_default`].
    pub async fn upsert_profile(&self, input: &ProfileInput) -> Result<(), PolicyStoreError> {
        let execution_mode = serde_json::to_value(input.policy.execution_mode)?;
        let fail_mode = serde_json::to_value(input.policy.fail_mode)?;
        let normalize = serde_json::to_value(input.policy.normalize)?;

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            r#"
            INSERT INTO profiles (id, description, execution_mode, fail_mode, normalize, updated_at)
            VALUES ($1, $2, $3, $4, $5, now())
            ON CONFLICT (id) DO UPDATE SET
                description = EXCLUDED.description,
                execution_mode = EXCLUDED.execution_mode,
                fail_mode = EXCLUDED.fail_mode,
                normalize = EXCLUDED.normalize,
                updated_at = now()
            "#,
        )
        .bind(&input.policy.id)
        .bind(&input.description)
        .bind(&execution_mode)
        .bind(&fail_mode)
        .bind(&normalize)
        .execute(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM checks WHERE profile_id = $1")
            .bind(&input.policy.id)
            .execute(&mut *tx)
            .await?;

        for check in &input.policy.checks {
            let on_fail = serde_json::to_value(check.on_fail)?;
            let mode = serde_json::to_value(check.mode)?;
            let fail_mode = serde_json::to_value(check.fail_mode)?;
            let options = serde_json::to_value(&check.options)?;
            let strategy = check
                .strategy
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?;
            let backends = serde_json::to_value(&check.backends)?;
            let scorecard = check
                .scorecard
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?;

            sqlx::query(
                r#"
                INSERT INTO checks (id, profile_id, category, enabled, on_fail, mode, fail_mode, options, strategy, backends, scorecard)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                "#,
            )
            .bind(Uuid::new_v4())
            .bind(&input.policy.id)
            .bind(&check.category)
            .bind(check.enabled)
            .bind(&on_fail)
            .bind(&mode)
            .bind(&fail_mode)
            .bind(&options)
            .bind(&strategy)
            .bind(&backends)
            .bind(&scorecard)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// `Err(ProfileInUse)` when an application still references this
    /// profile (the `applications.profile_id` foreign key has no `ON
    /// DELETE CASCADE` — see the migration) — the caller (`control_plane.rs`)
    /// maps this to `409`, never silently orphaning an application's
    /// assignment.
    pub async fn delete_profile(&self, id: &str) -> Result<(), PolicyStoreError> {
        let result = sqlx::query("DELETE FROM profiles WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await;

        match result {
            Ok(res) if res.rows_affected() == 0 => {
                Err(PolicyStoreError::ProfileNotFound(id.to_string()))
            }
            Ok(_) => Ok(()),
            Err(e) if is_fk_violation(&e) => Err(PolicyStoreError::ProfileInUse(id.to_string())),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn list_applications(&self) -> Result<Vec<ApplicationRow>, PolicyStoreError> {
        let rows = sqlx::query_as(
            "SELECT application_id, profile_id, name, updated_at FROM applications ORDER BY application_id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    pub async fn get_application(
        &self,
        application_id: &str,
    ) -> Result<Option<ApplicationRow>, PolicyStoreError> {
        let row = sqlx::query_as(
            "SELECT application_id, profile_id, name, updated_at FROM applications WHERE application_id = $1",
        )
        .bind(application_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    /// `Err(ApplicationInvalidProfile)` when `profile_id` doesn't name an
    /// existing profile (foreign-key violation) — the caller maps this to
    /// `400`.
    pub async fn upsert_application(
        &self,
        input: &ApplicationInput,
    ) -> Result<(), PolicyStoreError> {
        let result = sqlx::query(
            r#"
            INSERT INTO applications (application_id, profile_id, name, updated_at)
            VALUES ($1, $2, $3, now())
            ON CONFLICT (application_id) DO UPDATE SET
                profile_id = EXCLUDED.profile_id,
                name = EXCLUDED.name,
                updated_at = now()
            "#,
        )
        .bind(&input.application_id)
        .bind(&input.profile_id)
        .bind(&input.name)
        .execute(&self.pool)
        .await;

        match result {
            Ok(_) => Ok(()),
            Err(e) if is_fk_violation(&e) => Err(PolicyStoreError::ApplicationInvalidProfile(
                input.profile_id.clone(),
            )),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn delete_application(&self, application_id: &str) -> Result<(), PolicyStoreError> {
        let result = sqlx::query("DELETE FROM applications WHERE application_id = $1")
            .bind(application_id)
            .execute(&self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(PolicyStoreError::ApplicationNotFound(
                application_id.to_string(),
            ));
        }
        Ok(())
    }

    /// Everything needed to build a `ProfileResolver`
    /// (`crates/api/src/profiles.rs`): every profile as a `PolicyConfig`,
    /// plus every `(application_id, profile_id)` pair. Called at boot and
    /// after every mutating `/api/v1/*` call — see this module's doc
    /// comment.
    pub async fn load_all_policies(
        &self,
    ) -> Result<(Vec<PolicyConfig>, Vec<(String, String)>), PolicyStoreError> {
        let profile_rows: Vec<ProfileRow> = sqlx::query_as(
            "SELECT id, description, execution_mode, fail_mode, normalize, updated_at FROM profiles",
        )
        .fetch_all(&self.pool)
        .await?;

        let check_rows: Vec<CheckRow> = sqlx::query_as(
            "SELECT profile_id, category, enabled, on_fail, mode, fail_mode, options, \
                    strategy, backends, scorecard \
             FROM checks ORDER BY profile_id, category",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut checks_by_profile: HashMap<String, Vec<CheckRow>> = HashMap::new();
        for check in check_rows {
            checks_by_profile
                .entry(check.profile_id.clone())
                .or_default()
                .push(check);
        }

        let mut policies = Vec::with_capacity(profile_rows.len());
        for row in &profile_rows {
            let checks = checks_by_profile.remove(&row.id).unwrap_or_default();
            policies.push(assemble_policy(row, &checks)?);
        }

        let applications: Vec<(String, String)> =
            sqlx::query_as("SELECT application_id, profile_id FROM applications")
                .fetch_all(&self.pool)
                .await?;

        Ok((policies, applications))
    }
}
