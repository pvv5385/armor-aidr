//! Shared test scaffolding for the modules in this crate that talk to a
//! real database ([`crate::sessions`], [`crate::vault`]).
//!
//! These tests run against an actual Postgres rather than a mock, because
//! the things most worth testing here are database behaviors, not Rust
//! ones: that `ON CONFLICT DO UPDATE ... RETURNING` really does serialize
//! concurrent increments, that a unique index really does collapse a race
//! into one placeholder, that `ON DELETE CASCADE` really does take vault
//! rows with the session. A mock would assert that our own mock behaves the
//! way we assumed Postgres behaves, which is precisely the assumption under
//! test.
//!
//! Point `ARMOR_TEST_DATABASE_URL` at a scratch database and they run;
//! leave it unset and they skip with a notice. CI sets it (see
//! `.github/workflows/ci.yml`'s `postgres` service), so "skipped" is a
//! local-convenience state, not the state the merge gate runs in.

use std::sync::atomic::{AtomicBool, Ordering};

use sqlx::postgres::{PgPool, PgPoolOptions};
use uuid::Uuid;

/// Whether migrations have already been applied by some test in this
/// process. Only an optimization — `sqlx` migrations are idempotent and
/// take an advisory lock, so losing this race just means running the
/// (cheap, no-op) check twice.
static MIGRATED: AtomicBool = AtomicBool::new(false);

/// A migrated pool, or `None` when `ARMOR_TEST_DATABASE_URL` isn't set.
///
/// One pool **per test**, deliberately not shared: `#[tokio::test]` builds
/// a fresh runtime per test and tears it down at the end, and a pool
/// outliving the runtime that created it fails every later test with
/// "a Tokio 1.x context was found, but it is being shutdown". Small pools
/// keep the total connection count under Postgres's default limit when the
/// harness runs tests in parallel.
pub async fn test_pool() -> Option<PgPool> {
    let url = match std::env::var("ARMOR_TEST_DATABASE_URL") {
        Ok(url) if !url.trim().is_empty() => url,
        _ => {
            eprintln!(
                "SKIPPING database tests: ARMOR_TEST_DATABASE_URL is not set. \
                 Set it to a scratch Postgres database to run them."
            );
            return None;
        }
    };

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
        .expect("connecting to ARMOR_TEST_DATABASE_URL");

    if !MIGRATED.load(Ordering::Relaxed) {
        crate::policy_store::run_migrations(&pool)
            .await
            .expect("running migrations against the test database");
        MIGRATED.store(true, Ordering::Relaxed);
    }

    Some(pool)
}

/// A collision-free id, so tests sharing one database never contend over a
/// fixed session id and can run in parallel (which is the default).
pub fn unique_id(prefix: &str) -> String {
    format!("{prefix}-{}", Uuid::new_v4())
}
