//! Background retention sweep for TTL'd rows.
//!
//! `ARMOR_SESSION_TTL_SECONDS` and `ARMOR_VAULT_TTL_SECONDS` (`state.rs`)
//! only *set* `expires_at` on write — `sessions::touch`/`ensure` and
//! `vault::put` all do that already. Nothing previously read that column
//! back: `sessions::get`/`touch` don't filter by it, and no caller ever
//! invoked `sessions::purge_expired` or `vault::purge_expired` outside
//! tests, so an expired row sat in Postgres (and stayed readable) forever.
//! Configuring a retention window had no effect at all.
//!
//! This task closes that gap the same way `sessions::purge_expired`'s own
//! doc comment describes: periodically deleting rows whose `expires_at` has
//! passed. Deleting a session cascades (`ON DELETE CASCADE`) to its vault
//! entries, so the session sweep alone covers PII retention for sessions
//! with a TTL; the vault sweep additionally covers entries whose own TTL is
//! shorter than their session's.
//!
//! A no-op, not a stricter read-time filter: rows are still live (and
//! readable) between expiry and the next sweep, which matches every other
//! TTL-style cleanup in this codebase (`audit`'s spool rotation, `sync`'s
//! poll loop) — eventual, not transactional, retention.

use std::{sync::Arc, time::Duration};

use tokio::task::JoinHandle;

use armor_storage::{policy_store::PgPolicyStore, sessions, vault::Vault};

/// How often the sweep runs. Retention windows are measured in minutes to
/// days (`ARMOR_SESSION_TTL_SECONDS`/`ARMOR_VAULT_TTL_SECONDS`), so a fixed
/// five-minute cadence keeps expired rows around only briefly without
/// polling Postgres needlessly.
const SWEEP_INTERVAL: Duration = Duration::from_secs(300);

/// Periodic purge of expired `sessions`/`vault_entries` rows. Returned by
/// [`RetentionTask::spawn`]; call `stop()` during graceful shutdown.
pub struct RetentionTask {
    handle: JoinHandle<()>,
}

impl RetentionTask {
    /// Spawns the sweep loop. Returns immediately (does not block). When
    /// `db` is `None` — no control-plane database configured, so no
    /// `sessions`/`vault_entries` tables to sweep — the spawned task exits
    /// immediately and costs nothing.
    pub fn spawn(db: Option<Arc<PgPolicyStore>>, vault: Option<Arc<Vault>>) -> Self {
        let handle = tokio::spawn(async move {
            let Some(db) = db else {
                tracing::debug!("retention sweep disabled (no DATABASE_URL)");
                return;
            };
            let pool = db.pool().clone();

            tracing::info!(
                interval_secs = SWEEP_INTERVAL.as_secs(),
                "retention sweep task started"
            );

            loop {
                tokio::time::sleep(SWEEP_INTERVAL).await;

                match sessions::purge_expired(&pool, None).await {
                    Ok(count) if count > 0 => {
                        tracing::info!(count, "retention sweep: purged expired sessions");
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(error = %e, "retention sweep: session purge failed");
                    }
                }

                if let Some(vault) = &vault {
                    match vault.purge_expired(None).await {
                        Ok(count) if count > 0 => {
                            tracing::info!(count, "retention sweep: purged expired vault entries");
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!(error = %e, "retention sweep: vault purge failed");
                        }
                    }
                }
            }
        });

        Self { handle }
    }

    /// Cancels the sweep loop. Non-blocking.
    pub async fn stop(self) {
        self.handle.abort();
        let _ = self.handle.await;
    }
}
