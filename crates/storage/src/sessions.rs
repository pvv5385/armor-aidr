//! Multi-turn session/conversation state: the durable counters that
//! `armor_core::detectors::{abuse, unbounded_consumption}` need in order to
//! stay correct across replicas.
//!
//! # Why this exists
//!
//! Both detectors keep a process-global `Mutex<HashMap>` keyed by session
//! id. That is correct for a single instance and silently wrong the moment
//! a second replica appears behind a load balancer: each process counts
//! only the requests it personally handled, so a caller spreading a burst
//! across three replicas gets three times the intended budget. That is the
//! reason session state has to leave process memory. The mechanism is
//! unchanged — same fixed window, same lifetime budgets — only the storage
//! moves; this is a storage swap, not a redesign of the rate-limiting
//! mechanism.
//!
//! # Keeping I/O out of `armor-core`
//!
//! `armor-core` is synchronous and I/O-free by hard architectural rule, so
//! the detectors cannot call this module. The flow is inverted instead:
//! `armor-api` calls [`touch`] *before* the detector sweep and injects the
//! returned counters into `CheckOptions`, so the detector stays a pure
//! function of its inputs and simply reads a number somebody else looked
//! up. See `crates/api/src/aidr.rs`'s session-counter injection and each
//! detector's `session_*_count` option.
//!
//! # Tenancy — read this before running more than one tenant
//!
//! These tables key on `session_id` **alone**. `session_id` is
//! caller-supplied (`X-Armor-Session-Id`, minted by the gateway or the
//! customer's app as part of the header contract), which means that with
//! two tenants sharing one database, a caller who guesses or replays
//! another tenant's session id reads that tenant's counters and, far worse,
//! that tenant's [`crate::vault`] entries. A client-supplied session id is
//! therefore a cross-tenant read primitive against the vault.
//!
//! That is acceptable *only* because Armor has no tenant model at all today
//! — `middleware/auth.rs` treats an API key as a pass/fail credential with
//! no identity attached, so every deployment is single-tenant by
//! construction. **Before a tenant model lands, these tables must gain
//! `tenant_id` and key on `(tenant_id, session_id)`**, with `tenant_id`
//! resolved from the authenticated API key and never from the header or
//! body. This module is written so that closing that gap is an `ALTER
//! TABLE` plus an extra bind parameter, not a redesign.

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::postgres::PgPool;

/// Post-increment counter snapshot returned by [`touch`] — what the
/// detectors need to make a decision about *this* request, with this
/// request already counted.
#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct SessionCounters {
    /// Lifetime request count, including this request.
    pub request_count: i64,
    /// Lifetime token total, including this request's `estimated_tokens`.
    pub total_tokens: i64,
    /// Requests inside the current fixed rate window, including this one.
    pub window_request_count: i64,
    /// Start of the window `window_request_count` is counted against —
    /// returned so a caller can tell a fresh window from a continuing one.
    pub window_started_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Inputs to [`touch`].
#[derive(Debug, Clone, Copy)]
pub struct Touch<'a> {
    pub session_id: &'a str,
    /// Added to the lifetime token total. `armor-api` passes the caller's
    /// own count when they supplied one, falling back to the same
    /// `text.len() / 4` estimate `unbounded_consumption` uses, so the two
    /// paths never disagree about what a token is.
    pub estimated_tokens: i64,
    /// Width of the fixed rate window, matching `abuse`'s `window_seconds`
    /// option. A window that has fully elapsed is restarted by this call.
    pub window_seconds: f64,
    /// Retention. `Some(n)` sets `expires_at = now + n`, refreshed on every
    /// touch so an active session doesn't expire mid-conversation; `None`
    /// leaves the session with no expiry, and [`purge_expired`] will never
    /// collect it.
    pub ttl_seconds: Option<i64>,
    /// Clock override. `None` — the production path — uses the *database's*
    /// `now()`, which is the whole point: replicas with skewed clocks would
    /// otherwise disagree about where a rate window starts and hand a
    /// caller extra budget at every boundary. Tests pass `Some` to drive
    /// window rollover without sleeping.
    pub now: Option<DateTime<Utc>>,
}

/// Record one request against a session and return the resulting counters,
/// creating the session row if this is its first request.
///
/// Atomic by construction: the increment, the window-rollover decision, and
/// the read-back all happen inside one `INSERT ... ON CONFLICT DO UPDATE
/// ... RETURNING` statement, which takes a row lock for its duration. Two
/// replicas touching the same session concurrently therefore serialize —
/// neither can read a stale count and write back a number that loses the
/// other's increment, which is exactly the lost-update bug a
/// `SELECT`-then-`UPDATE` pair would have.
///
/// Note both `CASE` expressions read `sessions.window_started_at`, i.e. the
/// *existing* row's value, not the one being written in the same statement
/// — so the rollover test and the count it selects stay consistent with
/// each other.
pub async fn touch(pool: &PgPool, params: Touch<'_>) -> Result<SessionCounters, sqlx::Error> {
    sqlx::query_as::<_, SessionCounters>(
        r#"
        INSERT INTO sessions
            (session_id, created_at, last_seen_at, request_count, total_tokens,
             window_started_at, window_request_count, expires_at)
        VALUES
            ($1,
             COALESCE($4::timestamptz, now()),
             COALESCE($4::timestamptz, now()),
             1,
             $2,
             COALESCE($4::timestamptz, now()),
             1,
             CASE WHEN $5::bigint IS NULL THEN NULL
                  ELSE COALESCE($4::timestamptz, now()) + make_interval(secs => $5::bigint)
             END)
        ON CONFLICT (session_id) DO UPDATE SET
            last_seen_at = COALESCE($4::timestamptz, now()),
            request_count = sessions.request_count + 1,
            total_tokens = sessions.total_tokens + $2,
            window_started_at = CASE
                WHEN EXTRACT(EPOCH FROM (COALESCE($4::timestamptz, now()) - sessions.window_started_at)) >= $3
                THEN COALESCE($4::timestamptz, now())
                ELSE sessions.window_started_at
            END,
            window_request_count = CASE
                WHEN EXTRACT(EPOCH FROM (COALESCE($4::timestamptz, now()) - sessions.window_started_at)) >= $3
                THEN 1
                ELSE sessions.window_request_count + 1
            END,
            expires_at = CASE WHEN $5::bigint IS NULL THEN NULL
                              ELSE COALESCE($4::timestamptz, now()) + make_interval(secs => $5::bigint)
                         END
        RETURNING request_count, total_tokens, window_request_count,
                  window_started_at, created_at, last_seen_at, expires_at
        "#,
    )
    .bind(params.session_id)
    .bind(params.estimated_tokens)
    .bind(params.window_seconds)
    .bind(params.now)
    .bind(params.ttl_seconds)
    .fetch_one(pool)
    .await
}

/// Read a session's counters without recording a request against it.
///
/// Unused on the scan path (which always wants [`touch`]'s post-increment
/// view) — this is the primitive behind a future `GET /v1/sessions/{id}`
/// read endpoint, a plausible later addition but explicitly not part of
/// the header contract today.
pub async fn get(pool: &PgPool, session_id: &str) -> Result<Option<SessionCounters>, sqlx::Error> {
    sqlx::query_as::<_, SessionCounters>(
        "SELECT request_count, total_tokens, window_request_count, \
                window_started_at, created_at, last_seen_at, expires_at \
         FROM sessions WHERE session_id = $1",
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await
}

/// Create the session row if it doesn't exist, **without** recording a
/// request against it. Returns whether a row was created.
///
/// [`touch`] is the scan path's primitive and this is not a substitute for
/// it — the difference is the whole point. `vault_entries.session_id` is a
/// foreign key, so anything vaulting a value needs the session row to exist
/// first; but the scan that triggers it has usually already been counted by
/// [`touch`] (via `armor-api`'s `session_state::apply`), and counting it a
/// second time would inflate exactly the counters `abuse` and
/// `unbounded_consumption` enforce budgets against. Worse, it would do so
/// only for requests that happened to contain PII, which is an absurd
/// coupling to debug in production.
///
/// So this inserts the bare row — zero counters, window starting now — and
/// leaves an existing one completely untouched, `expires_at` included: a
/// session whose TTL is being refreshed on every [`touch`] must not have it
/// pinned back by a redaction pass.
pub async fn ensure(
    pool: &PgPool,
    session_id: &str,
    ttl_seconds: Option<i64>,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query(
        r#"
        INSERT INTO sessions (session_id, expires_at)
        VALUES ($1, CASE WHEN $2::bigint IS NULL THEN NULL
                         ELSE now() + make_interval(secs => $2::bigint)
                    END)
        ON CONFLICT (session_id) DO NOTHING
        "#,
    )
    .bind(session_id)
    .bind(ttl_seconds)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Delete every session whose `expires_at` has passed, returning how many
/// went. Sessions with no expiry are never collected.
///
/// `ON DELETE CASCADE` on `vault_entries.session_id` means this is also the
/// vault's retention sweep — an expired session takes its stored PII with
/// it, which is the behavior a retention policy has to have rather than
/// leaving orphaned ciphertext behind. [`crate::vault::purge_expired`]
/// covers the narrower case of vault entries that expire *before* their
/// session does.
pub async fn purge_expired(pool: &PgPool, now: Option<DateTime<Utc>>) -> Result<u64, sqlx::Error> {
    let result = sqlx::query(
        "DELETE FROM sessions \
         WHERE expires_at IS NOT NULL AND expires_at <= COALESCE($1::timestamptz, now())",
    )
    .bind(now)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

/// Right-to-erasure for one session: drops its counters and, via
/// `ON DELETE CASCADE`, every vault entry holding recoverable PII for it.
/// Returns whether a session actually existed.
///
/// Deliberately not exposed over HTTP. Erasure and deanonymization are
/// both privileged operations, and Armor has no RBAC model yet to gate who
/// is allowed to call them; publishing them as unauthenticated-by-default
/// control endpoints would be a worse outcome than making an operator run
/// them deliberately.
pub async fn erase(pool: &PgPool, session_id: &str) -> Result<bool, sqlx::Error> {
    let result = sqlx::query("DELETE FROM sessions WHERE session_id = $1")
        .bind(session_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{test_pool, unique_id};

    #[tokio::test]
    async fn first_touch_creates_the_session_and_counts_one_request() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess");

        let counters = touch(
            &pool,
            Touch {
                session_id: &session,
                estimated_tokens: 40,
                window_seconds: 60.0,
                ttl_seconds: None,
                now: None,
            },
        )
        .await
        .expect("touch");

        assert_eq!(counters.request_count, 1);
        assert_eq!(counters.total_tokens, 40);
        assert_eq!(counters.window_request_count, 1);
        assert!(counters.expires_at.is_none());
    }

    #[tokio::test]
    async fn repeated_touches_accumulate_lifetime_totals() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess");
        let params = Touch {
            session_id: &session,
            estimated_tokens: 25,
            window_seconds: 60.0,
            ttl_seconds: None,
            now: None,
        };

        touch(&pool, params).await.expect("touch 1");
        touch(&pool, params).await.expect("touch 2");
        let third = touch(&pool, params).await.expect("touch 3");

        assert_eq!(third.request_count, 3);
        assert_eq!(third.total_tokens, 75);
        assert_eq!(third.window_request_count, 3);
    }

    #[tokio::test]
    async fn window_rolls_over_once_it_has_fully_elapsed() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess");
        let start = Utc::now();
        let at = |offset_secs: i64| Touch {
            session_id: &session,
            estimated_tokens: 0,
            window_seconds: 10.0,
            ttl_seconds: None,
            now: Some(start + chrono::Duration::seconds(offset_secs)),
        };

        assert_eq!(touch(&pool, at(0)).await.unwrap().window_request_count, 1);
        assert_eq!(touch(&pool, at(5)).await.unwrap().window_request_count, 2);

        // Past the boundary: the window restarts and this request is its first.
        let rolled = touch(&pool, at(11)).await.unwrap();
        assert_eq!(rolled.window_request_count, 1);
        // ...but lifetime totals are untouched by a window rollover.
        assert_eq!(rolled.request_count, 3);
    }

    #[tokio::test]
    async fn distinct_sessions_do_not_share_counters() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let (a, b) = (unique_id("sess-a"), unique_id("sess-b"));
        fn params(id: &str) -> Touch<'_> {
            Touch {
                session_id: id,
                estimated_tokens: 10,
                window_seconds: 60.0,
                ttl_seconds: None,
                now: None,
            }
        }

        touch(&pool, params(&a)).await.unwrap();
        touch(&pool, params(&a)).await.unwrap();
        let b_counters = touch(&pool, params(&b)).await.unwrap();

        assert_eq!(b_counters.request_count, 1);
        assert_eq!(get(&pool, &a).await.unwrap().unwrap().request_count, 2);
    }

    #[tokio::test]
    async fn concurrent_touches_do_not_lose_increments() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess-race");

        // The reason `touch` is one statement rather than SELECT-then-UPDATE:
        // 32 concurrent touches must produce exactly 32, not "somewhere
        // between 1 and 32 depending on interleaving".
        let mut handles = Vec::new();
        for _ in 0..32 {
            let pool = pool.clone();
            let session = session.clone();
            handles.push(tokio::spawn(async move {
                touch(
                    &pool,
                    Touch {
                        session_id: &session,
                        estimated_tokens: 1,
                        window_seconds: 3600.0,
                        ttl_seconds: None,
                        now: None,
                    },
                )
                .await
                .expect("concurrent touch")
            }));
        }
        for handle in handles {
            handle.await.expect("join");
        }

        let final_counters = get(&pool, &session).await.unwrap().unwrap();
        assert_eq!(final_counters.request_count, 32);
        assert_eq!(final_counters.total_tokens, 32);
    }

    #[tokio::test]
    async fn ttl_sets_an_expiry_and_purge_collects_it() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess-ttl");
        let start = Utc::now();

        touch(
            &pool,
            Touch {
                session_id: &session,
                estimated_tokens: 0,
                window_seconds: 60.0,
                ttl_seconds: Some(30),
                now: Some(start),
            },
        )
        .await
        .unwrap();

        // Not yet expired.
        purge_expired(&pool, Some(start + chrono::Duration::seconds(10)))
            .await
            .unwrap();
        assert!(get(&pool, &session).await.unwrap().is_some());

        // Past the TTL.
        purge_expired(&pool, Some(start + chrono::Duration::seconds(31)))
            .await
            .unwrap();
        assert!(get(&pool, &session).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sessions_without_a_ttl_are_never_purged() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess-no-ttl");
        touch(
            &pool,
            Touch {
                session_id: &session,
                estimated_tokens: 0,
                window_seconds: 60.0,
                ttl_seconds: None,
                now: None,
            },
        )
        .await
        .unwrap();

        purge_expired(&pool, Some(Utc::now() + chrono::Duration::days(3650)))
            .await
            .unwrap();
        assert!(get(&pool, &session).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn erase_removes_the_session_and_reports_whether_it_existed() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess-erase");
        touch(
            &pool,
            Touch {
                session_id: &session,
                estimated_tokens: 0,
                window_seconds: 60.0,
                ttl_seconds: None,
                now: None,
            },
        )
        .await
        .unwrap();

        assert!(erase(&pool, &session).await.unwrap());
        assert!(get(&pool, &session).await.unwrap().is_none());
        assert!(!erase(&pool, &session).await.unwrap());
    }

    #[tokio::test]
    async fn ensure_creates_a_session_with_no_request_counted() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess-ensure");

        assert!(ensure(&pool, &session, None).await.unwrap());
        let counters = get(&pool, &session).await.unwrap().expect("row exists");
        assert_eq!(counters.request_count, 0);
        assert_eq!(counters.total_tokens, 0);
        assert_eq!(counters.window_request_count, 0);
    }

    #[tokio::test]
    async fn ensure_never_disturbs_an_existing_session() {
        let Some(pool) = test_pool().await else {
            return;
        };
        let session = unique_id("sess-ensure-existing");
        let touched = touch(
            &pool,
            Touch {
                session_id: &session,
                estimated_tokens: 40,
                window_seconds: 60.0,
                ttl_seconds: Some(3600),
                now: None,
            },
        )
        .await
        .unwrap();

        // The case the redaction path actually hits: the scan already
        // counted this request. A second count here would inflate the
        // budgets `unbounded_consumption` enforces, and pinning `expires_at`
        // back would shorten a live session's retention.
        assert!(!ensure(&pool, &session, Some(60)).await.unwrap());
        let after = get(&pool, &session).await.unwrap().expect("row exists");
        assert_eq!(after.request_count, 1);
        assert_eq!(after.total_tokens, 40);
        assert_eq!(after.expires_at, touched.expires_at);
    }
}
