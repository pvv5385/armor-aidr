//! Exercises `RedisLimiter` against a real Redis instance, verifying the
//! token-bucket Lua script actually enforces capacity/refill correctly
//! end-to-end (the in-process backend's equivalent logic is unit-tested
//! directly in `middleware::rate_limit::tests`; this is the Redis-script
//! analogue).
//!
//! Requires `ARMOR_TEST_REDIS_URL` (same naming convention as
//! `ARMOR_TEST_DATABASE_URL` in `vault_redaction_integration.rs`); skips
//! with a notice when it isn't set — e.g.
//! `docker run --rm -p 16399:6379 redis:7-alpine` and
//! `ARMOR_TEST_REDIS_URL=redis://127.0.0.1:16399 cargo test -p armor-api --test redis_rate_limit_integration`.

use std::net::IpAddr;

use armor_api::middleware::redis_rate_limit::RedisLimiter;

fn test_redis_url() -> Option<String> {
    std::env::var("ARMOR_TEST_REDIS_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
}

fn ip(s: &str) -> IpAddr {
    s.parse().unwrap()
}

macro_rules! skip_without_redis {
    () => {
        match test_redis_url() {
            Some(url) => url,
            None => {
                eprintln!(
                    "SKIPPING redis-rate-limit integration test: ARMOR_TEST_REDIS_URL is not set."
                );
                return;
            }
        }
    };
}

#[tokio::test]
async fn allows_up_to_burst_then_blocks() {
    let url = skip_without_redis!();
    let limiter =
        RedisLimiter::connect(&url, 1, 3, format!("armor:test:{}:", uuid::Uuid::new_v4()))
            .await
            .expect("connect to test redis");

    let client = ip("203.0.113.10");
    assert!(limiter.try_acquire(client).await);
    assert!(limiter.try_acquire(client).await);
    assert!(limiter.try_acquire(client).await);
    assert!(!limiter.try_acquire(client).await);
}

#[tokio::test]
async fn refills_over_time() {
    let url = skip_without_redis!();
    let limiter =
        RedisLimiter::connect(&url, 10, 1, format!("armor:test:{}:", uuid::Uuid::new_v4()))
            .await
            .expect("connect to test redis");

    let client = ip("203.0.113.11");
    assert!(limiter.try_acquire(client).await);
    assert!(!limiter.try_acquire(client).await);

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(limiter.try_acquire(client).await);
}

#[tokio::test]
async fn tracks_clients_independently() {
    let url = skip_without_redis!();
    let limiter =
        RedisLimiter::connect(&url, 1, 1, format!("armor:test:{}:", uuid::Uuid::new_v4()))
            .await
            .expect("connect to test redis");

    let a = ip("203.0.113.12");
    let b = ip("203.0.113.13");
    assert!(limiter.try_acquire(a).await);
    assert!(!limiter.try_acquire(a).await);
    assert!(limiter.try_acquire(b).await);
}

#[tokio::test]
async fn malformed_url_fails_to_connect() {
    // No network involved (this is a pure `Client::open` parse failure), so
    // it doesn't need `ARMOR_TEST_REDIS_URL` / a live server — unlike the
    // tests above, it verifies the "bad ARMOR_REDIS_URL fails the boot"
    // contract documented on `RedisLimiter::connect` without paying for a
    // real (and, for an unreachable host, potentially slow) connection
    // attempt.
    let result = RedisLimiter::connect("not-a-redis-url", 10, 10, "armor:test:".to_string()).await;
    assert!(result.is_err());
}
