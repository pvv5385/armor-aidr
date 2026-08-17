//! Redis-backed half of `rate_limit::RateLimiter` — the same token-bucket
//! algorithm as the in-process backend, but the bucket state lives in Redis
//! behind an atomic Lua script instead of an in-process `LruCache`, so every
//! replica behind a load balancer shares one limit per client instead of
//! each enforcing its own independently.
//!
//! Bucket state is a Redis hash (`{tokens, ts}`) per client key, refilled
//! lazily on each call using Redis's own `TIME` command rather than a
//! timestamp supplied by the caller — replicas can otherwise disagree on
//! "now" by however far their clocks have drifted, which would make the
//! shared bucket only as accurate as the least-synced instance. `EXPIRE` on
//! every call bounds memory to active clients: an idle bucket lives only
//! long enough to fully refill, then Redis reclaims it on its own.
//!
//! A Redis outage fails **open** (the request is allowed, with a `warn!`
//! log) rather than closed — the same degrade-gracefully posture as the
//! `armor-inference` sidecar's circuit breaker (`main::wire_inference`'s doc
//! comment): a dependency that only ever narrows what's allowed should never
//! itself become the reason every request fails.

use std::net::IpAddr;

use redis::{aio::ConnectionManager, Client, Script};

/// Atomic token-bucket check-and-decrement. `KEYS[1]` is the per-client key;
/// `ARGV[1]`/`ARGV[2]` are capacity/refill-per-sec; `ARGV[3]` is the TTL (in
/// whole seconds) to set on the key, long enough for a full refill from
/// empty. Uses Redis's own clock (`TIME`) so every replica calling this
/// script agrees on elapsed time regardless of its own clock skew.
const TOKEN_BUCKET_SCRIPT: &str = r#"
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local time = redis.call('TIME')
local now_ms = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
    ts = now_ms
end

local elapsed_ms = math.max(0, now_ms - ts)
tokens = math.min(capacity, tokens + (elapsed_ms / 1000.0) * refill_per_sec)

local allowed = 0
if tokens >= 1.0 then
    tokens = tokens - 1.0
    allowed = 1
end

redis.call('HMSET', key, 'tokens', tostring(tokens), 'ts', tostring(now_ms))
redis.call('EXPIRE', key, ttl)

return allowed
"#;

pub struct RedisLimiter {
    conn: ConnectionManager,
    script: Script,
    capacity: f64,
    refill_per_sec: f64,
    ttl_secs: i64,
    key_prefix: String,
}

impl RedisLimiter {
    /// Opens (lazily — `Client::open` just parses the URL) and connects
    /// (`ConnectionManager::new` does the actual handshake, then
    /// auto-reconnects on drops for the life of the process) to `redis_url`.
    /// Fails the boot on a bad URL or an unreachable server at startup time,
    /// same fail-fast posture as `wire_inference`'s `HttpTransport::connect`
    /// — a `ARMOR_REDIS_URL` that's set but wrong should stop the deploy,
    /// not silently run unlimited.
    pub async fn connect(
        redis_url: &str,
        requests_per_sec: u32,
        burst: u32,
        key_prefix: String,
    ) -> anyhow::Result<Self> {
        let client = Client::open(redis_url)?;
        let conn = ConnectionManager::new(client).await?;

        let capacity = burst.max(1) as f64;
        let refill_per_sec = requests_per_sec.max(1) as f64;
        // Time (in whole seconds, rounded up) to refill an empty bucket to
        // capacity, plus a one-second margin — an idle client's key outlives
        // its own refill window by at most a second before Redis reclaims it.
        let ttl_secs = (capacity / refill_per_sec).ceil() as i64 + 1;

        Ok(Self {
            conn,
            script: Script::new(TOKEN_BUCKET_SCRIPT),
            capacity,
            refill_per_sec,
            ttl_secs,
            key_prefix,
        })
    }

    /// `true` if the request may proceed, consuming a token; `false` if the
    /// caller is over budget right now. Fails open (returns `true`) on any
    /// Redis error — see module doc comment.
    pub async fn try_acquire(&self, ip: IpAddr) -> bool {
        let key = format!("{}{ip}", self.key_prefix);
        let mut conn = self.conn.clone();

        let result: redis::RedisResult<i64> = self
            .script
            .key(key)
            .arg(self.capacity)
            .arg(self.refill_per_sec)
            .arg(self.ttl_secs)
            .invoke_async(&mut conn)
            .await;

        match result {
            Ok(allowed) => allowed == 1,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "redis rate limiter unreachable; failing open (request allowed)"
                );
                true
            }
        }
    }
}
