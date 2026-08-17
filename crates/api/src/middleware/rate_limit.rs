//! Abuse/rate-limiting: a token-bucket counter keyed by client IP, on either
//! of two backends selected by `ARMOR_RATE_LIMIT_MODE` (default `none`):
//! in-process (`fixed`, single-instance, this file's `InProcessLimiter`) or
//! Redis (`redis`, shared across replicas — `redis_rate_limit::RedisLimiter`).
//! Both enforce the same `ARMOR_RATE_LIMIT_RPS`/`_BURST` semantics; only
//! where the bucket state lives differs.
//!
//! Client IP defaults to the TCP peer address (`ConnectInfo`). If the peer
//! matches an entry in `ARMOR_TRUSTED_PROXIES` (empty/default: nothing is
//! trusted), `X-Forwarded-For` is honored instead, via `resolve_client_ip`'s
//! rightmost-untrusted-hop rule — see its doc comment. This is a strict
//! opt-in allowlist: without a matching trusted-proxy entry, a direct
//! caller can never spoof this header to dodge its own bucket.

use lru::LruCache;
use std::{
    net::{IpAddr, SocketAddr},
    num::NonZeroUsize,
    sync::Mutex,
    time::Instant,
};

use axum::{
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use ipnet::IpNet;

use crate::{middleware::redis_rate_limit::RedisLimiter, state::AppState};

/// Resolves the client IP to rate-limit on. `peer` (the raw TCP peer
/// address) is used as-is unless it matches an entry in `trusted_proxies`,
/// in which case `X-Forwarded-For` is read right-to-left, skipping any
/// entry that is itself a trusted proxy, and the first entry that isn't is
/// returned — the "rightmost untrusted hop". This unwinds an arbitrary
/// chain of trusted proxies while never trusting an entry a non-proxy
/// caller could have injected into the header themselves. Falls back to
/// `peer` if the header is absent, unparseable, or every entry is a
/// trusted proxy.
fn resolve_client_ip(peer: IpAddr, headers: &HeaderMap, trusted_proxies: &[IpNet]) -> IpAddr {
    let is_trusted = |ip: &IpAddr| trusted_proxies.iter().any(|net| net.contains(ip));

    if trusted_proxies.is_empty() || !is_trusted(&peer) {
        return peer;
    }

    let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) else {
        return peer;
    };

    xff.rsplit(',')
        .map(str::trim)
        .filter_map(|s| s.parse::<IpAddr>().ok())
        .find(|ip| !is_trusted(ip))
        .unwrap_or(peer)
}

struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// The in-process token-bucket engine (`RateLimitMode::Fixed`) — one LRU of
/// buckets per running instance, so a multi-replica deployment enforces a
/// separate budget per replica rather than one shared budget. See
/// `redis_rate_limit::RedisLimiter` for the shared-budget alternative.
struct InProcessLimiter {
    capacity: f64,
    refill_per_sec: f64,
    buckets: Mutex<LruCache<IpAddr, Bucket>>,
}

impl InProcessLimiter {
    fn new(requests_per_sec: u32, burst: u32) -> Self {
        Self {
            capacity: burst.max(1) as f64,
            refill_per_sec: requests_per_sec.max(1) as f64,
            buckets: Mutex::new(LruCache::new(NonZeroUsize::new(100_000).unwrap())),
        }
    }

    /// `true` if the request may proceed, consuming a token; `false` if the
    /// caller is over budget right now.
    fn try_acquire(&self, ip: IpAddr) -> bool {
        self.try_acquire_at(ip, Instant::now())
    }

    fn try_acquire_at(&self, ip: IpAddr, now: Instant) -> bool {
        let mut buckets = self.buckets.lock().expect("rate limiter mutex poisoned");

        let bucket = match buckets.get_mut(&ip) {
            Some(b) => b,
            None => {
                buckets.put(
                    ip,
                    Bucket {
                        tokens: self.capacity,
                        last_refill: now,
                    },
                );
                buckets.get_mut(&ip).unwrap()
            }
        };

        let elapsed = now.saturating_duration_since(bucket.last_refill);
        bucket.tokens =
            (bucket.tokens + elapsed.as_secs_f64() * self.refill_per_sec).min(self.capacity);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

enum Backend {
    InProcess(InProcessLimiter),
    Redis(Box<RedisLimiter>),
}

/// Facade over the two rate-limit backends (`RateLimitMode::Fixed`/`Redis`)
/// — `middleware::enforce` and `AppState` only ever see this type, not which
/// backend is underneath.
pub struct RateLimiter {
    trusted_proxies: Vec<IpNet>,
    backend: Backend,
}

impl RateLimiter {
    pub fn in_process(requests_per_sec: u32, burst: u32, trusted_proxies: Vec<IpNet>) -> Self {
        Self {
            trusted_proxies,
            backend: Backend::InProcess(InProcessLimiter::new(requests_per_sec, burst)),
        }
    }

    /// Connects to Redis now (see `RedisLimiter::connect`'s doc comment on
    /// why this fails the boot rather than degrading silently).
    pub async fn redis(
        redis_url: &str,
        requests_per_sec: u32,
        burst: u32,
        key_prefix: String,
        trusted_proxies: Vec<IpNet>,
    ) -> anyhow::Result<Self> {
        let backend = RedisLimiter::connect(redis_url, requests_per_sec, burst, key_prefix).await?;
        Ok(Self {
            trusted_proxies,
            backend: Backend::Redis(Box::new(backend)),
        })
    }

    /// `true` if the request may proceed, consuming a token; `false` if the
    /// caller is over budget right now.
    async fn try_acquire(&self, ip: IpAddr) -> bool {
        match &self.backend {
            Backend::InProcess(limiter) => limiter.try_acquire(ip),
            Backend::Redis(limiter) => limiter.try_acquire(ip).await,
        }
    }

    #[cfg(test)]
    fn try_acquire_at(&self, ip: IpAddr, now: Instant) -> bool {
        match &self.backend {
            Backend::InProcess(limiter) => limiter.try_acquire_at(ip, now),
            Backend::Redis(_) => unreachable!("only the in-process backend supports injected time"),
        }
    }
}

pub async fn enforce(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(limiter) = state.rate_limiter.as_ref() else {
        return Ok(next.run(req).await);
    };

    let client_ip = resolve_client_ip(addr.ip(), req.headers(), &limiter.trusted_proxies);

    if limiter.try_acquire(client_ip).await {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::TOO_MANY_REQUESTS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn ip() -> IpAddr {
        "127.0.0.1".parse().unwrap()
    }

    #[test]
    fn allows_up_to_burst_capacity_then_blocks() {
        let limiter = RateLimiter::in_process(1, 3, Vec::new());
        let now = Instant::now();
        assert!(limiter.try_acquire_at(ip(), now));
        assert!(limiter.try_acquire_at(ip(), now));
        assert!(limiter.try_acquire_at(ip(), now));
        assert!(!limiter.try_acquire_at(ip(), now));
    }

    #[test]
    fn refills_over_time() {
        let limiter = RateLimiter::in_process(10, 1, Vec::new());
        let start = Instant::now();
        assert!(limiter.try_acquire_at(ip(), start));
        assert!(!limiter.try_acquire_at(ip(), start));

        let later = start + Duration::from_millis(200);
        assert!(limiter.try_acquire_at(ip(), later));
    }

    #[test]
    fn tracks_clients_independently() {
        let limiter = RateLimiter::in_process(1, 1, Vec::new());
        let now = Instant::now();
        let a: IpAddr = "127.0.0.1".parse().unwrap();
        let b: IpAddr = "127.0.0.2".parse().unwrap();
        assert!(limiter.try_acquire_at(a, now));
        assert!(!limiter.try_acquire_at(a, now));
        assert!(limiter.try_acquire_at(b, now));
    }

    fn headers_with_xff(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", value.parse().unwrap());
        headers
    }

    #[test]
    fn untrusted_peer_ignores_spoofed_header() {
        let peer: IpAddr = "203.0.113.9".parse().unwrap();
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let headers = headers_with_xff("198.51.100.1");

        // The peer isn't in trusted_proxies, so the header — which the peer
        // itself could have set — must never override its own address.
        assert_eq!(resolve_client_ip(peer, &headers, &trusted), peer);
    }

    #[test]
    fn no_trusted_proxies_configured_always_uses_peer() {
        let peer: IpAddr = "10.0.0.5".parse().unwrap();
        let headers = headers_with_xff("198.51.100.1");

        assert_eq!(resolve_client_ip(peer, &headers, &[]), peer);
    }

    #[test]
    fn trusted_proxy_header_is_honored() {
        let peer: IpAddr = "10.0.0.5".parse().unwrap();
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let headers = headers_with_xff("198.51.100.1");

        let real_client: IpAddr = "198.51.100.1".parse().unwrap();
        assert_eq!(resolve_client_ip(peer, &headers, &trusted), real_client);
    }

    #[test]
    fn multi_hop_chain_resolves_past_all_trusted_proxies() {
        let peer: IpAddr = "10.0.0.5".parse().unwrap(); // proxy2, our direct peer
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        // client, proxy1 — both appended by trusted hops before reaching us.
        let headers = headers_with_xff("198.51.100.1, 10.0.0.9");

        let real_client: IpAddr = "198.51.100.1".parse().unwrap();
        assert_eq!(resolve_client_ip(peer, &headers, &trusted), real_client);
    }

    #[test]
    fn all_hops_trusted_falls_back_to_peer() {
        let peer: IpAddr = "10.0.0.5".parse().unwrap();
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let headers = headers_with_xff("10.0.0.9, 10.0.0.10");

        assert_eq!(resolve_client_ip(peer, &headers, &trusted), peer);
    }

    #[test]
    fn malformed_header_from_trusted_proxy_falls_back_to_peer() {
        let peer: IpAddr = "10.0.0.5".parse().unwrap();
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let headers = headers_with_xff("not-an-ip");

        assert_eq!(resolve_client_ip(peer, &headers, &trusted), peer);
    }

    #[test]
    fn missing_header_from_trusted_proxy_falls_back_to_peer() {
        let peer: IpAddr = "10.0.0.5".parse().unwrap();
        let trusted: Vec<IpNet> = vec!["10.0.0.0/8".parse().unwrap()];

        assert_eq!(resolve_client_ip(peer, &HeaderMap::new(), &trusted), peer);
    }
}
