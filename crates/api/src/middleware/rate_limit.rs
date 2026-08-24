//! Abuse/rate-limiting: a token-bucket counter keyed by client IP, on either
//! of two backends selected by `ARMOR_RATE_LIMIT_MODE` (default `none`):
//! in-process (`fixed`, single-instance, this file's `InProcessLimiter`) or
//! Redis (`redis`, shared across replicas — `redis_rate_limit::RedisLimiter`).
//! Both enforce the same `ARMOR_RATE_LIMIT_RPS`/`_BURST` semantics; only
//! where the bucket state lives differs.
//!
//! What a bucket is keyed on depends on whether auth is turned on — see
//! [`BucketKey`] and `resolve_bucket`. With `ARMOR_AUTH_MODE=api_key`, a
//! request carrying a *valid* key gets a bucket of its own; everything else
//! is bucketed by client IP, as it always was.
//!
//! Client IP defaults to the TCP peer address (`ConnectInfo`). If the peer
//! matches an entry in `ARMOR_TRUSTED_PROXIES` (empty/default: nothing is
//! trusted), `X-Forwarded-For` is honored instead, via `resolve_client_ip`'s
//! rightmost-untrusted-hop rule — see its doc comment. This is a strict
//! opt-in allowlist: without a matching trusted-proxy entry, a direct
//! caller can never spoof this header to dodge its own bucket.

use lru::LruCache;
use std::{
    fmt,
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

/// What one token bucket belongs to.
///
/// With auth off (the default) this is always [`BucketKey::Ip`] and behaviour
/// is exactly as it was. With `ARMOR_AUTH_MODE=api_key`, a request presenting
/// a valid key is bucketed on that key instead, which fixes the case where a
/// whole office behind one NAT shares a single budget while an attacker
/// spread across many source addresses gets a full budget each.
///
/// **Only a valid key earns a key-shaped bucket.** Bucketing on the presented
/// key without checking it first would be a rate-limit bypass: send a fresh
/// random key on every request and every request gets a brand-new full
/// bucket. Unauthenticated and wrong-key traffic therefore stays on the IP
/// bucket, where it cannot mint new buckets at will. `resolve_bucket` is the
/// only place this decision is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BucketKey {
    Ip(IpAddr),
    /// SHA-256 of a key that `middleware::auth::match_key` accepted.
    ApiKey([u8; 32]),
}

impl fmt::Display for BucketKey {
    /// The Redis key suffix (`redis_rate_limit`). Namespaced so an IP-shaped
    /// bucket can never collide with a key-shaped one.
    ///
    /// Only the first 8 bytes of the digest are emitted. That is far more than
    /// enough to tell one API key from another, and it means a Redis instance
    /// — which this design explicitly allows sharing with other data — never
    /// holds the full SHA-256 of a live credential for an offline attack to
    /// work against.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BucketKey::Ip(ip) => write!(f, "ip:{ip}"),
            BucketKey::ApiKey(hash) => {
                write!(f, "key:")?;
                for byte in &hash[..8] {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
        }
    }
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
    buckets: Mutex<LruCache<BucketKey, Bucket>>,
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
    fn try_acquire(&self, key: BucketKey) -> bool {
        self.try_acquire_at(key, Instant::now())
    }

    fn try_acquire_at(&self, key: BucketKey, now: Instant) -> bool {
        let mut buckets = self.buckets.lock().expect("rate limiter mutex poisoned");

        let bucket = match buckets.get_mut(&key) {
            Some(b) => b,
            None => {
                buckets.put(
                    key,
                    Bucket {
                        tokens: self.capacity,
                        last_refill: now,
                    },
                );
                buckets.get_mut(&key).unwrap()
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
    async fn try_acquire(&self, key: BucketKey) -> bool {
        match &self.backend {
            Backend::InProcess(limiter) => limiter.try_acquire(key),
            Backend::Redis(limiter) => limiter.try_acquire(key).await,
        }
    }

    #[cfg(test)]
    fn try_acquire_at(&self, key: BucketKey, now: Instant) -> bool {
        match &self.backend {
            Backend::InProcess(limiter) => limiter.try_acquire_at(key, now),
            Backend::Redis(_) => unreachable!("only the in-process backend supports injected time"),
        }
    }
}

/// Picks the bucket this request spends from — see [`BucketKey`] for why a
/// key-shaped bucket requires a *valid* key rather than merely a present one.
///
/// `state.api_keys` is `Some` exactly when `ARMOR_AUTH_MODE=api_key`
/// (`main.rs`), so its absence is the "auth is off" signal and every request
/// stays on its IP bucket, unchanged from before this existed.
///
/// Note this runs *outside* the auth middleware, which is layered inside it
/// (`routes::router`) so an over-budget caller is rejected before a full auth
/// pass. Validating the key here costs one SHA-256 and a constant-time scan —
/// the same work `require_api_key` would do — and it is the only way to tell
/// a caller who owns a budget from one who is merely claiming to.
fn resolve_bucket(
    state: &AppState,
    req: &Request,
    peer: IpAddr,
    trusted_proxies: &[IpNet],
) -> BucketKey {
    if let Some(keys) = state.api_keys.as_deref() {
        if let Some(hash) = crate::middleware::auth::extract_key(req)
            .and_then(|presented| crate::middleware::auth::match_key(keys, &presented))
        {
            return BucketKey::ApiKey(hash);
        }
    }
    BucketKey::Ip(resolve_client_ip(peer, req.headers(), trusted_proxies))
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

    let bucket = resolve_bucket(&state, &req, addr.ip(), &limiter.trusted_proxies);

    if limiter.try_acquire(bucket).await {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::TOO_MANY_REQUESTS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The default bucket the pre-auth-aware tests below have always used —
    /// now spelled as a `BucketKey` rather than a bare address.
    fn ip() -> BucketKey {
        BucketKey::Ip("127.0.0.1".parse().unwrap())
    }

    // ── Bucket selection (`resolve_bucket`) ────────────────────────────

    use axum::body::Body;

    fn hashes_of(keys: &[&str]) -> std::sync::Arc<Vec<[u8; 32]>> {
        use sha2::{Digest, Sha256};
        std::sync::Arc::new(
            keys.iter()
                .map(|k| {
                    let mut h = Sha256::new();
                    h.update(k.as_bytes());
                    h.finalize().into()
                })
                .collect(),
        )
    }

    fn req_with(header: Option<(&str, &str)>) -> Request {
        let mut b = axum::http::Request::builder();
        if let Some((n, v)) = header {
            b = b.header(n, v);
        }
        b.body(Body::empty()).unwrap()
    }

    fn peer() -> IpAddr {
        "203.0.113.7".parse().unwrap()
    }

    /// Auth off — `state.api_keys` is `None` — must behave exactly as before.
    #[test]
    fn without_auth_every_request_buckets_by_ip() {
        let state = crate::state::test_support::state_with_api_keys(None);
        let req = req_with(Some(("x-api-key", "anything")));
        assert_eq!(
            resolve_bucket(&state, &req, peer(), &[]),
            BucketKey::Ip(peer())
        );
    }

    #[test]
    fn a_valid_key_gets_its_own_bucket() {
        let state = crate::state::test_support::state_with_api_keys(Some(hashes_of(&["good"])));
        let req = req_with(Some(("x-api-key", "good")));
        assert!(matches!(
            resolve_bucket(&state, &req, peer(), &[]),
            BucketKey::ApiKey(_)
        ));
    }

    /// The whole point of item 9: two callers behind one NAT, each with their
    /// own key, must not share a budget.
    #[test]
    fn two_valid_keys_from_one_ip_get_separate_buckets() {
        let state =
            crate::state::test_support::state_with_api_keys(Some(hashes_of(&["alice", "bob"])));
        let a = resolve_bucket(&state, &req_with(Some(("x-api-key", "alice"))), peer(), &[]);
        let b = resolve_bucket(&state, &req_with(Some(("x-api-key", "bob"))), peer(), &[]);
        assert_ne!(a, b, "two keys from one address shared a bucket");
    }

    /// The bypass this design exists to prevent. An unrecognized key must not
    /// mint a fresh bucket — otherwise an attacker sends a new random key per
    /// request and never runs out of budget.
    #[test]
    fn an_invalid_key_falls_back_to_the_ip_bucket() {
        let state = crate::state::test_support::state_with_api_keys(Some(hashes_of(&["good"])));
        for attempt in ["wrong-1", "wrong-2", "wrong-3"] {
            assert_eq!(
                resolve_bucket(&state, &req_with(Some(("x-api-key", attempt))), peer(), &[]),
                BucketKey::Ip(peer()),
                "an unrecognized key minted its own bucket ({attempt})"
            );
        }
    }

    #[test]
    fn a_missing_key_falls_back_to_the_ip_bucket() {
        let state = crate::state::test_support::state_with_api_keys(Some(hashes_of(&["good"])));
        assert_eq!(
            resolve_bucket(&state, &req_with(None), peer(), &[]),
            BucketKey::Ip(peer())
        );
    }

    /// A key-shaped bucket must ignore the address entirely — that is what
    /// lets one key be spent from several hosts against one budget.
    #[test]
    fn a_valid_keys_bucket_does_not_depend_on_its_source_address() {
        let state = crate::state::test_support::state_with_api_keys(Some(hashes_of(&["good"])));
        let req = || req_with(Some(("x-api-key", "good")));
        assert_eq!(
            resolve_bucket(&state, &req(), "198.51.100.1".parse().unwrap(), &[]),
            resolve_bucket(&state, &req(), "198.51.100.2".parse().unwrap(), &[])
        );
    }

    /// The Redis key must namespace the two shapes apart and must not carry a
    /// whole credential digest.
    #[test]
    fn bucket_keys_render_namespaced_and_truncated() {
        assert_eq!(BucketKey::Ip(peer()).to_string(), "ip:203.0.113.7");

        let hash = [0xabu8; 32];
        let rendered = BucketKey::ApiKey(hash).to_string();
        assert_eq!(rendered, "key:abababababababab");
        assert_eq!(
            rendered.len(),
            "key:".len() + 16,
            "expected 8 bytes of digest"
        );
    }

    #[test]
    fn an_ip_bucket_and_a_key_bucket_never_collide() {
        let ip = BucketKey::Ip(peer()).to_string();
        let key = BucketKey::ApiKey([0u8; 32]).to_string();
        assert_ne!(ip, key);
        assert!(ip.starts_with("ip:") && key.starts_with("key:"));
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
        let a = BucketKey::Ip("127.0.0.1".parse().unwrap());
        let b = BucketKey::Ip("127.0.0.2".parse().unwrap());
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
