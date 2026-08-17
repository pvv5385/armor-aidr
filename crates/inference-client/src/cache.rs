//! Client-side result cache for the sidecar hop.
//!
//! The sidecar has its own cache (it saves the forward pass); this one saves
//! the network hop entirely for repeated text — a secrets/PII scanner sees
//! the same sentence more than once, and the deterministic tier already made
//! the cheap decision that it wanted the second opinion.
//!
//! Two properties are load-bearing and both are deliberate:
//!
//! 1. **The key hashes the exact text.** No lowercasing, no whitespace
//!    collapsing, no normalization. `AKIAIOSFODNN7EXAMPLE` and its lowercase
//!    twin are different inputs to a secret scanner, and homoglyph-spaced
//!    variants are what an evasion test is made of. The lost hit rate is the
//!    price of the cache never being the layer that loses fidelity
//!    (`armor-core` already has explicit normalized views — the caller can
//!    opt into those by scoring a normalized view's text).
//! 2. **Only successes are cached.** An error — timeout, unavailable, 503 —
//!    carries no information worth reusing; caching it would turn a transient
//!    blip into a persistent verdict.
//!
//! This is a decorator over an [`InferenceTransport`] rather than state baked
//! into `HttpTransport`, so tests can wrap a [`MockTransport`] and the swap
//! stays behind the same trait.
//!
//! [`MockTransport`]: crate::MockTransport

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lru::LruCache;
use sha2::{Digest, Sha256};

use crate::contract::{InferRequest, InferResult};
use crate::transport::{InferError, InferenceTransport};

/// A bounded, content-addressed result cache in front of another transport.
pub struct CachingTransport {
    inner: Arc<dyn InferenceTransport>,
    cache: Mutex<LruCache<String, InferResult>>,
}

impl CachingTransport {
    pub fn new(inner: Arc<dyn InferenceTransport>, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(capacity.max(1)).expect("capacity >= 1"),
            )),
        }
    }
}

/// Feeds `bytes` into `hasher` preceded by its length, so concatenating two
/// fields with no framing at all (`"ab"` + `"c"` vs. `"a"` + `"bc"`) cannot
/// collapse two distinct requests onto the same digest.
fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// The exact-text content key, plus the task and (when present) the per-call
/// params, since a model asked for different params is a different question.
///
/// Hashed with SHA-256 rather than `DefaultHasher` (SipHash-1-3 with the
/// fixed all-zero keys `DefaultHasher::new()` always uses): SipHash's
/// collision resistance comes from its key being secret, and a public,
/// hardcoded key on a 64-bit output is well within reach of an offline
/// search — an attacker who can find *any* text that collides with a
/// previously-cached benign one gets that benign verdict handed back for
/// free (`get`, below), without ever exercising the actual model. SHA-256's
/// 256-bit output has no such search, and needs no secret key to get it.
fn content_key(task: &str, req: &InferRequest<'_>) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, task.as_bytes());
    // Presence is part of the identity: an absent field must not hash the
    // same as a present-but-empty one.
    match req.text {
        Some(text) => {
            hasher.update([1u8]);
            hash_field(&mut hasher, text.as_bytes());
        }
        None => hasher.update([0u8]),
    }
    match req.texts {
        Some(texts) => {
            hasher.update([1u8]);
            hash_field(&mut hasher, &(texts.len() as u64).to_le_bytes());
            for text in texts {
                hash_field(&mut hasher, text.as_bytes());
            }
        }
        None => hasher.update([0u8]),
    }
    match req.params {
        Some(params) => {
            hasher.update([1u8]);
            // A stable rendering of the params value — `to_string` of a JSON
            // value is deterministic for a given input order.
            let rendered = serde_json::to_string(params).unwrap_or_default();
            hash_field(&mut hasher, rendered.as_bytes());
        }
        None => hasher.update([0u8]),
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    format!("{task}\u{1f}{hex}")
}

#[async_trait]
impl InferenceTransport for CachingTransport {
    async fn infer(&self, task: &str, req: InferRequest<'_>) -> Result<InferResult, InferError> {
        let key = content_key(task, &req);
        if let Some(hit) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            return Ok(hit.clone());
        }
        let result = self.inner.infer(task, req).await;
        if let Ok(ok) = &result {
            self.cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .put(key, ok.clone());
        }
        result
    }

    async fn models(&self) -> Result<Vec<crate::contract::ModelInfo>, InferError> {
        self.inner.models().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::MlDecision;
    use crate::mock::{self, MockTransport};

    fn allow(confidence: f32) -> InferResult {
        mock::result(MlDecision::Allow, 0, Some(confidence))
    }

    fn block(confidence: f32) -> InferResult {
        mock::result(MlDecision::Block, 90, Some(confidence))
    }

    fn cache(inner: Arc<dyn InferenceTransport>, capacity: usize) -> CachingTransport {
        CachingTransport::new(inner, capacity)
    }

    #[tokio::test]
    async fn a_repeated_text_is_answered_from_cache_without_another_call() {
        let mock = Arc::new(MockTransport::new());
        mock.push_ok("prompt_injection", allow(0.9));
        let transport = cache(mock.clone(), 16);

        let first = transport
            .infer("prompt_injection", InferRequest::text("same text"))
            .await
            .unwrap();
        let second = transport
            .infer("prompt_injection", InferRequest::text("same text"))
            .await
            .unwrap();

        assert_eq!(first.decision, MlDecision::Allow);
        assert_eq!(second.decision, MlDecision::Allow);
        assert_eq!(
            mock.call_count(),
            1,
            "the second call must not reach the inner transport"
        );
    }

    #[tokio::test]
    async fn different_text_is_a_different_key() {
        let mock = Arc::new(MockTransport::new());
        mock.push_ok("prompt_injection", allow(0.9))
            .push_ok("prompt_injection", block(0.95));
        let transport = cache(mock.clone(), 16);

        let first = transport
            .infer("prompt_injection", InferRequest::text("a"))
            .await
            .unwrap();
        let second = transport
            .infer("prompt_injection", InferRequest::text("b"))
            .await
            .unwrap();

        assert_eq!(first.decision, MlDecision::Allow);
        assert_eq!(second.decision, MlDecision::Block);
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn the_key_does_not_normalize_the_text() {
        // The fidelity rule: case is a real input difference to a secret
        // scanner, so "AKIA..." and "akia..." must not share a verdict.
        let mock = Arc::new(MockTransport::new());
        mock.push_ok("prompt_injection", allow(0.9))
            .push_ok("prompt_injection", block(0.95));
        let transport = cache(mock.clone(), 16);

        let _ = transport
            .infer("prompt_injection", InferRequest::text("AKIAABC"))
            .await
            .unwrap();
        let _ = transport
            .infer("prompt_injection", InferRequest::text("akiaabc"))
            .await
            .unwrap();
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn errors_are_not_cached() {
        let mock = Arc::new(MockTransport::new());
        mock.push_err(
            "prompt_injection",
            InferError::Unavailable("down".to_string()),
        )
        .push_ok("prompt_injection", allow(0.9));
        let transport = cache(mock.clone(), 16);

        let first = transport
            .infer("prompt_injection", InferRequest::text("x"))
            .await;
        assert!(first.is_err());
        // A subsequent call with the same text must retry, not replay the error.
        let second = transport
            .infer("prompt_injection", InferRequest::text("x"))
            .await;
        assert!(second.is_ok());
        assert_eq!(mock.call_count(), 2);
    }

    #[test]
    fn the_key_does_not_collapse_under_naive_concatenation() {
        // Without a length prefix per field, `texts: ["ab", "c"]` and
        // `texts: ["a", "bc"]` would hash identically — the exact class of
        // collision that let an attacker force a cached "allow" verdict onto
        // different text than the one it was cached for.
        let a = vec!["ab".to_string(), "c".to_string()];
        let b = vec!["a".to_string(), "bc".to_string()];
        let req_a = InferRequest {
            texts: Some(&a),
            ..InferRequest::text("")
        };
        let req_b = InferRequest {
            texts: Some(&b),
            ..InferRequest::text("")
        };
        assert_ne!(
            content_key("prompt_injection", &req_a),
            content_key("prompt_injection", &req_b)
        );
    }

    #[test]
    fn an_absent_field_is_not_the_same_key_as_a_present_empty_one() {
        let absent = InferRequest {
            text: None,
            ..InferRequest::text("")
        };
        let present_empty = InferRequest::text("");
        assert_ne!(
            content_key("prompt_injection", &absent),
            content_key("prompt_injection", &present_empty)
        );
    }

    #[test]
    fn the_key_is_a_full_sha256_digest_not_a_64_bit_hash() {
        // The regression this guards: a 64-bit `DefaultHasher` digest is
        // small enough for an offline collision search; SHA-256's 64 hex
        // characters is the property that makes that search infeasible.
        let key = content_key("prompt_injection", &InferRequest::text("hello"));
        let digest = key.rsplit('\u{1f}').next().unwrap();
        assert_eq!(
            digest.len(),
            64,
            "expected a 256-bit hex digest, got {digest:?}"
        );
    }

    #[tokio::test]
    async fn params_are_part_of_the_key() {
        let mock = Arc::new(MockTransport::new());
        mock.push_ok("prompt_injection", allow(0.9))
            .push_ok("prompt_injection", block(0.95));
        let transport = cache(mock.clone(), 16);

        let params_a = serde_json::json!({"k": 1});
        let params_b = serde_json::json!({"k": 2});
        let _ = transport
            .infer(
                "prompt_injection",
                InferRequest::text("same").with_params(Some(&params_a)),
            )
            .await
            .unwrap();
        let _ = transport
            .infer(
                "prompt_injection",
                InferRequest::text("same").with_params(Some(&params_b)),
            )
            .await
            .unwrap();
        assert_eq!(mock.call_count(), 2);
    }
}
