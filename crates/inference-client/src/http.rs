//! [`InferenceTransport`] over HTTP/JSON, chosen over gRPC for the sidecar
//! hop: `reqwest` is already a direct dependency of `armor-api`, the sidecar
//! is a FastAPI service, and what actually mattered for this call — INT8
//! quantization, the sequence-length cap, request batching — lives in the
//! sidecar's model-serving path and is transport-independent, so gRPC would
//! have bought nothing there. `GrpcTransport` becomes a swap behind the same
//! trait if the many-to-many pool ever needs client-side load balancing.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::Instant;

use crate::breaker::CircuitBreaker;
use crate::contract::{InferRequest, InferResult, ModelInfo};
use crate::net_guard::{resolve_endpoint, EndpointError};
use crate::transport::{InferError, InferenceTransport};

#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// The deadline for **the whole call**, retries included. This is the
    /// number a policy's `timeout_ms` maps onto, and it has to bound the
    /// sequence rather than each attempt: a per-attempt deadline silently
    /// multiplies by the retry count, so a policy asking for 120ms would get
    /// 360ms and blow the escalation budget it was written to fit inside.
    pub timeout: Duration,
    /// Extra attempts after the first. Zero is a reasonable production value
    /// when the deadline is tight.
    pub max_retries: u32,
    /// Base backoff, doubled per retry. Kept small because the whole budget
    /// is typically ~120ms — this is jitter against a connection blip, not a
    /// wait for a service to restart.
    pub retry_backoff: Duration,
    /// Sent as `Authorization: Bearer` when set.
    pub auth_token: Option<String>,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(120),
            max_retries: 1,
            retry_backoff: Duration::from_millis(10),
            auth_token: None,
        }
    }
}

pub struct HttpTransport {
    client: reqwest::Client,
    base: reqwest::Url,
    config: HttpConfig,
    /// Shared with whatever else talks to this endpoint. `None` disables
    /// breaking entirely, which is what the tests of the retry logic want.
    breaker: Option<Arc<CircuitBreaker>>,
}

impl std::fmt::Debug for HttpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never derive: `HttpConfig` carries the bearer token.
        f.debug_struct("HttpTransport")
            .field("base", &self.base.as_str())
            .field("timeout", &self.config.timeout)
            .finish_non_exhaustive()
    }
}

impl HttpTransport {
    /// Validate and resolve `base_url`, then build a client pinned to the
    /// addresses that passed.
    ///
    /// Resolution happens here, once, at startup — not per request. Two
    /// consequences worth knowing: a bad endpoint is rejected immediately by
    /// this call rather than surfacing lazily on the first request, and DNS
    /// cannot rebind the target between the check and the connection (see
    /// [`crate::net_guard`]). What a caller does with that `Err` — fail the
    /// process, or log it and run without this transport — is up to them;
    /// `armor-api`'s own caller (`main::wire_inference`) chooses the latter.
    ///
    /// The flip side is that a sidecar whose address legitimately changes —
    /// a Kubernetes Service whose ClusterIP is recreated — needs a restart to
    /// be followed. That is the right trade for a fixed sidecar and the wrong
    /// one for a rotating pool; revisit it if the pool becomes dynamic.
    pub async fn connect(
        base_url: &str,
        config: HttpConfig,
        breaker: Option<Arc<CircuitBreaker>>,
    ) -> Result<Self, EndpointError> {
        let (base, addrs) = resolve_endpoint(base_url).await?;
        let host = base.host_str().unwrap_or_default().to_string();

        let client = reqwest::Client::builder()
            // Pin the checked addresses. reqwest still sends the original
            // Host header and SNI, so virtual hosting and TLS keep working.
            .resolve_to_addrs(&host, &addrs)
            // Connections are reused across escalations; without a pool the
            // TCP+TLS handshake alone would eat most of a 120ms budget.
            .pool_idle_timeout(Duration::from_secs(90))
            // `resolve_to_addrs` only pins *this* host. reqwest's default
            // policy follows up to 10 redirects and re-resolves each `Location`
            // through ordinary DNS, so a malicious or compromised sidecar could
            // otherwise redirect the client straight past `net_guard` to an
            // internal address (the cloud metadata service, say) — the exact
            // rebinding this module's resolve-once-and-pin design exists to
            // close. The sidecar is a fixed JSON API; it has no legitimate
            // reason to redirect, so refusing to follow one costs nothing.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| EndpointError::Malformed(base_url.to_string(), e.to_string()))?;

        Ok(Self {
            client,
            base,
            config,
            breaker,
        })
    }

    /// Build a transport over an existing client, skipping resolution.
    /// For tests that point at a local server on an already-known address.
    pub fn with_client(client: reqwest::Client, base: reqwest::Url, config: HttpConfig) -> Self {
        Self {
            client,
            base,
            config,
            breaker: None,
        }
    }

    pub fn with_breaker(mut self, breaker: Arc<CircuitBreaker>) -> Self {
        self.breaker = Some(breaker);
        self
    }

    fn url(&self, path: &str) -> Result<reqwest::Url, InferError> {
        self.base
            .join(path)
            .map_err(|e| InferError::Malformed(format!("bad path {path:?}: {e}")))
    }

    /// Run `attempt` under the breaker and the whole-call deadline.
    ///
    /// The deadline is computed once and every attempt is bounded by what
    /// remains of it, so N retries cannot exceed the caller's timeout.
    async fn call<F, Fut, T>(&self, attempt: F) -> Result<T, InferError>
    where
        F: Fn(reqwest::Client, Duration) -> Fut,
        Fut: std::future::Future<Output = Result<T, InferError>>,
    {
        if let Some(breaker) = &self.breaker {
            if !breaker.allow() {
                // Never made the call, so it says nothing about the endpoint
                // and must not be reported back as a failure.
                return Err(InferError::CircuitOpen);
            }
        }

        let deadline = Instant::now() + self.config.timeout;
        let mut last: InferError;
        let mut tries = 0u32;

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                last = InferError::Timeout {
                    elapsed_ms: self.config.timeout.as_millis() as u64,
                };
                break;
            }

            match attempt(self.client.clone(), remaining).await {
                Ok(value) => {
                    if let Some(breaker) = &self.breaker {
                        breaker.on_success();
                    }
                    return Ok(value);
                }
                Err(err) => {
                    // Only retry what a retry could plausibly fix. A `Status`
                    // is a considered answer from a service that is up —
                    // retrying a 429 from the saturation guard adds load to
                    // an overloaded pool, and retrying a 503 re-asks a
                    // question already answered. A `Malformed` body will be
                    // malformed again.
                    let retryable = matches!(err, InferError::Unavailable(_));
                    last = err;
                    if !retryable || tries >= self.config.max_retries {
                        break;
                    }
                    let backoff = self.config.retry_backoff * 2u32.pow(tries.min(8));
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if backoff >= remaining {
                        // Sleeping would consume the rest of the budget and
                        // leave no time for the attempt it precedes.
                        break;
                    }
                    tokio::time::sleep(backoff).await;
                    tries += 1;
                }
            }
        }

        if let Some(breaker) = &self.breaker {
            if last.is_breaker_signal() {
                breaker.on_failure();
            }
        }
        Err(last)
    }
}

/// Turn a transport-level `reqwest` failure into the typed error the caller
/// branches on. String-matching a message is how this goes wrong silently,
/// so the discrimination happens once, here.
fn transport_error(err: reqwest::Error, budget: Duration) -> InferError {
    if err.is_timeout() {
        return InferError::Timeout {
            elapsed_ms: budget.as_millis() as u64,
        };
    }
    if err.is_decode() || err.is_body() {
        return InferError::Malformed(err.to_string());
    }
    // Connect, DNS, pool exhaustion, connection reset — the pool could not be
    // reached, which is the one signal that says something about its health.
    InferError::Unavailable(err.to_string())
}

/// Map a non-2xx response onto the typed error.
async fn status_error(task: Option<&str>, response: reqwest::Response) -> InferError {
    let status = response.status().as_u16();
    // The sidecar puts the reason in `detail` (FastAPI's convention). Keep it
    // — "task 'toxicity' unavailable: sha256 mismatch" is the whole diagnosis,
    // and dropping it means an operator has to go read the sidecar's logs.
    let body = response.text().await.unwrap_or_default();
    let message = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("detail").and_then(|d| d.as_str()).map(str::to_string))
        .unwrap_or_else(|| body.chars().take(200).collect());

    match status {
        // 404: not in the registry. 503: in the registry, could not load —
        // missing artifact, digest mismatch, absent runner. Both mean "this
        // task cannot score for you", which is one decision for the caller.
        404 | 503 => InferError::UnknownTask(task.unwrap_or("").to_string()),
        _ => InferError::Status { status, message },
    }
}

#[async_trait]
impl InferenceTransport for HttpTransport {
    async fn infer(&self, task: &str, req: InferRequest<'_>) -> Result<InferResult, InferError> {
        let url = self.url(&format!("v1/infer/{task}"))?;
        // Serialize once rather than per attempt: the body is identical
        // across retries and `InferRequest` borrows its text anyway.
        let body = serde_json::to_vec(&req)
            .map_err(|e| InferError::Malformed(format!("could not encode request: {e}")))?;

        self.call(|client, remaining| {
            let url = url.clone();
            let body = body.clone();
            let token = self.config.auth_token.clone();
            async move {
                let mut builder = client
                    .post(url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .timeout(remaining)
                    .body(body);
                if let Some(token) = token {
                    builder = builder.bearer_auth(token);
                }
                let response = builder
                    .send()
                    .await
                    .map_err(|e| transport_error(e, remaining))?;
                if !response.status().is_success() {
                    return Err(status_error(Some(task), response).await);
                }
                // Out-of-range values clamp during deserialization (see
                // `contract`); reaching this arm means the shape itself is
                // wrong.
                response
                    .json::<InferResult>()
                    .await
                    .map_err(|e| InferError::Malformed(e.to_string()))
            }
        })
        .await
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, InferError> {
        #[derive(serde::Deserialize)]
        struct ModelsResponse {
            models: Vec<ModelInfo>,
        }

        let url = self.url("v1/models")?;
        self.call(|client, remaining| {
            let url = url.clone();
            let token = self.config.auth_token.clone();
            async move {
                let mut builder = client.get(url).timeout(remaining);
                if let Some(token) = token {
                    builder = builder.bearer_auth(token);
                }
                let response = builder
                    .send()
                    .await
                    .map_err(|e| transport_error(e, remaining))?;
                if !response.status().is_success() {
                    return Err(status_error(None, response).await);
                }
                response
                    .json::<ModelsResponse>()
                    .await
                    .map(|r| r.models)
                    .map_err(|e| InferError::Malformed(e.to_string()))
            }
        })
        .await
    }
}
