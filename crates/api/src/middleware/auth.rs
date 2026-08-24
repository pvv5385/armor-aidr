//! API key auth. Configurable via `ARMOR_AUTH_MODE` (default `none`); when
//! set to `api_key`, requests to protected routes must present a key from
//! `ARMOR_API_KEYS` via `Authorization: Bearer <key>` or `X-API-Key`.
//!
//! `tenant_id` must be bound to the authenticated API key, never to request
//! content — not wired up yet since there's no tenant model here;
//! each key is currently just a pass/fail credential.

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};

use crate::state::AppState;

pub async fn require_api_key(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(keys) = state.api_keys.as_ref() else {
        return Ok(next.run(req).await);
    };

    match extract_key(&req).and_then(|key| match_key(keys, &key)) {
        Some(_) => Ok(next.run(req).await),
        None => Err(StatusCode::UNAUTHORIZED),
    }
}

/// The SHA-256 of `presented` if it is one of `keys`, otherwise `None`.
///
/// Shared with `middleware::rate_limit`, which keys a caller's token bucket
/// on the returned hash so an authenticated caller gets a budget of their own
/// rather than sharing one with everyone behind the same NAT. That sharing is
/// the point of factoring this out: if the rate limiter decided "is this a
/// valid key" by any other rule than the one enforced here, a caller could be
/// bucketed as authenticated and then rejected as unauthenticated, or worse,
/// get a fresh bucket for a key auth would refuse.
///
/// Constant-time and non-short-circuiting: every stored hash is compared even
/// after a match, so the time this takes says nothing about *which* key
/// matched or how many were checked. The final branch reveals only whether
/// some key matched, which the response status makes public anyway.
pub(crate) fn match_key(keys: &[[u8; 32]], presented: &str) -> Option<[u8; 32]> {
    use sha2::{Digest, Sha256};
    use subtle::ConstantTimeEq;

    let mut hasher = Sha256::new();
    hasher.update(presented.as_bytes());
    let request_hash: [u8; 32] = hasher.finalize().into();

    let mut matched = 0u8;
    for valid_hash in keys {
        matched |= valid_hash.ct_eq(&request_hash).unwrap_u8();
    }

    (matched == 1).then_some(request_hash)
}

pub(crate) fn extract_key(req: &Request) -> Option<String> {
    if let Some(value) = req.headers().get("x-api-key") {
        return value.to_str().ok().map(str::to_string);
    }

    req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    fn request_with_header(name: &str, value: &str) -> Request {
        axum::http::Request::builder()
            .header(name, value)
            .body(Body::empty())
            .unwrap()
    }

    #[test]
    fn extracts_key_from_x_api_key_header() {
        let req = request_with_header("x-api-key", "secret-1");
        assert_eq!(extract_key(&req), Some("secret-1".to_string()));
    }

    #[test]
    fn extracts_key_from_bearer_authorization_header() {
        let req = request_with_header("authorization", "Bearer secret-2");
        assert_eq!(extract_key(&req), Some("secret-2".to_string()));
    }

    #[test]
    fn ignores_non_bearer_authorization_header() {
        let req = request_with_header("authorization", "Basic dXNlcjpwYXNz");
        assert_eq!(extract_key(&req), None);
    }

    #[test]
    fn no_headers_yields_no_key() {
        let req = axum::http::Request::builder().body(Body::empty()).unwrap();
        assert_eq!(extract_key(&req), None);
    }
}
