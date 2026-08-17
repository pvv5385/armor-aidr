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

    match extract_key(&req) {
        Some(key) => {
            use sha2::{Digest, Sha256};
            use subtle::ConstantTimeEq;

            let mut hasher = Sha256::new();
            hasher.update(key.as_bytes());
            let request_hash: [u8; 32] = hasher.finalize().into();

            let mut matched = false;
            for valid_hash in keys.iter() {
                if valid_hash.ct_eq(&request_hash).into() {
                    matched = true;
                }
            }

            if matched {
                Ok(next.run(req).await)
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

fn extract_key(req: &Request) -> Option<String> {
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
