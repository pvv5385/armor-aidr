//! Baseline security response headers, applied to every response
//! unconditionally (unlike auth/rate-limit, there's no `Settings` mode to
//! turn this off — it's cheap and there's no reason a deployment would want
//! it disabled). CORS decides *who* a browser lets read a response; these
//! decide how any client is allowed to treat it once it has it: no framing,
//! no MIME sniffing, no referrer leakage, HSTS once there's a real TLS
//! deployment behind this binary. `/ui` is a real HTML/JS surface rendered
//! in a browser (`crates/api/ui/`), so a strict same-origin CSP applies
//! here too — the data-plane JSON API ignores it, since browsers only
//! enforce CSP against a page they've navigated to, not a JSON response.

use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};

use crate::config::Environment;

pub async fn apply(environment: Environment, req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();

    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
    headers.insert(
        "referrer-policy",
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    headers.insert(
        "permissions-policy",
        HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
    );
    headers.insert("cache-control", HeaderValue::from_static("no-store"));
    headers.insert(
        "content-security-policy",
        HeaderValue::from_static("default-src 'self'; frame-ancestors 'none'"),
    );

    // HSTS only once ARMOR_ENV=production — asserting it in dev just breaks
    // the next plain-http://localhost run in browsers that cache the header.
    if environment == Environment::Production {
        headers.insert(
            "strict-transport-security",
            HeaderValue::from_static("max-age=63072000; includeSubDomains"),
        );
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, middleware::from_fn, routing::get, Router};
    use tower::ServiceExt;

    async fn app(environment: Environment) -> Router {
        Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(from_fn(move |req: Request, next: Next| async move {
                apply(environment, req, next).await
            }))
    }

    #[tokio::test]
    async fn sets_baseline_headers() {
        let response = app(Environment::Development)
            .await
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let headers = response.headers();
        assert_eq!(headers.get("x-content-type-options").unwrap(), "nosniff");
        assert_eq!(headers.get("x-frame-options").unwrap(), "DENY");
        assert!(headers.get("strict-transport-security").is_none());
    }

    #[tokio::test]
    async fn adds_hsts_only_in_production() {
        let response = app(Environment::Production)
            .await
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert!(response
            .headers()
            .get("strict-transport-security")
            .is_some());
    }
}
