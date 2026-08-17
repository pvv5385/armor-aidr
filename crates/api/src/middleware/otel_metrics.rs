//! HTTP request metrics, recorded through the OTel global meter. When
//! `ARMOR_OTEL`/metrics export isn't enabled, `opentelemetry::global` hands
//! back a no-op meter, so this middleware is always safe to install — no
//! `Settings`-gated branch needed here, unlike auth/rate-limit/CORS.
//!
//! Instrument names follow the OTel HTTP semantic conventions
//! (`http.server.request.duration` etc.) so any OTLP backend renders them
//! without custom dashboards. Route label is the raw request path rather
//! than axum's `MatchedPath` — `MatchedPath` is only populated for
//! middleware applied with `Router::route_layer`, not the whole-router
//! `Router::layer` this is installed with, and the current route set
//! (`/api/v1/aidr/scan`, `/integrations/*/v1/aidr/scan`, `/healthz`,
//! `/readyz`) is small enough that raw path cardinality isn't a concern.

use std::{sync::OnceLock, time::Instant};

use axum::{extract::Request, middleware::Next, response::Response};
use opentelemetry::{
    global,
    metrics::{Counter, Histogram},
    KeyValue,
};

struct Instruments {
    requests: Counter<u64>,
    duration: Histogram<f64>,
}

fn instruments() -> &'static Instruments {
    static INSTRUMENTS: OnceLock<Instruments> = OnceLock::new();
    INSTRUMENTS.get_or_init(|| {
        let meter = global::meter("armor-api");
        Instruments {
            requests: meter
                .u64_counter("http.server.request.count")
                .with_description("Total HTTP requests handled")
                .build(),
            duration: meter
                .f64_histogram("http.server.request.duration")
                .with_description("HTTP request duration")
                .with_unit("s")
                .build(),
        }
    })
}

pub async fn record(req: Request, next: Next) -> Response {
    let method = req.method().to_string();
    let route = req.uri().path().to_string();

    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();

    let attrs = [
        KeyValue::new("http.request.method", method),
        KeyValue::new("http.route", route),
        KeyValue::new(
            "http.response.status_code",
            response.status().as_u16() as i64,
        ),
    ];
    let inst = instruments();
    inst.requests.add(1, &attrs);
    inst.duration.record(elapsed, &attrs);

    response
}
