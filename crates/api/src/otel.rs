//! OpenTelemetry wiring: OTLP export for traces, metrics, and logs.
//!
//! Each signal is independently enabled — see
//! `config::Settings::from_env`'s `otlp_signal_enabled` for the rule (a
//! signal turns on iff its own or the generic `OTEL_EXPORTER_OTLP_*_ENDPOINT`
//! is set). This module only builds what's enabled; a disabled signal costs
//! nothing at runtime (no exporter, no background export thread) rather than
//! being wired to a no-op destination.
//!
//! Endpoint, protocol, headers, and compression are read directly by
//! `opentelemetry-otlp`'s exporter builders from the standard `OTEL_*` env
//! vars (`SpanExporter::builder().build()` etc. resolve them internally) —
//! this module never re-parses them, so there's exactly one place that can
//! get the OTel env var spec wrong, and it isn't this file. That also means
//! any OTLP-speaking collector or vendor backend works by pointing the env
//! var at it; nothing here is vendor-specific.

use anyhow::Context;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::{
    logs::SdkLoggerProvider, metrics::SdkMeterProvider, trace::SdkTracerProvider, Resource,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::ObservabilityConfig;

/// Holds whichever providers were actually constructed. An OTLP batch
/// exporter buffers spans/logs in memory and flushes on a timer — without an
/// explicit `shutdown()` on exit, the last (sub-batch-interval) of data is
/// silently lost. `main` calls `shutdown()` after the server stops accepting
/// connections but before the process exits.
#[derive(Default)]
pub struct OtelGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
    logger_provider: Option<SdkLoggerProvider>,
}

impl OtelGuard {
    pub fn shutdown(&self) {
        if let Some(p) = &self.tracer_provider {
            if let Err(e) = p.shutdown() {
                eprintln!("otel: tracer provider shutdown failed: {e}");
            }
        }
        if let Some(p) = &self.meter_provider {
            if let Err(e) = p.shutdown() {
                eprintln!("otel: meter provider shutdown failed: {e}");
            }
        }
        if let Some(p) = &self.logger_provider {
            if let Err(e) = p.shutdown() {
                eprintln!("otel: logger provider shutdown failed: {e}");
            }
        }
    }
}

fn resource(service_name: &str) -> Resource {
    Resource::builder()
        .with_service_name(service_name.to_string())
        .build()
}

/// Builds the process-wide `tracing` subscriber (fmt + optional OTel trace
/// and log layers) and, if enabled, installs the global OTel meter provider
/// for `middleware::otel_metrics`. Must run before any `tracing::*!` call.
pub fn init(config: &ObservabilityConfig) -> anyhow::Result<OtelGuard> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer();

    let mut guard = OtelGuard::default();

    let trace_layer = if config.traces_enabled {
        let resource = resource(&config.service_name);
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .build()
            .context("building OTLP span exporter")?;
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource)
            .build();
        let tracer = provider.tracer("armor-api");
        guard.tracer_provider = Some(provider);
        Some(tracing_opentelemetry::layer().with_tracer(tracer))
    } else {
        None
    };

    let log_layer = if config.logs_enabled {
        let resource = resource(&config.service_name);
        let exporter = opentelemetry_otlp::LogExporter::builder()
            .build()
            .context("building OTLP log exporter")?;
        let provider = SdkLoggerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource)
            .build();
        let bridge =
            opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(&provider);
        guard.logger_provider = Some(provider);
        Some(bridge)
    } else {
        None
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(trace_layer)
        .with(log_layer)
        .init();

    if config.metrics_enabled {
        let resource = resource(&config.service_name);
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .build()
            .context("building OTLP metric exporter")?;

        // Counters/histograms already aggregate in-process for free (see
        // `middleware::otel_metrics`) — nothing is exported per request.
        // Only this periodic flush costs a network call; the interval is the
        // SDK's own default (60s), standard-overridable via
        // `OTEL_METRIC_EXPORT_INTERVAL`.
        let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter).build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(resource)
            .build();
        opentelemetry::global::set_meter_provider(provider.clone());
        guard.meter_provider = Some(provider);
    }

    Ok(guard)
}
