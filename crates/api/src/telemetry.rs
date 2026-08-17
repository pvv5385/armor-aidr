//! Batched, fire-and-forget telemetry emitter. Collects metadata-only
//! [`EvaluationEvent`]s (the same type `audit.rs` durably spools) and ships
//! them to a control plane in batches. Never blocks evaluation: `emit` is a
//! bounded, drop-oldest buffer push; a flush failure is logged and the
//! batch is dropped, never retried.
//!
//! Off by default. If enabled without an endpoint configured, `new`
//! disables itself (logs a warning) rather than crashing or guessing a
//! destination — there is no baked-in default control-plane URL for this
//! project yet.

use std::{collections::VecDeque, sync::Mutex, time::Duration};

use tokio::task::JoinHandle;

use crate::audit::EvaluationEvent;

/// Ceiling on buffered-but-unsent events before the oldest is dropped — a
/// memory-pressure bound, not a behavior knob, so it's kept as a constant
/// rather than exposed through settings.
const MAX_BUFFER_SIZE: usize = 8_000;
const BATCH_SIZE: usize = 50;
const FLUSH_INTERVAL: Duration = Duration::from_secs(15);

pub struct TelemetryEmitter {
    endpoint: String,
    api_key: String,
    enabled: bool,
    buffer: Mutex<VecDeque<EvaluationEvent>>,
    client: reqwest::Client,
}

impl TelemetryEmitter {
    /// `enabled` is the operator's request; it's downgraded to `false` here
    /// (with a warning) if no endpoint was configured, so callers never
    /// need to re-check both fields.
    pub fn new(enabled: bool, endpoint: String, api_key: String) -> Self {
        let effective = enabled && !endpoint.trim().is_empty();
        if enabled && !effective {
            tracing::warn!(
                "ARMOR_TELEMETRY_ENABLED=true but ARMOR_TELEMETRY_URL is unset — telemetry stays disabled"
            );
        }
        Self {
            endpoint,
            api_key,
            enabled: effective,
            buffer: Mutex::new(VecDeque::new()),
            client: reqwest::Client::new(),
        }
    }

    #[cfg(test)]
    fn disabled() -> Self {
        Self::new(false, String::new(), String::new())
    }

    /// Non-blocking: pushes onto the in-process buffer. No-op when disabled.
    pub fn emit(&self, event: EvaluationEvent) {
        if !self.enabled {
            return;
        }
        let mut buf = self.buffer.lock().unwrap_or_else(|p| p.into_inner());
        if buf.len() >= MAX_BUFFER_SIZE {
            buf.pop_front();
        }
        buf.push_back(event);
    }

    /// Spawns the periodic flush loop. No-op (returns an already-finished
    /// task) when disabled.
    pub fn spawn(self: std::sync::Arc<Self>) -> JoinHandle<()> {
        tokio::spawn(async move {
            if !self.enabled {
                return;
            }
            tracing::info!("telemetry emitter started");
            loop {
                tokio::time::sleep(FLUSH_INTERVAL).await;
                self.flush().await;
            }
        })
    }

    /// Cancels the flush loop and sends whatever's left, best-effort.
    pub async fn stop(&self, handle: JoinHandle<()>) {
        handle.abort();
        let _ = handle.await;
        self.flush().await;
    }

    fn take_batch(&self) -> Vec<EvaluationEvent> {
        let mut buf = self.buffer.lock().unwrap_or_else(|p| p.into_inner());
        let mut batch = Vec::with_capacity(BATCH_SIZE.min(buf.len()));
        while batch.len() < BATCH_SIZE {
            match buf.pop_front() {
                Some(event) => batch.push(event),
                None => break,
            }
        }
        batch
    }

    async fn flush(&self) {
        let batch = self.take_batch();
        if batch.is_empty() {
            return;
        }

        let url = format!(
            "{}/telemetry/v1/ingest",
            self.endpoint.trim_end_matches('/')
        );
        let result = self
            .client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .json(&serde_json::json!({ "events": batch }))
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_client_error() || resp.status().is_server_error() => {
                tracing::warn!(status = %resp.status(), "telemetry send failed");
            }
            Ok(_) => tracing::debug!(count = batch.len(), "sent telemetry events"),
            // Events are lost — acceptable for fire-and-forget telemetry.
            Err(e) => tracing::debug!(error = %e, "telemetry flush failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: &str) -> EvaluationEvent {
        EvaluationEvent {
            scan_id: id.to_string(),
            session_id: "session-1".to_string(),
            client_request_id: None,
            occurred_at_unix_ms: 0,
            stage: "input".to_string(),
            verdict: "ALLOW".to_string(),
            checks: vec![],
            latency_ms: 0.0,
            application_id: None,
            profile_id: None,
            layers: None,
            model_version: None,
        }
    }

    #[test]
    fn disabled_emitter_never_buffers() {
        let emitter = TelemetryEmitter::disabled();
        emitter.emit(event("a"));
        assert!(emitter.buffer.lock().unwrap().is_empty());
    }

    #[test]
    fn enabled_without_endpoint_downgrades_to_disabled() {
        let emitter = TelemetryEmitter::new(true, String::new(), String::new());
        assert!(!emitter.enabled);
    }

    #[test]
    fn buffer_drops_oldest_once_full() {
        let emitter =
            TelemetryEmitter::new(true, "http://example.invalid".to_string(), String::new());
        {
            let mut buf = emitter.buffer.lock().unwrap();
            for i in 0..MAX_BUFFER_SIZE {
                buf.push_back(event(&i.to_string()));
            }
        }
        emitter.emit(event("newest"));

        let buf = emitter.buffer.lock().unwrap();
        assert_eq!(buf.len(), MAX_BUFFER_SIZE);
        assert_eq!(buf.front().unwrap().scan_id, "1");
        assert_eq!(buf.back().unwrap().scan_id, "newest");
    }
}
