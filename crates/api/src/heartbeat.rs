//! Anonymous broker heartbeat. Sends a daily ping with non-sensitive
//! metadata (broker_id, version, OS, detector count, eval count). No PII,
//! no request content, no tenant data.
//!
//! Strictly opt-in: off unless `ARMOR_HEARTBEAT_ENABLED=true`, and even
//! then inert (with a warning) unless `ARMOR_HEARTBEAT_URL` is set — there
//! is no baked-in default control-plane hostname for this project to phone
//! home to.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use serde::Serialize;
use tokio::task::JoinHandle;

/// Expressed as a product of its parts rather than the literal second count
/// so the "once a day" intent reads directly from the constant.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize)]
struct Payload<'a> {
    source: &'a str,
    event_type: &'a str,
    broker_id: &'a str,
    os: &'a str,
    arch: &'a str,
    version: &'a str,
    eval_count: u64,
    detector_count: usize,
}

pub struct Heartbeat {
    enabled: bool,
    endpoint: String,
    broker_id: String,
    detector_count: usize,
    eval_count: AtomicU64,
    client: reqwest::Client,
}

impl Heartbeat {
    /// `enabled` is the operator's request; downgraded to `false` (with a
    /// warning) if no endpoint was configured.
    pub fn new(enabled: bool, endpoint: String, broker_id: String, detector_count: usize) -> Self {
        let effective = enabled && !endpoint.trim().is_empty();
        if enabled && !effective {
            tracing::warn!(
                "ARMOR_HEARTBEAT_ENABLED=true but ARMOR_HEARTBEAT_URL is unset — heartbeat stays disabled"
            );
        }
        Self {
            enabled: effective,
            endpoint,
            broker_id,
            detector_count,
            eval_count: AtomicU64::new(0),
            client: reqwest::Client::new(),
        }
    }

    #[cfg(test)]
    fn disabled(broker_id: String, detector_count: usize) -> Self {
        Self::new(false, String::new(), broker_id, detector_count)
    }

    pub fn record_evaluation(&self) {
        self.eval_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Sends an immediate ping, then loops every 24h. No-op when disabled.
    pub fn spawn(self: std::sync::Arc<Self>) -> JoinHandle<()> {
        tokio::spawn(async move {
            if !self.enabled {
                return;
            }
            tracing::info!(broker_id = %short_id(&self.broker_id), "anonymous heartbeat enabled");
            self.send_ping("heartbeat").await;
            loop {
                tokio::time::sleep(HEARTBEAT_INTERVAL).await;
                self.send_ping("heartbeat").await;
            }
        })
    }

    pub async fn stop(&self, handle: JoinHandle<()>) {
        handle.abort();
        let _ = handle.await;
    }

    /// One-shot ping fired once, on a broker's actual first startup.
    pub async fn ping_on_install(&self) {
        if !self.enabled {
            return;
        }
        self.send_ping("first_run").await;
    }

    async fn send_ping(&self, event_type: &str) {
        let payload = Payload {
            source: "api",
            event_type,
            broker_id: &self.broker_id,
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            version: VERSION,
            eval_count: self.eval_count.load(Ordering::Relaxed),
            detector_count: self.detector_count,
        };

        let result = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_client_error() || resp.status().is_server_error() => {
                tracing::debug!(status = %resp.status(), "heartbeat response");
            }
            Ok(_) => tracing::debug!("heartbeat sent"),
            // Never fail the runtime because of telemetry.
            Err(e) => tracing::debug!(error = %e, "heartbeat failed (non-fatal)"),
        }
    }
}

fn short_id(id: &str) -> &str {
    &id[..id.len().min(8)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabled_without_endpoint_downgrades_to_disabled() {
        let hb = Heartbeat::new(true, String::new(), "id".to_string(), 3);
        assert!(!hb.enabled);
    }

    #[test]
    fn eval_count_increments() {
        let hb = Heartbeat::disabled("id".to_string(), 3);
        hb.record_evaluation();
        hb.record_evaluation();
        assert_eq!(hb.eval_count.load(Ordering::Relaxed), 2);
    }
}
