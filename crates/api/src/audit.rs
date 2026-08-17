//! Per-request decision log: rule hits, per-check verdicts, latency, final
//! action — metadata only, mirroring `aidr::run_scan`'s existing rule
//! ("category names and pass/fail, never the request body or a hit's
//! matched span").
//!
//! This durable local spool (`JsonlSpoolAuditSink`) is always on by default
//! — zero-DB deployments still get an audit trail. When `DATABASE_URL` is
//! set, `PgAuditSink` writes the same events to `armor_storage::audit_events`
//! (Postgres) too, fanned out via `MultiAuditSink` — see `build_audit_sink`.
//! `crates/api/src/control_plane.rs`'s `GET /api/v1/logs` reads the
//! Postgres copy back out for the management UI's "simple logging" view.
//!
//! Failure semantics: every sink here is written after the verdict is
//! already decided, so there's nothing left to deny on a write failure —
//! it's logged and swallowed rather than propagated (see `aidr::run_scan`'s
//! call site, which fires this off via `spawn_blocking` without awaiting
//! the result).

use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use armor_storage::{audit_events, policy_store::PgPolicyStore};
use serde::{Deserialize, Serialize};

/// One check's contribution to a decision — metadata only, no matched text
/// or span offsets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckSummary {
    pub category: String,
    pub passed: bool,
    pub action: String,
    pub severity: String,
}

/// Shared by both the audit sink and the telemetry emitter — built once per
/// request in `aidr::run_scan` rather than duplicated per consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationEvent {
    /// Armor's own per-request id, minted once in `aidr::run_scan` and
    /// reused as `ScanResponse.scan_id` — authoritative, always present.
    pub scan_id: String,
    /// Ties every event for one conversation together. Always populated by
    /// the time this event is built (`routes::resolve_session_id`
    /// self-mints one when the caller didn't supply `X-Armor-Session-Id`),
    /// so there's no null-session case to handle downstream.
    pub session_id: String,
    /// Caller-supplied correlation id (`metadata.request_id`, or an
    /// adapter's vendor-native equivalent) — `None` when the caller didn't
    /// supply one. `#[serde(default)]` so a spool file written before this
    /// field existed still parses.
    #[serde(default)]
    pub client_request_id: Option<String>,
    pub occurred_at_unix_ms: u64,
    pub stage: String,
    pub verdict: String,
    pub checks: Vec<CheckSummary>,
    pub latency_ms: f64,
    /// Which application/profile resolved this request (`aidr::run_scan`).
    /// `None` for the default profile / an absent `application_id`, same
    /// as `profiles::ProfileResolver::resolve`. `#[serde(default)]` for
    /// the same backward-compat reason as `client_request_id` above.
    #[serde(default)]
    pub application_id: Option<String>,
    #[serde(default)]
    pub profile_id: Option<String>,
    /// Per-check layer trace: which layer (deterministic, ML) produced each
    /// check's verdict, whether it was selected, and its model version.
    /// `None` when no escalation ran (deterministic-only path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<LayerSummary>>,
    /// Model version of the selected layer, when an ML layer ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSummary {
    pub layer: String,
    pub passed: bool,
    pub selected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Records `EvaluationEvent`s. Synchronous — `armor-core` and its callers
/// keep I/O off the hot async path by design; the one implementation that
/// does I/O (`JsonlSpoolAuditSink`) is always called from `spawn_blocking`.
pub trait AuditSink: Send + Sync {
    fn record(&self, event: &EvaluationEvent) -> io::Result<()>;
}

pub struct DiscardAuditSink;

impl AuditSink for DiscardAuditSink {
    fn record(&self, _event: &EvaluationEvent) -> io::Result<()> {
        Ok(())
    }
}

/// Durable append-only JSON-lines spool. Each `record` serializes the event,
/// then performs a lock-protected write-flush-fsync sequence so a recorded
/// event survives a crash even under concurrent callers.
pub struct JsonlSpoolAuditSink {
    path: PathBuf,
    lock: Mutex<()>,
    max_size_bytes: u64,
}

impl JsonlSpoolAuditSink {
    pub fn new(path: PathBuf, max_size_bytes: u64) -> Self {
        Self {
            path,
            lock: Mutex::new(()),
            max_size_bytes,
        }
    }

    /// Every spooled event, in file order. A single corrupt/partial line
    /// (e.g. an interrupted write) is logged and skipped rather than
    /// aborting the read. Unused outside tests today; this is the
    /// entrypoint a future drain job (the control plane reading the spool
    /// into Postgres) will call.
    #[allow(dead_code)]
    pub fn load_events(&self) -> io::Result<Vec<EvaluationEvent>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let contents = std::fs::read_to_string(&self.path)?;
        let mut out = Vec::new();
        for raw_line in contents.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<EvaluationEvent>(trimmed) {
                Ok(event) => out.push(event),
                Err(e) => tracing::warn!(error = %e, "skipping unparseable audit spool line"),
            }
        }
        Ok(out)
    }

    fn append_line(&self, line: &str) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        if let Ok(metadata) = std::fs::metadata(&self.path) {
            if metadata.len() > self.max_size_bytes {
                let rotated_path = self.path.with_extension("spool.1");
                let _ = std::fs::rename(&self.path, &rotated_path);
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()
    }
}

impl AuditSink for JsonlSpoolAuditSink {
    fn record(&self, event: &EvaluationEvent) -> io::Result<()> {
        let line = serde_json::to_string(event)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.append_line(&line)
    }
}

/// Postgres-backed twin of `JsonlSpoolAuditSink` — writes the same
/// `EvaluationEvent` to `armor_storage::audit_events` (the `evaluation_logs`
/// table `control_plane.rs`'s `GET /api/v1/logs` reads back out). Only
/// constructed when `AppState.db` is `Some` (`build_audit_sink`).
pub struct PgAuditSink {
    store: Arc<PgPolicyStore>,
}

impl PgAuditSink {
    pub fn new(store: Arc<PgPolicyStore>) -> Self {
        Self { store }
    }
}

impl AuditSink for PgAuditSink {
    /// `record` is a sync trait method (see the module doc: every sink is
    /// invoked from inside `spawn_blocking`, a blocking-pool thread, not a
    /// Tokio worker), but `armor_storage::audit_events::insert` is async
    /// `sqlx` I/O. `Handle::current().block_on(..)` from inside a
    /// `spawn_blocking` closure is Tokio's own documented pattern for
    /// bridging sync code into async I/O — safe here specifically because
    /// the caller never invokes this off a Tokio worker thread.
    fn record(&self, event: &EvaluationEvent) -> io::Result<()> {
        let checks = serde_json::to_value(&event.checks)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let layers = event
            .layers
            .as_ref()
            .map(|l| serde_json::to_value(l).unwrap_or(serde_json::Value::Null));
        let entry = audit_events::NewEvaluationLog {
            scan_id: &event.scan_id,
            session_id: &event.session_id,
            client_request_id: event.client_request_id.as_deref(),
            application_id: event.application_id.as_deref(),
            profile_id: event.profile_id.as_deref(),
            occurred_at_unix_ms: event.occurred_at_unix_ms as i64,
            stage: &event.stage,
            verdict: &event.verdict,
            checks,
            latency_ms: event.latency_ms,
            layers,
            model_version: event.model_version.as_deref(),
        };

        tokio::runtime::Handle::current()
            .block_on(audit_events::insert(self.store.pool(), &entry))
            .map_err(|e| io::Error::other(e.to_string()))
    }
}

/// Fans one `record` call out to every sink in `Vec`, logging (not
/// propagating) each sink's individual failure — one sink being briefly
/// down (e.g. Postgres) shouldn't stop the durable local spool from still
/// getting the event, or vice versa.
pub struct MultiAuditSink(pub Vec<Box<dyn AuditSink>>);

impl AuditSink for MultiAuditSink {
    fn record(&self, event: &EvaluationEvent) -> io::Result<()> {
        for sink in &self.0 {
            if let Err(e) = sink.record(event) {
                tracing::warn!(error = %e, "audit sink failed to record event");
            }
        }
        Ok(())
    }
}

/// `db` is `Some` when `DATABASE_URL` is configured (`main.rs`) — the
/// returned sink then fans out to both the local spool/noop sink
/// (`config.mode`) and Postgres, via `MultiAuditSink`.
pub fn build_audit_sink(
    config: &crate::config::AuditConfig,
    db: Option<Arc<PgPolicyStore>>,
) -> Box<dyn AuditSink> {
    use crate::config::AuditSinkMode;
    let primary: Box<dyn AuditSink> = match config.mode {
        AuditSinkMode::Noop => Box::new(DiscardAuditSink),
        AuditSinkMode::Spool => Box::new(JsonlSpoolAuditSink::new(
            PathBuf::from(&config.spool_path),
            config.max_size_bytes,
        )),
    };
    match db {
        Some(store) => Box::new(MultiAuditSink(vec![
            primary,
            Box::new(PgAuditSink::new(store)),
        ])),
        None => primary,
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
            checks: vec![CheckSummary {
                category: "secrets".to_string(),
                passed: true,
                action: "log".to_string(),
                severity: "low".to_string(),
            }],
            latency_ms: 1.5,
            application_id: None,
            profile_id: None,
            layers: None,
            model_version: None,
        }
    }

    #[test]
    fn spool_round_trips_events() {
        let dir = tempfile::tempdir().unwrap();
        let sink = JsonlSpoolAuditSink::new(dir.path().join("audit.spool"), 1024 * 1024);
        sink.record(&event("a")).unwrap();
        sink.record(&event("b")).unwrap();

        let events = sink.load_events().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].scan_id, "a");
        assert_eq!(events[1].scan_id, "b");
    }

    #[test]
    fn spool_skips_corrupt_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.spool");
        let sink = JsonlSpoolAuditSink::new(path.clone(), 1024 * 1024);
        sink.record(&event("a")).unwrap();

        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(file, "not valid json").unwrap();
        sink.record(&event("b")).unwrap();

        let events = sink.load_events().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].scan_id, "a");
        assert_eq!(events[1].scan_id, "b");
    }

    #[test]
    fn missing_spool_file_reads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let sink = JsonlSpoolAuditSink::new(dir.path().join("does-not-exist.spool"), 1024 * 1024);
        assert!(sink.load_events().unwrap().is_empty());
    }

    #[test]
    fn noop_sink_records_nothing_but_never_errors() {
        let sink = DiscardAuditSink;
        assert!(sink.record(&event("a")).is_ok());
    }
}
