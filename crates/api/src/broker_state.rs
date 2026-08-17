//! Shared broker state, persisted to `<state_dir>/state.json`. Tracks a
//! random `broker_id` (stable across runs), first/last run timestamps, and
//! a run counter, so `heartbeat.rs` has something stable and anonymous to
//! report. Named for where this is heading: the enterprise edition grows
//! this into the full AIDR broker identity (session/behavior correlation
//! across requests), not just an install fingerprint.
//!
//! Best-effort: a read/write failure is logged and swallowed. Telemetry
//! bookkeeping must never fail startup.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerState {
    pub broker_id: String,
    pub first_run_at_unix_ms: u64,
    pub last_run_at_unix_ms: u64,
    #[serde(default)]
    pub run_count: u64,
}

impl BrokerState {
    fn newly_installed(now: u64) -> Self {
        Self {
            broker_id: uuid::Uuid::new_v4().to_string(),
            first_run_at_unix_ms: now,
            last_run_at_unix_ms: now,
            run_count: 0,
        }
    }

    fn mark_started(&mut self, now: u64) {
        self.run_count += 1;
        self.last_run_at_unix_ms = now;
    }
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn state_path(state_dir: &Path) -> PathBuf {
    state_dir.join("state.json")
}

fn load(state_dir: &Path) -> Option<BrokerState> {
    let path = state_path(state_dir);
    let contents = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str(&contents) {
        Ok(state) => Some(state),
        Err(e) => {
            tracing::debug!(error = %e, "could not parse broker state file, will recreate");
            None
        }
    }
}

fn save(state_dir: &Path, state: &BrokerState) {
    if let Err(e) = std::fs::create_dir_all(state_dir) {
        tracing::debug!(error = %e, "could not create state dir (non-fatal)");
        return;
    }
    let path = state_path(state_dir);
    match serde_json::to_string_pretty(state) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::debug!(error = %e, "could not save broker state (non-fatal)");
            }
        }
        Err(e) => tracing::debug!(error = %e, "could not serialize broker state (non-fatal)"),
    }
}

/// Loads (or creates) broker state, marks this process startup against it,
/// persists, and returns the updated state. Call once per process startup.
pub fn track_process_start(state_dir: &Path) -> BrokerState {
    let now = now_unix_ms();
    let mut state = match load(state_dir) {
        Some(existing) => existing,
        None => BrokerState::newly_installed(now),
    };
    state.mark_started(now);
    save(state_dir, &state);
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_start_persists_broker_id_and_increments_count() {
        let dir = tempfile::tempdir().unwrap();

        let first = track_process_start(dir.path());
        assert_eq!(first.run_count, 1);

        let second = track_process_start(dir.path());
        assert_eq!(second.run_count, 2);
        assert_eq!(second.broker_id, first.broker_id);
        assert_eq!(second.first_run_at_unix_ms, first.first_run_at_unix_ms);
    }

    #[test]
    fn missing_dir_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("does/not/exist/yet");
        let state = track_process_start(&nested);
        assert_eq!(state.run_count, 1);
    }
}
