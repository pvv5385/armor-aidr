//! A circuit breaker for the sidecar hop.
//!
//! # Why this is a type and not a field on the transport
//!
//! Breaker state has to outlive any single call site, or it never
//! accumulates: construct-per-call means the count resets before it can
//! reach the threshold, so it reads like a breaker and behaves like a no-op.
//!
//! So the state is a shareable value here. One [`CircuitBreaker`] per
//! endpoint, held in `AppState`, cloned into whatever needs it, outliving
//! every request. That is the only arrangement in which "five failures in a
//! row" is a statement about the endpoint rather than about one request.
//!
//! # What counts as a failure
//!
//! Only [`InferError::is_breaker_signal`]. A timeout does not: it is the
//! caller's own deadline expiring, and tripping on it means a run of slow
//! requests takes out a healthy pool — the failure mode where the breaker
//! causes the outage it exists to contain.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Breaker configuration. The defaults are tuned for a sidecar on the same
/// network: fail fast, recover fast, because the fallback is a fully working
/// deterministic tier rather than an error page.
#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Consecutive breaker-signal failures before opening.
    pub failure_threshold: u32,
    /// How long to stay open before allowing a probe.
    pub cooldown: Duration,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            cooldown: Duration::from_secs(10),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Calls pass through.
    Closed,
    /// Calls are refused without being made.
    Open,
    /// One probe is in flight; everything else is still refused.
    HalfOpen,
}

#[derive(Debug)]
enum Inner {
    Closed,
    Open {
        until: Instant,
    },
    /// A probe has been handed out and we are waiting to hear how it went.
    /// Tracked explicitly so a burst during the cooldown sends exactly one
    /// call, not one per caller that happens to arrive after the deadline.
    ///
    /// `deadline` bounds how long we wait: a timed-out probe reports neither
    /// success nor failure (module doc above), so without a deadline
    /// `Probing` would stay stuck forever. Once `deadline` passes, `allow()`/
    /// `state()` treat the probe as failed and reopen with a fresh cooldown.
    Probing {
        deadline: Instant,
    },
}

/// Shareable breaker state for one endpoint.
#[derive(Debug)]
pub struct CircuitBreaker {
    config: BreakerConfig,
    inner: Mutex<Inner>,
    consecutive_failures: AtomicU32,
    /// Cumulative counters for `/metrics` and for tests. Not part of the
    /// decision — just what happened.
    pub trips: AtomicU32,
    pub rejected: AtomicU32,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(BreakerConfig::default())
    }
}

impl CircuitBreaker {
    pub fn new(config: BreakerConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner::Closed),
            consecutive_failures: AtomicU32::new(0),
            trips: AtomicU32::new(0),
            rejected: AtomicU32::new(0),
        }
    }

    /// Ask permission to make a call.
    ///
    /// `true` means go ahead, and the caller **must** report back through
    /// [`Self::on_success`] or [`Self::on_failure`] — a probe that is never
    /// reported leaves the breaker half-open until the next cooldown check.
    pub fn allow(&self) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match &*inner {
            Inner::Closed => true,
            Inner::Probing { deadline } => {
                if Instant::now() >= *deadline {
                    // The probe never reported back — neither `on_success`
                    // nor `on_failure` fired for it (a timeout is not a
                    // breaker signal, by design). Treat it as a failed probe
                    // rather than leaving `Probing` stuck forever.
                    *inner = Inner::Open {
                        until: Instant::now() + self.config.cooldown,
                    };
                    self.trips.fetch_add(1, Ordering::Relaxed);
                    self.rejected.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                // A probe is already out. Everyone else waits rather than
                // all piling onto a pool that just failed.
                self.rejected.fetch_add(1, Ordering::Relaxed);
                false
            }
            Inner::Open { until } => {
                if Instant::now() >= *until {
                    *inner = Inner::Probing {
                        deadline: Instant::now() + self.config.cooldown,
                    };
                    true
                } else {
                    self.rejected.fetch_add(1, Ordering::Relaxed);
                    false
                }
            }
        }
    }

    pub fn on_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *inner = Inner::Closed;
    }

    /// Report a failure that [`InferError::is_breaker_signal`] said counts.
    ///
    /// [`InferError::is_breaker_signal`]: crate::transport::InferError::is_breaker_signal
    pub fn on_failure(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        // A failed probe re-opens immediately, without waiting to re-reach
        // the threshold: the pool just told us it is still unhealthy, and
        // counting to five again would send four more doomed calls.
        if matches!(*inner, Inner::Probing { .. }) {
            *inner = Inner::Open {
                until: Instant::now() + self.config.cooldown,
            };
            self.trips.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if failures >= self.config.failure_threshold {
            *inner = Inner::Open {
                until: Instant::now() + self.config.cooldown,
            };
            self.trips.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn state(&self) -> State {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match &*inner {
            Inner::Closed => State::Closed,
            Inner::Probing { deadline } => {
                // Same staleness check as `allow()`, so a caller polling
                // `state()` alone (e.g. `/metrics`) doesn't see `HalfOpen`
                // forever either. Reports only — like the `Open` arm below,
                // `state()` never consumes the single-probe slot itself;
                // `allow()` is what actually reopens it on the next call.
                if Instant::now() >= *deadline {
                    State::Open
                } else {
                    State::HalfOpen
                }
            }
            Inner::Open { until } => {
                if Instant::now() >= *until {
                    // Report what a caller would actually get, not what the
                    // stored enum says — the cooldown has elapsed, so the
                    // next `allow()` returns true.
                    *inner = Inner::Open { until: *until };
                    State::HalfOpen
                } else {
                    State::Open
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breaker(threshold: u32, cooldown_ms: u64) -> CircuitBreaker {
        CircuitBreaker::new(BreakerConfig {
            failure_threshold: threshold,
            cooldown: Duration::from_millis(cooldown_ms),
        })
    }

    #[test]
    fn it_starts_closed_and_passes_calls() {
        let b = breaker(3, 50);
        assert_eq!(b.state(), State::Closed);
        assert!(b.allow());
    }

    #[test]
    fn it_opens_after_consecutive_failures() {
        let b = breaker(3, 10_000);
        b.on_failure();
        b.on_failure();
        assert_eq!(b.state(), State::Closed, "still under the threshold");
        b.on_failure();
        assert_eq!(b.state(), State::Open);
        assert!(!b.allow());
        assert_eq!(b.trips.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_success_resets_the_run() {
        let b = breaker(3, 10_000);
        b.on_failure();
        b.on_failure();
        b.on_success();
        b.on_failure();
        b.on_failure();
        assert_eq!(
            b.state(),
            State::Closed,
            "the threshold counts CONSECUTIVE failures; an intervening \
             success means the pool is serving and the run is over"
        );
    }

    #[test]
    fn state_persists_across_callers() {
        // Two call sites sharing one breaker accumulate one run of failures
        // between them — the property a per-call-site breaker would lose.
        let b = std::sync::Arc::new(breaker(2, 10_000));
        let a = std::sync::Arc::clone(&b);
        a.on_failure();
        b.on_failure();
        assert_eq!(b.state(), State::Open);
        assert!(!a.allow());
    }

    #[test]
    fn after_the_cooldown_exactly_one_probe_is_admitted() {
        let b = breaker(1, 0);
        b.on_failure();
        assert!(b.allow(), "cooldown elapsed, so one probe goes through");
        assert!(
            !b.allow(),
            "a probe is already in flight — a burst must not all be admitted"
        );
        assert_eq!(b.state(), State::HalfOpen);
    }

    #[test]
    fn a_probe_that_never_reports_does_not_wedge_the_breaker_forever() {
        // Simulates a probe that timed out: `InferError::is_breaker_signal`
        // is false for a timeout (module doc above), so `http.rs::call()`
        // never calls `on_success`/`on_failure` for it. The deadline on
        // `Probing` is the only thing that can still notice and recover.
        let b = breaker(1, 30);
        b.on_failure();
        std::thread::sleep(Duration::from_millis(40));
        assert!(b.allow(), "cooldown elapsed, one probe admitted");
        assert_eq!(b.state(), State::HalfOpen);

        std::thread::sleep(Duration::from_millis(40));

        assert_eq!(
            b.state(),
            State::Open,
            "a probe that never reported back must not leave the breaker \
             reporting HalfOpen forever"
        );
        assert!(
            !b.allow(),
            "the stale probe reopened the breaker; it must cool down again"
        );
        assert_eq!(
            b.trips.load(Ordering::Relaxed),
            2,
            "the stale probe counts as its own trip"
        );

        std::thread::sleep(Duration::from_millis(40));
        assert!(
            b.allow(),
            "after the fresh cooldown elapses, another probe is admitted"
        );
    }

    #[test]
    fn a_successful_probe_closes_the_breaker() {
        let b = breaker(1, 0);
        b.on_failure();
        assert!(b.allow());
        b.on_success();
        assert_eq!(b.state(), State::Closed);
        assert!(b.allow());
        assert!(b.allow(), "closed, so calls are no longer rationed");
    }

    #[test]
    fn a_failed_probe_reopens_immediately_without_recounting() {
        let b = breaker(5, 10_000);
        for _ in 0..5 {
            b.on_failure();
        }
        assert_eq!(b.trips.load(Ordering::Relaxed), 1);

        // Force the cooldown to have elapsed by rebuilding with no cooldown.
        let b = breaker(5, 0);
        for _ in 0..5 {
            b.on_failure();
        }
        assert!(b.allow(), "probe admitted");
        b.on_failure();
        assert_eq!(
            b.trips.load(Ordering::Relaxed),
            2,
            "a failed probe re-opens at once rather than sending four more \
             doomed calls to re-reach the threshold"
        );
    }

    #[test]
    fn rejections_are_counted() {
        let b = breaker(1, 10_000);
        b.on_failure();
        assert!(!b.allow());
        assert!(!b.allow());
        assert_eq!(b.rejected.load(Ordering::Relaxed), 2);
    }
}
