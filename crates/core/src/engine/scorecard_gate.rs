//! Advisory/enforcement gate for model-backed checks: a `block`/`redact`
//! action auto-downgrades to `warn` until the detector's metrics (F1, AUROC,
//! ECE, FP-rate, latency, per-language sample count, staleness) pass their
//! threshold — inert for now, since no model-backed checks ship yet. Built
//! ahead of them so a model-backed check can never ship without first
//! clearing this gate.
//!
//! # Two-tier design
//!
//! A model-backed check's results are gated by two quality thresholds:
//!
//! - **Advisory**: the check must pass this tier to run *at all*. When
//!   passing only advisory, the check runs pinned to `EnforcementMode::Warn`
//!   regardless of its configured mode. This ensures a model with unmeasured
//!   quality never blocks a request.
//!
//! - **Enforcement**: the check must pass this tier for its verdict to carry
//!   `Block`-mode authority. A check passing advisory but not enforcement
//!   still contributes findings (hits, severity) but its action is downgraded
//!   to `Warn`.
//!
//! A missing metric (not `None` but absent from the source) fails the gate
//! immediately — you cannot trust a model whose quality you cannot measure.
//!
//! # The routing signal vs. calibrated quality
//!
//! [`super::escalation::risk_score`] is an *ordinal routing signal* — it
//! decides whether to spend a forward pass. The scorecard gate measures
//! *calibrated quality* against a benchmark suite and decides whether the
//! model's verdict may be enforced. These are orthogonal concerns.

use serde::{Deserialize, Serialize};

/// Benchmark quality metrics for a model-backed check, derived from the
/// benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardMetrics {
    /// F1 score on the benchmark dataset. Range `[0.0, 1.0]`.
    pub f1: Option<f64>,
    /// Area under the ROC curve. Range `[0.0, 1.0]`.
    pub auroc: Option<f64>,
    /// Expected Calibration Error. Range `[0.0, 1.0]`; lower is better.
    pub ece: Option<f64>,
    /// False-positive rate on the benchmark dataset. Range `[0.0, 1.0]`.
    pub fp_rate: Option<f64>,
    /// 95th-percentile latency in milliseconds. Lower is better.
    pub p95_latency_ms: Option<f64>,
    /// Total sample count across all languages in the benchmark.
    pub sample_count: Option<u64>,
    /// Per-language sample counts. Keys are BCP-47 language tags.
    pub per_language_samples: Option<std::collections::HashMap<String, u64>>,
    /// How many days since this model was last re-evaluated against the
    /// benchmark. `None` means unknown (which fails the gate).
    pub staleness_days: Option<u64>,
}

/// Minimum quality thresholds for a check to run or enforce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardThresholds {
    /// Minimum F1 for advisory tier. Default: 0.6.
    pub advisory_f1: f64,
    /// Minimum AUROC for advisory tier. Default: 0.7.
    pub advisory_auroc: f64,
    /// Maximum ECE for advisory tier. Default: 0.15.
    pub advisory_max_ece: f64,
    /// Minimum sample count for advisory tier. Default: 100.
    pub advisory_min_samples: u64,
    /// Maximum staleness in days for advisory tier. Default: 90.
    pub advisory_max_staleness_days: u64,
    /// Minimum F1 for enforcement tier. Default: 0.8.
    pub enforcement_f1: f64,
    /// Minimum AUROC for enforcement tier. Default: 0.9.
    pub enforcement_auroc: f64,
    /// Maximum ECE for enforcement tier. Default: 0.08.
    pub enforcement_max_ece: f64,
    /// Maximum false-positive rate for enforcement tier. Default: 0.05.
    pub enforcement_max_fp_rate: f64,
    /// Maximum p95 latency in ms for enforcement tier. Default: 200.
    pub enforcement_max_p95_latency_ms: f64,
    /// Minimum sample count for enforcement tier. Default: 1000.
    pub enforcement_min_samples: u64,
    /// Minimum per-language sample count for enforcement. Default: 50.
    pub enforcement_min_per_lang_samples: u64,
    /// Maximum staleness in days for enforcement tier. Default: 30.
    pub enforcement_max_staleness_days: u64,
}

impl Default for ScorecardThresholds {
    fn default() -> Self {
        Self {
            advisory_f1: 0.6,
            advisory_auroc: 0.7,
            advisory_max_ece: 0.15,
            advisory_min_samples: 100,
            advisory_max_staleness_days: 90,
            enforcement_f1: 0.8,
            enforcement_auroc: 0.9,
            enforcement_max_ece: 0.08,
            enforcement_max_fp_rate: 0.05,
            enforcement_max_p95_latency_ms: 200.0,
            enforcement_min_samples: 1000,
            enforcement_min_per_lang_samples: 50,
            enforcement_max_staleness_days: 30,
        }
    }
}

/// Result of evaluating a check's scorecard metrics against thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateVerdict {
    /// Metrics pass enforcement thresholds — the check may block/redact.
    Pass,
    /// Metrics pass advisory but not enforcement — the check may run but
    /// only in warn mode (findings contribute, but action is downgraded).
    AdvisoryOnly,
    /// Metrics are missing or below advisory thresholds — the check
    /// cannot run at all and should be disabled.
    Fail,
}

impl GateVerdict {
    /// Returns `true` if the check is allowed to run at all.
    pub fn may_run(self) -> bool {
        self != GateVerdict::Fail
    }

    /// Returns `true` if the check may carry enforcement authority
    /// (block/redact in Block mode).
    pub fn may_enforce(self) -> bool {
        self == GateVerdict::Pass
    }
}

/// Evaluate scorecard metrics against thresholds.
///
/// A `None` metric fails the gate immediately (you cannot trust a model
/// whose quality you cannot measure).
pub fn evaluate(metrics: &ScorecardMetrics, thresholds: &ScorecardThresholds) -> GateVerdict {
    if !passes_advisory(metrics, thresholds) {
        return GateVerdict::Fail;
    }
    if !passes_enforcement(metrics, thresholds) {
        return GateVerdict::AdvisoryOnly;
    }
    GateVerdict::Pass
}

fn passes_advisory(m: &ScorecardMetrics, t: &ScorecardThresholds) -> bool {
    let f1 = match m.f1 {
        Some(v) => v,
        None => return false,
    };
    let auroc = match m.auroc {
        Some(v) => v,
        None => return false,
    };
    let ece = match m.ece {
        Some(v) => v,
        None => return false,
    };
    let samples = match m.sample_count {
        Some(v) => v,
        None => return false,
    };
    let staleness = match m.staleness_days {
        Some(v) => v,
        None => return false,
    };

    f1 >= t.advisory_f1
        && auroc >= t.advisory_auroc
        && ece <= t.advisory_max_ece
        && samples >= t.advisory_min_samples
        && staleness <= t.advisory_max_staleness_days
}

fn passes_enforcement(m: &ScorecardMetrics, t: &ScorecardThresholds) -> bool {
    let f1 = m.f1.unwrap_or(0.0);
    let auroc = m.auroc.unwrap_or(0.0);
    let ece = m.ece.unwrap_or(1.0);
    let fp_rate = match m.fp_rate {
        Some(v) => v,
        None => return false,
    };
    let p95 = match m.p95_latency_ms {
        Some(v) => v,
        None => return false,
    };
    let samples = m.sample_count.unwrap_or(0);
    let staleness = m.staleness_days.unwrap_or(u64::MAX);

    if f1 < t.enforcement_f1
        || auroc < t.enforcement_auroc
        || ece > t.enforcement_max_ece
        || fp_rate > t.enforcement_max_fp_rate
        || p95 > t.enforcement_max_p95_latency_ms
        || samples < t.enforcement_min_samples
        || staleness > t.enforcement_max_staleness_days
    {
        return false;
    }

    // Per-language minimum sample count — every language must meet the bar.
    if let Some(ref per_lang) = m.per_language_samples {
        for count in per_lang.values() {
            if *count < t.enforcement_min_per_lang_samples {
                return false;
            }
        }
    } else if t.enforcement_min_per_lang_samples > 0 {
        // Enforcement requires per-language breakdown; missing means we
        // cannot verify per-language coverage.
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_metrics() -> ScorecardMetrics {
        let mut per_lang = std::collections::HashMap::new();
        per_lang.insert("en".to_string(), 500);
        per_lang.insert("es".to_string(), 200);
        ScorecardMetrics {
            f1: Some(0.92),
            auroc: Some(0.95),
            ece: Some(0.04),
            fp_rate: Some(0.03),
            p95_latency_ms: Some(120.0),
            sample_count: Some(2000),
            per_language_samples: Some(per_lang),
            staleness_days: Some(10),
        }
    }

    #[test]
    fn full_metrics_pass_enforcement() {
        assert_eq!(
            evaluate(&full_metrics(), &ScorecardThresholds::default()),
            GateVerdict::Pass
        );
    }

    #[test]
    fn missing_f1_fails_gate() {
        let mut m = full_metrics();
        m.f1 = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn missing_auroc_fails_gate() {
        let mut m = full_metrics();
        m.auroc = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn missing_ece_fails_gate() {
        let mut m = full_metrics();
        m.ece = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn missing_sample_count_fails_gate() {
        let mut m = full_metrics();
        m.sample_count = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn missing_staleness_fails_gate() {
        let mut m = full_metrics();
        m.staleness_days = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn missing_fp_rate_fails_enforcement() {
        let mut m = full_metrics();
        m.fp_rate = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn missing_p95_fails_enforcement() {
        let mut m = full_metrics();
        m.p95_latency_ms = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn missing_per_lang_fails_enforcement() {
        let mut m = full_metrics();
        m.per_language_samples = None;
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn low_f1_passes_advisory_but_not_enforcement() {
        let mut m = full_metrics();
        m.f1 = Some(0.7); // passes advisory (>= 0.6), fails enforcement (>= 0.8)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn very_low_f1_fails_gate() {
        let mut m = full_metrics();
        m.f1 = Some(0.4); // fails advisory (< 0.6)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn high_ece_fails_enforcement() {
        let mut m = full_metrics();
        m.ece = Some(0.12); // passes advisory (<= 0.15), fails enforcement (<= 0.08)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn very_high_ece_fails_gate() {
        let mut m = full_metrics();
        m.ece = Some(0.3); // fails advisory (> 0.15)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn high_fp_rate_fails_enforcement() {
        let mut m = full_metrics();
        m.fp_rate = Some(0.08); // fails enforcement (> 0.05)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn high_latency_fails_enforcement() {
        let mut m = full_metrics();
        m.p95_latency_ms = Some(350.0); // fails enforcement (> 200)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn stale_model_fails_enforcement() {
        let mut m = full_metrics();
        m.staleness_days = Some(45); // passes advisory (<= 90), fails enforcement (> 30)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn very_stale_model_fails_gate() {
        let mut m = full_metrics();
        m.staleness_days = Some(120); // fails advisory (> 90)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn low_sample_count_fails_enforcement() {
        let mut m = full_metrics();
        m.sample_count = Some(500); // passes advisory (>= 100), fails enforcement (>= 1000)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn very_low_sample_count_fails_gate() {
        let mut m = full_metrics();
        m.sample_count = Some(50); // fails advisory (< 100)
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::Fail
        );
    }

    #[test]
    fn low_per_lang_sample_fails_enforcement() {
        let mut m = full_metrics();
        let mut per_lang = std::collections::HashMap::new();
        per_lang.insert("en".to_string(), 500);
        per_lang.insert("fr".to_string(), 30); // below enforcement_min_per_lang_samples (50)
        m.per_language_samples = Some(per_lang);
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }

    #[test]
    fn gate_verdict_may_run() {
        assert!(GateVerdict::Pass.may_run());
        assert!(GateVerdict::AdvisoryOnly.may_run());
        assert!(!GateVerdict::Fail.may_run());
    }

    #[test]
    fn gate_verdict_may_enforce() {
        assert!(GateVerdict::Pass.may_enforce());
        assert!(!GateVerdict::AdvisoryOnly.may_enforce());
        assert!(!GateVerdict::Fail.may_enforce());
    }

    #[test]
    fn custom_thresholds() {
        let thresholds = ScorecardThresholds {
            advisory_f1: 0.3,
            advisory_auroc: 0.4,
            advisory_max_ece: 0.5,
            advisory_min_samples: 10,
            advisory_max_staleness_days: 365,
            enforcement_f1: 0.5,
            enforcement_auroc: 0.6,
            enforcement_max_ece: 0.2,
            enforcement_max_fp_rate: 0.1,
            enforcement_max_p95_latency_ms: 500.0,
            enforcement_min_samples: 100,
            enforcement_min_per_lang_samples: 10,
            enforcement_max_staleness_days: 180,
        };
        let m = ScorecardMetrics {
            f1: Some(0.55),
            auroc: Some(0.65),
            ece: Some(0.18),
            fp_rate: Some(0.08),
            p95_latency_ms: Some(400.0),
            sample_count: Some(150),
            per_language_samples: Some({
                let mut h = std::collections::HashMap::new();
                h.insert("en".to_string(), 100);
                h
            }),
            staleness_days: Some(100),
        };
        assert_eq!(evaluate(&m, &thresholds), GateVerdict::Pass);
    }

    #[test]
    fn zero_per_lang_samples_fails_enforcement() {
        let mut m = full_metrics();
        let mut per_lang = std::collections::HashMap::new();
        per_lang.insert("en".to_string(), 500);
        per_lang.insert("fr".to_string(), 0); // zero samples
        m.per_language_samples = Some(per_lang);
        assert_eq!(
            evaluate(&m, &ScorecardThresholds::default()),
            GateVerdict::AdvisoryOnly
        );
    }
}
