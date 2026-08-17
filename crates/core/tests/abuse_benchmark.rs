//! Runs `config/benchmarks/abuse_v1.yaml` against
//! `armor_core::detectors::abuse`. Stateful and time-windowed, so each
//! sample carries its own options plus an optional `warmup_calls` count
//! (fired before the measured call, at `warmup_now` if given, else `now`)
//! to set up burst/rollover scenarios deterministically — see that corpus
//! file's header and `crates/core/src/detectors/abuse.rs`.

use armor_core::detectors::abuse;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;

#[derive(Debug, Deserialize, Default)]
struct Sample {
    id: String,
    text: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    max_requests_per_window: Option<f64>,
    #[serde(default)]
    window_seconds: Option<f64>,
    #[serde(default)]
    now: Option<f64>,
    #[serde(default)]
    warmup_calls: u32,
    #[serde(default)]
    warmup_now: Option<f64>,
    #[serde(default)]
    expect_rule: Option<String>,
    #[serde(default)]
    rule_ids: Vec<String>,
}

impl Sample {
    fn options_at(&self, now: Option<f64>) -> CheckOptions {
        let mut o = CheckOptions::default();
        if let Some(session_id) = &self.session_id {
            o.set_str("session_id", session_id);
        }
        if let Some(v) = self.max_requests_per_window {
            o.set_f64("max_requests_per_window", v);
        }
        if let Some(v) = self.window_seconds {
            o.set_f64("window_seconds", v);
        }
        if let Some(v) = now {
            o.set_f64("now", v);
        }
        o
    }

    /// Runs any configured warmup calls, then returns the result of the
    /// measured call at `now`.
    fn evaluate(&self) -> armor_core::models::DetectorResult {
        let warmup_now = self.warmup_now.or(self.now);
        for _ in 0..self.warmup_calls {
            abuse::evaluate(&self.text, &self.options_at(warmup_now));
        }
        abuse::evaluate(&self.text, &self.options_at(self.now))
    }
}

#[derive(Debug, Deserialize)]
struct Corpus {
    positive_samples: Vec<Sample>,
    negative_samples: Vec<Sample>,
    general_benign: Vec<Sample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!("../../../config/benchmarks/abuse_v1.yaml"))
        .expect("config/benchmarks/abuse_v1.yaml must parse")
}

#[test]
fn recall_is_100_percent_on_positive_samples() {
    let corpus = corpus();
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let expect_rule = sample
            .expect_rule
            .as_ref()
            .expect("positive sample must set expect_rule");
        let result = sample.evaluate();
        let rule_ids: Vec<String> = result.hits.iter().map(|h| h.rule_id.clone()).collect();
        if !rule_ids.contains(expect_rule) {
            misses.push(format!(
                "{} ({}): expected rule {:?}, got {:?}",
                sample.id, sample.text, expect_rule, rule_ids
            ));
        }
    }

    assert!(
        misses.is_empty(),
        "{} positive sample(s) failed to fire their labeled rule:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

#[test]
fn negative_and_general_benign_samples_have_zero_false_positives() {
    let corpus = corpus();
    let mut fps = Vec::new();

    for sample in corpus.negative_samples.iter().chain(&corpus.general_benign) {
        let result = sample.evaluate();
        let rule_ids: Vec<String> = result.hits.iter().map(|h| h.rule_id.clone()).collect();
        let watched: Vec<&String> = if sample.rule_ids.is_empty() {
            rule_ids.iter().collect()
        } else {
            rule_ids
                .iter()
                .filter(|r| sample.rule_ids.contains(r))
                .collect()
        };
        if !watched.is_empty() {
            fps.push(format!(
                "{} ({}): fired {:?}",
                sample.id, sample.text, watched
            ));
        }
    }

    assert!(
        fps.is_empty(),
        "{} sample(s) fired a rule:\n{}",
        fps.len(),
        fps.join("\n")
    );
}
