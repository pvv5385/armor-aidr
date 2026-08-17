//! Runs `config/benchmarks/unbounded_consumption_v1.yaml` against
//! `armor_core::detectors::unbounded_consumption`. Stateful (per-session
//! counters), so unlike every other benchmark here each sample carries its
//! own options and a unique `session_id` — see that corpus file's header.

use armor_core::detectors::unbounded_consumption;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;

#[derive(Debug, Deserialize, Default)]
struct Sample {
    id: String,
    text: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    estimated_tokens: Option<f64>,
    #[serde(default)]
    max_tokens_per_session: Option<f64>,
    #[serde(default)]
    max_requests_per_session: Option<f64>,
    #[serde(default)]
    loop_depth: Option<f64>,
    #[serde(default)]
    max_loop_depth: Option<f64>,
    #[serde(default)]
    expect_rule: Option<String>,
    #[serde(default)]
    rule_ids: Vec<String>,
}

impl Sample {
    fn options(&self) -> CheckOptions {
        let mut o = CheckOptions::default();
        if let Some(session_id) = &self.session_id {
            o.set_str("session_id", session_id);
        }
        if let Some(v) = self.estimated_tokens {
            o.set_f64("estimated_tokens", v);
        }
        if let Some(v) = self.max_tokens_per_session {
            o.set_f64("max_tokens_per_session", v);
        }
        if let Some(v) = self.max_requests_per_session {
            o.set_f64("max_requests_per_session", v);
        }
        if let Some(v) = self.loop_depth {
            o.set_f64("loop_depth", v);
        }
        if let Some(v) = self.max_loop_depth {
            o.set_f64("max_loop_depth", v);
        }
        o
    }
}

#[derive(Debug, Deserialize)]
struct Corpus {
    positive_samples: Vec<Sample>,
    negative_samples: Vec<Sample>,
    general_benign: Vec<Sample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!(
        "../../../config/benchmarks/unbounded_consumption_v1.yaml"
    ))
    .expect("config/benchmarks/unbounded_consumption_v1.yaml must parse")
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
        let result = unbounded_consumption::evaluate(&sample.text, &sample.options());
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
        let result = unbounded_consumption::evaluate(&sample.text, &sample.options());
        let rule_ids: Vec<String> = result.hits.iter().map(|h| h.rule_id.clone()).collect();
        let watched = if sample.rule_ids.is_empty() {
            rule_ids.iter().collect::<Vec<_>>()
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
