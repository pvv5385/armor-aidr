//! Runs `config/benchmarks/tool_output_injection_v1.yaml` against
//! `armor_core::detectors::tool_output_injection`. See that file's header
//! for why this corpus isn't per-rule-coverage shaped like the other Phase
//! 1 benchmarks — the detector wraps two already-benchmarked rule banks
//! rather than owning one of its own.

use armor_core::detectors::tool_output_injection;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PositiveSample {
    id: String,
    text: String,
    expect_rule: String,
}

#[derive(Debug, Deserialize)]
struct NegativeSample {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Corpus {
    positive_samples: Vec<PositiveSample>,
    negative_samples: Vec<NegativeSample>,
    general_benign: Vec<NegativeSample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!(
        "../../../config/benchmarks/tool_output_injection_v1.yaml"
    ))
    .expect("config/benchmarks/tool_output_injection_v1.yaml must parse")
}

#[test]
fn recall_is_100_percent_on_positive_samples() {
    let corpus = corpus();
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let result = tool_output_injection::evaluate(&sample.text, &CheckOptions::default());
        let rule_ids: Vec<String> = result.hits.iter().map(|h| h.rule_id.clone()).collect();
        if !rule_ids.contains(&sample.expect_rule) {
            misses.push(format!(
                "{} ({}): expected rule {:?}, got {:?}",
                sample.id, sample.text, sample.expect_rule, rule_ids
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
        let result = tool_output_injection::evaluate(&sample.text, &CheckOptions::default());
        if !result.hits.is_empty() {
            fps.push(format!(
                "{} ({}): fired {:?}",
                sample.id,
                sample.text,
                result.hits.iter().map(|h| &h.rule_id).collect::<Vec<_>>()
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
