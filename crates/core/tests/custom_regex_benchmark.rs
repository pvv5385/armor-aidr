//! Runs `config/benchmarks/custom_regex_v1.yaml` against
//! `armor_core::detectors::custom_regex` with two deployment-supplied
//! patterns configured.

use armor_core::detectors::custom_regex;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;
use serde_yaml::Value;

#[derive(Debug, Deserialize)]
struct Sample {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Corpus {
    positive_samples: Vec<Sample>,
    negative_samples: Vec<Sample>,
    general_benign: Vec<Sample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!(
        "../../../config/benchmarks/custom_regex_v1.yaml"
    ))
    .expect("config/benchmarks/custom_regex_v1.yaml must parse")
}

fn opts() -> CheckOptions {
    let mut o = CheckOptions::default();
    let entry = |rule_id: &str, pattern: &str| -> Value {
        let mut m = serde_yaml::Mapping::new();
        m.insert("rule_id".into(), rule_id.into());
        m.insert("pattern".into(), pattern.into());
        Value::Mapping(m)
    };
    let seq = vec![
        entry("employee-id", r"EMP-\d{4}"),
        entry("codeword", "PROJECTFALCON"),
    ];
    o.set_raw("patterns", Value::Sequence(seq));
    o
}

#[test]
fn recall_is_100_percent_on_positive_samples() {
    let corpus = corpus();
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let result = custom_regex::evaluate(&sample.text, &opts());
        if result.hits.is_empty() {
            misses.push(format!("{} ({})", sample.id, sample.text));
        }
    }

    assert!(
        misses.is_empty(),
        "{} positive sample(s) failed to fire:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

#[test]
fn negative_and_general_benign_samples_have_zero_false_positives() {
    let corpus = corpus();
    let mut fps = Vec::new();

    for sample in corpus.negative_samples.iter().chain(&corpus.general_benign) {
        let result = custom_regex::evaluate(&sample.text, &opts());
        if !result.hits.is_empty() {
            fps.push(format!("{} ({})", sample.id, sample.text));
        }
    }

    assert!(
        fps.is_empty(),
        "{} sample(s) fired a rule:\n{}",
        fps.len(),
        fps.join("\n")
    );
}
