//! Runs `config/benchmarks/copyright_v1.yaml` against
//! `armor_core::detectors::copyright`. Word-shingle Jaccard similarity (no
//! rules.yaml bank, no real copyrighted fixture text), so this follows
//! `gibberish_benchmark.rs`'s shape — one rule, recall + zero-FP only.

use armor_core::detectors::copyright;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;

fn opts(passages: &[String]) -> CheckOptions {
    let refs: Vec<&str> = passages.iter().map(String::as_str).collect();
    let mut o = CheckOptions::default();
    o.set_str_list("protected_passages", &refs);
    o
}

#[derive(Debug, Deserialize)]
struct PositiveSample {
    id: String,
    text: String,
    #[serde(default)]
    protected_passages: Vec<String>,
    expect_rule: String,
}

#[derive(Debug, Deserialize)]
struct NegativeSample {
    id: String,
    text: String,
    #[serde(default)]
    protected_passages: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Corpus {
    positive_samples: Vec<PositiveSample>,
    negative_samples: Vec<NegativeSample>,
    general_benign: Vec<NegativeSample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!("../../../config/benchmarks/copyright_v1.yaml"))
        .expect("config/benchmarks/copyright_v1.yaml must parse")
}

#[test]
fn recall_is_100_percent_on_positive_samples() {
    let corpus = corpus();
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let result = copyright::evaluate(&sample.text, &opts(&sample.protected_passages));
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
        let result = copyright::evaluate(&sample.text, &opts(&sample.protected_passages));
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
