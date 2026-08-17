//! Runs `config/benchmarks/tool_allowlist_v1.yaml` against
//! `armor_core::detectors::tool_allowlist` with `allow: [search,
//! read_file], deny: [delete_database]`.

use armor_core::detectors::tool_allowlist;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;

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
        "../../../config/benchmarks/tool_allowlist_v1.yaml"
    ))
    .expect("config/benchmarks/tool_allowlist_v1.yaml must parse")
}

fn opts() -> CheckOptions {
    let mut o = CheckOptions::default();
    o.set_str_list("allow", &["search", "read_file"]);
    o.set_str_list("deny", &["delete_database"]);
    o
}

#[test]
fn recall_is_100_percent_on_positive_samples() {
    let corpus = corpus();
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let result = tool_allowlist::evaluate(&sample.text, &opts());
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
        let result = tool_allowlist::evaluate(&sample.text, &opts());
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
