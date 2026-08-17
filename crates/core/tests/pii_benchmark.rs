//! Runs `config/benchmarks/pii_v1.yaml` against
//! `armor_core::detectors::pii`. See that file's header — a REPRESENTATIVE
//! 12-of-66-entity pass, not exhaustive; checksum-validated entities are
//! covered by dedicated inline unit tests instead (see `pii.rs`).

use std::collections::{HashMap, HashSet};

use armor_core::detectors::pii;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PositiveSample {
    id: String,
    rule_id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct NegativeSample {
    id: String,
    #[serde(default)]
    rule_ids: Vec<String>,
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeneralBenignSample {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Corpus {
    entities_covered: Vec<String>,
    positive_samples: Vec<PositiveSample>,
    negative_samples: Vec<NegativeSample>,
    general_benign: Vec<GeneralBenignSample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!("../../../config/benchmarks/pii_v1.yaml"))
        .expect("config/benchmarks/pii_v1.yaml must parse")
}

fn all_entities_enabled(entities: &[String]) -> CheckOptions {
    let mut o = CheckOptions::default();
    for e in entities {
        o.set_bool(e, true);
    }
    o
}

fn hit_rule_ids(text: &str, options: &CheckOptions) -> HashSet<String> {
    pii::evaluate(text, options)
        .hits
        .into_iter()
        .map(|h| h.rule_id)
        .collect()
}

#[test]
fn every_covered_rule_meets_the_coverage_target() {
    let corpus = corpus();

    let mut positive_counts: HashMap<String, usize> = Default::default();
    for s in &corpus.positive_samples {
        *positive_counts.entry(s.rule_id.clone()).or_default() += 1;
    }
    let mut negative_counts: HashMap<String, usize> = Default::default();
    for s in &corpus.negative_samples {
        for rule_id in &s.rule_ids {
            *negative_counts.entry(rule_id.clone()).or_default() += 1;
        }
    }

    for rule_id in positive_counts.keys() {
        assert!(
            positive_counts.get(rule_id).copied().unwrap_or(0) >= 2,
            "rule {rule_id:?} has fewer than 2 positive samples"
        );
        assert!(
            negative_counts.get(rule_id).copied().unwrap_or(0) >= 2,
            "rule {rule_id:?} has fewer than 2 negative samples"
        );
    }
}

#[test]
fn recall_is_100_percent_on_positive_samples() {
    let corpus = corpus();
    let options = all_entities_enabled(&corpus.entities_covered);
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let hits = hit_rule_ids(&sample.text, &options);
        if !hits.contains(&sample.rule_id) {
            misses.push(format!(
                "{} ({}): expected rule {:?}, got {:?}",
                sample.id, sample.text, sample.rule_id, hits
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
fn near_miss_negatives_have_zero_false_positives() {
    let corpus = corpus();
    let options = all_entities_enabled(&corpus.entities_covered);
    let mut fps = Vec::new();

    for sample in &corpus.negative_samples {
        let hits = hit_rule_ids(&sample.text, &options);
        for rule_id in &sample.rule_ids {
            if hits.contains(rule_id) {
                fps.push(format!(
                    "{} ({}): fired {:?}",
                    sample.id, sample.text, rule_id
                ));
            }
        }
    }

    assert!(
        fps.is_empty(),
        "{} near-miss false positive(s):\n{}",
        fps.len(),
        fps.join("\n")
    );
}

#[test]
fn general_benign_has_zero_false_positives() {
    let corpus = corpus();
    let options = all_entities_enabled(&corpus.entities_covered);
    let mut fps = Vec::new();

    for sample in &corpus.general_benign {
        let hits = hit_rule_ids(&sample.text, &options);
        if !hits.is_empty() {
            fps.push(format!("{} ({}): fired {:?}", sample.id, sample.text, hits));
        }
    }

    assert!(
        fps.is_empty(),
        "{} of {} general-benign sample(s) fired a rule:\n{}",
        fps.len(),
        corpus.general_benign.len(),
        fps.join("\n")
    );
}
