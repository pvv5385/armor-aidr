//! Runs `config/benchmarks/compliance_v1.yaml` against
//! `armor_core::detectors::compliance`. Compliance is a two-part check
//! (advice trigger present AND no configured disclaimer present), so each
//! positive sample is evaluated twice: once with no `disclaimer_phrases`
//! configured (trigger alone should deny) and once with the sample's
//! matching disclaimer configured (should allow, since the disclaimer is
//! now present in the same text).

use std::collections::{HashMap, HashSet};

use armor_core::detectors::compliance;
use armor_core::policy::schema::CheckOptions;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PositiveSample {
    id: String,
    rule_id: String,
    text: String,
    disclaimer: String,
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
    positive_samples: Vec<PositiveSample>,
    negative_samples: Vec<NegativeSample>,
    general_benign: Vec<GeneralBenignSample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!(
        "../../../config/benchmarks/compliance_v1.yaml"
    ))
    .expect("config/benchmarks/compliance_v1.yaml must parse")
}

fn hit_rule_ids(text: &str, disclaimers: &[&str]) -> HashSet<String> {
    let mut o = CheckOptions::default();
    o.set_str_list("disclaimer_phrases", disclaimers);
    compliance::evaluate(text, &o)
        .hits
        .into_iter()
        .map(|h| h.rule_id)
        .collect()
}

#[derive(Deserialize)]
struct RawRule {
    id: String,
}

fn rule_ids() -> Vec<String> {
    let rules: Vec<RawRule> =
        serde_yaml::from_str(include_str!("../../../rules/compliance/rules.yaml")).unwrap();
    rules.into_iter().map(|r| r.id).collect()
}

#[test]
fn every_rule_meets_the_coverage_target() {
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

    for rule_id in rule_ids() {
        assert!(
            positive_counts.get(&rule_id).copied().unwrap_or(0) >= 2,
            "rule {rule_id:?} has fewer than 2 positive samples"
        );
        assert!(
            negative_counts.get(&rule_id).copied().unwrap_or(0) >= 2,
            "rule {rule_id:?} has fewer than 2 negative samples"
        );
    }
}

#[test]
fn trigger_without_disclaimer_denies_and_with_disclaimer_allows() {
    let corpus = corpus();
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let hits_without = hit_rule_ids(&sample.text, &[]);
        if !hits_without.contains(&sample.rule_id) {
            misses.push(format!(
                "{} ({}): expected rule {:?} without a disclaimer, got {:?}",
                sample.id, sample.text, sample.rule_id, hits_without
            ));
        }

        let with_disclaimer = format!("{} {}", sample.text, sample.disclaimer);
        let hits_with = hit_rule_ids(&with_disclaimer, &[sample.disclaimer.as_str()]);
        if !hits_with.is_empty() {
            misses.push(format!(
                "{} ({}): expected no hits once the disclaimer is present, got {:?}",
                sample.id, with_disclaimer, hits_with
            ));
        }
    }

    assert!(
        misses.is_empty(),
        "{} sample(s) failed:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

#[test]
fn near_miss_negatives_have_zero_false_positives() {
    let corpus = corpus();
    let mut fps = Vec::new();

    for sample in &corpus.negative_samples {
        let hits = hit_rule_ids(&sample.text, &[]);
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
    let mut fps = Vec::new();

    for sample in &corpus.general_benign {
        let hits = hit_rule_ids(&sample.text, &[]);
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
