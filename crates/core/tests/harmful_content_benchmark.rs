//! Runs `config/benchmarks/harmful_content_v1.yaml` against
//! `armor_core::detectors::harmful_content` and asserts the acceptance
//! criteria documented in that file's header. This is deliberately a real
//! test, not just a corpus — a benchmark YAML with no runner behind it is
//! documentation, not evaluation.

use std::collections::HashSet;

use armor_core::detectors::harmful_content;
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
    positive_samples: Vec<PositiveSample>,
    negative_samples: Vec<NegativeSample>,
    general_benign: Vec<GeneralBenignSample>,
}

fn corpus() -> Corpus {
    serde_yaml::from_str(include_str!(
        "../../../config/benchmarks/harmful_content_v1.yaml"
    ))
    .expect("config/benchmarks/harmful_content_v1.yaml must parse")
}

fn hit_rule_ids(text: &str) -> HashSet<String> {
    harmful_content::evaluate(text, &CheckOptions::default())
        .hits
        .into_iter()
        .map(|h| h.rule_id)
        .collect()
}

/// Every rule must have at least 3 positive and 3 negative samples — the
/// coverage target stated in the corpus header. Catches a rule added to
/// `rules/harmful_content/rules.yaml` without matching corpus entries.
#[test]
fn every_rule_meets_the_coverage_target() {
    let corpus = corpus();

    let mut positive_counts: std::collections::HashMap<String, usize> = Default::default();
    for s in &corpus.positive_samples {
        *positive_counts.entry(s.rule_id.clone()).or_default() += 1;
    }
    let mut negative_counts: std::collections::HashMap<String, usize> = Default::default();
    for s in &corpus.negative_samples {
        for rule_id in &s.rule_ids {
            *negative_counts.entry(rule_id.clone()).or_default() += 1;
        }
    }

    #[derive(Deserialize)]
    struct RawRule {
        id: String,
    }
    let rules: Vec<RawRule> =
        serde_yaml::from_str(include_str!("../../../rules/harmful_content/rules.yaml")).unwrap();

    for rule in &rules {
        assert!(
            positive_counts.get(&rule.id).copied().unwrap_or(0) >= 3,
            "rule {:?} has fewer than 3 positive samples in the benchmark corpus",
            rule.id
        );
        assert!(
            negative_counts.get(&rule.id).copied().unwrap_or(0) >= 3,
            "rule {:?} has fewer than 3 negative samples in the benchmark corpus",
            rule.id
        );
    }
}

/// Recall: every positive sample must fire its labeled rule_id.
/// Acceptance criterion: 100% (these are literal patterns written from
/// these exact phrasings — a miss here is a regex bug, not noise).
#[test]
fn recall_is_100_percent_on_positive_samples() {
    let corpus = corpus();
    let mut misses = Vec::new();

    for sample in &corpus.positive_samples {
        let hits = hit_rule_ids(&sample.text);
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

/// False positives: per-rule near-miss negatives must not fire the rule
/// they're scoped to. Reports rather than hard-fails for
/// `csam-material-mention` — documented in the corpus header as expected
/// to be chattier since legitimate legal/policy/prevention discourse
/// necessarily names the bare phrase "child pornography"/"child sexual
/// abuse material". Every other rule (weapons_howto / drugs_howto /
/// violence_howto, plus csam's other three rules, all of which require an
/// instructional verb phrase, not just a topic mention) is held to a
/// strict bound since they're candidates for `mode: block` promotion.
#[test]
fn near_miss_negatives_false_positive_rate() {
    let corpus = corpus();

    const CHATTY_RULES: &[&str] = &["csam-material-mention"];

    let mut strict_total = 0;
    let mut strict_fps = Vec::new();
    let mut chatty_total = 0;
    let mut chatty_fps = Vec::new();

    for sample in &corpus.negative_samples {
        let hits = hit_rule_ids(&sample.text);
        for rule_id in &sample.rule_ids {
            let is_fp = hits.contains(rule_id);
            if CHATTY_RULES.contains(&rule_id.as_str()) {
                chatty_total += 1;
                if is_fp {
                    chatty_fps.push(format!(
                        "{} ({}): fired {:?}",
                        sample.id, sample.text, rule_id
                    ));
                }
            } else {
                strict_total += 1;
                if is_fp {
                    strict_fps.push(format!(
                        "{} ({}): fired {:?}",
                        sample.id, sample.text, rule_id
                    ));
                }
            }
        }
    }

    eprintln!(
        "[harmful_content eval] strict-category FPR: {}/{} ({:.1}%)",
        strict_fps.len(),
        strict_total,
        100.0 * strict_fps.len() as f64 / strict_total.max(1) as f64
    );
    eprintln!(
        "[harmful_content eval] chatty-category (warn-only) FPR: {}/{} ({:.1}%)",
        chatty_fps.len(),
        chatty_total,
        100.0 * chatty_fps.len() as f64 / chatty_total.max(1) as f64
    );
    if !chatty_fps.is_empty() {
        eprintln!(
            "[harmful_content eval] chatty-category false positives (reported, not gated):\n{}",
            chatty_fps.join("\n")
        );
    }

    assert!(
        strict_fps.is_empty(),
        "{} strict-category false positive(s), expected 0:\n{}",
        strict_fps.len(),
        strict_fps.join("\n")
    );
}

/// General benign corpus: realistic non-adversarial prompts, concentrated
/// on the categories most likely to over-fire (true crime, policy/legal
/// research, fiction, trust & safety). ANY hit here counts as a false
/// positive, per the org's existing PII/secrets/jailbreak benchmark
/// convention. Gated at 0/20 — unlike jailbreak's chattier
/// fictional_framing/ethics_bypass_declaration bank, every harmful_content
/// rule requires an instructional verb phrase or an explicit-solicitation
/// phrase, not just topic vocabulary, so this corpus is expected to be
/// clean.
#[test]
fn general_benign_false_positive_rate() {
    let corpus = corpus();
    let mut fps = Vec::new();

    for sample in &corpus.general_benign {
        let hits = hit_rule_ids(&sample.text);
        if !hits.is_empty() {
            fps.push(format!("{} ({}): fired {:?}", sample.id, sample.text, hits));
        }
    }

    eprintln!(
        "[harmful_content eval] general_benign FPR: {}/{}",
        fps.len(),
        corpus.general_benign.len()
    );

    assert!(
        fps.is_empty(),
        "{} of {} general-benign sample(s) fired a harmful_content rule:\n{}",
        fps.len(),
        corpus.general_benign.len(),
        fps.join("\n")
    );
}
