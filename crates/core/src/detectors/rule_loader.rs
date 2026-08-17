//! Shared parse/compile boilerplate for the `static RULES: Lazy<Vec<...>>`
//! block every rules.yaml-backed detector needs — one place for "parse this
//! YAML or panic with a message naming the file, then compile each rule's
//! regex or panic naming the rule id," instead of ~25 near-identical copies
//! of that logic.
//!
//! [`compile_simple_rules`] is the one-line drop-in for the ~20 detectors
//! whose rules have no fields beyond `id`/`description`/`category`/`regex`.
//! Detectors with extra per-rule fields (validators, providers, entropy
//! requirements, language scoping — `pii`, `secrets`, `malicious_url`,
//! `code_safety`) keep their own `RawRule`/`CompiledRule` pair, but still
//! call [`parse_rules`] and [`compile_regex`] here for the parse-or-panic
//! and compile-or-panic steps, so every detector panics with the same
//! message shape and there is one place to change it.
//!
//! **Startup validation seam**: every one of these panics fires the first
//! time something touches the `Lazy`, which today is the first request that
//! exercises that detector, not process start — a malformed rule file takes
//! down the *n*th request instead of failing the boot. `validate_all` in
//! `detectors::mod` forces every detector's `RULES` static up front so the
//! panic (if any) happens before the process starts accepting traffic.

use regex::{Regex, RegexBuilder};
use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::models::{CheckAction, DetectorResult, RuleHit, Severity};
use crate::policy::schema::CheckOptions;

/// Parses `yaml` into `Vec<T>`, panicking with `source` in the message on
/// failure — the parse half of every detector's `Lazy` block.
pub fn parse_rules<T: DeserializeOwned>(yaml: &str, source: &str) -> Vec<T> {
    serde_yaml::from_str(yaml).unwrap_or_else(|e| panic!("{source} must parse: {e}"))
}

/// Compiles one rule's regex, panicking with `source` and the rule id on an
/// invalid pattern — the compile half of every detector's `Lazy` block.
pub fn compile_regex(pattern: &str, rule_id: &str, source: &str, case_insensitive: bool) -> Regex {
    RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .unwrap_or_else(|e| panic!("invalid regex in {source} for {rule_id}: {e}"))
}

/// The `id`/`description`/`category`/`regex` shape shared by every detector
/// with no per-rule fields of its own.
#[derive(Debug, Deserialize)]
pub struct SimpleRawRule {
    pub id: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub description: String,
    #[serde(default)]
    pub category: String,
    pub regex: String,
}

/// A compiled [`SimpleRawRule`]. `category` is carried through uncompiled —
/// most callers ignore it (it exists for test-side coverage checks and
/// `redacted_text` placeholder labels), a couple (`excessive_agency`,
/// `harmful_content`) use it to grade per-hit severity.
pub struct SimpleRule {
    pub id: String,
    pub category: String,
    pub pattern: Regex,
}

/// Parses and compiles a `SimpleRawRule` bank in one call — the common case.
pub fn compile_simple_rules(yaml: &str, source: &str) -> Vec<SimpleRule> {
    parse_rules::<SimpleRawRule>(yaml, source)
        .into_iter()
        .map(|r| SimpleRule {
            pattern: compile_regex(&r.regex, &r.id, source, false),
            id: r.id,
            category: r.category,
        })
        .collect()
}

/// `options.bool_option("pattern_match", true)` — the gate almost every
/// `SimpleRule`-backed detector uses to let a caller disable regex scanning
/// for a single check without disabling the whole category.
pub fn pattern_match_enabled(options: &CheckOptions) -> bool {
    options.bool_option("pattern_match", true)
}

/// Where a matched [`SimpleRule`]'s hit [`Severity`] comes from.
#[derive(Clone, Copy)]
pub enum HitSeverity {
    /// Every hit against the rule bank gets the same severity.
    Fixed(Severity),
    /// Severity depends on the matched rule's `category` (e.g.
    /// `excessive_agency`'s `destructive` actions outrank the rest).
    ByCategory(fn(&str) -> Severity),
}

impl HitSeverity {
    fn for_rule(&self, rule: &SimpleRule) -> Severity {
        match self {
            HitSeverity::Fixed(severity) => *severity,
            HitSeverity::ByCategory(grade) => grade(&rule.category),
        }
    }
}

/// How a run's overall [`DetectorResult::severity`] is derived from its hits.
#[derive(Clone, Copy)]
pub enum ResultSeverity {
    /// Always this severity, regardless of what (if anything) matched.
    Fixed(Severity),
    /// The highest severity among the hits, or this default when there are
    /// none — for detectors whose hits don't all carry the same severity
    /// (graded-by-category rules, or a fixed-severity rule bank combined
    /// with a differently-severe `extra` side-channel).
    MaxOfHits(Severity),
}

impl ResultSeverity {
    fn resolve(&self, hits: &[RuleHit]) -> Severity {
        match self {
            ResultSeverity::Fixed(severity) => *severity,
            ResultSeverity::MaxOfHits(default) => {
                hits.iter().map(|h| h.severity).max().unwrap_or(*default)
            }
        }
    }
}

/// Declarative evaluation for the "scan a [`SimpleRule`] bank, deny/flag on
/// any hit" shape shared by most `rules.yaml`-backed detectors, instead of
/// each hand-rolling its own near-identical `pattern_match`/`evaluate`
/// pair. Build one `const` per detector
/// with [`SimpleDetector::new`] (or [`SimpleDetector::by_category`] /
/// [`SimpleDetector::new_max_severity`] for the handful that grade
/// severity), then call [`SimpleDetector::evaluate`] — or
/// [`SimpleDetector::evaluate_with`] for the handful that also filter
/// matches (an allowlist) or fold in a side-channel bank
/// (`injection_markers`, `ammonia`, invisible-text).
pub struct SimpleDetector {
    pub detector_id: &'static str,
    pub hit_severity: HitSeverity,
    pub result_severity: ResultSeverity,
    pub action_on_hit: CheckAction,
}

impl SimpleDetector {
    /// Every hit, and the overall result, get the same severity — the
    /// common case.
    pub const fn new(
        detector_id: &'static str,
        severity: Severity,
        action_on_hit: CheckAction,
    ) -> Self {
        Self {
            detector_id,
            hit_severity: HitSeverity::Fixed(severity),
            result_severity: ResultSeverity::Fixed(severity),
            action_on_hit,
        }
    }

    /// Per-hit severity graded by the matched rule's `category`; the
    /// overall result takes the highest hit severity, falling back to
    /// `default` when there are no hits.
    pub const fn by_category(
        detector_id: &'static str,
        grade: fn(&str) -> Severity,
        default: Severity,
        action_on_hit: CheckAction,
    ) -> Self {
        Self {
            detector_id,
            hit_severity: HitSeverity::ByCategory(grade),
            result_severity: ResultSeverity::MaxOfHits(default),
            action_on_hit,
        }
    }

    /// Fixed per-hit severity from the rule bank, but the overall result
    /// takes the max across hits — for detectors whose `extra` side-channel
    /// hits carry their own (different) severity.
    pub const fn new_max_severity(
        detector_id: &'static str,
        severity: Severity,
        action_on_hit: CheckAction,
    ) -> Self {
        Self {
            detector_id,
            hit_severity: HitSeverity::Fixed(severity),
            result_severity: ResultSeverity::MaxOfHits(severity),
            action_on_hit,
        }
    }

    fn build_result(&self, hits: Vec<RuleHit>) -> DetectorResult {
        let action = if hits.is_empty() {
            CheckAction::Log
        } else {
            self.action_on_hit
        };
        let severity = self.result_severity.resolve(&hits);
        DetectorResult {
            detector_id: self.detector_id.to_string(),
            action,
            severity,
            hits,
            confidence: None,
        }
    }

    /// The common case: scan `rules` against `text`, gated by the
    /// `pattern_match` option (default on).
    pub fn evaluate(
        &self,
        rules: &[SimpleRule],
        text: &str,
        options: &CheckOptions,
    ) -> DetectorResult {
        self.evaluate_with(
            rules,
            text,
            options,
            |_, o| pattern_match_enabled(o),
            |_, _| true,
            |_, _| Vec::new(),
        )
    }

    /// Like [`Self::evaluate`], with three extension points:
    /// - `scan_gate`: whether to run the rule-bank scan at all (replaces
    ///   the default `pattern_match` check for detectors with extra
    ///   preconditions, e.g. `compliance`'s disclaimer check).
    /// - `keep`: per-match veto (e.g. `sensitive_business_data`'s
    ///   allowlist) — called with the matched rule and the matched substring.
    /// - `extra`: hits from a side-channel bank, appended unconditionally
    ///   (the closure is expected to check its own governing option, since
    ///   some side channels — e.g. `document_metadata_leakage`'s invisible-
    ///   text scan — are gated independently of `scan_gate`).
    pub fn evaluate_with(
        &self,
        rules: &[SimpleRule],
        text: &str,
        options: &CheckOptions,
        scan_gate: impl Fn(&str, &CheckOptions) -> bool,
        keep: impl Fn(&SimpleRule, &str) -> bool,
        extra: impl Fn(&str, &CheckOptions) -> Vec<RuleHit>,
    ) -> DetectorResult {
        let mut hits: Vec<RuleHit> = Vec::new();
        if scan_gate(text, options) {
            for rule in rules {
                for m in rule.pattern.find_iter(text) {
                    if keep(rule, m.as_str()) {
                        hits.push(RuleHit {
                            rule_id: rule.id.clone(),
                            span: (m.start(), m.end()),
                            severity: self.hit_severity.for_rule(rule),
                        });
                    }
                }
            }
        }
        hits.extend(extra(text, options));

        self.build_result(hits)
    }
}
