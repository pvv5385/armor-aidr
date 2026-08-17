//! Policy config schema — the `serde`-deserializable shape of
//! `config/policies.yaml`, mirroring `app/guardrails/types.py`'s
//! `CheckSpec`/`GuardrailSpec`. Tenant/app/env layering (Python's more
//! elaborate resolution) is not yet implemented here (`policy::resolver`);
//! this loads one flat policy.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_yaml::Value;

use crate::engine::normalize::NormalizeOptions;
use crate::engine::scorecard_gate::ScorecardMetrics;
use crate::models::{CheckAction, EnforcementMode};

/// Per-check `options` bag — an untyped key/value map, same shape as
/// Python's `CheckSpec.options: dict`. Each detector reads out the keys it
/// understands and ignores the rest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckOptions(HashMap<String, Value>);

impl CheckOptions {
    pub fn bool_option(&self, key: &str, default: bool) -> bool {
        self.0.get(key).and_then(Value::as_bool).unwrap_or(default)
    }

    pub fn f64_option(&self, key: &str, default: f64) -> f64 {
        self.0.get(key).and_then(Value::as_f64).unwrap_or(default)
    }

    /// Like [`Self::f64_option`], but distinguishes "absent" from "set to
    /// the default value" — needed where the caller must branch on whether
    /// an option was supplied at all, e.g. the session counters
    /// `armor-api` injects into `abuse`/`unbounded_consumption` (absent
    /// means "no session store configured, use the in-process fallback",
    /// which `0.0` would not).
    pub fn opt_f64(&self, key: &str) -> Option<f64> {
        self.0.get(key).and_then(Value::as_f64)
    }

    pub fn str_option<'a>(&'a self, key: &str) -> Option<&'a str> {
        self.0.get(key).and_then(Value::as_str)
    }

    /// A YAML sequence of strings, e.g. a customer-supplied keyword or
    /// tool-name list. Non-string entries are dropped rather than erroring —
    /// same permissive spirit as the other `*_option` getters.
    pub fn str_list_option(&self, key: &str) -> Vec<String> {
        self.0
            .get(key)
            .and_then(Value::as_sequence)
            .map(|seq| {
                seq.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Test/construction helper — production configs come from YAML via `serde`.
    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.0.insert(key.to_string(), Value::Bool(value));
    }

    /// Test/construction helper — production configs come from YAML via `serde`.
    pub fn set_str(&mut self, key: &str, value: &str) {
        self.0
            .insert(key.to_string(), Value::String(value.to_string()));
    }

    /// Test/construction helper — production configs come from YAML via `serde`.
    pub fn set_f64(&mut self, key: &str, value: f64) {
        self.0.insert(
            key.to_string(),
            Value::Number(serde_yaml::Number::from(value)),
        );
    }

    /// Test/construction helper — production configs come from YAML via `serde`.
    pub fn set_str_list(&mut self, key: &str, values: &[&str]) {
        self.0.insert(
            key.to_string(),
            Value::Sequence(
                values
                    .iter()
                    .map(|v| Value::String(v.to_string()))
                    .collect(),
            ),
        );
    }

    /// A YAML sequence of structured objects — for option shapes the plain
    /// `*_option` getters above can't express, e.g. `custom_regex`'s
    /// `{rule_id, pattern, severity}` per entry. `Ok(vec![])` when the key
    /// is absent (same "unset = empty" convention as `str_list_option`);
    /// `Err` only when the key IS present but doesn't deserialize into `T`
    /// — a config-shape mistake the caller should treat as fatal (e.g.
    /// `detectors::custom_regex::validate`, called at startup so this
    /// surfaces before the server ever accepts a request), not silently
    /// dropped like a bad individual list entry is elsewhere.
    pub fn struct_list_option<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Vec<T>, String> {
        match self.0.get(key) {
            None => Ok(Vec::new()),
            Some(value) => {
                serde_yaml::from_value(value.clone()).map_err(|e| format!("option {key:?}: {e}"))
            }
        }
    }

    /// General setter for an arbitrary parsed YAML value — used by
    /// `armor-api`'s custom-rules-directory loader to fold external file
    /// content into a check's options at startup. The `set_bool`/`set_str`/
    /// etc. helpers above exist only for test-code readability; this is the
    /// one real production caller.
    pub fn set_raw(&mut self, key: &str, value: Value) {
        self.0.insert(key.to_string(), value);
    }
}

/// Fail-open vs. fail-closed resolution on a check's timeout or error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailMode {
    #[default]
    FailOpen,
    FailClosed,
}

fn default_true() -> bool {
    true
}

fn default_on_fail() -> CheckAction {
    CheckAction::Deny
}

fn default_mode() -> EnforcementMode {
    EnforcementMode::Block
}

/// One detector's configuration within a [`PolicyConfig`], mirroring `CheckSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckConfig {
    pub category: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub options: CheckOptions,
    #[serde(default = "default_on_fail")]
    pub on_fail: CheckAction,
    #[serde(default)]
    pub fail_mode: FailMode,
    #[serde(default = "default_mode")]
    pub mode: EnforcementMode,
    /// How this check escalates beyond the deterministic layer.
    /// **No `strategy` ⇒ the deterministic path, byte-identical** — that
    /// rule is why every field here is `Option`/`#[serde(default)]` and why
    /// every shipped profile in `config/profiles/` keeps parsing unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<Strategy>,
    /// Which backend serves each layer named in `strategy.order`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub backends: HashMap<ExecutionLayer, Backend>,
    /// Benchmark quality metrics for this check's model, used by the
    /// scorecard gate to decide whether the model's verdict may be enforced
    /// or must be advisory-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scorecard: Option<ScorecardMetrics>,
}

/// Mirrors the `serde` defaults above field-for-field, so a hand-built
/// `CheckConfig { category, ..Default::default() }` and one deserialized from
/// a minimal YAML `- category: pci` are the same value. Exists mainly so
/// construction sites don't have to be touched every time this struct grows
/// an optional field.
impl Default for CheckConfig {
    fn default() -> Self {
        Self {
            category: String::new(),
            enabled: default_true(),
            options: CheckOptions::default(),
            on_fail: default_on_fail(),
            fail_mode: FailMode::default(),
            mode: default_mode(),
            strategy: None,
            backends: HashMap::new(),
            scorecard: None,
        }
    }
}

/// Where a check's verdict came from. Ordered cheapest-to-strongest, which
/// is the order `Strategy::order` normally lists them in.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionLayer {
    /// Regex, Luhn, entropy, allow/deny lists — the deterministic checks
    /// this crate ships today.
    #[default]
    LocalDeterministic,
    /// A local classifier or NER head served by the sidecar.
    LocalMl,
    LocalEmbedding,
    /// The generative judge, run as the `judge` task.
    LocalLlm,
    RemoteLlm,
}

/// What to do with a check whose backend failed. The default keeps the
/// deterministic answer, which is why a sidecar outage degrades detection
/// quality rather than availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fallback {
    #[default]
    FallbackToDeterministic,
    /// Treat the failure as a pass. Only sane for a check whose ML layer is
    /// the *only* real signal and whose deterministic tier is a stub.
    FailOpen,
    /// Treat the failure as a deny. Fails the request closed on a sidecar
    /// outage — correct for a small number of high-assurance checks, and a
    /// self-inflicted outage for anything else.
    FailClosed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    /// Layers to try, in order. The first entry is normally
    /// `local_deterministic`; a strategy that omits it still runs the
    /// deterministic sweep (that is unconditional) and simply treats its
    /// result as the escalation input.
    pub order: Vec<ExecutionLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalate_when: Option<EscalateWhen>,
    #[serde(default)]
    pub fallback: Fallback,
    /// Merge the ML layer's hits into the deterministic ones instead of
    /// replacing them. Required for `pii`, where the NER layer adds
    /// unstructured findings and must never erase the regex layer's
    /// redaction.
    #[serde(default)]
    pub additive: bool,
    /// Whole-strategy budget. The per-backend deadline nests inside it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

/// The escalation predicate. Both terms are optional and ANDed; a `Strategy`
/// with no `escalate_when` always escalates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EscalateWhen {
    /// Escalate only when the deterministic layer's [`risk_score`] falls
    /// **inside** `[low, high]`, inclusive. A high score means the rules are
    /// already confident and a forward pass would buy nothing; score 0 means
    /// they found nothing, which is exactly where a classifier earns its
    /// keep. The band subsumes the routing modes — `[0,100]` is "always",
    /// `[20,70]` is "gray zone only", and no `strategy` at all is "never".
    ///
    /// [`risk_score`]: crate::engine::escalation::risk_score
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic_score_between: Option<(u8, u8)>,
    /// Escalate to the next layer only when the layer that just ran reported
    /// confidence below this. **A missing confidence escalates** — an
    /// abstention should fail toward the stronger layer, not resolve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ml_confidence_below: Option<f32>,
}

/// Which remote task serves a layer, and how it is pinned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backend {
    /// Registry task name on the sidecar, e.g. `"prompt_injection"`.
    pub task: String,
    /// `None` ⇒ the default sidecar URL from `ARMOR_INFERENCE_URL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    #[default]
    Parallel,
    Sequential,
}

/// A named set of checks and how to run them, mirroring `GuardrailSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub id: String,
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    #[serde(default)]
    pub fail_mode: FailMode,
    #[serde(default)]
    pub normalize: NormalizeConfig,
    #[serde(default)]
    pub checks: Vec<CheckConfig>,
}

/// Wire (YAML) shape of the normalize toggles — mirrors
/// [`NormalizeOptions`] but as an independently-derivable serde type so
/// `NormalizeOptions` itself stays free of a serde dependency edge.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NormalizeConfig {
    #[serde(default)]
    pub unicode_nfkc: bool,
    #[serde(default)]
    pub strip_invisible: bool,
    #[serde(default)]
    pub deleet: bool,
    #[serde(default)]
    pub html_entities: bool,
    #[serde(default)]
    pub homoglyph: bool,
    #[serde(default)]
    pub collapse_spacing: bool,
    #[serde(default)]
    pub rot13: bool,
    #[serde(default)]
    pub base64: bool,
}

impl From<NormalizeConfig> for NormalizeOptions {
    fn from(c: NormalizeConfig) -> Self {
        Self {
            unicode_nfkc: c.unicode_nfkc,
            strip_invisible: c.strip_invisible,
            deleet: c.deleet,
            html_entities: c.html_entities,
            homoglyph: c.homoglyph,
            collapse_spacing: c.collapse_spacing,
            rot13: c.rot13,
            base64: c.base64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The non-negotiable rule: a check with no `strategy` is the existing
    /// deterministic path, byte-identical. Every shipped profile in
    /// `config/profiles/` predates these fields, so this is what keeps them
    /// parsing.
    #[test]
    fn a_check_without_the_new_fields_still_parses() {
        let check: CheckConfig = serde_yaml::from_str("category: pci").unwrap();
        assert_eq!(check.category, "pci");
        assert!(check.strategy.is_none());
        assert!(check.backends.is_empty());
        assert!(check.enabled);
        assert_eq!(check.mode, EnforcementMode::Block);
    }

    #[test]
    fn a_minimal_check_deserializes_to_the_same_value_as_default() {
        let parsed: CheckConfig = serde_yaml::from_str("category: pci").unwrap();
        let built = CheckConfig {
            category: "pci".to_string(),
            ..Default::default()
        };
        assert_eq!(
            serde_yaml::to_string(&parsed).unwrap(),
            serde_yaml::to_string(&built).unwrap()
        );
    }

    #[test]
    fn a_full_strategy_parses_from_yaml() {
        let yaml = r#"
category: prompt_injection
mode: warn
strategy:
  order: [local_deterministic, local_ml, local_llm]
  escalate_when:
    deterministic_score_between: [0, 70]
    ml_confidence_below: 0.8
  fallback: fallback_to_deterministic
  additive: false
  timeout_ms: 120
backends:
  local_ml:
    task: prompt_injection
    model_id: protectai/deberta-v3-base-prompt-injection-v2
    revision: main
    threshold: 0.7
  local_llm:
    task: judge
    timeout_ms: 50
"#;
        let check: CheckConfig = serde_yaml::from_str(yaml).unwrap();
        let strategy = check.strategy.expect("strategy present");

        assert_eq!(
            strategy.order,
            vec![
                ExecutionLayer::LocalDeterministic,
                ExecutionLayer::LocalMl,
                ExecutionLayer::LocalLlm,
            ]
        );
        let w = strategy.escalate_when.expect("escalate_when present");
        assert_eq!(w.deterministic_score_between, Some((0, 70)));
        assert_eq!(w.ml_confidence_below, Some(0.8));
        assert_eq!(strategy.fallback, Fallback::FallbackToDeterministic);
        assert_eq!(strategy.timeout_ms, Some(120));

        assert_eq!(
            check.backends[&ExecutionLayer::LocalMl].task,
            "prompt_injection"
        );
        assert_eq!(check.backends[&ExecutionLayer::LocalLlm].task, "judge");
        assert_eq!(
            check.backends[&ExecutionLayer::LocalLlm].timeout_ms,
            Some(50)
        );
    }

    #[test]
    fn the_pii_additive_shape_parses() {
        // NER adds to the regex layer, never replaces it.
        let yaml = r#"
category: pii
strategy:
  order: [local_deterministic, local_ml]
  additive: true
  escalate_when:
    deterministic_score_between: [0, 100]
backends:
  local_ml:
    task: ner
"#;
        let check: CheckConfig = serde_yaml::from_str(yaml).unwrap();
        let strategy = check.strategy.unwrap();
        assert!(strategy.additive);
        assert_eq!(
            strategy.escalate_when.unwrap().deterministic_score_between,
            Some((0, 100))
        );
    }

    #[test]
    fn a_check_with_no_strategy_serializes_without_the_new_keys() {
        // Keeps the audit/profile-export shape stable for policies that
        // never opted into an inference tier.
        let yaml = serde_yaml::to_string(&CheckConfig {
            category: "pci".to_string(),
            ..Default::default()
        })
        .unwrap();
        assert!(!yaml.contains("strategy"), "{yaml}");
        assert!(!yaml.contains("backends"), "{yaml}");
    }

    #[test]
    fn execution_layer_names_are_snake_case_on_the_wire() {
        let layer: ExecutionLayer = serde_yaml::from_str("local_ml").unwrap();
        assert_eq!(layer, ExecutionLayer::LocalMl);
        assert_eq!(
            serde_yaml::to_string(&ExecutionLayer::RemoteLlm)
                .unwrap()
                .trim(),
            "remote_llm"
        );
    }
}
