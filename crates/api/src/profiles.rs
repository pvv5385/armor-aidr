//! File-based `application_id -> profile` resolution — the lightweight
//! stand-in for a planned Postgres-backed multi-tenant policy store
//! (`crates/storage/src/policy_store.rs`, still a stub). No new
//! infrastructure: profiles are policy YAML files with the same schema
//! `config/policies.yaml` already uses (`armor_core::policy::schema::PolicyConfig`,
//! one file per profile, each with its own top-level `id`), and the
//! application/profile mapping is one more YAML file. Both are read once at
//! startup, same as `config/policies.yaml` and `ARMOR_CUSTOM_RULES_DIR`
//! already are — no hot-reload, no runtime CRUD API.
//!
//! Off by default: an unconfigured deployment (`ARMOR_PROFILES_DIR`/
//! `ARMOR_APPLICATIONS_PATH` pointing at directories/files that don't
//! exist, the shipped default) gets `ProfileResolver::single(default)` —
//! every request resolves to the one default policy, byte-for-byte the
//! same behavior the engine always had before profiles existed.
//! `metadata.application_id` was already
//! threaded through every request (`aidr::AidrScanRequest`); this is what
//! makes it actually change which checks run.

use std::{collections::HashMap, path::Path, sync::Arc};

use anyhow::Context;
use armor_core::policy::schema::PolicyConfig;
use serde::Deserialize;

/// Resolves an `application_id` to the `PolicyConfig` its assigned profile
/// should run — falls back to the default profile for an absent or unknown
/// `application_id`, so an unrecognized id is never a hard error, just a
/// silent "run the default checks" (logged at `debug` so it's discoverable
/// without being noisy).
#[derive(Clone)]
pub struct ProfileResolver {
    default: Arc<PolicyConfig>,
    by_application_id: Arc<HashMap<String, Arc<PolicyConfig>>>,
}

impl ProfileResolver {
    /// No named profiles configured — every request runs `default`. Used
    /// both by `main.rs` when `profiles_dir`/`applications_path` don't
    /// exist, and directly by tests that don't care about multi-profile
    /// resolution.
    pub fn single(default: Arc<PolicyConfig>) -> Self {
        Self {
            default,
            by_application_id: Arc::new(HashMap::new()),
        }
    }

    /// Construct from pre-built parts. Used by `sync.rs` after it has
    /// compiled a new policy set from the control-plane sync payload —
    /// avoids re-reading the filesystem.
    pub fn from_parts(
        default: Arc<PolicyConfig>,
        by_application_id: HashMap<String, Arc<PolicyConfig>>,
    ) -> Self {
        Self {
            default,
            by_application_id: Arc::new(by_application_id),
        }
    }

    /// Every distinct policy this resolver can return — the default plus
    /// each uniquely-`id`'d profile referenced by `by_application_id`,
    /// deduped so a profile mapped to N application ids is counted once.
    /// Used by `sync.rs`'s pre-swap diff, which needs the whole old/new
    /// policy sets rather than just whatever one `resolve()` call returns.
    pub fn all_policies(&self) -> Vec<Arc<PolicyConfig>> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for policy in std::iter::once(&self.default).chain(self.by_application_id.values()) {
            if seen.insert(policy.id.clone()) {
                result.push(policy.clone());
            }
        }
        result
    }

    pub fn resolve(&self, application_id: Option<&str>) -> Arc<PolicyConfig> {
        let Some(id) = application_id else {
            return self.default.clone();
        };
        match self.by_application_id.get(id) {
            Some(policy) => policy.clone(),
            None => {
                tracing::debug!(
                    application_id = %id,
                    "no profile mapped for this application_id, using default profile"
                );
                self.default.clone()
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApplicationsFile {
    #[serde(default)]
    applications: Vec<ApplicationEntry>,
}

#[derive(Debug, Deserialize)]
struct ApplicationEntry {
    application_id: String,
    profile_id: String,
}

/// Parses `yaml` as a `PolicyConfig` (same validation `policy::loader::load`
/// applies to `config/policies.yaml`: unknown check categories fail fast),
/// then hardens it — factored out here so a named profile gets the
/// identical treatment as the default policy, not a lighter pass.
fn load_and_harden_policy(yaml: &str, custom_rules_dir: &Path) -> anyhow::Result<PolicyConfig> {
    let policy = armor_core::policy::loader::load(yaml)
        .map_err(|e| anyhow::anyhow!("loading policy: {e}"))?;
    harden(policy, custom_rules_dir)
}

/// Folds in `custom_rules_dir` and validates any `custom_regex` check —
/// every policy this process ever runs goes through this exact step
/// regardless of source: a file-based profile (via
/// [`load_and_harden_policy`] above), a control-plane sync payload
/// (`sync.rs`), or a Postgres-backed profile (`main.rs`'s DB boot wiring,
/// `control_plane.rs`'s post-mutation resolver rebuild) — one hardening
/// implementation, not three copies that can drift.
pub(crate) fn harden(
    mut policy: PolicyConfig,
    custom_rules_dir: &Path,
) -> anyhow::Result<PolicyConfig> {
    crate::custom_rules::apply(&mut policy, custom_rules_dir)
        .with_context(|| format!("applying custom rules from {}", custom_rules_dir.display()))?;

    let thresholds = armor_core::engine::scorecard_gate::ScorecardThresholds::default();

    for check in &mut policy.checks {
        if check.category == "custom_regex" && check.enabled {
            armor_core::detectors::custom_regex::validate(&check.options)
                .map_err(|e| anyhow::anyhow!("invalid custom_regex options: {e}"))?;
        }

        // Scorecard gate at policy load: if the model's benchmark metrics
        // fail the gate, disable the check rather than letting it run with
        // potentially untrustworthy results.
        if let Some(ref metrics) = check.scorecard {
            let verdict = armor_core::engine::scorecard_gate::evaluate(metrics, &thresholds);
            if !verdict.may_run() {
                tracing::warn!(
                    category = %check.category,
                    "scorecard gate FAIL at policy load; disabling model-backed check"
                );
                check.enabled = false;
            }
        }
    }

    Ok(policy)
}

/// A per-task model override sourced from the `inference_pins` table
/// (`armor_storage::inference_pins`, DB-backed deployments — see
/// [`pins_from_rows`]) or a control-plane sync payload's `pins` array
/// (`sync.rs`, edge deployments). Kept separate from
/// `armor_storage::inference_pins::InferencePin` / `sync::InferencePinMapping`
/// so this module doesn't need to know which of those two wire shapes a
/// caller happened to load from — both convert into this before calling
/// [`apply_pins`].
#[derive(Debug, Clone)]
pub struct PinOverride {
    pub model_id: String,
    pub revision: String,
}

/// `armor_storage::inference_pins::InferencePin` rows, keyed by `task`, in
/// the shape [`apply_pins`] wants. Factored out because both `main.rs`'s
/// boot-time load and `control_plane.rs`'s post-mutation reload do this
/// exact conversion.
pub fn pins_from_rows(
    rows: Vec<armor_storage::inference_pins::InferencePin>,
) -> HashMap<String, PinOverride> {
    rows.into_iter()
        .map(|row| {
            (
                row.task,
                PinOverride {
                    model_id: row.model_id,
                    revision: row.revision,
                },
            )
        })
        .collect()
}

/// Overwrites every `LocalMl` backend's `model_id`/`revision` with the pin
/// for its `task`, if one exists. This is what makes an `inference_pins` row
/// (written through the control-plane UI, `control_plane.rs`) or a synced pin
/// (`sync.rs`) actually change which model `ml::run_one` asks the sidecar
/// for — without this step, a pin is persisted and displayed back by
/// `GET /api/v1/inference-pins`, but the escalation pass still sends whatever
/// `model_id`/`revision` shipped in the policy's own YAML/DB row, because
/// that `Backend` is all `ml::run_one` ever looks at.
pub(crate) fn apply_pins(policy: &mut PolicyConfig, pins: &HashMap<String, PinOverride>) {
    for check in &mut policy.checks {
        for backend in check.backends.values_mut() {
            if let Some(pin) = pins.get(&backend.task) {
                backend.model_id = Some(pin.model_id.clone());
                backend.revision = Some(pin.revision.clone());
            }
        }
    }
}

/// Hardens every `policy`, applies any `pins` override to their `LocalMl`
/// backends (`apply_pins`), and assembles them (plus every `(application_id,
/// profile_id)` pair) into a `ProfileResolver` — the shape both
/// `sync.rs::build_resolver` (control-plane push) and `main.rs`/
/// `control_plane.rs` (Postgres-backed profiles) need, factored out
/// here since it's now the third caller. Errs if no profile with id
/// `"default"` is present (the resolver's fallback must always exist) or if
/// an application references an unknown `profile_id`.
pub fn resolver_from_policies(
    policies: Vec<PolicyConfig>,
    applications: Vec<(String, String)>,
    custom_rules_dir: &Path,
    pins: &HashMap<String, PinOverride>,
) -> anyhow::Result<ProfileResolver> {
    let mut by_profile_id: HashMap<String, Arc<PolicyConfig>> = HashMap::new();
    for policy in policies {
        let mut hardened = harden(policy, custom_rules_dir)?;
        apply_pins(&mut hardened, pins);
        by_profile_id.insert(hardened.id.clone(), Arc::new(hardened));
    }

    let default = by_profile_id.get("default").cloned().ok_or_else(|| {
        anyhow::anyhow!(
            "no profile with id \"default\" — the resolver's fallback profile must always exist"
        )
    })?;

    let mut by_application_id = HashMap::new();
    for (application_id, profile_id) in applications {
        let policy = by_profile_id.get(&profile_id).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "application_id {application_id:?} references unknown profile_id {profile_id:?}"
            )
        })?;
        by_application_id.insert(application_id, policy);
    }

    Ok(ProfileResolver::from_parts(default, by_application_id))
}

/// Loads every `*.yaml`/`*.yml` file in `profiles_dir` as a named profile
/// keyed by its own `id`, then reads `applications_path`
/// (`application_id -> profile_id` pairs) and resolves each into a
/// ready-to-use `Arc<PolicyConfig>`. `default` is always reachable too,
/// both as the fallback and under its own `id` (so `applications.yaml` can
/// reference it explicitly, not just rely on the fallback).
///
/// Fails fast at startup — a profile referencing an unknown category, an
/// `applications.yaml` entry naming a `profile_id` that doesn't exist among
/// the loaded profiles, or a duplicate profile `id` — rather than silently
/// misrouting requests later. Neither `profiles_dir` nor `applications_path`
/// existing is not an error: that's just "no named profiles configured",
/// same posture `custom_rules_dir` already has.
pub fn load(
    default: Arc<PolicyConfig>,
    profiles_dir: &Path,
    applications_path: &Path,
    custom_rules_dir: &Path,
) -> anyhow::Result<ProfileResolver> {
    let mut by_profile_id: HashMap<String, Arc<PolicyConfig>> = HashMap::new();
    by_profile_id.insert(default.id.clone(), default.clone());

    if profiles_dir.exists() {
        for entry in std::fs::read_dir(profiles_dir)
            .with_context(|| format!("reading {}", profiles_dir.display()))?
        {
            let path = entry
                .with_context(|| format!("reading {}", profiles_dir.display()))?
                .path();
            let is_yaml = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| ext == "yaml" || ext == "yml");
            if !path.is_file() || !is_yaml {
                continue;
            }

            let yaml = std::fs::read_to_string(&path)
                .with_context(|| format!("reading profile {}", path.display()))?;
            let policy = load_and_harden_policy(&yaml, custom_rules_dir)
                .with_context(|| format!("loading profile {}", path.display()))?;

            if by_profile_id.contains_key(&policy.id) {
                anyhow::bail!(
                    "duplicate profile id {:?} ({} collides with an earlier profile or the default policy's own id)",
                    policy.id,
                    path.display()
                );
            }
            tracing::info!(profile_id = %policy.id, checks = policy.checks.len(), path = %path.display(), "loaded profile");
            by_profile_id.insert(policy.id.clone(), Arc::new(policy));
        }
    }

    let mut by_application_id = HashMap::new();
    if applications_path.exists() {
        let yaml = std::fs::read_to_string(applications_path)
            .with_context(|| format!("reading {}", applications_path.display()))?;
        let applications: ApplicationsFile = serde_yaml::from_str(&yaml)
            .with_context(|| format!("parsing {}", applications_path.display()))?;

        for entry in applications.applications {
            let Some(policy) = by_profile_id.get(&entry.profile_id) else {
                anyhow::bail!(
                    "application_id {:?} in {} references unknown profile_id {:?}",
                    entry.application_id,
                    applications_path.display(),
                    entry.profile_id
                );
            };
            tracing::info!(
                application_id = %entry.application_id,
                profile_id = %entry.profile_id,
                "mapped application to profile"
            );
            by_application_id.insert(entry.application_id, policy.clone());
        }
    }

    Ok(ProfileResolver {
        default,
        by_application_id: Arc::new(by_application_id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use armor_core::policy::schema::{
        Backend, CheckConfig, ExecutionLayer, ExecutionMode, FailMode, NormalizeConfig,
    };

    fn policy(id: &str) -> Arc<PolicyConfig> {
        Arc::new(PolicyConfig {
            id: id.to_string(),
            execution_mode: ExecutionMode::Parallel,
            fail_mode: FailMode::FailOpen,
            normalize: NormalizeConfig::default(),
            checks: Vec::new(),
        })
    }

    fn ml_backend(task: &str) -> Backend {
        Backend {
            task: task.to_string(),
            endpoint_url: None,
            model_id: None,
            revision: None,
            threshold: None,
            timeout_ms: None,
            params: None,
        }
    }

    fn check_with_backend(category: &str, backend: Backend) -> CheckConfig {
        let mut backends = HashMap::new();
        backends.insert(ExecutionLayer::LocalMl, backend);
        CheckConfig {
            category: category.to_string(),
            backends,
            ..Default::default()
        }
    }

    #[test]
    fn a_matching_pin_overrides_the_backends_model_id_and_revision() {
        let mut policy = PolicyConfig {
            id: "default".to_string(),
            execution_mode: ExecutionMode::Parallel,
            fail_mode: FailMode::FailOpen,
            normalize: NormalizeConfig::default(),
            checks: vec![check_with_backend(
                "prompt_injection",
                ml_backend("prompt_injection"),
            )],
        };

        let mut pins = HashMap::new();
        pins.insert(
            "prompt_injection".to_string(),
            PinOverride {
                model_id: "pinned-model".to_string(),
                revision: "v3".to_string(),
            },
        );

        apply_pins(&mut policy, &pins);

        let backend = &policy.checks[0].backends[&ExecutionLayer::LocalMl];
        assert_eq!(backend.model_id.as_deref(), Some("pinned-model"));
        assert_eq!(backend.revision.as_deref(), Some("v3"));
    }

    #[test]
    fn a_backend_with_no_matching_pin_is_left_untouched() {
        let mut policy = PolicyConfig {
            id: "default".to_string(),
            execution_mode: ExecutionMode::Parallel,
            fail_mode: FailMode::FailOpen,
            normalize: NormalizeConfig::default(),
            checks: vec![check_with_backend("secrets", ml_backend("secrets"))],
        };

        let mut pins = HashMap::new();
        pins.insert(
            "prompt_injection".to_string(),
            PinOverride {
                model_id: "pinned-model".to_string(),
                revision: "v3".to_string(),
            },
        );

        apply_pins(&mut policy, &pins);

        let backend = &policy.checks[0].backends[&ExecutionLayer::LocalMl];
        assert_eq!(backend.model_id, None);
        assert_eq!(backend.revision, None);
    }

    #[test]
    fn single_always_resolves_to_default() {
        let resolver = ProfileResolver::single(policy("default"));
        assert_eq!(resolver.resolve(None).id, "default");
        assert_eq!(resolver.resolve(Some("unknown-app")).id, "default");
    }

    #[test]
    fn unconfigured_dirs_fall_back_to_default_only() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = load(
            policy("default"),
            &dir.path().join("no-such-profiles-dir"),
            &dir.path().join("no-such-applications.yaml"),
            &dir.path().join("no-such-custom-rules-dir"),
        )
        .unwrap();
        assert_eq!(resolver.resolve(Some("anything")).id, "default");
    }

    #[test]
    fn named_profile_is_resolved_by_mapped_application_id() {
        let dir = tempfile::tempdir().unwrap();
        let profiles_dir = dir.path().join("profiles");
        std::fs::create_dir_all(&profiles_dir).unwrap();
        std::fs::write(profiles_dir.join("strict.yaml"), "id: strict\nchecks: []\n").unwrap();

        let applications_path = dir.path().join("applications.yaml");
        std::fs::write(
            &applications_path,
            "applications:\n  - application_id: travel-assistant\n    profile_id: strict\n",
        )
        .unwrap();

        let resolver = load(
            policy("default"),
            &profiles_dir,
            &applications_path,
            &dir.path().join("no-such-custom-rules-dir"),
        )
        .unwrap();

        assert_eq!(resolver.resolve(Some("travel-assistant")).id, "strict");
        assert_eq!(resolver.resolve(Some("some-other-app")).id, "default");
        assert_eq!(resolver.resolve(None).id, "default");
    }

    #[test]
    fn default_profile_is_reachable_by_its_own_id() {
        let dir = tempfile::tempdir().unwrap();
        let applications_path = dir.path().join("applications.yaml");
        std::fs::write(
            &applications_path,
            "applications:\n  - application_id: some-app\n    profile_id: default\n",
        )
        .unwrap();

        let resolver = load(
            policy("default"),
            &dir.path().join("no-such-profiles-dir"),
            &applications_path,
            &dir.path().join("no-such-custom-rules-dir"),
        )
        .unwrap();

        assert_eq!(resolver.resolve(Some("some-app")).id, "default");
    }

    /// Guards against the shipped `.example` files (never auto-loaded —
    /// `profiles_dir`/`applications_path` only pick up real `.yaml`/`.yml`
    /// files, and the default `applications_path` doesn't exist out of the
    /// box) silently rotting out of sync with the real schema.
    #[test]
    fn shipped_example_profile_and_applications_file_parse() {
        let dir = tempfile::tempdir().unwrap();
        let profiles_dir = dir.path().join("profiles");
        std::fs::create_dir_all(&profiles_dir).unwrap();
        let example_profile = include_str!("../../../config/profiles/minimal-example.yaml.example");
        std::fs::write(profiles_dir.join("minimal.yaml"), example_profile).unwrap();

        let applications_path = dir.path().join("applications.yaml");
        let example_applications = include_str!("../../../config/applications.yaml.example");
        std::fs::write(&applications_path, example_applications).unwrap();

        let resolver = load(
            policy("default"),
            &profiles_dir,
            &applications_path,
            &dir.path().join("no-such-custom-rules-dir"),
        )
        .unwrap();

        assert_eq!(
            resolver.resolve(Some("travel-assistant-prod")).id,
            "minimal"
        );
    }

    #[test]
    fn unknown_profile_id_in_applications_file_fails_fast() {
        let dir = tempfile::tempdir().unwrap();
        let applications_path = dir.path().join("applications.yaml");
        std::fs::write(
            &applications_path,
            "applications:\n  - application_id: some-app\n    profile_id: does-not-exist\n",
        )
        .unwrap();

        let result = load(
            policy("default"),
            &dir.path().join("no-such-profiles-dir"),
            &applications_path,
            &dir.path().join("no-such-custom-rules-dir"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn duplicate_profile_id_fails_fast() {
        let dir = tempfile::tempdir().unwrap();
        let profiles_dir = dir.path().join("profiles");
        std::fs::create_dir_all(&profiles_dir).unwrap();
        std::fs::write(profiles_dir.join("a.yaml"), "id: dup\nchecks: []\n").unwrap();
        std::fs::write(profiles_dir.join("b.yaml"), "id: dup\nchecks: []\n").unwrap();

        let result = load(
            policy("default"),
            &profiles_dir,
            &dir.path().join("no-such-applications.yaml"),
            &dir.path().join("no-such-custom-rules-dir"),
        );
        assert!(result.is_err());
    }
}
