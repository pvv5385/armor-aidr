//! Loads customer-supplied rule data from `ARMOR_CUSTOM_RULES_DIR` and folds
//! it into the loaded policy's check `options` before the server starts.
//! This is the one common place any detector's runtime-editable data lives
//! — `custom_regex` patterns today, but the same mechanism works for
//! `keyword_blocklist.keywords`, `tool_allowlist.allow`/`deny`,
//! `malicious_url.popular_domains`, or any other detector's option keys —
//! without a bespoke loader per detector. This is where the file I/O for
//! that data belongs: `armor-core` stays synchronous with no I/O, so
//! reading and parsing customer files happens here and the result is
//! handed to `armor-core` as plain already-parsed option values.
//!
//! Convention: one YAML file per check *category*, named `<category>.yaml`
//! (e.g. `custom_rules/custom_regex.yaml`), whose top-level mapping is
//! merged into `options` for every check in the policy with that category.
//! A key here overwrites the same key if `config/policies.yaml` also set
//! it, since this directory represents the deployment's latest
//! customer-owned truth — the *check itself* (`enabled`/`mode`/`on_fail`)
//! still has to exist in `config/policies.yaml`; this only supplies data
//! for a check that's already there, it never creates one. Read once at
//! startup — no hot-reload, same as `config/policies.yaml` itself.

use std::{collections::HashMap, path::Path};

use anyhow::Context;
use armor_core::policy::schema::PolicyConfig;

/// `dir` missing entirely is not an error — the feature is simply off
/// (the default `ARMOR_CUSTOM_RULES_DIR` points at a directory most
/// deployments won't have created).
pub fn apply(policy: &mut PolicyConfig, dir: &Path) -> anyhow::Result<()> {
    if !dir.exists() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry
            .with_context(|| format!("reading {}", dir.display()))?
            .path();
        let is_yaml = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "yaml" || e == "yml");
        if !is_yaml {
            continue;
        }

        let category = path
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("{}: non-UTF-8 filename", path.display()))?
            .to_string();

        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let overrides: HashMap<String, serde_yaml::Value> =
            serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;

        let mut matched = false;
        for check in policy.checks.iter_mut().filter(|c| c.category == category) {
            matched = true;
            for (key, value) in &overrides {
                check.options.set_raw(key, value.clone());
            }
        }

        if matched {
            tracing::info!(file = %path.display(), category = %category, keys = overrides.len(), "applied custom rules file");
        } else {
            tracing::warn!(
                file = %path.display(),
                category = %category,
                "custom rules file has no matching check (category) in the loaded policy — ignored"
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use armor_core::policy::schema::PolicyConfig;

    fn policy_with_check(category: &str) -> PolicyConfig {
        let yaml = format!("id: test\nchecks:\n  - category: {category}\n");
        armor_core::policy::loader::load(&yaml).unwrap()
    }

    #[test]
    fn missing_dir_is_not_an_error() {
        let mut policy = policy_with_check("custom_regex");
        apply(&mut policy, Path::new("/nonexistent/does/not/exist")).unwrap();
        assert!(policy.checks[0]
            .options
            .struct_list_option::<serde_yaml::Value>("patterns")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn merges_matching_category_file_into_options() {
        let dir = tempdir();
        std::fs::write(
            dir.path().join("custom_regex.yaml"),
            "patterns:\n  - rule_id: employee-id\n    pattern: 'EMP-\\d{4}'\n",
        )
        .unwrap();

        let mut policy = policy_with_check("custom_regex");
        apply(&mut policy, dir.path()).unwrap();

        let patterns = policy.checks[0]
            .options
            .struct_list_option::<serde_yaml::Value>("patterns")
            .unwrap();
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn file_with_no_matching_check_is_ignored_not_fatal() {
        let dir = tempdir();
        std::fs::write(
            dir.path().join("nonexistent_category.yaml"),
            "keywords: [foo]\n",
        )
        .unwrap();

        let mut policy = policy_with_check("custom_regex");
        apply(&mut policy, dir.path()).unwrap();
    }

    #[test]
    fn non_yaml_files_are_skipped() {
        let dir = tempdir();
        std::fs::write(dir.path().join("README.md"), "not yaml").unwrap();

        let mut policy = policy_with_check("custom_regex");
        apply(&mut policy, dir.path()).unwrap();
    }

    /// Minimal scratch-dir helper — avoids pulling in `tempfile` just for
    /// these three tests.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TempDir {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "armor-api-custom-rules-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
}
