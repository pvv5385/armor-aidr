//! Parses `config/policies.yaml` into [`super::schema::PolicyConfig`] and
//! validates every check's `category` resolves to a real detector — fail
//! fast on a config typo at startup rather than silently no-op'ing that
//! check at request time.

use thiserror::Error;

use crate::detectors::get_check;
use crate::policy::schema::PolicyConfig;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("failed to parse policy YAML: {0}")]
    Parse(#[from] serde_yaml::Error),
    #[error("policy {policy_id:?} check #{index} has unknown category {category:?}")]
    UnknownCategory {
        policy_id: String,
        index: usize,
        category: String,
    },
}

pub fn load(yaml: &str) -> Result<PolicyConfig, LoadError> {
    let policy: PolicyConfig = serde_yaml::from_str(yaml)?;
    validate(&policy)?;
    Ok(policy)
}

/// Checks that every check's `category` resolves to a real detector —
/// factored out of [`load`] so callers that already have a parsed
/// `PolicyConfig` from a non-YAML source (the control-plane CRUD API,
/// `armor-storage`'s DB-backed profiles) get the same fail-fast validation
/// a YAML file gets, instead of a check silently no-op'ing at request time.
pub fn validate(policy: &PolicyConfig) -> Result<(), LoadError> {
    for (index, check) in policy.checks.iter().enumerate() {
        if get_check(&check.category).is_none() {
            return Err(LoadError::UnknownCategory {
                policy_id: policy.id.clone(),
                index,
                category: check.category.clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_the_shipped_default_policy() {
        let yaml = include_str!("../../../../config/policies.yaml");
        let policy = load(yaml).expect("config/policies.yaml must load");
        assert_eq!(policy.id, "default");
        assert!(!policy.checks.is_empty());
    }

    #[test]
    fn rejects_unknown_category() {
        let yaml = "id: bad\nchecks:\n  - category: not_a_real_check\n";
        let err = load(yaml).unwrap_err();
        assert!(matches!(err, LoadError::UnknownCategory { .. }));
    }
}
