# config/custom_rules/

Per-deployment rule *data*, kept separate from `config/policies.yaml` so it
can change without a policy redeploy. Loaded once at startup by
`armor-api`'s `custom_rules` module (`ARMOR_CUSTOM_RULES_DIR`, defaults to
this directory) and folded into the matching check's `options` before the
server starts — there's no hot-reload, same as `config/policies.yaml`
itself.

## Convention

One file per check *category*, named `<category>.yaml`. Its top-level keys
are merged into `options` for every check in the loaded policy with that
`category` — a key here overwrites the same key if `policies.yaml` also set
it. The check itself (`enabled`/`mode`/`on_fail`) still has to
already exist in `policies.yaml`; a file here only supplies data for a check
that's already declared, it never creates one.

This isn't `custom_regex`-specific — the same mechanism works for any
detector's option keys, e.g. `keyword_blocklist.yaml` with a `keywords:`
list, or `tool_allowlist.yaml` with `allow:`/`deny:` lists.

## `custom_regex.yaml`

```yaml
patterns:
  - rule_id: employee-id       # required — becomes the RuleHit's rule_id
    pattern: 'EMP-\d{4}'       # required — Rust `regex` syntax (no lookaround/backreferences)
    severity: high             # optional, default: medium (low|medium|high|critical)
    case_sensitive: false      # optional, default: false
```

See `custom_regex.yaml.example` in this directory for a runnable copy —
rename it to `custom_regex.yaml` to activate it (and flip
`policies.yaml`'s `custom_regex` check to `enabled: true`).

These patterns get none of the recall/false-positive vetting the shipped
rulesets in `rules/` get (see `crates/core/tests/*_benchmark.rs`) — that's
why the shipped policy ships this check `mode: warn`. Promote to `block`
only once you've validated your own patterns against real traffic.
