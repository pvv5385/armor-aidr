<!--
Security fixes: please do not open a public PR for an unreported
vulnerability. See SECURITY.md.
-->

## What this changes

<!-- What and why. Link the issue if there is one. -->

## Checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo fmt --all -- --check` passes
- [ ] Every commit is signed off (`git commit -s`) — see [`DCO.md`](DCO.md)

## If this touches `rules/`

- [ ] Patterns use no lookaround and no backreferences
- [ ] At least two positive samples per new rule in `config/benchmarks/`
- [ ] Negative samples added for near-misses that must *not* fire
- [ ] `NOTICE` updated if third-party pattern text was incorporated
- [ ] New categories ship `mode: warn` in `config/policies.yaml`

**False-positive impact:** <!-- Did the category's FPR move? By how much? -->

## If this touches the engine or the API

- [ ] `armor-core` remains synchronous with no I/O
- [ ] No change to the `/api/v1/aidr/scan` response shape, or the change is
      called out below and discussed in an issue

## Anything reviewers should look at closely

<!-- Tradeoffs you made, alternatives you rejected, parts you are unsure about. -->
