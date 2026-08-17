# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because AI Armor is a security control, two kinds of change get called out
explicitly wherever they occur, regardless of version:

- **Detection changes** — a new rule, a widened or narrowed pattern, a
  category promoted from `warn` to `block`. These change what a deployment
  catches, and an operator needs to know before upgrading.
- **Default changes** — anything that alters behavior for someone who has not
  edited `config/policies.yaml`.

## [Unreleased]

Everything to date. The project is pre-1.0 and has not cut a tagged release
yet; `main` is the only supported line. See [`SECURITY.md`](SECURITY.md).

### Added

- Detection engine (`armor-core`): synchronous, no I/O, with rulesets for
  secrets, PII, PCI, prompt injection, jailbreak, harmful content, code
  safety, exfiltration, system-prompt leakage, MCP manifest scanning, and
  others. Patterns are embedded at compile time.
- HTTP service (`armor-api`): the `/api/v1/aidr/scan` data plane, a
  control-plane API and browser UI, and auth, audit, and rate-limit
  middleware.
- Postgres-backed storage (`armor-storage`): session state and an encrypted
  PII vault with anonymize/deanonymize and right-to-erasure.
- Optional model-backed sidecar (`armor-inference`): a separate Python
  deployable serving classifier, embedding, NER, and NLI runners over ONNX.
  Boots and serves on stub runners with no ML stack and no weights present;
  `armor-core` reaches it only when `ARMOR_INFERENCE_URL` is set.
- Gateway integrations for LiteLLM and Portkey.
- Benchmark corpora under `config/benchmarks/` with per-category
  precision/recall tests.

### Notes on defaults

The shipped defaults optimize for a first run on a laptop, not a hardened
deployment: `ARMOR_AUTH_MODE=none`, `ARMOR_RATE_LIMIT_MODE=none`, most
detectors at `mode: warn`, and `fail_mode: fail_open`. These are documented
choices rather than oversights — see the "A note on defaults" section of
[`SECURITY.md`](SECURITY.md) before exposing a deployment to a network.

[Unreleased]: https://github.com/pvv5385/armor-aidr/commits/main
