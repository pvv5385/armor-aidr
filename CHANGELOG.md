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

### Changed

- `/readyz` performs a real readiness check instead of returning `200`
  unconditionally. With `DATABASE_URL` configured, it probes the pool (2s
  deadline) and returns `503
  {"status":"not_ready","dependency":"database"}` when it cannot answer, so
  a replica that has lost its control-plane database stops receiving traffic.
  `/healthz` stays unconditional — liveness and readiness are now distinct.
  Deliberately not gated on the Redis rate limiter (fails open) or the
  inference sidecar (absorbed by the circuit breaker). **Operators pointing
  a Kubernetes `readinessProbe` at `/readyz` will see pods go NotReady during
  a database outage where they previously stayed Ready.**
- The `armor-inference` serving image installs from a hash-pinned lock
  (`inference/requirements-serve.lock`) rather than resolving `>=` floors at
  build time, making the image reproducible. The `[export]` stack and
  `--build-arg WITH_EXPORT=true` are unchanged and remain unlocked.
- `Dockerfile` copies `rust-toolchain.toml` into the builder, so release
  images are built with the same pinned toolchain CI tests with rather than
  whatever `rust:1-bookworm` resolves to.
- Rate limiting counts an authenticated request against its API key rather
  than its client IP. Only a key that `ARMOR_API_KEYS` actually contains earns
  a bucket of its own — an absent or unrecognized key still counts against the
  client IP, because bucketing on an unvalidated key would let a caller mint a
  fresh budget per request. No effect when `ARMOR_AUTH_MODE=none` (the
  default). **Operators running a fleet of workers that share one API key now
  need `ARMOR_RATE_LIMIT_RPS` sized for the fleet: the budget follows the
  credential, not the address.**
- `cargo deny check` replaces `cargo audit` in the dependency-audit workflow.
  Same RustSec advisory database, plus license policy (this repository is
  Apache-2.0 with a hand-written NOTICE, and nothing previously stopped a
  copyleft crate arriving transitively) and dependency provenance. Policy and
  per-license reasoning live in `deny.toml`.

### Added

- `.dockerignore` — the build context no longer includes `target/`, `.git/`,
  `.env`, or Python build artifacts.
- `pip-audit` runs against the sidecar's serving lock in the dependency-audit
  workflow, alongside the Rust-side advisory gate.
- CI asserts `inference/requirements-serve.lock` is in sync with
  `inference/pyproject.toml`.
- Cross-boundary contract job: CI now boots `armor-inference` on its stub
  runners and runs `it_speaks_to_the_real_sidecar` against it with a real
  `HttpTransport`. The wire contract is written out by hand in
  `contract.rs`, `contract.py`, and `inference.proto`; until now every test
  on either side answered with a fixture that side also wrote, so a renamed
  field or a changed enum spelling would not have surfaced until a
  deployment. The job also asserts the test actually ran, since
  `cargo test -- --ignored` exits 0 when it matches nothing.

### Security

- `h2` 0.4.15 → 0.4.18, fixing RUSTSEC-2026-0258 (unbounded queuing of empty
  DATA frames — memory exhaustion, low severity). Reached through
  hyper/reqwest/tonic; lockfile-only change.
- `lru` 0.16 → 0.18, fixing RUSTSEC-2026-0253 (use-after-free: `LruCache::pop()`
  was not panic-safe and could leave dangling pointers in its intrusive list).
  Used by the in-process rate-limit buckets and the inference result cache.
  Both advisories were already present in the tree and were surfaced by the
  `cargo deny` gate above.

### Fixed

- README described the deep-semantic judge as a shipped capability, which
  contradicted [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) and
  the code: the `guard_llm` runner kind is a declared seam with no module
  implemented. README and `config/ml_catalog.yaml` now say so.

### Notes on defaults

The shipped defaults optimize for a first run on a laptop, not a hardened
deployment: `ARMOR_AUTH_MODE=none`, `ARMOR_RATE_LIMIT_MODE=none`, most
detectors at `mode: warn`, and `fail_mode: fail_open`. These are documented
choices rather than oversights — see the "A note on defaults" section of
[`SECURITY.md`](SECURITY.md) before exposing a deployment to a network.

[Unreleased]: https://github.com/pvv5385/armor-aidr/commits/main
