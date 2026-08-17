# Contributing to AI Armor

Thanks for considering a contribution.

AI Armor is a security control. Something that ships here becomes part of
someone else's defenses, and a rule that quietly stops matching is worse than
one that was never written — nobody gets an alert when a guardrail goes
silent. That shapes most of what follows: we ask for evidence a change works,
not just a claim that it does.

**Found a vulnerability? Do not open a pull request or a public issue.** See
[`SECURITY.md`](SECURITY.md) — that includes detector bypasses and detector
crashes, which are in scope there, not here.

## Before you start

For anything larger than a bug fix or a single new pattern, open an issue
first. It is a cheap way to find out that a category is already scoped, or
that a design constraint you cannot see from the outside rules the approach
out — better learned before you write the code than in review.

Small, obviously-correct fixes need no issue. Send them.

## Development setup

```
cargo build
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

The toolchain is pinned in [`rust-toolchain.toml`](rust-toolchain.toml);
rustup will select it automatically. Those four commands are exactly what CI
runs, so a clean local run means a clean CI run — with two exceptions worth
knowing about.

### Tests that need a service

Some tests exercise behavior that only a real backing service has —
transactional races, `ON DELETE CASCADE`, the Redis token-bucket Lua script.
They **skip with a printed notice rather than failing** when the service is
absent, so a green `cargo test` on a laptop does not mean they ran. CI sets
both variables and fails the build if any `SKIPPING` notice appears.

To run them locally:

```
docker run --rm -d -p 15432:5432 \
  -e POSTGRES_DB=armor_test -e POSTGRES_USER=armor -e POSTGRES_PASSWORD=armor \
  postgres:16-alpine
docker run --rm -d -p 16399:6379 redis:7-alpine

export ARMOR_TEST_DATABASE_URL=postgres://armor:armor@127.0.0.1:15432/armor_test
export ARMOR_TEST_REDIS_URL=redis://127.0.0.1:16399
```

`armor-storage`'s tests share one database, and `sessions::purge_expired` and
`vault::purge_expired` are table-wide `DELETE`s by design. Tests that assert a
row survives a sweep call them with a far-future cutoff, which will collect
rows another test is mid-assertion on. Run that crate single-threaded:

```
cargo test -p armor-storage -- --test-threads=1
```

CI does the same. If you add a test to `armor-storage` that depends on
parallel isolation, it will not get it.

### The Python sidecar

`inference/` is a separate deployable with its own test suite:

```
pip install -e "./inference[dev]"
pytest inference/tests
```

Install `[dev]` only — not `[onnx]`, not `[export]`. The sidecar is designed
to boot and serve on the stub runner with no ML stack present, and CI checks
that property by never installing one. If your change needs torch to pass its
tests, that is a signal worth discussing in an issue first.

## Making the change

### Commits need a sign-off

Every commit must carry a `Signed-off-by` trailer:

```
git commit -s
```

This certifies the [Developer Certificate of
Origin](https://developercertificate.org/) — that you wrote the contribution
or have the right to submit it under Apache-2.0. It is an attestation, not a
copyright assignment; see [`DCO.md`](DCO.md).

The trailer uses your git `user.name` and `user.email`, and git history is
permanent and public. **The address does not have to be one you read mail
at** — a GitHub noreply address
(`ID+username@users.noreply.github.com`, under **Settings → Emails**) signs
off just as validly and keeps a personal address out of the log. Nothing in
this project requires you to publish a working inbox.

CI checks every commit in the PR and tells you how to fix a branch that is
missing one:

```
git rebase --signoff <base>
```

### Contributing to `rules/`

The rulesets are the accumulated asset of this project, and the bar is
correspondingly higher. `rules/` is a symlink to `crates/core/rules/`, which
is where the files actually live.

- **No lookarounds, no backreferences.** Every pattern must carry over
  cleanly to Rust's `regex` crate, which has neither. This is not a style
  preference — it is what guarantees linear-time matching, and therefore that
  a hostile input cannot turn a detector into a denial of service.
- **Two positive samples minimum per new rule**, in the matching
  `config/benchmarks/<category>_v1.yaml`, keyed by `rule_id`.
- **Negative samples for the near-misses.** The input that looks like a hit
  and must not fire is the more valuable of the two. A rule with no negatives
  has not been evaluated, it has only been demonstrated.
- **New categories ship `mode: warn`** in `config/policies.yaml`, never
  `block`, until a precision/recall pass justifies promoting them. A `warn`
  check records the hit and allows the request.
- **Report the false-positive impact in the PR.** Did the category's FPR
  move, and by how much? The benchmark tests in `crates/core/tests/` print
  this.
- If you incorporated third-party pattern text, say so, and update
  [`NOTICE`](NOTICE). Read its "Detection rulesets" section first — this
  repository deliberately claims no derivative-work relationship to any
  third-party pattern bank, and we intend to keep that true.

Rule files carry a header comment explaining the ruleset's scope, what it
deliberately does *not* cover, and its last eval result. Keep that header
accurate; it is what tells a security researcher whether a gap they found is
a bug or a documented non-goal.

### Contributing to the engine or the API

- `armor-core` is synchronous and does no I/O. Keep it that way — it is what
  makes the detection logic testable without a runtime and embeddable without
  a network.
- Changes to the `/api/v1/aidr/scan` response shape need an issue and a
  discussion first. People parse it in production.
- Logs, metrics, the audit spool, and telemetry carry **metadata only** —
  category names, pass/fail, latency. Never request content, never matched
  spans. `SECURITY.md` treats a violation of this as a vulnerability rather
  than a bug, so a PR that adds content to any of those paths will be sent
  back.

## Opening the pull request

The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) has the
checklist. Beyond ticking it:

- Say what you are unsure about. A PR that flags its own weak spot gets a
  better review than one that hides it.
- Keep licensing, `SECURITY.md`, and ruleset changes out of unrelated PRs.
  These are owned separately (see [`.github/CODEOWNERS`](.github/CODEOWNERS))
  precisely so they never ride along as a drive-by edit.
- Rebase rather than merge `main` into your branch, so the DCO check and the
  reviewer both see a clean commit range.

Review is by a maintainer. We aim to respond within a few days; this is a
small team, so a nudge on a quiet thread is welcome rather than rude.

## Code of conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

Contributions are accepted under the Apache License 2.0, the license covering
this entire repository. There is no split licensing and no relicensing plan.
