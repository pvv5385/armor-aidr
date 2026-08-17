# Security policy

AI Armor is a security control. A vulnerability here is not just a bug in our
software — it may be a silent gap in someone else's defenses. We treat reports
accordingly.

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report it through GitHub's private vulnerability reporting: go to the
[Security tab](../../security/advisories/new) of this repository and click
**Report a vulnerability**. The report is visible only to the maintainers
until we publish an advisory.

If GitHub private reporting is unavailable to you, open a public issue
containing only the words "security report, need a private channel" and no
technical detail, and a maintainer will open a private channel with you.

### What to include

- The version, commit SHA, or container digest you tested.
- Configuration relevant to the finding — most importantly your
  `config/policies.yaml` and any non-default `ARMOR_*` environment variables.
- A minimal reproduction. For a detector bypass, the exact input text is
  enough; please note whether it was sent through the HTTP API or evaluated
  directly against `armor-core`.
- What you believe the impact is.

### What to expect

| Stage | Target |
|---|---|
| Acknowledgement of your report | 3 business days |
| Initial assessment and severity call | 10 business days |
| Fix or documented mitigation for High/Critical | 30 days from assessment |
| Public advisory | Coordinated with you, after a fix ships |

These are targets for a small maintainer team, not a contractual SLA. If a
report goes quiet past these windows, escalating in the same thread is
welcome and appropriate.

We will credit you in the advisory unless you ask us not to. We do not
currently run a paid bug bounty.

## Scope

### In scope

- **Detector bypasses.** Input that should be caught by a shipped ruleset and
  is not — encoding tricks, normalization gaps, Unicode handling, or a pattern
  that is narrower than its rule ID claims.
- **False-negative-by-crash.** Any input that makes a detector panic, hang, or
  time out. Because checks default to `fail_mode: fail_open`, a detector that
  crashes silently stops protecting. We consider this at least as serious as a
  pattern bypass.
- Authentication or rate-limiting bypass in `armor-api`.
- Denial of service against `armor-api` — unbounded memory or disk growth,
  thread exhaustion, or pathological CPU from a single request.
- Leakage of evaluated content into logs, metrics, the audit spool, or
  telemetry. The design rule is that these carry **metadata only** — category
  names, pass/fail, latency — and never request content or matched spans. Any
  path that violates this is a vulnerability, not a bug.
- Policy-parsing flaws that cause a check to be silently skipped or downgraded.

### Out of scope

- **Missing coverage for a threat no shipped rule claims to detect.** A
  category we have not implemented is a feature request. A rule that claims
  coverage it does not deliver is a vulnerability. Read the ruleset header if
  you are unsure which one you have found.
- Detector false positives. These are quality bugs — file them as normal
  issues, they are genuinely useful.
- Findings that require an attacker to already control `config/policies.yaml`,
  `ARMOR_CUSTOM_RULES_DIR`, or the process environment. Anyone with that
  access has already replaced the security policy.
- Results from running with `ARMOR_AUTH_MODE=none` (the default) on a
  network-exposed deployment. See the hardening notes below.
- Vulnerabilities in a dependency with no exploitable path through this
  codebase. Report those upstream; tell us if we should pin or patch.

## A note on defaults

The shipped defaults optimize for a first-run experience on a laptop, not for
a hardened deployment. In particular:

- `ARMOR_AUTH_MODE` defaults to `none` — the API accepts any caller.
- `ARMOR_RATE_LIMIT_MODE` defaults to `none`.
- Most detectors ship `mode: warn`, not `block`, because their rulesets have
  not completed a precision/recall evaluation pass. A `warn` check records
  the hit and allows the request.
- `fail_mode` defaults to `fail_open` — a check that errors or times out does
  not block the request.

None of these are vulnerabilities on their own; they are documented defaults.
Reports that a default deployment is insecure are out of scope, but reports
that the **documentation misrepresents** what a default does are in scope, and
we want them.

## Supported versions

The project is pre-1.0 and has not cut a tagged release yet. **`main` is the
only supported line** — security fixes land there, and the commit SHA or
container digest you deployed is what identifies your version in a report.

Once releases are tagged, this section will name the supported ones; a full
support policy for released versions follows at 1.0. Notable changes,
including every change to what the shipped rulesets detect, are recorded in
[`CHANGELOG.md`](CHANGELOG.md).
