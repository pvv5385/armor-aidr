# Known Limitations

This document exists so nothing about what AI Armor's current detection tier
actually catches has to be taken on faith. It covers the deterministic
engine and the ML inference tier shipped in this repository today — not
aspirational roadmap items. See the [`README`](README.md)'s Status line for
what has shipped beyond it.

**For anyone writing external-facing copy:** describe the deterministic tier as
*deterministic, pattern-based detection with single-pass normalization
against common obfuscation techniques*, and the ML tier as *classifier-backed
escalation for gray-zone risk signals*. Do not claim complete evasion
resistance, semantic understanding of novel attacks, or handling of
arbitrarily nested/stacked encodings — none of that is fully closed by
either tier, and the items below are why.

## Detection is pattern-based, not semantic

Every detector in `crates/core/src/detectors/` is regex/literal-pattern
matching, checksum validation (Luhn, CPF/CNPJ, etc.), or entropy gating —
there is no ML model and no LLM judge in this deterministic tier. A payload
that shares no vocabulary or structure with any rule's pattern bank —
paraphrased instructions, a novel jailbreak framing nobody has written a
rule for yet — will not be flagged, no matter how well-tuned the existing
rules are. Deeper semantic judgment is what the ML inference tier
(`armor-inference`) is for — see "ML tier limitations" below for what it
does and doesn't cover.

## Normalization is single-pass, not recursive

`crates/core/src/engine/normalize.rs` builds a fixed set of "views" of the
input text — NFKC, invisible-char-stripped, de-leeted, HTML-unescaped,
homoglyph-folded, spacing-collapsed (chained cumulatively), plus independent
`rot13` and `base64` views built once from the cumulative cleaned text. None
of this loops or recurses:

- **Stacked encodings are not unwound.** Base64-inside-base64,
  rot13-inside-base64, or any other multi-layer encoding decodes at most one
  level and then stops — the still-encoded remainder is checked as opaque
  text and won't match a plaintext pattern bank.
- **This is by design for the current tier, not a bug.** See the module's
  own doc comment for exactly which stages chain and which don't
  (`rot13_and_base64_are_independent_not_chained` in that file's test suite
  documents the exact behavior).
- Tracked as a roadmap item, not yet built: an adversarial benchmark with
  stacked evasions (base64-in-base64, UTF-7, chunk-splitting) and
  nested/iterative decode passes to catch them. Until that lands, there is
  no measured evasion-resistance number for this engine — treat any
  percentage quoted for "evasion resistance" as unverified unless it cites
  that benchmark.

## Homoglyph coverage is a fixed table, not the full Unicode confusables spec

`crates/core/src/homoglyphs.rs`'s confusables table is deliberately not
exhaustive — full Unicode confusables (UTS #39) run to thousands of entries;
this table covers what shows up in practice in registerable phishing domains
and hand-typed obfuscation. A confusable character outside that table passes
through unfolded.

## Pattern banks are English-language

Rule patterns are literal/regex matches against English words and phrasing.
An attack phrased in another language is a broader bypass vector than
normalization alone — it isn't a stacked-encoding problem, it's that the
rule text itself never matches.

## Hit spans point into the normalized view, not the raw input

`RuleHit.span` is a byte offset into whichever view matched (e.g., after
HTML-entity decoding shifts every subsequent offset), not the caller's
original string, and the API response doesn't return which view text the
offset applies to. Do not use these offsets to redact or highlight positions
in the raw input as received — this is an open, tracked issue.

## Deployments are single-tenant

The session store and the PII vault (`crates/storage/src/sessions.rs`,
`vault.rs`) key on `session_id` alone. There is no `tenant_id` column and no
tenant model — `middleware/auth.rs` treats an API key as a pass/fail
credential with no identity attached.

`session_id` is caller-supplied via `X-Armor-Session-Id`, so if two tenants
ever share one database, a caller who guesses or replays another tenant's
session id can read that tenant's counters and its vault entries. **Run one
deployment per tenant.** Closing this needs a key→tenant mapping resolved in
auth plus a migration widening both primary keys to
`(tenant_id, session_id)`.

## The vault has no network surface

Reversible anonymization works as a library: stable per-session
placeholders, AES-256-GCM at rest, an HMAC blind index for value lookup,
retention, and right-to-erasure. Nothing exposes `deanonymize` or `erase`
over HTTP, deliberately — there is no RBAC model yet, and an
unauthenticated deanonymize endpoint would turn the vault into a PII
disclosure API. Callers that need it link the crate.

The *write* side is now reachable through a policy: a check configured
`on_fail: redact` composes to `Verdict::Redact` and, when `ARMOR_VAULT_KEY`
is set alongside `DATABASE_URL`, its spans are vaulted rather than discarded
(`crates/api/src/redaction.rs`). Reading them back is what still has no
route. So a deployment can produce placeholders it cannot resolve over HTTP
— by design, but worth knowing before you build a workflow that depends on
resolving them.

One more consequence worth stating: the vault key lives in
`ARMOR_VAULT_KEY`, so an attacker holding both the database and the process
environment recovers plaintext. Splitting those is what the `KeyProvider`
trait exists for — a KMS-backed implementation drops in without a schema
change.

## Cross-request detection is counters, not semantics

Durable session state covers `abuse` (windowed request rate) and
`unbounded_consumption` (lifetime token/request budgets), and it is correct
across replicas when `DATABASE_URL` is configured. Without a database, both
fall back to per-instance counters and under-count behind a load balancer.
Both ship `enabled: false`.

What that does *not* give you: cumulative risk scoring across a
conversation, or multi-turn intent drift. Those need semantic judgment over
turn history that a counter, by design, cannot provide.

## ML tier limitations

The ML inference tier has the following known limitations:

- **Classifier quality depends on training data.** The ONNX models shipped
  with `armor-inference` are trained on specific datasets (prompt injection,
  toxicity, PII patterns). Attacks outside those distributions — novel
  jailbreak framings, domain-specific PII formats, adversarial examples
  crafted against the classifier — may evade detection. The scorecard gate
  enforces minimum quality thresholds (F1, AUROC, ECE) but cannot close the
  gap on distribution-shifted attacks.

- **Single forward pass per check.** Each escalating check gets one model
  call against the view that fired (or `raw`). There is no multi-pass
  ensemble, no judge-over-judge validation, and no adversarial robustness
  layer. A deeper semantic-judge design would address this; nothing in this
  repo implements one yet.

- **English-language models.** The shipped classifiers are trained on
  English text. Attacks in other languages bypass the ML tier entirely,
  falling back to the deterministic rules (which are also English-only —
  see "Pattern banks are English-language" above).

- **No streaming support.** The ML tier scores complete request/response
  payloads, not token streams. Streaming LLM output is scored after the
  full response arrives, not incrementally.

- **Scorecard metrics are benchmark-derived, not live-calibrated.** The
  scorecard gate measures quality against benchmark suites, not against
  live traffic. A model that degrades in production (e.g., due to concept
  drift) will not be detected until the next benchmark re-evaluation.

- **Per-task pins are not yet synced to edge instances.** The
  `inference_pins` table and sync bundle are built but the pull-based
  desired-state loop (`pin_sync.py`) is not yet wired into the sync
  task.

## What's not evaluated at all yet

The ML/semantic inference tier ships deterministic escalation against an
`armor-inference` sidecar with ONNX-based classifiers (prompt injection,
toxicity, over-refusal, PII NER, topic/intent). A deep-semantic judge layer
is not started — nothing in this repo today provides multi-turn intent
analysis, novel jailbreak framing detection, or cumulative risk scoring
across a conversation.
