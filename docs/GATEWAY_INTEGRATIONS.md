# Gateway integration capability matrix

The integration strategy deliberately ships **API integration
first — LiteLLM/Portkey guardrail hooks, no traffic interception.** This
document is the resulting capability matrix, so gaps get discovered here,
not in a support ticket. Adapter code:

- LiteLLM: [`integrations/litellm/armor_guardrail.py`](../integrations/litellm/armor_guardrail.py)
  (a plugin that runs inside the LiteLLM proxy process and calls Armor).
- Portkey: [`crates/api/src/integrations/portkey.rs`](../crates/api/src/integrations/portkey.rs)
  (a webhook endpoint Portkey calls, on Portkey's own request/response
  schema — see [`integrations/portkey/README.md`](../integrations/portkey/README.md)).

## Why a matrix at all

Both gateways are **binary allow/deny** at the hook boundary — neither has
a human-in-the-loop lane. Armor's own canonical verdict model
(`crates/core/src/models.rs::Verdict`) has five values: `Allow`, `Warn`,
`Redact`, `Block`, `Ask`. Every integration has to collapse that down to
whatever its own hook model supports, and where it can't, that's a
**documented, stored decision** — a "degradation policy" — not a silent
default discovered later.

## Capability matrix

| Armor verdict | LiteLLM (`armor_guardrail.py`) | Portkey (`integrations/portkey.rs`) |
|---|---|---|
| `Block` | Hook raises `ArmorGuardrailBlocked`; LiteLLM returns an error to the caller. | `verdict: false`; Portkey applies whatever failure action you've configured for the guardrail. |
| `Warn` | Passes through, logged at `WARNING`. No "flag but allow" lane exists in LiteLLM's hook model. | `verdict: true`. Same reason — Portkey's model is binary too. |
| `Allow` | Passes through silently. | `verdict: true`. |
| `Redact` | **Not supported.** Never reachable today (see below) — if it were, this adapter has no rewritten text to substitute into `data`, since there's no vault yet (vaulting is planned for later). | **Not supported**, same reason. Portkey's schema *does* have a `transformedData` field built for exactly this, but the adapter never populates it, for the same "no vault yet" reason. |
| `Ask` | **Not reachable today** (no shipped detector emits it). If that changes: degrades the same as `Warn` — logged, not blocking. | Same: not reachable today; degrades the same as `Warn` if it becomes reachable. |

**Why `Ask` and `Redact` are "not reachable" and not just "unsupported":**
grep the whole detector registry (`crates/core/src/detectors/`) and no
detector's `evaluate()` ever constructs `CheckAction::Redact`, and the
orchestrator (`crates/core/src/engine/decision.rs::compose`) never produces
`Verdict::Ask` from today's shipped check set — both are part of the
canonical model's vocabulary for capabilities not yet built (a real redaction path needs
a vault that doesn't exist yet; a real `ask` lane needs a human-in-the-loop
UI that doesn't exist yet), not something either adapter is choosing to
ignore. The degradation policy above is written down now specifically so
that when a detector eventually does emit one of these, the fallback
behavior is a conscious choice already on record, not a gap someone finds
in production.

## REDACT does not round-trip — by gateway, not just by us

Independent of Armor's own current scope, **REDACT may not round-trip
through either gateway even once Armor can produce one**: LiteLLM's
`async_pre_call_hook` can return a modified `data` dict (so a redacted
prompt could theoretically flow back in), but `async_post_call_success_hook`
(the output/post-call side) cannot rewrite the model's response — LiteLLM's
contract only lets it observe or block. Portkey's `transformedData` field
can rewrite either side, per its schema — closer to a real REDACT lane, but
still gated on Armor having something trustworthy to put in it, which is
the vault dependency above.

## Latency

Measuring and publishing Armor's own latency, not end-to-end, is the
deliberate approach here — `crates/core/tests/latency_benchmark.rs` (see
the workspace README's "Benchmarking" section) measures Armor's own
p50/p95/p99 against the shipped default policy, with no gateway hop
involved. As measured on the machine this test was last run on (release
build, 474-byte payload, the shipped policy's 33 checks, its default
`normalize` view set): **p95 ~1.6-1.7ms, p99 ~3.2-3.4ms**. The `<2ms`
budget referenced elsewhere is a p95 target, and it holds — but say "p95"
out loud when quoting it; p99 does not clear 2ms. A gateway integration
adds its own network + hook-dispatch overhead on top of these numbers;
don't attribute that overhead to Armor, and don't expect either
integration's README above to restate it — they link back here instead.

## Fail-open alignment

Both adapters inherit Armor's own `fail_mode: fail_open` default
(`config/policies.yaml`): Portkey's documented behavior on a webhook
timeout/error is to default to `verdict: true`; LiteLLM guardrail hooks
that raise an *unexpected* exception (a network error talking to
`armor-api`, not an intentional `ArmorGuardrailBlocked`) are the
customer's own retry/timeout configuration to set, not something this
adapter overrides. Keeping both integrations' failure posture consistent
with Armor's own default avoids a customer accidentally running
fail-closed at the Armor layer and fail-open at the gateway layer (or vice
versa) without realizing it.
