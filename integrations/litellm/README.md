# Armor + LiteLLM

Wires Armor into [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail)
as a **custom guardrail** — LiteLLM calls Armor over HTTP from inside the
proxy process; Armor is not a network proxy and does not see traffic it
isn't explicitly called with, an "API integration first" deliberate
choice. See `armor_guardrail.py`'s module docstring for the full
BLOCK/REDACT/WARN/ASK capability matrix and degradation policy this
adapter follows, or [`docs/GATEWAY_INTEGRATIONS.md`](../../docs/GATEWAY_INTEGRATIONS.md)
for how it compares side by side with the Portkey adapter.

## Setup

1. Run `armor-api` somewhere LiteLLM's proxy process can reach it (same
   host, same cluster, etc.) — see the top-level README's "Running"
   section.
2. Copy `armor_guardrail.py` next to your LiteLLM proxy's `config.yaml`
   (or anywhere on `PYTHONPATH`).
3. Add **one** guardrail entry to `config.yaml` — `mode` takes a list, so a
   single class instance covers every hook it implements (`pre_call` ->
   `async_pre_call_hook`, `post_call` -> `async_post_call_success_hook` /
   `async_post_call_streaming_iterator_hook`). This matches how LiteLLM's
   own vendor guardrails (Pillar, AIM, etc.) register:

   ```yaml
   guardrails:
     - guardrail_name: "armor"
       litellm_params:
         guardrail: armor_guardrail.ArmorGuardrail
         mode: ["pre_call", "post_call"]
   ```

   `during_call` is also supported (`async_moderation_hook`) as an
   alternative to `pre_call` for lower added latency — it runs in parallel
   with the LLM call instead of serially before it. Use one or the other,
   not both; they scan the same prompt text, so combining them just doubles
   the armor-api calls for no extra coverage:

   ```yaml
         mode: ["during_call", "post_call"]
   ```

4. Set environment variables for the proxy process:

   | Variable | Default | Purpose |
   |---|---|---|
   | `ARMOR_BASE_URL` | `http://localhost:8080` | Where `armor-api` is running |
   | `ARMOR_API_KEY` | unset | Only needed if `armor-api` runs with `ARMOR_AUTH_MODE=api_key` |
   | `ARMOR_TIMEOUT_SECONDS` | `2.0` | Per-call HTTP timeout |
   | `ARMOR_STREAM_SAMPLE_EVERY` | `5` | For streamed responses: check accumulated output every N chunks, plus once at end-of-stream. Lower = catches unsafe output sooner but costs more armor-api round trips per response. |

5. Apply the guardrail to a request the same way as any other LiteLLM
   guardrail — either globally or per-call via `guardrails: ["armor-input",
   "armor-output"]` in the request body.

## Wire format

The plugin POSTs to `POST {ARMOR_BASE_URL}/integrations/litellm/v1/aidr/scan`
(`crates/api/src/integrations/litellm.rs`), forwarding LiteLLM's own
`data["messages"]` array close to verbatim for `pre_call`/`during_call`, or
a single synthesized `{"role": "assistant", "content": "..."}` message for
`post_call`/streaming (LiteLLM hands those hooks a response object, not a
`messages` array, so the plugin still has to pull text out of it
client-side — there's no way around that part being Python). Armor's Rust
side normalizes this into the shared `AidrScanRequest` schema
(`crates/api/src/aidr.rs`, `docs/AIDR_IMPLEMENTATION.md`) and responds with
the same `Decision` JSON `/api/v1/aidr/scan` returns — LiteLLM doesn't
dictate a response contract here, unlike Portkey.

## What this adapter does and doesn't do

- **BLOCK**: a `Decision.verdict == "block"` raises inside the hook, which
  LiteLLM turns into an error response to the caller — enforced on both
  input and output. On the streaming path this instead raises LiteLLM's
  `StreamingCallbackError` to cut the in-flight SSE response — see
  **Streaming** below for what that does and doesn't prevent.
- **Streaming**: `async_post_call_streaming_iterator_hook` checks the
  accumulated response text every `ARMOR_STREAM_SAMPLE_EVERY` chunks and
  once more at end-of-stream, rather than per non-streaming call. This is
  necessarily weaker than the non-streaming path: chunks are yielded to the
  client as they arrive, so a block only stops chunks that haven't streamed
  *yet* — content in an already-sent chunk cannot be un-sent. Lower
  `ARMOR_STREAM_SAMPLE_EVERY` catches unsafe output sooner at the cost of
  more armor-api round trips per response.
- **WARN**: logged at `WARNING`, request/response passes through
  unmodified. There is no "flag but tell the caller" lane in LiteLLM's
  binary hook model.
- **REDACT**: **not wired up yet.** Armor's response does carry
  `redacted_text`, and `async_pre_call_hook` is technically capable of
  returning modified `data`, but this adapter doesn't yet substitute one
  into the other — advisory-only for now, same as `WARN`.
- **ASK**: **not reachable today** — no shipped detector emits
  `Verdict::Ask`. If a future detector does, this adapter degrades it the
  same way as `WARN` (logged, not blocking) rather than silently treating
  it as `BLOCK` or `ALLOW` — see `armor_guardrail.py`'s module docstring
  for the reasoning.

## Known gaps

- No automated test suite ships with this file — there's no Python test
  harness anywhere else in this repo, and adding `litellm`/`httpx`/`pytest`
  as dependencies just to cover a reference plugin felt like scope creep
  for this pass. Treat this as a reviewed reference implementation,
  not a covered one; if you fork it into your own LiteLLM deployment,
  bring your own tests for your fork.
- `data.get("litellm_session_id")` is forwarded as Armor's
  `X-Armor-Session-Id` header on a best-effort basis — if your LiteLLM
  version doesn't populate that key, Armor just self-mints a session ID
  per call instead (see `crates/api/src/routes.rs`'s
  `resolve_session_id`), which means per-session state (e.g. the `abuse`/
  `unbounded_consumption` detectors' budgets) won't accumulate correctly
  across turns of the same conversation. Wire in your own session-ID
  extraction if that matters for your deployment.
