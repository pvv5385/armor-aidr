# Armor + Portkey

Wires Armor into [Portkey's "Bring Your Own Guardrail"](https://portkey.ai/docs/integrations/guardrails/bring-your-own-guardrails)
mechanism. Unlike the LiteLLM integration (a plugin that runs inside the
gateway process and *calls* Armor), Portkey works the other way around:
Portkey calls a webhook you host, on its own request/response schema.
That webhook is implemented in this repo, not in this directory — see
`crates/api/src/integrations/portkey.rs` — because it has to be part of
`armor-api` itself to reach the loaded policy directly. This directory just
documents how to point Portkey at it.

See `crates/api/src/integrations/portkey.rs`'s module doc for the full
BLOCK/REDACT/WARN/ASK capability matrix and degradation policy, or
[`docs/GATEWAY_INTEGRATIONS.md`](../../docs/GATEWAY_INTEGRATIONS.md) for how
it compares side by side with the LiteLLM adapter.

## Setup

1. Run `armor-api` somewhere Portkey's control plane (or your self-hosted
   Portkey gateway) can reach over HTTPS.
2. In Portkey, configure a custom guardrail of type "webhook" pointing at:

   ```
   POST https://<your-armor-host>/integrations/portkey/v1/aidr/scan
   ```

3. Attach it to your Portkey config for both the `beforeRequestHook` stage
   (checks the prompt, Armor's `input` mode) and `afterRequestHook` stage
   (checks the completion, Armor's `output` mode) — the same endpoint
   handles both; it reads `eventType` off the request body to build Armor's
   standard `AidrScanRequest` (`crates/api/src/aidr.rs`,
   `docs/AIDR_IMPLEMENTATION.md`) with `metadata.mode` set accordingly, the
   same request schema `/api/v1/aidr/scan` and every other integration use.
4. If `armor-api` is running with `ARMOR_AUTH_MODE=api_key`, add the
   corresponding header (`X-API-Key`) to Portkey's webhook configuration —
   Portkey supports custom headers per webhook guardrail.

## Request/response contract

Verified against Portkey's own docs (see the link above) — this is
Portkey's schema, not Armor's:

```json
// Portkey -> armor-api
{
  "request": {"json": {...}, "text": "the user's prompt", "isStreamingRequest": false},
  "response": {"json": {...}, "text": "", "statusCode": null},
  "provider": "openai",
  "requestType": "chatComplete",
  "metadata": {...},
  "eventType": "beforeRequestHook"
}
```

```json
// armor-api -> Portkey
{"verdict": true}
```

## What this adapter does and doesn't do

- **BLOCK**: `Decision.verdict == "block"` -> `verdict: false`. Portkey
  fails the guardrail check and applies whatever action you've configured
  for a failed guardrail (deny the request, etc.).
- **WARN** / **ALLOW**: both -> `verdict: true`. Same as the LiteLLM
  adapter, Portkey's model is binary — there's no "flag but allow" lane to
  put a `Warn` verdict into.
- **REDACT**: **not supported.** Portkey's response schema has a
  `transformedData` field specifically for this (a rewritten
  request/response), but this adapter's response struct has no such field
  and never populates one.
- **ASK**: **not reachable today** — no shipped detector emits
  `Verdict::Ask`. If that changes, this adapter's degradation policy
  treats it the same as `WARN` (`verdict: true`, not blocking) — see
  `crates/api/src/integrations/portkey.rs`'s module docstring.
- If the webhook times out or errors, Portkey's documented behavior is to
  default to `verdict: true` (fail-open) — consistent with Armor's own
  `fail_mode: fail_open` default (see `config/policies.yaml`), so there's
  no policy mismatch at the boundary.

## Known gaps

- No session correlation: Portkey's webhook schema has no equivalent of
  Armor's `X-Armor-Session-Id` contract, and inventing one from `metadata`
  felt like guessing at a shape rather than building to a real requirement.
  Every call gets a fresh self-minted session id, so per-session detector
  state (e.g. `unbounded_consumption`) won't accumulate across turns of the
  same Portkey conversation. This *does* still write to Armor's audit sink
  and telemetry, same as `/api/v1/aidr/scan` — both go through the shared
  `aidr::run_scan` path now (`crates/api/src/integrations/portkey.rs`), so
  that part of the previous "known gap" here is closed.
