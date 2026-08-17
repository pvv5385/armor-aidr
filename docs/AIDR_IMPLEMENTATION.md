# AIDR scan request/response schema

One request shape, one endpoint family, covers every stage — plain text,
full OpenAI-shaped `messages` (including `tool_calls`), and agent
plan/memory state. There's no separate route for "advanced" agent checks;
a richer payload on the same schema is what triggers them. Source of
truth: `crates/api/src/aidr.rs` (`AidrScanRequest` / `ScanResponse`).

## Endpoints

- `POST /api/v1/aidr/scan` — the direct endpoint.
- `POST /integrations/litellm/v1/aidr/scan`, `POST
  /integrations/portkey/v1/aidr/scan` — gateway adapters. Each normalizes
  its vendor's own wire format into `AidrScanRequest` before handing off to
  the same `aidr::run_scan`, so every caller gets the same detectors, the
  same audit trail, and (mostly) the same response — see
  [`GATEWAY_INTEGRATIONS.md`](GATEWAY_INTEGRATIONS.md) for where Portkey's
  own webhook contract forces a different response shape.

Both take an optional `X-Armor-Session-Id` request header (1-128 visible-ASCII
bytes) tying repeated calls to one conversation, for detectors with
per-session state (`abuse`, `unbounded_consumption`). Omit it and Armor
mints a UUID v4 and echoes it back on the same header — pass that value on
the next call to keep the session going.

## Request schema

```json
{
  "text": "Find me flights to London.",
  "messages": [
    { "role": "user", "content": "Find me flights to London." }
  ],
  "metadata": {
    "mode": "input",
    "application_id": "travel-assistant",
    "request_id": "req_tool_benign_001",
    "user_id": "user_42",
    "labels": ["production", "customer-facing"],
    "model": "gpt-4o",
    "provider": "openai"
  }
}
```

- **`text`** — plain-string case. Scanned as-is.
- **`messages`** — OpenAI-shaped array. `content` may be a string or a
  content-part array (multimodal — only `{"type": "text", "text": ...}`
  parts are pulled in); an assistant message's `tool_calls` are
  JSON-serialized and scanned alongside its `content`, not field-plucked,
  so pattern-based detectors see them without `armor-core` needing to know
  their shape.
- **`metadata`** — routing, policy-matching, and telemetry data, kept
  separate from the content being scanned:
  - `mode` — where in the request lifecycle this is (`input`, `output`, or
    any caller-chosen string, e.g. `tool`/`agent-plan`). Accepted
    permissively and forwarded into the audit trail, not validated against
    a fixed enum. Defaults to `"input"`.
  - `application_id` — which profile's checks run (`crates/api/src/profiles.rs`).
    An unmapped or absent value falls back to the default profile — never
    a hard failure.
  - `request_id` — caller-supplied correlation id, echoed back as
    `ScanResponse.request_id` and stored on the audit trail as
    `client_request_id`. Same validation as the session header: 1-128
    visible-ASCII bytes; a bad value is a `400`. Distinct from `scan_id`
    (Armor's own id, always present, authoritative for correlating with
    the audit trail even when a caller never sends `request_id`).
  - `agent_state` — untyped agent plan/memory context (its shape varies by
    deployment), e.g. `current_plan`/`proposed_action`/`authorization_level`.
    JSON-serialized and folded into the scanned text the same way
    `tool_calls` are, so checks like `excessive_agency` can see it.
  - Anything else the caller sends is preserved on the audit trail via
    `#[serde(flatten)]` but never scanned — only `text`,
    `messages[*].content`, `messages[*].tool_calls`, and
    `metadata.agent_state` feed the engine.

### OpenAI chat-completions compatibility

A caller forwarding a raw OpenAI-shaped request can put `request_id`,
`application`, and `user_id` at the **root**, alongside `messages`, instead
of nesting them under `metadata`:

```json
{
  "request_id": "abc123",
  "application": "customer-support",
  "user_id": "user-123",
  "metadata": { "tenant": "enterprise-a", "region": "us-east-1" },
  "messages": [
    { "role": "system", "content": "You are a banking assistant." },
    { "role": "user", "content": "My SSN is 123-45-6789." }
  ]
}
```

Both shapes are accepted on the same endpoint. Root-level `request_id` /
`application` / `user_id` are folded into their `metadata` counterparts
before resolution (`AidrScanRequest::normalize`), so routing, profile
matching, and telemetry work identically either way. **`metadata` is
authoritative** — a root field is only consulted when the matching
`metadata` field is absent, never overriding it.

## Response schema

```json
{
  "scan_id": "b8f2b7b0-6f7b-4e2e-9b1e-2e6b7e6b1a2c",
  "request_id": "req_tool_benign_001",
  "verdict": "BLOCK",
  "redacted_text": "Find me flights to [REDACTED_LOCATION].",
  "latency_ms": 1.62,
  "checks": [
    {
      "category": "prompt_injection",
      "flagged": true,
      "action_taken": "blocked",
      "severity": "high",
      "hits": 1,
      "latency_ms": 0.31
    }
  ]
}
```

- **`verdict`** — one of Armor's five canonical values, serialized
  UPPERCASE: `ALLOW`, `WARN`, `REDACT`, `BLOCK`, `ASK`
  (`armor_core::models::Verdict`). See
  [`GATEWAY_INTEGRATIONS.md`](GATEWAY_INTEGRATIONS.md) for how each gateway
  adapter collapses this down to its own binary allow/deny hook model.
- **`checks`** — every enabled check in the resolved profile is listed,
  not just the ones that fired; `flagged` distinguishes the two.
  `action_taken` is one of `none` / `blocked` / `redacted` / `warned` /
  `logged`, read off the check's `(mode, action)` pair
  (`crates/api/src/aidr.rs::build_checks`) — e.g. `redacted` is a `Redact`
  action on a `Block`-mode check, whose spans are masked in
  `redacted_text`. `hits` is a count, not the matched spans themselves —
  this contract has never exposed raw hit text beyond what
  `redacted_text` already shows.
- **`redacted_text`** — the input with matched spans replaced by
  placeholders. Present even when nothing was flagged (equal to the
  original text in that case).
- **`request_id`** is only present when the caller sent
  `metadata.request_id` (or its OpenAI-compat root alias); `scan_id` is
  always present and is Armor's own identity for the request.

Portkey's adapter does **not** return this shape — Portkey's own webhook
contract expects `{"verdict": bool}`, so `crates/api/src/integrations/portkey.rs`
builds that instead after running the same `aidr::run_scan` path. See
[`GATEWAY_INTEGRATIONS.md`](GATEWAY_INTEGRATIONS.md).

## Why one endpoint instead of splitting by content type

An agent attempting something like a memory-instruction-injection might do
so via plain text, a tool call, or retrieved content — splitting evaluation
across `/v1/guardrails/text` and `/v1/guardrails/tools` loses the ability
to evaluate the full context together. A single endpoint takes the entire
request (text, messages, tool calls, agent state) and lets the resolved
profile decide what applies. It also matches how gateway hooks are shaped:
LiteLLM and Portkey each expect to send one payload to one URL and get one
instruction back, not fan out to multiple guardrail endpoints per LLM call.
