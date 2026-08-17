"""Armor guardrail plugin for LiteLLM's Custom Guardrail interface.

Calls Armor's own `POST /integrations/litellm/v1/aidr/scan` endpoint (see
`crates/api/src/integrations/litellm.rs`) from inside the LiteLLM proxy
process — an "API integration first, no traffic interception" design, not
a network proxy. Armor runs as its own service; this file is the thin
adapter LiteLLM loads. It forwards LiteLLM's own OpenAI-shaped `messages`
array almost as-is — Armor's Rust side does the normalization into its
standard `AidrScanRequest` schema (`crates/api/src/aidr.rs`), the same
schema every other integration and the direct `/api/v1/aidr/scan`
endpoint share.

Register ONE entry in LiteLLM's `config.yaml` with `mode` as a list (e.g.
`mode: [pre_call, post_call]`) — one class implementing multiple hook
methods, gated internally via `should_run_guardrail`, the same shape used
by LiteLLM's own vendor guardrails (Pillar, Javelin, AIM). See this
directory's README for the full example.

Hook -> Armor stage mapping:
  - `async_pre_call_hook`                  -> `input`  (blocking, serial, before the LLM call)
  - `async_moderation_hook`                -> `input`  (parallel with the LLM call — pick this
    OR pre_call, not both; they check the same prompt text)
  - `async_post_call_success_hook`         -> `output` (non-streaming responses only)
  - `async_post_call_streaming_iterator_hook` -> `output` (streaming responses; sampled every
    `ARMOR_STREAM_SAMPLE_EVERY` chunks plus once at end-of-stream — LiteLLM's own
    `generic_guardrail_api` uses the same sampling shape, since checking every chunk means
    one armor-api round trip per token)

Degradation policy: LiteLLM guardrail hooks are binary — a hook either lets
the call through or raises. Armor's `verdict` maps as:
  - `"block"`  -> raise, LiteLLM returns an error to the caller (streaming
    path raises `StreamingCallbackError` instead, LiteLLM's dedicated
    exception for cutting an in-flight SSE stream).
  - `"warn"`   -> pass through, logged at WARNING level.
  - `"allow"`  -> pass through, silently.
  - `"ask"` -> not reachable today (no shipped detector produces it yet).
    If that changes before this adapter does, it degrades the same way
    `"warn"` does: logged, never blocking — a gateway hook has no
    human-in-the-loop lane for `ask`.
  - `"redact"` -> reached the same way `"warn"` is: logged, pass through.
    A check configured `on_fail: redact` can produce this verdict today,
    but this adapter is deliberately advisory-only for it — see below for
    why `redacted_text` isn't substituted back into the call.
  - anything else (a missing `verdict` key, or a string outside Armor's own
    vocabulary — `crates/core::models::Verdict`) -> raise, logged at ERROR.
    This is a guardrail: a 2xx response this adapter cannot parse into a
    known verdict is a schema drift or a bug, and the safe failure mode for
    "I don't understand the answer" is to block loudly, not to fall through
    every known branch and allow the call through with no log line at all.

REDACT does not round-trip through this adapter today: LiteLLM's
`async_pre_call_hook` *can* return a modified `data` dict, and Armor's
response does carry `redacted_text`, but this adapter doesn't yet wire it
back into that dict — advisory-only for now, same as `"warn"`. The
streaming path has a separate, harder limit regardless: chunks already
yielded to the client can't be un-sent, so a block only stops what streams
*next*, not what already went out — an inherent property of streaming, not
something this adapter can paper over (LiteLLM's own
`generic_guardrail_api` documents the identical tradeoff for its
`streaming_end_of_stream_only` option).
"""

import logging
import os
from typing import Any, AsyncGenerator, List, Literal, Optional, Union

import httpx
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks

logger = logging.getLogger(__name__)

ARMOR_BASE_URL = os.environ.get("ARMOR_BASE_URL", "http://localhost:8080")
# Optional — only needed if armor-api is running with ARMOR_AUTH_MODE=api_key.
ARMOR_API_KEY = os.environ.get("ARMOR_API_KEY")
ARMOR_TIMEOUT_SECONDS = float(os.environ.get("ARMOR_TIMEOUT_SECONDS", "2.0"))
# Streaming responses are checked incrementally rather than per-chunk — one
# armor-api round trip per token would dominate total latency. Matches the
# default sampling cadence LiteLLM's own `generic_guardrail_api` guardrail
# uses (`streaming_sampling_rate`, default 5).
ARMOR_STREAM_SAMPLE_EVERY = int(os.environ.get("ARMOR_STREAM_SAMPLE_EVERY", "5"))

_CallType = Literal[
    "completion",
    "text_completion",
    "embeddings",
    "image_generation",
    "moderation",
    "audio_transcription",
    "responses",
    "mcp_call",
    "anthropic_messages",
]


class ArmorGuardrailBlocked(Exception):
    """Raised to make LiteLLM reject the call; message becomes the error LiteLLM returns."""


def _extract_response_text(response) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError):
        return ""


def _extract_chunk_text(chunk) -> str:
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, KeyError):
        return ""


ARMOR_SCAN_URL = f"{ARMOR_BASE_URL}/integrations/litellm/v1/aidr/scan"


async def _scan(
    mode: str,
    messages: List[dict],
    session_id: Optional[str],
    metadata: Optional[dict] = None,
    call_id: Optional[str] = None,
) -> dict:
    """POSTs to Armor's LiteLLM integration endpoint and returns the parsed
    `ScanResponse` JSON — see `crates/api/src/aidr.rs`'s `ScanResponse`/
    `ScanCheckResult` for the exact shape: `{"scan_id": str, "request_id":
      str|omitted, "verdict": "allow"|"warn"|"block"|"ask"|"redact",
      "checks": [{"category": str, "flagged": bool, "action_taken":
      "none"|"blocked"|"warned"|"logged", "severity": str|omitted, "hits":
      int|omitted, "error": str|omitted}], "redacted_text": str,
      "latency_ms": float}`.
    `checks` lists every check the resolved profile ran, not just the ones
    that fired — `flagged`/`action_taken` mark which entries actually did
    something, so `_enforce` below could name the tripped category if it
    ever needed to, though today it only reads the overall `verdict`.

    `messages` is forwarded close to verbatim (OpenAI chat format,
    `data["messages"]` for pre/during-call, or a synthesized single message
    for post-call/streaming where LiteLLM hands us a response object rather
    than a `messages` array) — the endpoint on Armor's side does the
    normalization into its standard request schema, not this plugin.

    `call_id` (LiteLLM's own `data["litellm_call_id"]`, one per gateway
    call — distinct from `session_id`, which spans a whole conversation)
    becomes `metadata.request_id` on Armor's side unless the caller already
    set `metadata["request_id"]` themselves, in which case theirs wins.
    """
    headers = {}
    if ARMOR_API_KEY:
        headers["x-api-key"] = ARMOR_API_KEY

    body = {"mode": mode, "messages": messages, "metadata": metadata or {}}
    if session_id:
        body["litellm_session_id"] = session_id
    if call_id:
        body["litellm_call_id"] = call_id

    async with httpx.AsyncClient(timeout=ARMOR_TIMEOUT_SECONDS) as client:
        resp = await client.post(ARMOR_SCAN_URL, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


_ADVISORY_VERDICTS = ("warn", "ask", "redact")


def _enforce(decision: dict, stage: str) -> None:
    # Armor's `ScanResponse.verdict` serializes uppercase (`#[serde(rename_all =
    # "UPPERCASE")]` on `crates/core::models::Verdict`) — e.g. `"BLOCK"`, not
    # `"block"`. Normalize once here rather than relying on every caller to
    # remember the casing.
    verdict = str(decision.get("verdict", "")).lower()
    if verdict == "block":
        raise ArmorGuardrailBlocked(f"Armor blocked this {stage} (verdict={verdict})")
    if verdict in _ADVISORY_VERDICTS:
        logger.warning(
            "Armor flagged this %s request (verdict=%s) — allowed through",
            stage,
            verdict,
        )
        return
    if verdict == "allow":
        return
    # A missing `verdict` key or a value outside Armor's own vocabulary —
    # matches neither the blocking nor the advisory set above, so without
    # this branch it would previously fall through and allow the call with
    # no log line at all. That's a 2xx response this adapter cannot parse as
    # a known verdict: a schema drift or a bug on one side of the wire, and
    # this is a guardrail, so the safe failure mode is to block loudly.
    logger.error(
        "Armor returned an unrecognized verdict %r for this %s request — blocking",
        decision.get("verdict"),
        stage,
    )
    raise ArmorGuardrailBlocked(
        f"Armor returned an unrecognized verdict for this {stage} request"
    )


async def _enforce_streaming(decision: dict, stage: str) -> None:
    """Same block/warn rule as `_enforce`, but a block on the streaming path has to
    raise LiteLLM's `StreamingCallbackError` — an in-flight SSE response can't be
    turned into a different HTTP error response the way a pre/post-call hook can.
    """
    try:
        _enforce(decision, stage=stage)
    except ArmorGuardrailBlocked as exc:
        from litellm.proxy.proxy_server import StreamingCallbackError

        raise StreamingCallbackError(str(exc)) from exc


class ArmorGuardrail(CustomGuardrail):
    """One class, registered once in config.yaml with `mode` as a list — see
    the module docstring for the hook -> Armor stage mapping and the README
    for the full config.yaml example.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: _CallType,
    ) -> Optional[Union[Exception, str, dict]]:
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is not True:
            return data

        messages = data.get("messages") or []
        if not messages:
            return data

        decision = await _scan(
            "input",
            messages,
            data.get("litellm_session_id"),
            data.get("metadata"),
            data.get("litellm_call_id"),
        )
        _enforce(decision, stage="input")
        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: _CallType,
    ) -> None:
        """Runs in parallel with the LLM call instead of blocking before it — lower
        added latency, at the cost of the LLM call having already started (its output
        is simply never returned to the caller if this then blocks). Configure this
        OR `pre_call` per guardrail entry, not both — they scan the same prompt text.
        """
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.during_call) is not True:
            return

        messages = data.get("messages") or []
        if not messages:
            return

        decision = await _scan(
            "input",
            messages,
            data.get("litellm_session_id"),
            data.get("metadata"),
            data.get("litellm_call_id"),
        )
        _enforce(decision, stage="input")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ) -> None:
        """Only fires for non-streaming responses — LiteLLM never calls this hook for a
        streamed completion, hence `async_post_call_streaming_iterator_hook` below.
        """
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return

        text = _extract_response_text(response)
        if not text:
            return

        decision = await _scan(
            "output",
            [{"role": "assistant", "content": text}],
            data.get("litellm_session_id"),
            data.get("metadata"),
            data.get("litellm_call_id"),
        )
        _enforce(decision, stage="output")

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call) is not True:
            async for chunk in response:
                yield chunk
            return

        session_id = request_data.get("litellm_session_id")
        metadata = request_data.get("metadata")
        call_id = request_data.get("litellm_call_id")
        accumulated = ""
        chunks_since_check = 0

        async for chunk in response:
            accumulated += _extract_chunk_text(chunk)
            chunks_since_check += 1
            yield chunk

            if accumulated and chunks_since_check >= ARMOR_STREAM_SAMPLE_EVERY:
                chunks_since_check = 0
                decision = await _scan(
                    "output",
                    [{"role": "assistant", "content": accumulated}],
                    session_id,
                    metadata,
                    call_id,
                )
                await _enforce_streaming(decision, stage="output")

        # Final check over the full response — catches content that only appears
        # after the last sample point. By now every chunk has already reached the
        # client, so this is an audit/logging backstop more than a preventive one
        # (see the module docstring's note on why streaming can't fully prevent this).
        if accumulated:
            decision = await _scan(
                "output",
                [{"role": "assistant", "content": accumulated}],
                session_id,
                metadata,
                call_id,
            )
            await _enforce_streaming(decision, stage="output")
