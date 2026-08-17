"""Tests for `_enforce`'s verdict handling — the block/warn/allow mapping
`armor_guardrail`'s module docstring documents as this adapter's
degradation policy.
"""

from __future__ import annotations

import pytest

from armor_guardrail import ArmorGuardrailBlocked, _enforce


def test_allow_passes_through_silently():
    _enforce({"verdict": "ALLOW"}, stage="input")


@pytest.mark.parametrize("verdict", ["WARN", "ASK", "REDACT"])
def test_advisory_verdicts_pass_through(verdict, caplog):
    _enforce({"verdict": verdict}, stage="input")
    assert "allowed through" in caplog.text


def test_block_raises():
    with pytest.raises(ArmorGuardrailBlocked, match="blocked"):
        _enforce({"verdict": "BLOCK"}, stage="input")


def test_verdict_casing_is_normalized():
    # Armor's wire format is uppercase (`#[serde(rename_all = "UPPERCASE")]`
    # on `crates/core::models::Verdict`); lowercase must behave identically.
    with pytest.raises(ArmorGuardrailBlocked):
        _enforce({"verdict": "block"}, stage="input")


def test_missing_verdict_key_is_blocked_not_silently_allowed():
    """Regression guard: a malformed 2xx body with no `verdict` key used to
    match neither the block nor the warn/ask/redact sets and fall through,
    allowing the call with no log line at all."""
    with pytest.raises(ArmorGuardrailBlocked, match="unrecognized verdict"):
        _enforce({}, stage="input")


def test_unrecognized_verdict_string_is_blocked_not_silently_allowed(caplog):
    with pytest.raises(ArmorGuardrailBlocked, match="unrecognized verdict"):
        _enforce({"verdict": "QUARANTINE"}, stage="input")
    assert "QUARANTINE" in caplog.text


# `_enforce_streaming` (`armor_guardrail.py`) is a thin wrapper — same
# `_enforce` call, `ArmorGuardrailBlocked` translated to LiteLLM's
# `StreamingCallbackError` — deliberately not re-tested here: it needs
# `litellm.proxy.proxy_server`, whose own import chain is unrelated to this
# adapter and brittle across litellm/fastapi version combinations. The
# verdict-parsing logic under test above is identical either way.
