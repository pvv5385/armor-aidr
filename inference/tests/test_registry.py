"""Failure isolation: a broken model costs one task, never the service."""

from __future__ import annotations

from armor_inference.config import InferenceConfig
from armor_inference.registry import RunnerRegistry, variant_key


def _registry(**specs) -> RunnerRegistry:
    return RunnerRegistry(InferenceConfig(task_specs=specs))


def test_stub_tasks_load_with_no_ml_stack():
    reg = _registry(prompt_injection={"runner": "stub", "threshold": 0.5})
    info = reg.list_models()[0]
    assert info.available
    assert info.runner == "stub"
    assert reg.get("prompt_injection") is not None


def test_an_unloadable_task_is_isolated_from_a_healthy_one():
    """The property that matters operationally: losing a model loses a
    detection layer, not the tier."""
    reg = _registry(
        prompt_injection={"runner": "stub"},
        toxicity={"runner": "classifier", "model_id": "org/nope", "revision": "main"},
    )
    by_task = {m.task: m for m in reg.list_models()}

    assert by_task["prompt_injection"].available
    assert not by_task["toxicity"].available
    assert reg.get("toxicity") is None
    assert reg.get("prompt_injection") is not None


def test_an_unknown_runner_kind_is_unavailable_not_fatal():
    reg = _registry(weird={"runner": "quantum_oracle"})
    info = reg.list_models()[0]
    assert not info.available
    assert "unknown runner kind" in (info.detail or "")


def test_an_unavailable_task_still_reports_its_configured_pin():
    """Erasing the pin on failure destroys the provenance an operator needs to
    work out what was supposed to be there."""
    reg = _registry(
        toxicity={
            "runner": "classifier",
            "model_id": "unitary/toxic-bert",
            "revision": "main",
            "sha256": "deadbeef",
        }
    )
    info = reg.list_models()[0]
    assert not info.available
    assert info.model_id == "unitary/toxic-bert"
    assert info.revision == "main"
    assert info.sha256 == "deadbeef"
    assert info.model_version == "unitary/toxic-bert@main"


def test_a_pin_becomes_the_served_identity():
    """So that pinned routing, the response's model_version, and the cache key
    all name the same thing."""
    reg = _registry(prompt_injection={"runner": "stub", "model_id": "acme/stub-1", "revision": "v3"})
    assert reg.model_version("prompt_injection") == "acme/stub-1@v3"
    assert reg.serves("prompt_injection", "acme/stub-1", "v3")
    assert reg.serves("prompt_injection", "acme/stub-1")  # revision optional
    assert not reg.serves("prompt_injection", "acme/stub-1", "v4")
    assert not reg.serves("prompt_injection", "other/model")


def test_known_task_distinguishes_unconfigured_from_unloadable():
    """404 and 503 are different answers: one says "no such task", the other
    says "that task exists and is broken"."""
    reg = _registry(toxicity={"runner": "classifier", "model_id": "org/nope"})
    assert reg.known_task("toxicity")
    assert reg.get("toxicity") is None
    assert not reg.known_task("nonexistent")


async def test_a_variant_loads_beside_the_active_slot():
    reg = _registry(prompt_injection={"runner": "stub", "model_id": "acme/default"})
    info = await reg.load_variant(
        "prompt_injection", {"runner": "stub", "model_id": "acme/pinned", "revision": "v2"}
    )
    try:
        assert info.available
        assert not info.active
        # Both are servable, each under its own identity.
        assert reg.serves("prompt_injection", "acme/default")
        assert reg.serves("prompt_injection", "acme/pinned", "v2")
        assert reg.model_version("prompt_injection") == "acme/default@main"
    finally:
        await reg.stop_all()


async def test_loading_a_variant_twice_is_a_no_op():
    reg = _registry(prompt_injection={"runner": "stub"})
    spec = {"runner": "stub", "model_id": "acme/pinned", "revision": "v2"}
    try:
        first = await reg.load_variant("prompt_injection", spec)
        second = await reg.load_variant("prompt_injection", spec)
        assert first is second
        assert len([m for m in reg.list_models() if not m.active]) == 1
    finally:
        await reg.stop_all()


async def test_unloading_a_variant_stops_reporting_it():
    reg = _registry(prompt_injection={"runner": "stub"})
    await reg.load_variant(
        "prompt_injection", {"runner": "stub", "model_id": "acme/pinned", "revision": "v2"}
    )
    try:
        assert await reg.unload_variant("prompt_injection", "acme/pinned", "v2")
        assert not reg.serves("prompt_injection", "acme/pinned", "v2")
        assert not await reg.unload_variant("prompt_injection", "acme/pinned", "v2")
    finally:
        await reg.stop_all()


async def test_reload_swaps_the_active_slot():
    reg = _registry(prompt_injection={"runner": "stub", "model_id": "acme/v1"})
    try:
        info = await reg.reload_task(
            "prompt_injection", {"runner": "stub", "model_id": "acme/v2", "revision": "main"}
        )
        assert info.available
        assert reg.model_version("prompt_injection") == "acme/v2@main"
        assert not reg.serves("prompt_injection", "acme/v1")
    finally:
        await reg.stop_all()


async def test_a_failed_reload_leaves_the_task_unavailable_not_stale():
    """Serving the previous model after an operator asked for a different one
    is the silent-wrong-model failure. Unavailable is the honest state."""
    reg = _registry(prompt_injection={"runner": "stub", "model_id": "acme/v1"})
    try:
        info = await reg.reload_task("prompt_injection", {"runner": "classifier", "model_id": "org/nope"})
        assert not info.available
        assert reg.get("prompt_injection") is None
        assert reg.known_task("prompt_injection")
    finally:
        await reg.stop_all()


def test_variant_key_normalizes_a_missing_revision():
    assert variant_key("org/model", None) == "org/model@main"
    assert variant_key("org/model", "abc") == "org/model@abc"
