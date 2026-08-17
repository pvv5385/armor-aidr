"""One runner and one batcher per configured task, with failures isolated to
the task that caused them.

The property worth stating plainly: **a broken model never takes the sidecar
down.** A missing dependency, an absent artifact, a digest mismatch, a corrupt
tokenizer — each marks its own task `available: false` (503 on that path) and
leaves every other task serving. On a security product this is the difference
between losing one detection layer and losing the whole tier.

Beyond the per-task *active* slot the registry can hold additional loaded
**variants**, keyed by `(task, model_id@revision)`, so several deployments
sharing one sidecar are each served the model they pinned rather than
colliding on one slot.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Dict, List, Optional, Tuple

from armor_inference.batching import BatchProcessor
from armor_inference.config import InferenceConfig
from armor_inference.contract import ModelInfo
from armor_inference.runners.base import Runner, RunnerUnavailable, StubRunner

logger = logging.getLogger(__name__)

# The dependency-free kind is always constructible.
RUNNER_FACTORIES: Dict[str, Callable[[str, dict], Runner]] = {
    "stub": lambda task, spec: StubRunner(task=task, threshold=float(spec.get("threshold", 0.5))),
}

# Heavy kinds → the module exposing `make_runner(task, spec)`. Imported only
# when a task actually asks for one, and those modules import their ML stack
# inside `load()` rather than at module scope — so an image without
# onnxruntime still boots, and the task that needed it is the only casualty.
#
# A kind listed here with no module actually implemented yet (e.g.
# `guard_llm`) fails as "unavailable" with a clear detail, rather than as
# "unknown runner kind".
_HEAVY_KIND_MODULES = {
    "classifier": "armor_inference.runners.classifier",
    "ner": "armor_inference.runners.ner",
    "embedding": "armor_inference.runners.embedding",
    "nli": "armor_inference.runners.nli",
    "guard_llm": "armor_inference.runners.guard_llm",
}


def _factory_for(kind: str) -> Callable[[str, dict], Runner]:
    if kind in RUNNER_FACTORIES:
        return RUNNER_FACTORIES[kind]
    module_path = _HEAVY_KIND_MODULES.get(kind)
    if module_path is None:
        raise RunnerUnavailable(
            f"unknown runner kind '{kind}'; known kinds: "
            f"{sorted(set(RUNNER_FACTORIES) | set(_HEAVY_KIND_MODULES))}"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RunnerUnavailable(
            f"runner kind '{kind}' is not available in this build: {exc}"
        ) from exc
    return module.make_runner


def variant_key(model_id: Optional[str], revision: Optional[str]) -> str:
    """The identity of one loaded model: `model_id@revision`."""
    return f"{model_id or 'model'}@{revision or 'main'}"


class RunnerRegistry:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._batchers: Dict[str, BatchProcessor] = {}
        self._models: Dict[str, ModelInfo] = {}
        self._variant_batchers: Dict[Tuple[str, str], BatchProcessor] = {}
        self._variant_models: Dict[Tuple[str, str], ModelInfo] = {}
        for task, spec in config.task_specs.items():
            self._build(task, spec or {})

    # ── Construction ───────────────────────────────────────────────────────

    def _make(
        self, task: str, spec: dict, *, active: bool
    ) -> Tuple[Optional[BatchProcessor], ModelInfo]:
        """Build `(batcher, info)` for one spec, or `(None, unavailable-info)`.

        Shared by the active slot and by variants so both isolate load
        failures identically — a variant that fails must not be able to take
        down the slot it was loaded beside.
        """
        kind = spec.get("runner", "stub")
        model_id, revision = spec.get("model_id"), spec.get("revision")
        try:
            runner = _factory_for(kind)(task, spec)
            runner.load()  # heavy runners verify the pin here
            batcher = BatchProcessor(
                runner,
                max_batch_size=self.config.max_batch_size,
                max_wait_ms=self.config.max_wait_ms,
                max_queue=self.config.max_queue,
                budget_ms=self.config.budget_ms,
            )
            # When the spec pins a model, the pin is the served identity. That
            # keeps three things lined up that otherwise drift: pinned request
            # routing, the `model_version` in the response, and the cache key.
            model_version = variant_key(model_id, revision) if model_id else runner.model_version
            device = getattr(runner, "device", None)
            if kind != "stub":
                logger.info(
                    "task '%s' (runner=%s) loaded: %s%s%s",
                    task,
                    kind,
                    model_version,
                    f" [{device}]" if device else "",
                    "" if active else " (variant)",
                )
            return batcher, ModelInfo(
                task=task,
                model_version=model_version,
                runner=kind,
                available=True,
                model_id=model_id,
                revision=revision,
                sha256=spec.get("sha256"),
                active=active,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 — isolation is the point
            # RunnerUnavailable is an expected outcome (no artifact yet, no
            # deps in this image) and logs as a warning. Anything else is a
            # bug or a corrupt artifact and gets a traceback — but neither
            # propagates past this task.
            expected = isinstance(exc, RunnerUnavailable)
            logger.log(
                logging.WARNING if expected else logging.ERROR,
                "task '%s' (runner=%s) unavailable: %s",
                task,
                kind,
                exc,
                exc_info=not expected,
            )
            # Report the *configured* pin even though it did not load: we know
            # what this task was supposed to serve, and erasing that to a
            # sentinel destroys the provenance an operator needs to fix it.
            model_version = variant_key(model_id, revision) if model_id else "-"
            return None, ModelInfo(
                task=task,
                model_version=model_version,
                runner=kind,
                available=False,
                model_id=model_id,
                revision=revision,
                sha256=spec.get("sha256"),
                detail=str(exc) if expected else f"load failed: {exc}",
                active=active,
            )

    def _build(self, task: str, spec: dict) -> None:
        batcher, info = self._make(task, spec, active=True)
        if batcher is not None:
            self._batchers[task] = batcher
        self._models[task] = info

    # ── Lookup ─────────────────────────────────────────────────────────────

    def get(self, task: str) -> Optional[BatchProcessor]:
        return self._batchers.get(task)

    def model_version(self, task: str) -> Optional[str]:
        info = self._models.get(task)
        return info.model_version if info and info.available else None

    def known_task(self, task: str) -> bool:
        """Configured at all — as opposed to configured but not loadable. The
        two are different answers to the caller (404 vs 503)."""
        return task in self._models

    @staticmethod
    def _matches(info: ModelInfo, model_id: str, revision: Optional[str]) -> bool:
        """Does a loaded model satisfy a `model_id[@revision]` pin?"""
        if info.model_id:
            loaded_id, loaded_rev = info.model_id, (info.revision or "main")
        else:
            loaded_id, _, loaded_rev = info.model_version.partition("@")
            loaded_rev = loaded_rev or "main"
        return model_id == loaded_id and (revision is None or revision == loaded_rev)

    def get_for(
        self, task: str, model_id: str, revision: Optional[str] = None
    ) -> Optional[Tuple[BatchProcessor, str]]:
        """The batcher serving exactly `model_id[@revision]` for `task` — the
        active slot when it matches, else a loaded variant, else `None`.

        Returns the model version alongside it so the response and the cache
        key both carry the identity of the model that actually ran.
        """
        active = self._models.get(task)
        if active is not None and active.available and self._matches(active, model_id, revision):
            batcher = self._batchers.get(task)
            if batcher is not None:
                return batcher, active.model_version
        for (v_task, v_key), batcher in self._variant_batchers.items():
            if v_task != task:
                continue
            info = self._variant_models.get((v_task, v_key))
            if info is not None and info.available and self._matches(info, model_id, revision):
                return batcher, info.model_version
        return None

    def serves(self, task: str, model_id: str, revision: Optional[str] = None) -> bool:
        return self.get_for(task, model_id, revision) is not None

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values()) + list(self._variant_models.values())

    # ── Mutation ───────────────────────────────────────────────────────────

    async def reload_task(self, task: str, spec: dict) -> ModelInfo:
        """Hot-swap a task's active runner — what an install job calls when it
        finishes. Stops the old batcher, builds and starts the new one."""
        old = self._batchers.pop(task, None)
        if old is not None:
            await old.stop()
        self._build(task, spec)
        new = self._batchers.get(task)
        if new is not None:
            await new.start()
        return self._models[task]

    async def load_variant(self, task: str, spec: dict) -> ModelInfo:
        """Load an additional model for `task` without touching the active
        slot. Idempotent per `(task, model_id@revision)`: an already-available
        variant is a no-op, a previously failed one is retried."""
        key = (task, variant_key(spec.get("model_id"), spec.get("revision")))
        existing = self._variant_models.get(key)
        if existing is not None and existing.available:
            return existing

        old = self._variant_batchers.pop(key, None)
        if old is not None:
            await old.stop()

        batcher, info = self._make(task, spec, active=False)
        if batcher is not None:
            self._variant_batchers[key] = batcher
            await batcher.start()
        self._variant_models[key] = info
        return info

    async def unload_variant(
        self, task: str, model_id: str, revision: Optional[str] = None
    ) -> bool:
        """`load_variant`'s mirror. Returns False when nothing matched."""
        key = (task, variant_key(model_id, revision))
        batcher = self._variant_batchers.pop(key, None)
        had_info = self._variant_models.pop(key, None) is not None
        if batcher is not None:
            await batcher.stop()
        return had_info or batcher is not None

    async def start_all(self) -> None:
        for batcher in list(self._batchers.values()) + list(self._variant_batchers.values()):
            await batcher.start()

    async def stop_all(self) -> None:
        for batcher in list(self._batchers.values()) + list(self._variant_batchers.values()):
            await batcher.stop()

    def batcher_stats(self) -> Dict[str, dict]:
        stats = {task: b.stats() for task, b in self._batchers.items()}
        stats.update(
            {f"{task}::{key}": b.stats() for (task, key), b in self._variant_batchers.items()}
        )
        return stats
