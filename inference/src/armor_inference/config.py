"""Env-driven configuration, following `crates/api/src/config.rs`'s `ARMOR_*`
convention so one deployment has one naming scheme.

The defaults matter more than usual here: they are what makes
`docker run armor-inference` work with no ML dependencies installed, no
weights on disk, and no network. Every heavier posture is opt-in.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def default_artifacts_dir() -> str:
    """Fallback when `ARMOR_INFERENCE_ARTIFACTS_DIR` is unset.

    `<repo root>/models` — the same directory `ARMOR_MODELS_DIR=./models
    docker compose --profile ml up` bind-mounts to `/models` (see the root
    `.gitignore` and `docker-compose.yml`), so a bare local run and a
    bind-mounted container run share one place for weights instead of the
    sidecar inventing a second, home-directory convention. The image itself
    never hits this: `Dockerfile.inference` sets the env var explicitly, so
    this only fires for a local dev run from the checkout.

    `parents[3]` mirrors `catalog.py`'s `_SEARCH_PATHS`: four levels up from
    this file (armor_inference/ -> src/ -> inference/ -> repo root).
    """
    return str(Path(__file__).resolve().parents[3] / "models")


# The boot default: three tasks on the dependency-free StubRunner. Enough that
# the contract, cache, batching, saturation and registry paths are all live and
# testable on an image with nothing in it.
_DEFAULT_TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "prompt_injection": {"runner": "stub", "threshold": 0.5},
    "toxicity": {"runner": "stub", "threshold": 0.5},
    "pii_ner": {"runner": "stub", "threshold": 0.5},
}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("%s is not an integer; using %d", name, default)
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class InferenceConfig:
    task_specs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _DEFAULT_TASK_SPECS.items()}
    )
    max_batch_size: int = 16
    max_wait_ms: int = 10
    max_queue: int = 256
    # The sidecar's own ceiling on how long an item may sit in the queue.
    # Deliberately far above the caller's `ARMOR_INFERENCE_TIMEOUT_MS` (120ms):
    # the caller's deadline is authoritative for the request path, and this
    # exists only to keep a wedged runner from growing an unbounded queue.
    budget_ms: int = 2000
    cache_maxsize: int = 4096
    # Where pinned artifacts live. Empty means `default_artifacts_dir()`
    # (`<repo root>/models`); the image sets it to /models, which is the
    # mount point.
    artifacts_dir: str = ""
    # Whether POST /v1/models/install may download. Off unless asked: a
    # security product that can be told to fetch and load new weights over
    # HTTP has a supply-chain hole, and the honest default is that weights
    # arrive by an operator action. `docker compose --profile ml` turns it on
    # explicitly, which is that operator action.
    allow_install: bool = False
    # When set, every /v1 route requires `Authorization: Bearer <token>`.
    # Unset by default because the sidecar's intended posture is an internal
    # network with no route from outside — but "internal" is an assumption
    # about someone else's network, so the knob exists. Pairs with the API's
    # ARMOR_INFERENCE_AUTH_TOKEN.
    auth_token: str = ""
    # Where `lifespan` (main.py) publishes the auto-generated mutation token
    # when `auth_token` above is unset — armor-core reads it from the same
    # path (`ARMOR_INFERENCE_TOKEN_FILE`, `crates/api/src/config.rs`) off a
    # volume `docker-compose.yml` mounts into both containers, so
    # install/reload work with zero config regardless of whether the stack
    # was started via `make` or plain `docker compose`. Never consulted when
    # `auth_token` is set — an operator-supplied token always wins and is
    # never written here. Writing (and reading) is best-effort: a bare,
    # non-compose run just has nothing mounted at this path and silently
    # keeps today's behavior — a fresh token logged once per boot.
    token_file: str = "/var/run/armor/inference-token"

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        cfg = cls(
            max_batch_size=_int("ARMOR_INFERENCE_MAX_BATCH", 16),
            max_wait_ms=_int("ARMOR_INFERENCE_MAX_WAIT_MS", 10),
            max_queue=_int("ARMOR_INFERENCE_MAX_QUEUE", 256),
            budget_ms=_int("ARMOR_INFERENCE_BUDGET_MS", 2000),
            cache_maxsize=_int("ARMOR_INFERENCE_CACHE_SIZE", 4096),
            artifacts_dir=os.getenv("ARMOR_INFERENCE_ARTIFACTS_DIR", ""),
            allow_install=_bool("ARMOR_INFERENCE_ALLOW_INSTALL", False),
            auth_token=os.getenv("ARMOR_INFERENCE_AUTH_TOKEN", "").strip(),
            token_file=os.getenv("ARMOR_INFERENCE_TOKEN_FILE", "/var/run/armor/inference-token"),
        )
        cfg.task_specs = _resolve_task_specs()
        return cfg

    def artifact_path(self, *parts: str) -> str:
        base = self.artifacts_dir or default_artifacts_dir()
        return str(Path(base, *parts))


def _resolve_task_specs() -> Dict[str, Dict[str, Any]]:
    """Three ways to say what this sidecar serves, most specific first.

    1. `ARMOR_INFERENCE_TASKS` — a JSON task→spec map. Full control, used by
       compose and by anyone running a model the catalog does not list.
    2. `ARMOR_INFERENCE_PROFILE=catalog` — every task in `ml_catalog.yaml`,
       at its pinned model. The normal way to run the real tier.
    3. Neither — the stub defaults.

    A malformed `ARMOR_INFERENCE_TASKS` raises rather than falling through to
    the stubs. Booting on keyword heuristics because an env var had a trailing
    comma, while `/v1/models` cheerfully reports `available: true`, is the kind
    of silent downgrade that gets noticed a quarter later.
    """
    raw = os.getenv("ARMOR_INFERENCE_TASKS")
    if raw and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"ARMOR_INFERENCE_TASKS is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("ARMOR_INFERENCE_TASKS must be a non-empty JSON object")
        for task, spec in parsed.items():
            if not isinstance(spec, dict):
                raise ValueError(f"ARMOR_INFERENCE_TASKS['{task}'] must be an object")
        return parsed

    profile = os.getenv("ARMOR_INFERENCE_PROFILE", "stub").strip().lower()
    if profile == "catalog":
        from armor_inference.catalog import heavy_task_specs

        specs = heavy_task_specs()
        if specs:
            return specs
        logger.warning(
            "ARMOR_INFERENCE_PROFILE=catalog but the catalog has no tasks; using stub defaults"
        )
    elif profile not in ("stub", ""):
        logger.warning("unknown ARMOR_INFERENCE_PROFILE=%r; using stub defaults", profile)

    return {k: dict(v) for k, v in _DEFAULT_TASK_SPECS.items()}
