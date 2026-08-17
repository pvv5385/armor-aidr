"""Reader for `config/ml_catalog.yaml` — the cross-language task catalog.

This module is the Python half of a source of truth that `armor-core` also
reads. It stays a thin, dependency-light reader on purpose: the file is the
authority, and anything this module *decides* rather than *reports* is a place
the two languages can drift.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Where to look, in order:
#
#   1. `ARMOR_ML_CATALOG` — an operator pointing at a catalog of their own.
#      The image sets it, so in a container this is the only one that runs.
#   2. `/app/config` — the image's copy, if the env var was cleared.
#   3. `./config` — running from a repo root.
#   4. Three parents up from this file — an editable install or a
#      `PYTHONPATH=src` run, where the package is still inside the checkout.
#      This one does NOT fire for a normal `pip install`, since site-packages
#      has no `config/` above it; that case is what (1) and (3) are for.
_SEARCH_PATHS = (
    lambda: os.getenv("ARMOR_ML_CATALOG"),
    lambda: "/app/config/ml_catalog.yaml",
    lambda: str(Path.cwd() / "config" / "ml_catalog.yaml"),
    lambda: str(Path(__file__).resolve().parents[3] / "config" / "ml_catalog.yaml"),
)


class CatalogError(RuntimeError):
    """The catalog exists but does not parse, or names a runner the service
    cannot construct. Distinct from "no catalog found", which is fine — the
    service falls back to stub task specs."""


def catalog_path() -> Optional[str]:
    for candidate in _SEARCH_PATHS:
        raw = candidate()
        if raw and Path(raw).is_file():
            return raw
    return None


@functools.lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    """Parse the catalog, or return an empty one when there is no file.

    Cached: it is read at boot, on every install-target resolution, and by the
    tests. `load_catalog.cache_clear()` exists for tests that write their own.
    """
    path = catalog_path()
    if path is None:
        logger.info("no ml_catalog.yaml found; only stub tasks are available")
        return {"version": 1, "tasks": {}, "candidates": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"could not read ML catalog at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"ML catalog at {path} is not a mapping")
    data.setdefault("tasks", {})
    data.setdefault("candidates", {})
    logger.info("loaded ML catalog from %s (%d tasks)", path, len(data["tasks"]))
    return data


def task_names() -> List[str]:
    return sorted(load_catalog().get("tasks", {}))


def task_spec(task: str) -> Optional[Dict[str, Any]]:
    """The catalog entry for `task`, in the shape `InferenceConfig.task_specs`
    wants: runner + pin + threshold. The descriptive fields (license, size,
    detail) are dropped — they are provenance for the control plane's models
    view, not inputs to loading a model."""
    entry = load_catalog().get("tasks", {}).get(task)
    if entry is None:
        return None
    spec = {
        "runner": entry.get("runner", "stub"),
        "threshold": float(entry.get("threshold", 0.5)),
    }
    for key in ("model_id", "revision", "sha256"):
        if entry.get(key):
            spec[key] = entry[key]
    return spec


def heavy_task_specs() -> Dict[str, Dict[str, Any]]:
    """Every catalogued task, as runner specs. This is what
    `ARMOR_INFERENCE_PROFILE=catalog` boots — the tasks whose artifacts are
    absent simply come up `available: false`, which is the honest state and
    exactly what `GET /v1/models` is for."""
    return {task: task_spec(task) or {} for task in task_names()}


def vetted_model_ids(task: str) -> List[str]:
    """The model ids an operator may install for `task` without an explicit
    override: the pinned default plus its shortlist."""
    cat = load_catalog()
    ids = []
    default = cat.get("tasks", {}).get(task, {}).get("model_id")
    if default:
        ids.append(default)
    for cand in cat.get("candidates", {}).get(task, []) or []:
        model_id = cand.get("model_id")
        if model_id and model_id not in ids:
            ids.append(model_id)
    return ids


def task_overview() -> List[Dict[str, Any]]:
    """Per-task descriptive metadata for the control plane's models view:
    display name, one-line rationale, and the vetted shortlist an operator
    may install instead of the default pin (`candidates()`).

    Distinct from `task_spec`, which strips these same fields on purpose —
    they are provenance for a UI, not inputs to loading a runner. Also
    distinct from `GET /v1/models` (`registry.list_models`), which reports
    live load state rather than static catalog data.
    """
    cat = load_catalog()
    rows = [
        {
            "task": task,
            "display_name": entry.get("display_name") or task,
            "detail": entry.get("detail"),
            "candidates": candidates(task),
        }
        for task, entry in cat.get("tasks", {}).items()
    ]
    return sorted(rows, key=lambda r: r["task"])


def candidates(task: Optional[str] = None, *, open_only: bool = False) -> List[Dict[str, Any]]:
    """The vetted shortlist, flattened, for the fetch CLI's `--list` and the
    control plane's models view. `open_only` drops rows whose license carries
    use restrictions rather than being OSI-permissive."""
    cat = load_catalog()
    pins = {t: e.get("model_id") for t, e in cat.get("tasks", {}).items()}
    rows: List[Dict[str, Any]] = []
    for cand_task, entries in (cat.get("candidates", {}) or {}).items():
        if task is not None and cand_task != task:
            continue
        for entry in entries or []:
            if open_only and not entry.get("open_license", False):
                continue
            row = dict(entry)
            row["task"] = cand_task
            row["runner"] = cat.get("tasks", {}).get(cand_task, {}).get("runner", "stub")
            row["is_current_pin"] = entry.get("model_id") == pins.get(cand_task)
            rows.append(row)
    return rows
