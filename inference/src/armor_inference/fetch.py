"""Operator-run model fetch — the one deliberate exception to "no network on
the serving path".

Weights are a supply-chain artifact. Downloading one is an auditable act that
produces a sha256 the operator pins, so that what serves traffic is exactly
what was reviewed and a silent swap cannot happen underneath. That is why this
module is never reachable from `/v1/infer`: it is invoked from the CLI
(`python -m armor_inference.fetch`) or from an install job
(`POST /v1/models/install`), both of which are an operator asking for it.

Export is also the only step that *may* need the heavy stack (torch +
optimum, the `[export]` extra). Serving needs just `[onnx]`. That asymmetry
is the point: the multi-GB payload lives here, offline, and never ships in
the serving image. "May", not "always": some repos already publish a
ready-made ONNX graph alongside the original checkpoint (see
`_find_prebuilt_onnx`) — for those, this module downloads it directly via
`huggingface_hub` and never imports torch/transformers/optimum at all, which
matters beyond just saving bandwidth: it also sidesteps needing `transformers`
to recognize whatever custom architecture class the checkpoint uses, which a
from-source export otherwise requires.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from armor_inference.catalog import (
    candidates,
    load_catalog,
    task_names,
    vetted_model_ids,
)
from armor_inference.runners._artifacts import artifact_sha256, resolve_artifact_dir
from armor_inference.runners.base import RunnerUnavailable

logger = logging.getLogger(__name__)

# Runner kinds served on CPU through onnxruntime: their artifact must be an
# ONNX graph, so the fetch exports and quantizes the checkpoint rather than
# snapshotting it. `guard_llm` runs on torch and keeps a raw snapshot.
_ONNX_RUNNERS = {"classifier", "ner", "embedding", "nli"}

_ORT_EXPORT_CLASS = {
    "classifier": "ORTModelForSequenceClassification",
    "nli": "ORTModelForSequenceClassification",
    "ner": "ORTModelForTokenClassification",
    "embedding": "ORTModelForFeatureExtraction",
}

# Repo-relative paths a pre-built ONNX graph is conventionally published
# under, checked in this order. The `onnx/` subfolder is the modern
# convention (used by transformers.js-tagged repos, e.g.
# openai/privacy-filter's `onnx/model_quantized.onnx`); bare root filenames
# are the older/alternate convention some optimum-exported repos use.
# Quantized is preferred over fp32 within each, matching `_heavy.py`'s own
# `_find_onnx` preference — so serving behaves the same regardless of which
# path fetched the artifact.
_PREBUILT_ONNX_CANDIDATES = (
    "onnx/model_quantized.onnx",
    "onnx/model.onnx",
    "model_quantized.onnx",
    "model.onnx",
)

# Repo-root files worth grabbing alongside a pre-built ONNX graph, if present.
# `tokenizer.json` is required (checked separately, in `_find_prebuilt_onnx`);
# the rest are optional context `_heavy.py`/`ner.py`/`classifier.py` read
# when available. `viterbi_calibration.json` is `_viterbi.py`'s calibration
# artifact — see that module's docstring.
_PREBUILT_SIDECAR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "viterbi_calibration.json",
)


def _require(import_fn: Callable[[], Any], what: str):
    """Lazy-import with an error that names the extra to install. A missing
    export dependency is a normal state for the serving image, not a crash."""
    try:
        return import_fn()
    except ImportError as exc:
        raise RunnerUnavailable(
            f"{what} is not installed — this step needs the [export] extra: "
            f"pip install './inference[export]'"
        ) from exc


def resolve_fetch_target(
    task: str,
    *,
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    allow_unvetted: bool = False,
) -> Dict[str, Any]:
    """What to fetch for `task`, and where it lands.

    Defaults to the catalog's pin. `model_id` may override it, but only to
    another model on that task's vetted shortlist — an off-list model needs
    `allow_unvetted`, which the HTTP install path does not expose. The easy
    path is a reviewed model; the other one requires saying so.
    """
    entry = load_catalog().get("tasks", {}).get(task)
    if entry is None:
        raise ValueError(f"unknown task '{task}'. Known tasks: {task_names()}")

    chosen_id = model_id or entry.get("model_id")
    if not chosen_id:
        raise ValueError(f"task '{task}' has no model_id in the catalog and none was given")
    chosen_rev = revision or entry.get("revision") or "main"

    if model_id and not allow_unvetted:
        vetted = vetted_model_ids(task)
        if model_id not in vetted:
            raise ValueError(
                f"'{model_id}' is not on the vetted shortlist for task '{task}'. "
                f"Choose one of {vetted}, or pass --allow-unvetted."
            )

    # The catalog itself never ships a digest (see this module's header) —
    # there is nothing to trust it against on a model's *first* fetch. But an
    # operator who has already fetched and reviewed the catalog's default pin
    # can record its digest back into `ml_catalog.yaml`'s `sha256` field, and
    # from then on every automatic install verifies against it instead of
    # just recording whatever the download happened to produce. Only applies
    # to the catalog's own default pin: a shortlist candidate's bytes are a
    # different artifact, and pinning the default's digest against it would
    # reject a legitimate install rather than catch a tampered one.
    expected_sha256 = None
    if chosen_id == entry.get("model_id") and chosen_rev == (entry.get("revision") or "main"):
        expected_sha256 = entry.get("sha256")

    return {
        "task": task,
        "runner": entry.get("runner", "stub"),
        "model_id": chosen_id,
        "revision": chosen_rev,
        "dest_dir": resolve_artifact_dir({"model_id": chosen_id}),
        "expected_sha256": expected_sha256,
    }


def _snapshot_download(model_id: str, revision: str, dest_dir: str) -> None:
    """Raw HuggingFace snapshot, for torch-served runners."""
    hub = _require(
        lambda: __import__("huggingface_hub", fromlist=["snapshot_download"]),
        "huggingface_hub",
    )
    hub.snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=dest_dir,
        # Weights, tokenizer and config only — skip the other frameworks'
        # duplicates and git metadata, which would otherwise land in the
        # digest and make it depend on what the hub happened to serve.
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", ".gitattributes"],
    )


def _repo_dir(repo_path: str) -> str:
    """The directory portion of a repo-relative path, with a trailing slash
    (empty string for a root-level file) — for checking whether a sidecar
    file sits next to a given candidate rather than assuming repo root."""
    return repo_path.rsplit("/", 1)[0] + "/" if "/" in repo_path else ""


def _find_prebuilt_onnx(model_id: str, revision: str) -> Optional[str]:
    """Check whether `model_id`'s repo already ships a usable ONNX graph, so
    `_default_downloader` can fetch it directly instead of exporting from the
    original checkpoint. Returns the repo-relative path of the best match, or
    `None` if the repo doesn't ship one (the common case — this is a bonus
    path, not something every model is expected to have).

    Requires `tokenizer.json` alongside it: `_heavy.py`'s `_find_tokenizer`
    has no fallback for repos that only ship a slow-tokenizer format, so a
    graph with no fast tokenizer isn't actually servable and isn't worth
    preferring over a from-source export that builds one. "Alongside" checks
    the candidate's own directory first, then repo root — repos are
    inconsistent about whether the tokenizer travels with the graph (e.g.
    Davlan/bert-base-multilingual-cased-ner-hrl ships both under `onnx/`) or
    stays at root while only the graph is nested (e.g. Xenova/toxic-bert).
    Checking root alone, as this used to, misses the former case entirely.
    """
    hub = _require(
        lambda: __import__("huggingface_hub", fromlist=["list_repo_files"]),
        "huggingface_hub",
    )
    try:
        files = set(hub.list_repo_files(model_id, revision=revision))
    except Exception as exc:  # noqa: BLE001 — network/repo lookup, not fatal
        logger.info(
            "could not list %s@%s's files (%s); falling back to export",
            model_id,
            revision,
            exc,
        )
        return None

    for candidate in _PREBUILT_ONNX_CANDIDATES:
        if candidate not in files:
            continue
        if _repo_dir(candidate) + "tokenizer.json" in files or "tokenizer.json" in files:
            return candidate
    return None


def _download_prebuilt_onnx(
    model_id: str, revision: str, dest_dir: str, onnx_repo_path: str
) -> None:
    """Download an already-exported ONNX graph plus its tokenizer/config —
    no torch, no transformers, no optimum needed for this path, just
    `huggingface_hub`.
    """
    hub = _require(
        lambda: __import__("huggingface_hub", fromlist=["hf_hub_download", "list_repo_files"]),
        "huggingface_hub",
    )
    files = set(hub.list_repo_files(model_id, revision=revision))

    # Keep the graph's own filename when picking siblings — a large ONNX
    # graph's external-data shard(s) (model_q4.onnx_data,
    # model_q4.onnx_data_1, ...) are referenced by that exact name inside the
    # graph's protobuf, so renaming the .onnx file without renaming those to
    # match would leave the graph unable to find its own weights.
    onnx_basename = onnx_repo_path.rsplit("/", 1)[-1]
    onnx_dir_prefix = _repo_dir(onnx_repo_path)
    sibling_data_files = sorted(
        f
        for f in files
        if f.startswith(onnx_dir_prefix + onnx_basename) and f != onnx_repo_path
    )

    wanted: List[str] = [onnx_repo_path, *sibling_data_files]
    # Prefer a sidecar file colocated with the graph over one at repo root —
    # some repos (Davlan) publish a distinct tokenizer.json under `onnx/`
    # rather than sharing the root one; picking root here would silently grab
    # a different, less specific artifact than the one _find_prebuilt_onnx
    # actually verified exists alongside this graph.
    for name in _PREBUILT_SIDECAR_FILES:
        colocated = onnx_dir_prefix + name
        if colocated in files:
            wanted.append(colocated)
        elif name in files:
            wanted.append(name)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for repo_path in wanted:
        local_path = hub.hf_hub_download(repo_id=model_id, revision=revision, filename=repo_path)
        # Flatten onnx/... into dest_dir's root — _heavy.py's _find_onnx and
        # _find_tokenizer look directly in dest_dir, not a nested subfolder.
        target_name = repo_path.rsplit("/", 1)[-1]
        shutil.copy2(local_path, dest / target_name)


def _should_skip_quantization(dest_dir: str) -> bool:
    """DeBERTa-v3's disentangled attention degrades badly under int8 dynamic
    quantization — the graph runs and the scores are noise, which is worse
    than being slow. Detect it from config.json and keep fp32."""
    cfg_path = Path(dest_dir) / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return str(cfg.get("model_type", "")).lower() in ("deberta-v2", "deberta-v3")


def _quantize_onnx(dest_dir: str) -> None:
    """Best-effort int8 dynamic quantization. Non-fatal: if it is unavailable
    for this op set the fp32 graph still serves, just slower."""
    if _should_skip_quantization(dest_dir):
        logger.info("skipping int8 quantization for the DeBERTa-v3 model in %s", dest_dir)
        return
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(dest_dir)
        quantizer.quantize(
            save_dir=dest_dir,
            quantization_config=AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            ),
        )
        logger.info("wrote model_quantized.onnx (int8) into %s", dest_dir)
    except Exception as exc:  # noqa: BLE001 — an optimization, not a requirement
        logger.warning("ONNX quantization skipped (%s); serving fp32 model.onnx", exc)


def _export_onnx(model_id: str, revision: str, dest_dir: str, runner: str) -> None:
    """Export to an ONNX graph plus `tokenizer.json`, then quantize.

    Only reached when `model_id` has no pre-built ONNX graph on the hub (see
    `_find_prebuilt_onnx`) — the common case needs none of this. The error
    below names that specifically, rather than a generic "not installed",
    because the fix differs by deployment: rebuild the Docker image with the
    `WITH_EXPORT` build arg for the HTTP install path, or reach for
    `make ml-fetch` / an offline `pip install` for the CLI path. A bare
    "optimum is not installed" leaves an operator guessing which.
    """
    try:
        ort_mod = __import__("optimum.onnxruntime", fromlist=list(_ORT_EXPORT_CLASS.values()))
        tfm = __import__("transformers", fromlist=["AutoTokenizer"])
    except ImportError as exc:
        raise RunnerUnavailable(
            f"{model_id} publishes no pre-built ONNX graph, so installing it needs "
            f"a local export (torch + optimum-onnx + transformers — the [export] "
            f"extra), which isn't installed here. If this is the Docker image: it "
            f"was built with WITH_EXPORT=false (the default); rebuild with "
            f"`--build-arg WITH_EXPORT=true` (or `ARMOR_INFERENCE_WITH_EXPORT=true "
            f"docker compose --profile ml build inference`) to install this model "
            f"over HTTP. Otherwise: `make ml-fetch TASK=<task>`, or "
            f"`pip install './inference[export]'` followed by "
            f"`armor-inference-fetch`."
        ) from exc
    ort_cls = getattr(ort_mod, _ORT_EXPORT_CLASS[runner])
    model = ort_cls.from_pretrained(model_id, revision=revision, export=True)
    model.save_pretrained(dest_dir)
    # The serving runner loads `tokenizer.json` through the standalone
    # `tokenizers` crate, so it needs the fast tokenizer written out here —
    # `transformers` is an export-time dependency only.
    tfm.AutoTokenizer.from_pretrained(model_id, revision=revision).save_pretrained(dest_dir)
    _quantize_onnx(dest_dir)


def _default_downloader(model_id: str, revision: str, dest_dir: str, runner: Optional[str]) -> None:
    if runner in _ONNX_RUNNERS:
        prebuilt = _find_prebuilt_onnx(model_id, revision)
        if prebuilt is not None:
            logger.info(
                "%s@%s ships a pre-built ONNX graph (%s) — downloading it directly, "
                "no export needed",
                model_id,
                revision,
                prebuilt,
            )
            _download_prebuilt_onnx(model_id, revision, dest_dir, prebuilt)
        else:
            _export_onnx(model_id, revision, dest_dir, runner)
    else:
        _snapshot_download(model_id, revision, dest_dir)


def fetch_model(
    model_id: str,
    revision: str,
    dest_dir: str,
    *,
    runner: Optional[str] = None,
    downloader: Optional[Callable[[str, str, str], None]] = None,
    expected_sha256: Optional[str] = None,
) -> str:
    """Fetch `model_id@revision` into `dest_dir`; return the sha256 to pin.

    The digest is computed by the same `artifact_sha256` the runner verifies
    with, over the tree that actually landed — so what the operator pins is
    exactly what will be loaded, not a hash of what the hub advertised.

    Without `expected_sha256`, that digest is self-certifying: it proves the
    artifact matches itself, not that it matches anything an operator
    reviewed. That is fine for the CLI's normal use (print the digest, let a
    human read it and paste it into config as a separate, later step) but it
    is not a verification step. A caller that intends to treat the returned
    digest as a trusted pin without a human in between — e.g. re-installing a
    model that already has a known-good sha256 on file — must pass that value
    as `expected_sha256`; a mismatch raises instead of silently returning
    whatever was downloaded.

    `downloader` is injectable so the pipeline is testable without the network
    or the export stack.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    effective = downloader or (lambda m, r, d: _default_downloader(m, r, d, runner))
    effective(model_id, revision, dest_dir)
    digest = artifact_sha256(dest_dir)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RunnerUnavailable(
            f"integrity check failed for {model_id}@{revision}: downloaded "
            f"artifact has sha256={digest}, expected {expected_sha256} — "
            f"refusing to treat the download as verified"
        )
    logger.info("fetched %s@%s into %s (sha256=%s)", model_id, revision, dest_dir, digest)
    return digest


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="armor-inference-fetch",
        description=(
            "Download and export a pinned model into the artifacts directory, "
            "and print the sha256 to pin. Needs the [export] extra."
        ),
    )
    parser.add_argument("--task", help="catalog task to fetch, e.g. prompt_injection")
    parser.add_argument("--model-id", help="override the catalog's pinned model")
    parser.add_argument("--revision", help="override the catalog's pinned revision")
    parser.add_argument(
        "--allow-unvetted",
        action="store_true",
        help="permit a model that is not on the task's vetted shortlist",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help=(
            "verify the download against this digest instead of self-certifying it "
            "(e.g. a value already pinned in config, or obtained out of band); "
            "fetch fails if the artifact that lands does not match"
        ),
    )
    parser.add_argument(
        "--list", action="store_true", help="list the vetted models and exit"
    )
    parser.add_argument(
        "--open-only", action="store_true", help="with --list, only OSI-permissive licenses"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.list:
        rows = candidates(args.task, open_only=args.open_only)
        if not rows:
            print("no catalogued models match")
            return 0
        width = max(len(r["model_id"]) for r in rows)
        for row in rows:
            marker = "*" if row.get("is_current_pin") else " "
            print(
                f"{marker} {row['task']:<18} {row['model_id']:<{width}}  "
                f"{row.get('license', '?'):<14} {row.get('size', '?'):<6} "
                f"{row.get('hardware', '')}"
            )
        print("\n* = the catalog's current pin")
        return 0

    if not args.task:
        parser.error("--task is required (or use --list)")

    try:
        target = resolve_fetch_target(
            args.task,
            model_id=args.model_id,
            revision=args.revision,
            allow_unvetted=args.allow_unvetted,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"fetching {target['model_id']}@{target['revision']} → {target['dest_dir']}")
    try:
        digest = fetch_model(
            target["model_id"],
            target["revision"],
            target["dest_dir"],
            runner=target["runner"],
            expected_sha256=args.expected_sha256 or target.get("expected_sha256"),
        )
    except RunnerUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(f"\nsha256: {digest}\n")
    print("Pin it so the runner verifies what it loads:")
    print(
        json.dumps(
            {
                args.task: {
                    "runner": target["runner"],
                    "model_id": target["model_id"],
                    "revision": target["revision"],
                    "sha256": digest,
                    "threshold": 0.5,
                }
            },
            indent=2,
        )
    )
    print("\n…as ARMOR_INFERENCE_TASKS on the inference service.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
