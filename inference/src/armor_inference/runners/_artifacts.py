"""Pinned-artifact resolution and verification.

The rule this module enforces: **the serving path never touches the network.**
A runner loads weights only from an operator-supplied local directory, and
verifies the pinned digest before use. There is no fallback that downloads
"just this once" — that fallback is how a model swap becomes invisible, and a
guardrail whose weights can change underneath it is not a guardrail.

Getting weights onto disk is a separate, deliberate act: mount them at
`/models`, or run an install job (`armor_inference.install`), which is the
same fetch pipeline triggered by an operator rather than by traffic.

Dependency-free — stdlib hashlib only, so it is testable with no ML stack.
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from typing import Optional

from armor_inference.config import default_artifacts_dir
from armor_inference.runners.base import RunnerUnavailable

_CHUNK = 1 << 20  # 1 MiB


def artifact_dirname(model_id: str) -> str:
    """`org/name` → `org__name`. One directory per model id, flat, so a mount
    is browsable and a path traversal in a model id cannot escape the base."""
    return model_id.replace("/", "__").replace("\\", "__")


def _hash_file(path: Path, h) -> None:
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)


def artifact_sha256(path: str) -> str:
    """A stable digest for one artifact: a file's contents, or a directory
    hashed over its sorted relative paths *and* their contents.

    Hashing the names as well as the bytes is what makes it a digest of the
    tree rather than of a bag of files — renaming `model.onnx` to
    `model_quantized.onnx` changes which graph gets served, so it has to
    change the digest.
    """
    p = Path(path)
    h = sha256()
    if p.is_file():
        _hash_file(p, h)
        return h.hexdigest()
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(f.relative_to(p)).encode("utf-8"))
            h.update(b"\0")
            _hash_file(f, h)
        return h.hexdigest()
    raise RunnerUnavailable(f"model artifact not found: {path}")


def resolve_artifact_dir(spec: dict) -> str:
    """Where `spec`'s model lives locally.

    Order: an explicit `artifacts_dir` in the spec →
    `$ARMOR_INFERENCE_ARTIFACTS_DIR/<model_id>` →
    `default_artifacts_dir()/<model_id>` (`<repo root>/models` outside a
    container). Never a URL.
    """
    explicit = spec.get("artifacts_dir")
    if explicit:
        return str(explicit)
    base = os.getenv("ARMOR_INFERENCE_ARTIFACTS_DIR") or default_artifacts_dir()
    return str(Path(base) / artifact_dirname(spec.get("model_id") or "model"))


def verify_pinned(path: str, expected_sha256: Optional[str]) -> None:
    """Fail closed unless the artifact is present and matches its pin.

    An *unpinned* artifact (no `sha256` in the spec) loads. That is a real gap
    and a deliberate one: an operator who mounts a directory they built is
    trusting their own filesystem, and requiring a digest before anything can
    run would push people to skip the mechanism entirely. `GET /v1/models`
    reports `sha256: null` for those, and the control plane's models view is
    where that becomes visible.
    """
    p = Path(path)
    if not p.exists():
        raise RunnerUnavailable(
            f"model artifact not found at {path} — mount it at "
            f"$ARMOR_INFERENCE_ARTIFACTS_DIR or install it (no implicit download)"
        )
    if expected_sha256:
        actual = artifact_sha256(path)
        if actual != expected_sha256:
            raise RunnerUnavailable(
                f"sha256 mismatch for {path}: pinned {expected_sha256}, found {actual}"
            )
