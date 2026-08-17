"""Sequence classifier runner — prompt_injection, toxicity, over_refusal.

Turns ONNX logits into `InferOutput` via softmax (single-label models) or
sigmoid (multi-label models — see `_load_is_multi_label`), picks the
"unsafe" label by regex against the model's label names, and maps the
probability onto decision/risk_score/confidence. The unsafe-label regex is
the one piece of per-model knowledge that cannot be generic:
`unitary/toxic-bert` names its positive class `"toxic"`,
`protectai/deberta-v3-base-prompt-injection-v2` names it `"LABEL_1"`, and a
wrong pick inverts the signal.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from armor_inference.runners._heavy import OnnxTextRunner
from armor_inference.runners.base import (
    InferOutput,
    _sigmoid,
    _softmax,
)

# Per-task regex patterns that identify the "positive" (unsafe) label.
# Checked against the ONNX graph's label names (read from config.json or
# inferred from the output dimension).  First match wins; a task that
# matches none defaults to the last label (the convention for binary
# classifiers that output [safe, unsafe]).
_UNSAFE_PATTERNS = {
    "prompt_injection": re.compile(r"^(LABEL_1|INJECTION|UNSAFE|POSITIVE|YES)$", re.I),
    "toxicity": re.compile(r"^(toxic|unsafe|harmful|positive|LABEL_1)$", re.I),
    "over_refusal": re.compile(r"^(refusal|LABEL_1|UNSAFE|POSITIVE|YES)$", re.I),
}


def _load_label_names(spec: dict) -> Optional[List[str]]:
    """Try to read label names from the artifact's config.json."""
    import json
    from pathlib import Path

    from armor_inference.runners._artifacts import resolve_artifact_dir

    try:
        cfg_path = Path(resolve_artifact_dir(spec)) / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            names = cfg.get("id2label") or cfg.get("label2id")
            if isinstance(names, dict):
                # id2label is {0: "LABEL_0", 1: "LABEL_1"} or {"0": "LABEL_0", ...}
                return [names[str(i)] for i in range(len(names))]
            if isinstance(names, list):
                return names
    except Exception:
        pass
    return None


def _load_is_multi_label(spec: dict) -> bool:
    """`config.json`'s `problem_type` — `"multi_label_classification"` means
    the labels are independent (a text can be `toxic` AND `insult` AND
    `obscene` at once, each with its own probability), so each logit needs
    its own sigmoid rather than one softmax shared across all labels.
    Missing/anything else defaults to single-label (softmax), the prior
    behavior — a model that doesn't say otherwise is assumed mutually
    exclusive, same convention transformers itself uses."""
    import json
    from pathlib import Path

    from armor_inference.runners._artifacts import resolve_artifact_dir

    try:
        cfg_path = Path(resolve_artifact_dir(spec)) / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return cfg.get("problem_type") == "multi_label_classification"
    except Exception:
        pass
    return False


class ClassifierRunner(OnnxTextRunner):
    """ONNX sequence classifier. Softmax (single-label) or sigmoid
    (multi-label) over logits → pick unsafe label by regex → map to
    `InferOutput`."""

    runner_kind: str = "classifier"

    def __init__(self, task: str, spec: Dict[str, Any]):
        super().__init__(task, spec)
        self._label_names = _load_label_names(spec) or []
        self._unsafe_idx: Optional[int] = None
        self._unsafe_pattern = _UNSAFE_PATTERNS.get(task)
        self._multi_label = _load_is_multi_label(spec)

    def load(self) -> None:
        super().load()
        # Resolve the unsafe label index from model metadata or regex
        if self._label_names:
            self._unsafe_idx = _pick_unsafe_idx(self._label_names, self._unsafe_pattern)

    def _postprocess_single(self, logits, params: Optional[Dict[str, Any]] = None) -> InferOutput:
        probs = _sigmoid(logits) if self._multi_label else _softmax(logits)
        n_labels = len(probs)

        # Determine which index is the "unsafe" one
        unsafe_idx = self._unsafe_idx
        if unsafe_idx is None:
            # Fallback: last label is positive (binary convention)
            unsafe_idx = n_labels - 1

        unsafe_prob = probs[unsafe_idx]
        safe_prob = 1.0 - unsafe_prob

        risk_score = min(100, max(0, int(round(unsafe_prob * 100))))

        if unsafe_prob >= self._threshold:
            decision = "BLOCK"
        elif unsafe_prob > 0.3:
            decision = "WARN"
        else:
            decision = "ALLOW"

        label_scores = {}
        if self._label_names and len(self._label_names) >= n_labels:
            for i, name in enumerate(self._label_names[:n_labels]):
                label_scores[name] = round(probs[i], 4)
        else:
            label_scores["unsafe"] = round(unsafe_prob, 4)
            label_scores["safe"] = round(safe_prob, 4)

        return InferOutput(
            decision=decision,
            risk_score=risk_score,
            confidence=round(unsafe_prob, 4),
            label_scores=label_scores,
            calibrated_score=None,
            threshold=self._threshold,
        )


def _pick_unsafe_idx(label_names: List[str], pattern: Optional[re.Pattern]) -> Optional[int]:
    """Find the index of the unsafe/positive label."""
    if pattern:
        for i, name in enumerate(label_names):
            if pattern.match(name):
                return i
    return None


def make_runner(task: str, spec: dict) -> ClassifierRunner:
    """Factory entry point — called by the registry."""
    return ClassifierRunner(task, spec)
