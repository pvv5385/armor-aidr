"""The `Runner` protocol and the dependency-free `StubRunner`.

A runner turns a batch of texts into the model-native half of the contract.
Heavy runners lazy-import their stack in `load()`; the stub needs nothing,
which is what lets the whole service — contract, cache, batching, registry,
saturation, install jobs — run and be tested on an image with no ML
dependencies and no weights.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _softmax(logits):
    """Numerically-stable softmax (pure Python — no numpy).

    Accepts a 1-D sequence of logits or a 2-D sequence of sequences
    (softmax applied per row). Numpy arrays work transparently because
    they iterate as sequences. Returns a list shaped like the input, so
    it stays usable on the dependency-free base install.
    """

    def _row(values):
        xs = [float(x) for x in values]
        peak = max(xs) if xs else 0.0
        exps = [math.exp(x - peak) for x in xs]
        total = sum(exps)
        return [e / total for e in exps]

    first = next(iter(logits), None)
    is_row = isinstance(first, (list, tuple)) or (hasattr(first, "ndim") and first.ndim >= 1)
    if is_row:
        return [_row(row) for row in logits]
    return _row(logits)


def _sigmoid(logits):
    """Element-wise sigmoid (pure Python — no numpy), for multi-label
    classifiers whose labels are independent rather than mutually exclusive
    (`unitary/toxic-bert`'s `problem_type: "multi_label_classification"`:
    a text can be simultaneously `toxic` and `insult` and `obscene`, each
    with its own probability — `_softmax` would force those to compete for
    a shared 1.0 of probability mass, which is wrong for this shape of
    model and inflates whichever logit is least-negative into a false
    positive on inputs where every label is actually confidently absent.

    Same shape contract as `_softmax`: 1-D sequence or 2-D sequence of
    sequences (applied per element either way, since sigmoid has no
    cross-label normalization to do)."""

    def _row(values):
        return [1.0 / (1.0 + math.exp(-float(x))) for x in values]

    first = next(iter(logits), None)
    is_row = isinstance(first, (list, tuple)) or (hasattr(first, "ndim") and first.ndim >= 1)
    if is_row:
        return [_row(row) for row in logits]
    return _row(logits)


class RunnerUnavailable(Exception):
    """`load()` could not make this runner serve: the heavy dependencies are
    absent, the pinned artifact is missing, or its digest did not match. The
    registry catches it, marks the task `available: false`, and leaves every
    other task running."""


@dataclass
class InferOutput:
    """One item's result, before the service attaches `model_version`."""

    decision: str
    risk_score: int
    confidence: Optional[float] = None
    label_scores: Optional[Dict[str, float]] = None
    calibrated_score: Optional[float] = None
    threshold: Optional[float] = None


class Runner:
    """Subclasses set `task` / `model_version` and implement `load` +
    `infer_batch`.

    `infer_batch` MUST return exactly one output per input, in order — the
    batcher hands results back by position, and a runner that reorders or
    drops one answers a caller with someone else's score.
    """

    task: str = "generic"
    runner_kind: str = "base"
    model_version: str = "base@v0"
    # "cpu", "cuda", or "rocm" — set by `load()` for ONNX-backed runners
    # (`_heavy.py`) once the execution provider actually initializes. None
    # for the stub, which has no notion of a device.
    device: Optional[str] = None

    def load(self) -> None:
        """Prepare to serve: load weights, verify the pinned digest. May raise
        `RunnerUnavailable`."""

    def infer_batch(
        self, texts: List[str], params: Optional[Dict[str, Any]] = None
    ) -> List[InferOutput]:
        raise NotImplementedError


@dataclass
class StubRunner(Runner):
    """A deterministic keyword scorer.

    It exists so the infrastructure runs anywhere and so the escalation path
    can be exercised end to end with no ML stack. It is **not** a detection
    tier: its recall on anything it has no pattern for is zero, which is the
    exact weakness the real classifier exists to fix. `GET /v1/models` reports
    it as `runner: "stub"` so nothing downstream can mistake it for a model,
    and the scorecard gate will refuse to let a stub-backed check enforce.
    """

    task: str = "prompt_injection"
    threshold: float = 0.5
    model_version: str = "stub@v1"
    runner_kind: str = "stub"
    _signals: Dict[str, List] = field(default_factory=dict)

    # Per-task signal sets. Weights accumulate and saturate at 1.0.
    _SIGNALS = {
        "prompt_injection": [
            (r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) instructions", 0.7),
            (r"\bdisregard\b.{0,40}\b(?:rules?|instructions?|polic(?:y|ies)|guidelines?)\b", 0.6),
            (r"\bdeveloper mode\b|\bDAN\b", 0.6),
            (r"\b(?:reveal|show|print|repeat)\b.{0,40}\b(?:system prompt|instructions)\b", 0.5),
            (r"</?(?:system|assistant)>", 0.4),
        ],
        "jailbreak": [
            (r"\bpretend (?:you are|to be)\b|\bact as if\b", 0.6),
            (r"\bno restrictions\b|\bwithout any filter\b|\bunfiltered\b", 0.6),
            (r"ignore (?:all |any |the )?(?:previous|prior) instructions", 0.7),
        ],
        "toxicity": [
            (r"\b(?:idiot|moron|stupid|kill yourself|hate you)\b", 0.7),
            (r"\b(?:slur|racist|sexist)\b", 0.5),
        ],
        # No stub signals: a keyword list is not a plausible stand-in for
        # token-level NER or embedding similarity, and pretending otherwise
        # would make an empty result look like a clean scan. These score 0
        # (ALLOW) until a real runner backs them.
        "pii_ner": [],
        "over_refusal": [],
        "topic_intent": [],
    }

    def __post_init__(self):
        if not self._signals:
            self._signals = {
                task: [(re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in sigs]
                for task, sigs in self._SIGNALS.items()
            }

    def _score_one(self, text: str) -> InferOutput:
        prob = 0.0
        for pattern, weight in self._signals.get(self.task, []):
            if pattern.search(text or ""):
                prob = min(1.0, prob + weight)
        risk = int(round(prob * 100))
        decision = "BLOCK" if prob >= self.threshold else ("WARN" if prob > 0 else "ALLOW")
        return InferOutput(
            decision=decision,
            risk_score=risk,
            confidence=round(prob, 4),
            # The stub reports no `calibrated_score`. It has never been
            # measured against a benchmark suite, and a number here would be a
            # claim it cannot support — the scorecard gate reads this field.
            calibrated_score=None,
            label_scores={"unsafe": round(prob, 4), "safe": round(1.0 - prob, 4)},
            threshold=self.threshold,
        )

    def infer_batch(
        self, texts: List[str], params: Optional[Dict[str, Any]] = None
    ) -> List[InferOutput]:
        return [self._score_one(t) for t in texts]
