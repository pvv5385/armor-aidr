"""NLI (Natural Language Inference) runner — for future use and the
`guard_llm` path.

NLI models score (premise, hypothesis) pairs as
entailment / contradiction / neutral. This runner is structurally similar
to the classifier but operates on **paired** inputs: the premise is the
scanned text, and the hypothesis is a label description.

It is not used by any catalog task today (the catalogued tasks use
classifier, ner, or embedding runners), but the registry already maps
`nli` to this module so a future task can use it without code changes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from armor_inference.runners._heavy import OnnxTextRunner
from armor_inference.runners.base import InferOutput, _softmax

logger = logging.getLogger(__name__)

# Standard NLI label order: contradiction, neutral, entailment
_NLI_LABELS = ["contradiction", "neutral", "entailment"]


class NliRunner(OnnxTextRunner):
    """ONNX NLI runner. Scores (premise, hypothesis) pairs."""

    runner_kind: str = "nli"

    def __init__(self, task: str, spec: Dict[str, Any]):
        super().__init__(task, spec)
        self._hypothesis: str = spec.get("hypothesis", "This text is unsafe.")
        self._entailment_idx: int = 2  # "entailment" is typically index 2

    def _postprocess_single(self, logits, params: Optional[Dict[str, Any]] = None) -> InferOutput:
        probs = _softmax(logits)

        # NOTE: neither `self._hypothesis` nor a per-request
        # `params["hypothesis"]` override reaches the model. The hypothesis
        # has to be applied at tokenization time — it is half of the
        # (premise, hypothesis) pair the graph scores — and by the time
        # postprocessing sees raw logits the pair has already been encoded.
        # This method previously read the override into a local and dropped
        # it, which looked like support for a knob that does not exist. The
        # dead read is gone; the gap it implied is not fixed. Wiring the
        # hypothesis into the tokenizer is a behavior change to a detection
        # path and belongs in its own change with its own tests.
        entailment_prob = probs[self._entailment_idx]
        contradiction_prob = probs[0]
        neutral_prob = probs[1] if len(probs) > 1 else 0.0

        risk_score = min(100, max(0, int(round(entailment_prob * 100))))

        if entailment_prob >= self._threshold:
            decision = "BLOCK"
        elif entailment_prob > 0.3:
            decision = "WARN"
        else:
            decision = "ALLOW"

        return InferOutput(
            decision=decision,
            risk_score=risk_score,
            confidence=round(entailment_prob, 4),
            label_scores={
                "contradiction": round(contradiction_prob, 4),
                "neutral": round(neutral_prob, 4),
                "entailment": round(entailment_prob, 4),
            },
            calibrated_score=None,
            threshold=self._threshold,
        )


def make_runner(task: str, spec: dict) -> NliRunner:
    """Factory entry point — called by the registry."""
    return NliRunner(task, spec)
