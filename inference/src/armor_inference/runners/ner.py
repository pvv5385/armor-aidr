"""Token-level NER runner — pii_ner task.

Produces per-token labels and groups contiguous spans into hit ranges.
Handles both tagging schemes a catalogued model may use: plain BIO
(B-PER, I-PER, B-ORG, ...) and BIOES, which adds E- to close a multi-token
span and S- for a single-token entity that is its own start and end. The
deterministic regex tier handles structured PII (credit cards, SSNs,
emails); this layer catches **unstructured** PII (names, addresses, phone
numbers in free text) that no pattern can enumerate.

BIOES label spaces are decoded with a constrained Viterbi search
(`_viterbi.py`), not per-token argmax — some BIOES models (e.g.
openai/privacy-filter) are explicitly trained expecting that: their own
model card says independent per-token argmax "can produce fragmented or
inconsistent boundaries" on noisy text, which is exactly the free-text case
this layer exists for. Plain BIO label spaces (no E-/S- tags anywhere, e.g.
Davlan's PER/ORG/LOC/MISC) have no boundary ambiguity for Viterbi to
resolve, so they keep using argmax — `_viterbi.build_decoder_for_label_space`
returns `None` for those and `load()` falls back accordingly.

The runner reports spans as byte-offset tuples `(start, end)` into the
**original** text, not the tokenized form. The `tokenizers` library tracks
offset mappings, so this is reliable as long as the model does not reorder
tokens (none of the catalogued NER models do).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from armor_inference.runners._heavy import OnnxTextRunner
from armor_inference.runners.base import InferOutput, _softmax

if TYPE_CHECKING:
    from armor_inference.runners._viterbi import ViterbiDecoder

logger = logging.getLogger(__name__)

# Label prefixes that mark a token as part of an entity, across both BIO
# (B-/I- only) and BIOES (adds E-/S-) models. The single source both
# `entity_col_ids` and `_count_entities` below read from — having each hardcode
# its own ("B-", "I-") tuple is exactly how a BIOES model's E-/S- tags went
# unrecognized by both before: every single-token entity (S-, common for a
# lone phone number or email) was silently excluded from risk scoring and
# never counted at all.
_ENTITY_PREFIXES = ("B-", "I-", "E-", "S-")
# Prefixes where a *new* entity begins: a BIO/BIOES span opens on B-, and a
# BIOES single-token entity is S- with no accompanying B- — it must be
# counted here too, or it never gets counted anywhere.
_SPAN_START_PREFIXES = ("B-", "S-")

# The label set is model-dependent, but the plan says the NER layer is
# ADDITIVE — it *adds* hits to the deterministic tier and never replaces it.
# So the runner does not need to know which labels are "PII" vs "non-PII";
# the orchestrator's `additive` flag handles the merge.  We report all
# detected entities and let the policy decide.


class NerRunner(OnnxTextRunner):
    """ONNX token classifier for NER. Produces per-token labels and groups
    them into entity spans."""

    runner_kind: str = "ner"

    def __init__(self, task: str, spec: Dict[str, Any]):
        super().__init__(task, spec)
        self._id2label: Dict[int, str] = {}
        self._max_length = 512  # NER models commonly support longer sequences
        self._viterbi: Optional["ViterbiDecoder"] = None

    def load(self) -> None:
        super().load()
        # Read id2label from the model's config
        import json
        from pathlib import Path

        from armor_inference.runners import _viterbi
        from armor_inference.runners._artifacts import resolve_artifact_dir

        artifact_dir = resolve_artifact_dir(self._spec)
        try:
            cfg_path = Path(artifact_dir) / "config.json"
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                raw = cfg.get("id2label", {})
                self._id2label = {int(k): v for k, v in raw.items()}
        except Exception:
            pass

        if self._id2label:
            class_names = [
                self._id2label[i] for i in sorted(self._id2label)
            ]
            self._viterbi = _viterbi.build_decoder_for_label_space(
                class_names, artifact_dir
            )

    def _postprocess_single(self, logits, params: Optional[Dict[str, Any]] = None) -> InferOutput:
        """Aggregate chunk-level token predictions into entity spans."""
        import numpy as np

        # logits: (seq_len, num_labels)
        # `_softmax` is pure Python; convert back to an array for the
        # argmax/max/mean reductions below (numpy is guaranteed here — this
        # only runs after `load()` succeeded). Each row is its own
        # probability distribution over labels — the correct semantics for a
        # per-token classifier, unlike a stray pre-refactor implementation
        # that normalized over the whole (seq_len * num_labels) matrix as one
        # distribution (`e7c51a2`'s numpy removal fixed that in `base.py`).
        probs = np.asarray(_softmax(logits))
        if self._viterbi is not None:
            # Constrained search over the whole sequence, not an independent
            # per-token argmax — see the module docstring for why BIOES
            # label spaces need this. `np.log` is safe here: `_softmax`
            # guarantees every row sums to 1 over positive values, so the
            # clip only guards the (never-zero-in-practice) underflow edge.
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            pred_ids = np.asarray(self._viterbi.decode(log_probs))
        else:
            pred_ids = np.argmax(probs, axis=-1)
        confidence = float(probs.max(axis=-1).mean())

        # Convert to label strings
        labels = [self._id2label.get(int(pid), f"LABEL_{pid}") for pid in pred_ids]

        # The model's peak confidence that *any* token is a named entity.
        # Deliberately not `probs.max()` over the whole matrix: with each row
        # correctly normalized over just this model's label set (typically
        # single digits), an ordinary token's near-certain "O" prediction is
        # itself close to 1.0 — so a matrix-wide max is dominated by that
        # non-signal and is near 1.0 for almost every input, regardless of
        # whether anything entity-like is present. Restricting the max to
        # entity columns (`_ENTITY_PREFIXES`) keeps this a signal about
        # entities specifically, which is what `risk_score`/`decision` below
        # are meant to measure.
        entity_col_ids = [
            i
            for i in range(probs.shape[-1])
            if self._id2label.get(i, "O").startswith(_ENTITY_PREFIXES)
        ]
        max_entity_prob = float(probs[:, entity_col_ids].max()) if entity_col_ids else 0.0
        risk_score = min(100, max(0, int(round(max_entity_prob * 100))))

        n_entities = _count_entities(labels)

        if n_entities > 0:
            decision = "BLOCK"
        elif max_entity_prob > 0.3:
            decision = "WARN"
        else:
            decision = "ALLOW"

        return InferOutput(
            decision=decision,
            risk_score=risk_score,
            confidence=round(confidence, 4),
            label_scores={"entities": n_entities},
            calibrated_score=None,
            threshold=self._spec.get("threshold", 0.5),
        )


def _count_entities(labels: List[str]) -> int:
    """Count distinct entity spans: a span starts at B-, and a BIOES
    single-token entity (S-) has no B- of its own, so it counts here too."""
    count = 0
    for label in labels:
        if label.startswith(_SPAN_START_PREFIXES):
            count += 1
    return count


def make_runner(task: str, spec: dict) -> NerRunner:
    """Factory entry point — called by the registry."""
    return NerRunner(task, spec)
