"""Embedding similarity runner — topic_intent task.

Computes cosine distance between the input text's embedding and
precomputed label vectors. The plan calls this "option (b)" — embedding
similarity rather than NLI — because a cross-encoder costs a forward pass
per candidate label, which is the wrong shape for a request-path check
that must score in <100ms.

The label vectors are embedded at startup from a small label set (the
`topic_labels` param), and each inference call is a single embedding
forward pass + cosine comparison — O(n_labels × embed_dim), not
O(n_labels × seq_len × embed_dim) like NLI.

If no `topic_labels` param is supplied, the runner defaults to a generic
"this is a topic match" signal based on embedding norm — a fallback that
the policy can choose to ignore.

Heavy deps (numpy) are imported inside `load()` and methods, never at
module scope, so the registry can decide a task is unavailable without
the import cost or the crash.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from armor_inference.runners._heavy import OnnxTextRunner
from armor_inference.runners.base import InferOutput

logger = logging.getLogger(__name__)

# topic_labels is caller-supplied (request body). Cap how many distinct label
# embeddings this process will ever hold, and how many labels one request may
# ask to embed, so an attacker can't force unbounded memory growth or an
# unbounded number of forward passes via novel label strings.
_MAX_LABEL_CACHE = 256
_MAX_LABELS_PER_REQUEST = 64


def _cosine_similarity(a, b) -> float:
    """Stable cosine similarity between two equal-length sequences.

    Pure Python (no numpy), so it is testable on the dependency-free base
    install; numpy arrays work transparently because they iterate as
    sequences. Zero-norm vectors score 0.0 rather than raising.
    """
    a = [float(x) for x in a]
    b = [float(y) for y in b]

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    # strict=True: two embeddings of different length is a bug in the caller
    # or the model spec, never a legitimate input. Silently truncating to the
    # shorter one returns a plausible-looking similarity score computed over
    # part of the vector, which is the kind of wrong that never gets noticed.
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


class EmbeddingRunner(OnnxTextRunner):
    """ONNX embedding model. Produces a fixed-size vector and compares it
    against precomputed label vectors via cosine similarity."""

    runner_kind: str = "embedding"

    def __init__(self, task: str, spec: Dict[str, Any]):
        super().__init__(task, spec)
        self._label_embeddings: "OrderedDict[str, Any]" = OrderedDict()
        self._embed_dim: int = 0

    def load(self) -> None:
        super().load()
        # Read embedding dimension from the ONNX graph output
        if self._session is not None:
            outputs = self._session.get_outputs()
            if outputs:
                shape = outputs[0].shape
                if isinstance(shape, list) and len(shape) >= 2:
                    self._embed_dim = shape[-1]
                else:
                    self._embed_dim = 384  # common default (all-MiniLM-L6-v2)

    def _embed(self, texts: List[str]):
        """Embed one or more texts. Returns (n_texts, embed_dim) numpy array."""
        if self._session is None or self._tokenizer is None:
            raise RuntimeError(f"task '{self.task}' not loaded")

        import numpy as np

        inputs, chunk_map, n_original = self._tokenize_batch(texts)
        raw_outputs = self._session.run(self._output_names, inputs)

        # The output is typically (batch, seq_len, embed_dim) or (batch, embed_dim)
        embedding = raw_outputs[0]

        # If 3-D, mean-pool over sequence length (excluding padding)
        if embedding.ndim == 3:
            mask_name = None
            for k in inputs:
                if "attention" in k.lower() or "mask" in k.lower():
                    mask_name = k
                    break
            mask = inputs.get(mask_name, None) if mask_name else None
            if mask is not None:
                mask_f = mask.astype(np.float32)
                mask_expanded = np.expand_dims(mask_f, axis=-1)
                sum_emb = (embedding * mask_expanded).sum(axis=1)
                sum_mask = mask_expanded.sum(axis=1).clip(min=1e-9)
                embedding = sum_emb / sum_mask
            else:
                embedding = embedding.mean(axis=1)

        return embedding  # (n_texts, embed_dim)

    def _compute_label_embeddings(self, labels: List[str]) -> Dict[str, Any]:
        """Embed label texts once, cache for the request."""
        result = {}
        for label in labels:
            vec = self._embed([label])[0]
            result[label] = vec
        return result

    def _postprocess_single(self, logits, params: Optional[Dict[str, Any]] = None) -> InferOutput:
        """For embeddings, `logits` is actually the embedding vector."""
        return self._score_with_embeddings(logits, params)

    def _score_with_embeddings(self, embedding, params: Optional[Dict[str, Any]] = None) -> InferOutput:
        """Score by comparing embedding against label vectors."""
        import numpy as np

        params = params or {}
        topic_labels: List[str] = params.get("topic_labels", [])
        topic_threshold: float = float(params.get("topic_threshold", self._threshold))

        if len(topic_labels) > _MAX_LABELS_PER_REQUEST:
            logger.warning(
                "topic_labels truncated from %d to %d for task '%s'",
                len(topic_labels), _MAX_LABELS_PER_REQUEST, self.task,
            )
            topic_labels = topic_labels[:_MAX_LABELS_PER_REQUEST]

        if not topic_labels:
            # No labels to compare — return a neutral signal
            return InferOutput(
                decision="ALLOW",
                risk_score=0,
                confidence=0.0,
                label_scores={"embedding_norm": float(np.linalg.norm(embedding))},
                calibrated_score=None,
                threshold=topic_threshold,
            )

        # Compare against each label
        label_scores: Dict[str, float] = {}
        best_score = -1.0

        for label in topic_labels:
            if label in self._label_embeddings:
                self._label_embeddings.move_to_end(label)
            else:
                self._label_embeddings[label] = self._embed([label])[0]
                if len(self._label_embeddings) > _MAX_LABEL_CACHE:
                    self._label_embeddings.popitem(last=False)
            label_vec = self._label_embeddings[label]
            sim = _cosine_similarity(embedding, label_vec)
            label_scores[label] = round(sim, 4)
            if sim > best_score:
                best_score = sim

        risk_score = min(100, max(0, int(round(max(0.0, best_score) * 100))))

        if best_score >= topic_threshold:
            decision = "BLOCK"
        elif best_score > 0.3:
            decision = "WARN"
        else:
            decision = "ALLOW"

        return InferOutput(
            decision=decision,
            risk_score=risk_score,
            confidence=round(max(0.0, best_score), 4),
            label_scores=label_scores,
            calibrated_score=None,
            threshold=topic_threshold,
        )

    def infer_batch(
        self, texts: List[str], params: Optional[Dict[str, Any]] = None
    ) -> List[InferOutput]:
        """Override to handle embedding aggregation differently from classification."""
        if self._session is None or self._tokenizer is None:
            from armor_inference.runners.base import RunnerUnavailable
            raise RunnerUnavailable(f"task '{self.task}' not loaded")

        embeddings = self._embed(texts)
        return [self._score_with_embeddings(emb, params) for emb in embeddings]


def make_runner(task: str, spec: dict) -> EmbeddingRunner:
    """Factory entry point — called by the registry."""
    return EmbeddingRunner(task, spec)
