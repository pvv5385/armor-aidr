"""Constrained Viterbi decoding for BIOES-tagged NER label sequences.

Ported from openai/privacy-filter's `opf/_core/decoding.py` +
`sequence_labeling.py` (Apache-2.0, https://github.com/openai/privacy-filter)
into pure numpy — no torch, so it stays inside the `[onnx]` serving extra
instead of pulling the `[export]` stack's ~2GB into the always-running
service. Scope is trimmed to what this repo's single-sequence-at-a-time
runner needs: the CPU decode path only, not the reference's CUDA-batched
`decode_many` — that exists for serving many long sequences in one batch at
once; this runner decodes one escalation call's text at a time, and the
decode itself is a tiny O(seq_len * num_classes^2) DP pass next to the
transformer forward pass it follows, so there's nothing to batch.

Only meaningful for genuinely BIOES-tagged label spaces (every entity type
carries B-/I-/E-/S-). A plain BIO model (B-/I- only, e.g. Davlan's
PER/ORG/LOC/MISC) has no boundary ambiguity for Viterbi to resolve —
`build_label_info` requires the full BIOES set per entity and raises
`ValueError` if it's missing, so `ner.py` catches that at load time and
falls back to plain argmax for label spaces that were never BIOES to begin
with, rather than force this decoder onto a scheme it wasn't designed for.

Why this exists at all: `openai/privacy-filter`'s own model card says per-
token argmax is the wrong decoding method for it — "decode labels with a
constrained Viterbi decoder using linear-chain transition scoring, rather
than taking an independent argmax for each token... especially in noisy or
mixed-format text where local token decisions alone can produce fragmented
or inconsistent boundaries." The six bias parameters below correspond
exactly to `viterbi_calibration.json`'s `operating_points.default.biases`,
shipped alongside the model artifact the same way `model.onnx`/
`tokenizer.json` are.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_NEG_INF = -1e9
_BOUNDARY_PREFIXES = ("B", "I", "E", "S")
_BACKGROUND_LABEL = "O"

# Matches viterbi_calibration.json's operating_points.default.biases keys.
VITERBI_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)

DEFAULT_CALIBRATION_FILENAME = "viterbi_calibration.json"


class LabelInfo:
    """Resolved BIOES label-space lookup tables for one model's id2label."""

    __slots__ = (
        "token_to_span_label",
        "token_boundary_tags",
        "background_token_label",
        "background_span_label",
    )

    def __init__(
        self,
        token_to_span_label: Dict[int, int],
        token_boundary_tags: Dict[int, Optional[str]],
        background_token_label: int,
        background_span_label: int,
    ) -> None:
        self.token_to_span_label = token_to_span_label
        self.token_boundary_tags = token_boundary_tags
        self.background_token_label = background_token_label
        self.background_span_label = background_span_label


def build_label_info(class_names: Sequence[str]) -> LabelInfo:
    """Build BIOES label-space lookup tables from an ordered id2label list.

    Raises `ValueError` if the label space has no background class, or any
    entity type is missing one of B-/I-/E-/S- — the caller's job to catch
    that and fall back to argmax for label spaces that were never BIOES.
    """
    span_label_lookup: Dict[str, int] = {_BACKGROUND_LABEL: 0}
    boundary_label_lookup: Dict[str, Dict[str, int]] = {}
    token_to_span_label: Dict[int, int] = {}
    token_boundary_tags: Dict[int, Optional[str]] = {}
    background_idx: Optional[int] = None

    for idx, name in enumerate(class_names):
        if name == _BACKGROUND_LABEL:
            background_idx = idx
            token_to_span_label[idx] = 0
            token_boundary_tags[idx] = None
            continue
        boundary, sep, base_label = name.partition("-")
        if not sep or not base_label or boundary not in _BOUNDARY_PREFIXES:
            raise ValueError(
                f"unrecognized label {name!r}: expected 'O' or '<B|I|E|S>-<name>'"
            )
        span_idx = span_label_lookup.setdefault(base_label, len(span_label_lookup))
        token_to_span_label[idx] = span_idx
        token_boundary_tags[idx] = boundary
        boundary_label_lookup.setdefault(base_label, {})[boundary] = idx

    if background_idx is None:
        raise ValueError("label space has no background class 'O'")

    for base_label, mapping in boundary_label_lookup.items():
        missing = set(_BOUNDARY_PREFIXES) - set(mapping)
        if missing:
            raise ValueError(
                f"label {base_label!r} is missing boundary tags {sorted(missing)} "
                "— not a BIOES label space"
            )

    return LabelInfo(
        token_to_span_label=token_to_span_label,
        token_boundary_tags=token_boundary_tags,
        background_token_label=background_idx,
        background_span_label=0,
    )


def zero_biases() -> Dict[str, float]:
    """All-zero transition biases — still fully constrained to valid BIOES
    paths, just uncalibrated. The fallback when a model ships no
    `viterbi_calibration.json`."""
    return {key: 0.0 for key in VITERBI_BIAS_KEYS}


def load_calibration(artifact_dir: str) -> Dict[str, float]:
    """Load transition biases from `viterbi_calibration.json` in
    `artifact_dir`, or return all-zero biases if the model didn't ship one."""
    path = Path(artifact_dir) / DEFAULT_CALIBRATION_FILENAME
    if not path.is_file():
        return zero_biases()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        biases = payload["operating_points"]["default"]["biases"]
        return {key: float(biases[key]) for key in VITERBI_BIAS_KEYS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "%s is malformed (%s); decoding with all-zero transition biases",
            path,
            exc,
        )
        return zero_biases()


class ViterbiDecoder:
    """Decode `[seq_len, num_classes]` per-token log-probabilities into a
    BIOES-valid label-id path via constrained Viterbi search.

    The transition table is built once, from the label space and the six
    calibration biases — not learned, not touched again per request. Every
    (prev, next) class pair is either structurally forbidden (masked to
    `_NEG_INF` — e.g. a span can't jump straight from B- to a different
    entity's I-) or scored with the matching bias (0.0 for any legal edge
    the six named biases don't cover).
    """

    def __init__(self, label_info: LabelInfo, biases: Mapping[str, float]) -> None:
        import numpy as np

        num_classes = len(label_info.token_to_span_label)
        tags = label_info.token_boundary_tags
        spans = label_info.token_to_span_label
        bg_token = label_info.background_token_label
        bg_span = label_info.background_span_label

        start = np.full(num_classes, _NEG_INF, dtype=np.float32)
        end = np.full(num_classes, _NEG_INF, dtype=np.float32)
        transition = np.full((num_classes, num_classes), _NEG_INF, dtype=np.float32)

        for i in range(num_classes):
            tag_i = tags.get(i)
            if tag_i in ("B", "S") or i == bg_token:
                start[i] = 0.0
            if tag_i in ("E", "S") or i == bg_token:
                end[i] = 0.0

        for i in range(num_classes):
            tag_i = tags.get(i)
            i_is_bg = spans.get(i) == bg_span
            for j in range(num_classes):
                tag_j = tags.get(j)
                span_j = spans.get(j)
                j_is_bg = span_j == bg_span

                if i_is_bg:
                    if j_is_bg:
                        transition[i, j] = biases["transition_bias_background_stay"]
                    elif tag_j in ("B", "S"):
                        transition[i, j] = biases["transition_bias_background_to_start"]
                elif tag_i in ("B", "I"):
                    same_span = spans.get(i) == span_j
                    if same_span and tag_j == "I":
                        transition[i, j] = biases["transition_bias_inside_to_continue"]
                    elif same_span and tag_j == "E":
                        transition[i, j] = biases["transition_bias_inside_to_end"]
                elif tag_i in ("E", "S"):
                    if j_is_bg:
                        transition[i, j] = biases["transition_bias_end_to_background"]
                    elif tag_j in ("B", "S"):
                        transition[i, j] = biases["transition_bias_end_to_start"]

        self._start = start
        self._end = end
        self._transition = transition

    def decode(self, log_probs: np.ndarray) -> List[int]:
        """Decode one `[seq_len, num_classes]` log-probability array into the
        highest-scoring BIOES-valid label-id path."""
        import numpy as np

        log_probs = np.asarray(log_probs, dtype=np.float32)
        seq_len = log_probs.shape[0]
        if seq_len == 0:
            return []

        scores = log_probs[0] + self._start
        num_classes = scores.shape[0]
        backpointers = np.empty((seq_len - 1, num_classes), dtype=np.int64)

        for t in range(1, seq_len):
            # transitions[prev, next] = best-so-far score of ending at `prev`
            # plus the cost of moving prev -> next.
            transitions = scores[:, None] + self._transition
            best_prev = np.argmax(transitions, axis=0)
            best_scores = transitions[best_prev, np.arange(num_classes)]
            scores = best_scores + log_probs[t]
            backpointers[t - 1] = best_prev

        if not np.isfinite(scores).any():
            # No legal path scored above -inf (a pathological/degenerate
            # input) — fail open to argmax rather than return garbage.
            return np.argmax(log_probs, axis=-1).tolist()

        scores = scores + self._end
        path = np.empty(seq_len, dtype=np.int64)
        path[-1] = int(np.argmax(scores))
        for t in range(seq_len - 2, -1, -1):
            path[t] = backpointers[t, path[t + 1]]
        return path.tolist()


def build_decoder_for_label_space(
    class_names: Sequence[str], artifact_dir: str
) -> Optional[ViterbiDecoder]:
    """Build a `ViterbiDecoder` for `class_names` if it's a genuine BIOES
    label space, loading calibration biases from `artifact_dir` if the model
    shipped one. Returns `None` (not a raised error) for a plain BIO label
    space — the caller's signal to keep using argmax."""
    try:
        label_info = build_label_info(class_names)
    except ValueError as exc:
        logger.info("not a BIOES label space (%s); decoding with argmax", exc)
        return None
    biases = load_calibration(artifact_dir)
    return ViterbiDecoder(label_info, biases)
