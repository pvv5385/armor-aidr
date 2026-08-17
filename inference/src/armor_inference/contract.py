"""The wire contract, kept in lockstep with `crates/inference-client/src/contract.rs`
and `proto/inference.proto`.

The three files describe one shape in three languages, and the Rust side is
the one with teeth: it clamps out-of-range values on parse rather than
erroring, because a misbehaving sidecar must degrade a live request rather
than poison it. This side validates strictly instead — a runner that produces
`confidence: 1.2` has a bug, and the sidecar is where that bug should surface,
not two services downstream.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# The decision vocabulary. Deliberately the model's own — `armor-core` speaks
# CheckAction/Verdict, and mapping between them is policy-dependent, so it
# happens in `armor-core::engine::escalation` rather than here. A runner
# picks from this set and cannot invent a fourth verdict.
DECISIONS = ("ALLOW", "WARN", "BLOCK", "REDACT")

# Bounds `text`/`texts` before anything downstream (the content-hash cache
# key, the tokenizer, a batch queue) does any work on it. There is no HTTP
# body-size middleware in front of this service (unlike `armor-api`'s
# `DefaultBodyLimit` — `routes.rs`), so without a limit here a single
# oversized request is fully buffered, hashed and tokenized before any check
# ever runs. `MAX_TEXT_CHARS` is generous relative to any real prompt/response
# a check would score; `MAX_BATCH_ITEMS` bounds how many such items one
# request can queue at once, independent of the runner's own
# `max_batch_size` (`config.py`), which only caps one forward pass, not how
# many pile up waiting for one.
MAX_TEXT_CHARS = 100_000
MAX_BATCH_ITEMS = 256


class InferRequest(BaseModel):
    """One scoring call. Exactly one of `text` / `texts` — a request that sets
    both, or neither, is a 422 rather than a guess about which was meant."""

    text: Optional[str] = Field(default=None, max_length=MAX_TEXT_CHARS)
    texts: Optional[List[str]] = Field(default=None, max_length=MAX_BATCH_ITEMS)
    model_id: Optional[str] = None
    revision: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

    @field_validator("texts")
    @classmethod
    def _texts_items_are_bounded(cls, v):
        if v is not None:
            for i, t in enumerate(v):
                if len(t) > MAX_TEXT_CHARS:
                    raise ValueError(
                        f"'texts[{i}]' exceeds the {MAX_TEXT_CHARS}-character limit"
                    )
        return v

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "InferRequest":
        has_text, has_texts = self.text is not None, self.texts is not None
        if has_text == has_texts:
            raise ValueError("provide exactly one of 'text' or 'texts'")
        if has_text and not (self.text or "").strip():
            raise ValueError("'text' must be non-empty")
        if has_texts and (not self.texts or any(not (t or "").strip() for t in self.texts)):
            raise ValueError("'texts' must be a non-empty list of non-empty strings")
        # A revision without a model_id pins nothing: there is no way to honour
        # it, so silently ignoring it would score the caller against a model it
        # did not ask for. That is the one failure mode the pin exists to
        # prevent, so it is an error instead.
        if self.revision is not None and not self.model_id:
            raise ValueError("'revision' pin requires 'model_id'")
        return self

    def items(self) -> List[str]:
        if self.texts is not None:
            return list(self.texts)
        return [self.text or ""]


class InferResult(BaseModel):
    """One scored item."""

    decision: str
    risk_score: int = Field(ge=0, le=100)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    label_scores: Optional[Dict[str, float]] = None
    # Distinct from `confidence` on purpose: `confidence` is whatever the head
    # emitted, `calibrated_score` is what the scorecard gate measured against
    # a benchmark suite. Collapsing them loses the only signal that says
    # whether a threshold means anything.
    calibrated_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    threshold: Optional[float] = None
    model_version: str

    @field_validator("decision")
    @classmethod
    def _known_decision(cls, v: str) -> str:
        if v not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {v!r}")
        return v

    @field_validator("label_scores")
    @classmethod
    def _label_scores_are_finite(cls, v):
        """`label_scores` is a bag of named metrics, not uniformly a
        probability — unlike `confidence`/`calibrated_score` above, which
        always are. `ner.py` reports an entity *count* under `"entities"`
        (can be > 1), `embedding.py` reports an L2 `"embedding_norm"` (can be
        > 1) and cosine similarities (can be negative). A blanket [0, 1]
        bound here 500s every real NER/embedding response — NER always
        emits `entities`, so it 500s on *every* call, not just an edge case.
        What every one of these must be, regardless of what it measures, is
        a finite number: NaN/inf is always a bug worth surfacing here rather
        than two services downstream.
        """
        if v is not None:
            for key, val in v.items():
                if not math.isfinite(float(val)):
                    raise ValueError(f"label score for '{key}' must be finite, got {val!r}")
        return v


class InferResponse(InferResult):
    """Single-item response: the contract plus what the service measured
    serving it. Both extra fields are `#[serde(default)]`-shaped on the Rust
    side, so adding them here does not break the client."""

    latency_ms: int = 0
    cached: bool = False


class BatchInferResponse(BaseModel):
    results: List[InferResult]
    latency_ms: int = 0
    model_version: str


class ModelInfo(BaseModel):
    """What a task's backing model is and whether it can serve right now."""

    task: str
    model_version: str
    runner: str
    # False when the heavy deps or the pinned artifact are missing, or the
    # artifact's sha256 did not match. The task returns 503; the service stays
    # up.
    available: bool
    model_id: Optional[str] = None
    revision: Optional[str] = None
    sha256: Optional[str] = None
    detail: Optional[str] = None
    # True for the task's active slot, False for an additionally-loaded variant
    # that a pinned request routes to. Defaults True so an older payload that
    # predates variants reads as the active model.
    active: bool = True
    # "cpu" / "cuda" / "rocm" — the execution provider actually serving this
    # task, not just the one requested (`ARMOR_INFERENCE_DEVICE=cuda` on a
    # host with no matching provider fails to load rather than silently
    # landing here as "cpu"; this field is what shows a *working* GPU auto-
    # detected, or a `device: "cuda"` override falling back after a failed
    # init). None for the stub and for a task that failed to load.
    device: Optional[str] = None
