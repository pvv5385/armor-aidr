"""The contract's validation rules — the ones that exist to stop a caller
being answered by something other than what it asked for."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from armor_inference.contract import (
    MAX_BATCH_ITEMS,
    MAX_TEXT_CHARS,
    InferRequest,
    InferResult,
)


def test_exactly_one_input_form():
    assert InferRequest(text="hello").items() == ["hello"]
    assert InferRequest(texts=["a", "b"]).items() == ["a", "b"]

    with pytest.raises(ValidationError, match="exactly one"):
        InferRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        InferRequest(text="a", texts=["b"])


def test_empty_inputs_are_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        InferRequest(text="   ")
    with pytest.raises(ValidationError, match="non-empty"):
        InferRequest(texts=[])
    with pytest.raises(ValidationError, match="non-empty"):
        InferRequest(texts=["ok", ""])


def test_revision_without_model_id_is_an_error():
    """A revision alone pins nothing. Accepting it would mean scoring the
    caller against a model it did not ask for while it believes otherwise —
    the exact failure the pin exists to prevent."""
    with pytest.raises(ValidationError, match="requires 'model_id'"):
        InferRequest(text="hi", revision="abc123")

    InferRequest(text="hi", model_id="org/model", revision="abc123")  # fine


def test_result_rejects_out_of_range_values():
    InferResult(decision="BLOCK", risk_score=100, model_version="stub@v1")

    with pytest.raises(ValidationError):
        InferResult(decision="BLOCK", risk_score=101, model_version="stub@v1")
    with pytest.raises(ValidationError):
        InferResult(decision="BLOCK", risk_score=50, confidence=1.5, model_version="stub@v1")


def test_label_scores_are_not_forced_into_unit_range():
    """Regression guard: `label_scores` is a bag of named metrics, not
    uniformly a probability. NER's `entities` is a count, embedding's
    `embedding_norm` is an L2 norm, and cosine similarities range over
    [-1, 1] — a blanket [0, 1] bound previously 500'd every real NER/
    embedding response."""
    InferResult(
        decision="BLOCK",
        risk_score=90,
        label_scores={"entities": 3},
        model_version="stub@v1",
    )
    InferResult(
        decision="ALLOW",
        risk_score=0,
        label_scores={"embedding_norm": 15.2, "similarity": -0.5},
        model_version="stub@v1",
    )


def test_label_scores_reject_non_finite_values():
    with pytest.raises(ValidationError, match="finite"):
        InferResult(
            decision="ALLOW",
            risk_score=0,
            label_scores={"broken": math.nan},
            model_version="stub@v1",
        )
    with pytest.raises(ValidationError, match="finite"):
        InferResult(
            decision="ALLOW",
            risk_score=0,
            label_scores={"broken": math.inf},
            model_version="stub@v1",
        )


def test_text_is_bounded():
    InferRequest(text="x" * MAX_TEXT_CHARS)
    with pytest.raises(ValidationError):
        InferRequest(text="x" * (MAX_TEXT_CHARS + 1))


def test_texts_batch_and_items_are_bounded():
    InferRequest(texts=["x"] * MAX_BATCH_ITEMS)
    with pytest.raises(ValidationError):
        InferRequest(texts=["x"] * (MAX_BATCH_ITEMS + 1))
    with pytest.raises(ValidationError, match="exceeds"):
        InferRequest(texts=["x" * (MAX_TEXT_CHARS + 1)])


def test_result_rejects_an_invented_decision():
    """A runner picks from the shared vocabulary or it fails here. Letting an
    unknown verdict through means the Rust client's enum rejects it two hops
    later, where the diagnostic is far worse."""
    with pytest.raises(ValidationError, match="decision must be one of"):
        InferResult(decision="QUARANTINE", risk_score=50, model_version="stub@v1")


def test_calibrated_score_is_not_confidence():
    """The two fields are independent by design: `confidence` is the head's
    output, `calibrated_score` is what a benchmark measured. The scorecard
    gate reads the second one, so nothing may quietly populate it from the
    first."""
    result = InferResult(decision="WARN", risk_score=40, confidence=0.4, model_version="stub@v1")
    assert result.confidence == 0.4
    assert result.calibrated_score is None
