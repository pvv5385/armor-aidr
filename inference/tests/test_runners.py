"""Unit tests for the runner implementations.

These tests exercise the pure-Python logic (softmax, cosine similarity,
entity counting, label picking, factory construction) without requiring
onnxruntime or tokenizers — the same no-ML-deps guarantee the rest of the
suite enforces.
"""

from __future__ import annotations

import pytest

# ── classifier helpers ─────────────────────────────────────────────────


class TestClassifierSoftmax:
    def test_softmax_basic(self):
        from armor_inference.runners.classifier import _softmax

        probs = _softmax([0.0, 0.0, 0.0])
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(abs(p - 1 / 3) < 1e-6 for p in probs)

    def test_softmax_large_values(self):
        from armor_inference.runners.classifier import _softmax

        probs = _softmax([1000.0, 1001.0, 1002.0])
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6
        # The largest value should have the highest probability
        assert probs[2] > probs[1] > probs[0]

    def test_softmax_negative_values(self):
        from armor_inference.runners.classifier import _softmax

        probs = _softmax([-10.0, -5.0, 0.0])
        assert abs(sum(probs) - 1.0) < 1e-6
        assert probs[2] > probs[1] > probs[0]


class TestClassifierLabelPicking:
    def test_pick_unsafe_idx_matches_pattern(self):
        from armor_inference.runners.classifier import _pick_unsafe_idx

        labels = ["LABEL_0", "LABEL_1"]
        idx = _pick_unsafe_idx(labels, None)
        assert idx is None  # no pattern

    def test_pick_unsafe_idx_with_pattern(self):
        import re

        from armor_inference.runners.classifier import _pick_unsafe_idx

        labels = ["LABEL_0", "LABEL_1"]
        pattern = re.compile(r"^LABEL_1$", re.I)
        idx = _pick_unsafe_idx(labels, pattern)
        assert idx == 1

    def test_pick_unsafe_idx_toxic_label(self):
        import re

        from armor_inference.runners.classifier import _pick_unsafe_idx

        labels = ["safe", "toxic"]
        pattern = re.compile(r"^(toxic|unsafe|harmful|positive|LABEL_1)$", re.I)
        idx = _pick_unsafe_idx(labels, pattern)
        assert idx == 1

    def test_pick_unsafe_idx_no_match_returns_none(self):
        import re

        from armor_inference.runners.classifier import _pick_unsafe_idx

        labels = ["neutral", "positive"]
        pattern = re.compile(r"^(toxic|unsafe|harmful|LABEL_1)$", re.I)
        idx = _pick_unsafe_idx(labels, pattern)
        assert idx is None


class TestClassifierSigmoid:
    def test_sigmoid_zero_is_half(self):
        from armor_inference.runners.base import _sigmoid

        probs = _sigmoid([0.0, 0.0, 0.0])
        assert all(abs(p - 0.5) < 1e-6 for p in probs)

    def test_sigmoid_does_not_normalize_across_labels(self):
        # Unlike softmax, independent labels do not have to sum to 1 —
        # every confidently-negative logit stays near 0, none of them get
        # inflated by competing against the others for shared probability
        # mass.
        from armor_inference.runners.base import _sigmoid

        probs = _sigmoid([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
        assert sum(probs) < 0.05
        assert all(p < 0.01 for p in probs)

    def test_sigmoid_confident_positive_logit(self):
        from armor_inference.runners.base import _sigmoid

        probs = _sigmoid([10.0])
        assert probs[0] > 0.999


class TestClassifierPostprocessMultiLabel:
    """Regression guard for the toxic-bert false-positive bug: `unitary/
    toxic-bert` is `problem_type: "multi_label_classification"` (six
    independent labels — a text can be `toxic` and `insult` and `obscene`
    at once), but `_postprocess_single` ran every classifier through one
    shared softmax regardless of `problem_type`. Six confidently-negative
    logits (every label genuinely absent) still sum to 1 after softmax, so
    whichever logit is *least* negative gets inflated into a false
    "unsafe" reading — reproduced here with `unitary/toxic-bert`'s actual
    real-world logits for the benign text "how much is 4+2 * 2?", which
    softmax turned into a 77% `toxic` score and sigmoid correctly reads as
    <1%.
    """

    _LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # unitary/toxic-bert's real ONNX output for "how much is 4+2 * 2?"
    _BENIGN_LOGITS = [-5.821287, -9.281568, -8.032668, -9.292836, -8.413818, -8.835085]

    def _runner(self, multi_label: bool):
        from armor_inference.runners.classifier import (
            _UNSAFE_PATTERNS,
            ClassifierRunner,
            _pick_unsafe_idx,
        )

        runner = ClassifierRunner("toxicity", {"threshold": 0.5})
        runner._label_names = list(self._LABELS)
        runner._unsafe_idx = _pick_unsafe_idx(self._LABELS, _UNSAFE_PATTERNS["toxicity"])
        runner._multi_label = multi_label
        return runner

    def test_softmax_on_multi_label_logits_falsely_inflates_a_benign_text(self):
        # Documents the bug as it behaved before the fix: forcing these six
        # confidently-negative logits through one softmax still yields a
        # dominant "toxic" reading purely from cross-label competition.
        out = self._runner(multi_label=False)._postprocess_single(self._BENIGN_LOGITS)
        assert out.risk_score > 70

    def test_sigmoid_on_multi_label_logits_correctly_reads_as_safe(self):
        out = self._runner(multi_label=True)._postprocess_single(self._BENIGN_LOGITS)
        assert out.risk_score < 5
        assert out.label_scores["toxic"] < 0.05

    def test_sigmoid_label_scores_do_not_sum_to_one(self):
        # The defining difference from softmax: independent per-label
        # probabilities have no reason to normalize to a shared total.
        out = self._runner(multi_label=True)._postprocess_single(self._BENIGN_LOGITS)
        assert sum(out.label_scores.values()) < 0.05


class TestClassifierPostprocessDecisionBoundary:
    """Regression guard: the WARN/ALLOW boundary used to compare against
    exactly 0.0 (`elif unsafe_prob > 0.0`), but a softmax/sigmoid output is
    practically never exact zero in floating point — even a confidently
    safe classification leaves some infinitesimal tail probability on the
    "unsafe" label, which is `> 0.0` and forced WARN regardless of how safe
    the input actually was. `ner.py`/`nli.py` already use a real floor
    (0.3) for exactly this reason; `classifier.py` was the one runner
    missing it, which meant every classifier-backed check (prompt_injection,
    toxicity, over_refusal) could return WARN but effectively never ALLOW.
    """

    def _runner(self, threshold: float = 0.5):
        from armor_inference.runners.classifier import ClassifierRunner

        runner = ClassifierRunner("prompt_injection", {"threshold": threshold})
        runner._label_names = ["SAFE", "INJECTION"]
        runner._unsafe_idx = 1
        runner._multi_label = False
        return runner

    def test_a_confidently_safe_text_allows_despite_a_nonzero_softmax_tail(self):
        # Softmax of logits this far apart leaves an unsafe probability on
        # the order of 1e-18 — representable, nonzero, and exactly the kind
        # of value real inference on genuinely safe text produces.
        out = self._runner()._postprocess_single([20.0, -20.0])
        assert out.decision == "ALLOW"

    def test_a_genuinely_borderline_score_still_warns(self):
        out = self._runner()._postprocess_single([0.5, 0.3])
        assert out.decision == "WARN"


class TestClassifierIsMultiLabel:
    def test_defaults_to_single_label_without_config(self):
        from armor_inference.runners.classifier import _load_is_multi_label

        assert _load_is_multi_label({"model_id": "org/nonexistent-model"}) is False


# ── NER helpers ────────────────────────────────────────────────────────


class TestNerEntityCounting:
    def test_count_entities_bio(self):
        from armor_inference.runners.ner import _count_entities

        labels = ["O", "B-PER", "I-PER", "O", "B-LOC", "O"]
        assert _count_entities(labels) == 2

    def test_count_entities_no_entities(self):
        from armor_inference.runners.ner import _count_entities

        labels = ["O", "O", "O", "O"]
        assert _count_entities(labels) == 0

    def test_count_entities_all_b_tags(self):
        from armor_inference.runners.ner import _count_entities

        labels = ["B-PER", "B-ORG", "B-LOC"]
        assert _count_entities(labels) == 3

    def test_count_entities_consecutive(self):
        from armor_inference.runners.ner import _count_entities

        labels = ["B-PER", "I-PER", "I-PER", "B-PER"]
        assert _count_entities(labels) == 2

    def test_count_entities_bioes_single_token(self):
        # A BIOES model tags a one-token entity S- with no accompanying B- —
        # it must still be counted, or it's invisible to the detector.
        from armor_inference.runners.ner import _count_entities

        labels = ["O", "S-EMAIL", "O"]
        assert _count_entities(labels) == 1

    def test_count_entities_bioes_multi_token_span(self):
        from armor_inference.runners.ner import _count_entities

        labels = ["O", "B-PER", "I-PER", "E-PER", "O"]
        assert _count_entities(labels) == 1

    def test_count_entities_bioes_mixed(self):
        from armor_inference.runners.ner import _count_entities

        labels = ["S-EMAIL", "O", "B-ADDR", "I-ADDR", "E-ADDR", "O", "S-PHONE"]
        assert _count_entities(labels) == 3


class TestNerPostprocess:
    """Regression guard for the e7c51a2 softmax-semantics bug: switching
    `_softmax` to correct per-row normalization (each token's logits sum to
    1 over its own label set) means an ordinary, obviously-non-entity token's
    near-certain "O" prediction is itself close to 1.0. Scoring risk off the
    max probability anywhere in the whole (seq_len x num_labels) matrix — as
    opposed to specifically among entity (`B-`/`I-`) columns — means that
    non-signal alone pushes risk to ~100 on every input, entity or not.
    """

    _ID2LABEL = {0: "O", 1: "B-PER", 2: "I-PER"}

    def _runner(self):
        from armor_inference.runners.ner import NerRunner

        runner = NerRunner("pii_ner", {"threshold": 0.5})
        runner._id2label = dict(self._ID2LABEL)
        return runner

    def test_a_confident_non_entity_token_does_not_score_as_risky(self):
        pytest.importorskip("numpy")
        # Every token confidently predicted "O" — no entity anywhere, but a
        # matrix-wide max would still see ~1.0 from the dominant "O" logit.
        logits = [[10.0, -5.0, -5.0]] * 6
        out = self._runner()._postprocess_single(logits)
        assert out.decision == "ALLOW"
        assert out.risk_score < 30

    def test_a_confident_entity_token_scores_as_risky(self):
        pytest.importorskip("numpy")
        logits = [[10.0, -5.0, -5.0], [-5.0, 10.0, -5.0], [10.0, -5.0, -5.0]]
        out = self._runner()._postprocess_single(logits)
        assert out.decision == "BLOCK"
        assert out.risk_score > 90

    def test_a_borderline_entity_signal_warns_without_winning_the_argmax(self):
        pytest.importorskip("numpy")
        # The entity label never wins the per-token argmax (so no B-/I- tag
        # is ever assigned and n_entities stays 0), but it carries enough
        # probability mass to be worth a soft flag rather than a silent ALLOW.
        logits = [[1.0, 0.6, -5.0]] * 4
        out = self._runner()._postprocess_single(logits)
        assert out.decision == "WARN"


class TestNerPostprocessBioes:
    """Regression guard for the BIOES undercounting bug: `entity_col_ids`
    and `_count_entities` used to hardcode a `("B-", "I-")` check, so a
    BIOES model's E-/S- tags — S- in particular, the *only* tag a
    single-token entity like a lone phone number or email ever gets — were
    invisible to both risk scoring and the BLOCK/WARN/ALLOW decision.
    """

    _ID2LABEL = {0: "O", 1: "S-PHONE", 2: "B-ADDR", 3: "I-ADDR", 4: "E-ADDR"}

    def _runner(self):
        from armor_inference.runners.ner import NerRunner

        runner = NerRunner("pii_ner", {"threshold": 0.5})
        runner._id2label = dict(self._ID2LABEL)
        return runner

    def test_a_confident_single_token_entity_scores_as_risky_and_blocks(self):
        pytest.importorskip("numpy")
        # One token confidently tagged S-PHONE, surrounded by "O" — before
        # the fix, S- was excluded from entity_col_ids (risk_score stayed
        # near 0) and from _count_entities (decision stayed ALLOW).
        logits = [
            [10.0, -5.0, -5.0, -5.0, -5.0],
            [-5.0, 10.0, -5.0, -5.0, -5.0],
            [10.0, -5.0, -5.0, -5.0, -5.0],
        ]
        out = self._runner()._postprocess_single(logits)
        assert out.decision == "BLOCK"
        assert out.risk_score > 90

    def test_a_confident_multi_token_bioes_span_scores_as_risky_and_blocks(self):
        pytest.importorskip("numpy")
        # B-ADDR, I-ADDR, E-ADDR closing a multi-token span: one entity, not
        # three, and E- must count toward risk_score same as B-/I- do.
        logits = [
            [-5.0, -5.0, 10.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, 10.0, -5.0],
            [-5.0, -5.0, -5.0, -5.0, 10.0],
        ]
        out = self._runner()._postprocess_single(logits)
        assert out.decision == "BLOCK"
        assert out.risk_score > 90
        assert out.label_scores["entities"] == 1


class TestViterbiDecoder:
    """The decoder ported from openai/privacy-filter's opf/_core/decoding.py
    — see runners/_viterbi.py's module docstring for why plain argmax is
    the wrong decoding method for a genuinely BIOES-trained model."""

    def test_build_label_info_rejects_plain_bio(self):
        # PER/ORG only have B-/I- — not a BIOES label space, and
        # build_label_info's job is to say so rather than silently build a
        # decoder for a boundary scheme the model was never trained on.
        from armor_inference.runners._viterbi import build_label_info

        with pytest.raises(ValueError):
            build_label_info(["O", "B-PER", "I-PER", "B-ORG", "I-ORG"])

    def test_build_label_info_accepts_full_bioes(self):
        from armor_inference.runners._viterbi import build_label_info

        info = build_label_info(["O", "B-ADDR", "I-ADDR", "E-ADDR", "S-ADDR"])
        assert info.background_token_label == 0
        assert info.token_boundary_tags[1] == "B"
        assert info.token_boundary_tags[4] == "S"

    def test_build_decoder_for_label_space_returns_none_for_bio(self):
        # `ner.py`'s load() relies on this None to know it should keep using
        # argmax for a BIO-only model instead of raising at load time.
        from armor_inference.runners._viterbi import build_decoder_for_label_space

        decoder = build_decoder_for_label_space(
            ["O", "B-PER", "I-PER"], artifact_dir="/nonexistent"
        )
        assert decoder is None

    def test_decode_repairs_a_boundary_orphaned_by_independent_argmax(self):
        # Token 0's raw logits favor I-ADDR (9) over B-ADDR (8) — a
        # continuation tag with no legal start of its own. Independent
        # argmax would emit ["I-ADDR", "E-ADDR"]: an orphaned I- that
        # `_count_entities` never counts, so the whole span goes missing
        # despite strong emission signal. The constrained decoder can't
        # reach I-ADDR as a start at all (start score is masked to -inf for
        # any non-B/S/background class), so it's forced onto the
        # next-best-scoring legal path that can still reach token 1's
        # strongly-favored E-ADDR: B-ADDR, which is the only other class
        # that may legally transition into E-ADDR of the same span.
        np = pytest.importorskip("numpy")
        from armor_inference.runners._viterbi import (
            ViterbiDecoder,
            build_label_info,
            zero_biases,
        )

        labels = ["O", "B-ADDR", "I-ADDR", "E-ADDR", "S-ADDR"]
        info = build_label_info(labels)
        decoder = ViterbiDecoder(info, zero_biases())

        log_probs = np.array(
            [
                # O,    B-ADDR, I-ADDR, E-ADDR, S-ADDR
                [-5.0, 8.0, 9.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, 10.0, -5.0],
            ]
        )

        # Naive independent argmax would pick the orphaned I-ADDR start.
        assert np.argmax(log_probs, axis=-1).tolist() == [2, 3]

        # The constrained decoder repairs it to a legal, fully-closed span.
        assert decoder.decode(log_probs) == [1, 3]

    def test_decode_empty_sequence(self):
        np = pytest.importorskip("numpy")
        from armor_inference.runners._viterbi import (
            ViterbiDecoder,
            build_label_info,
            zero_biases,
        )

        info = build_label_info(["O", "B-ADDR", "I-ADDR", "E-ADDR", "S-ADDR"])
        decoder = ViterbiDecoder(info, zero_biases())
        assert decoder.decode(np.empty((0, 5))) == []

    def test_load_calibration_missing_file_returns_zero_biases(self):
        from armor_inference.runners._viterbi import VITERBI_BIAS_KEYS, load_calibration

        biases = load_calibration("/nonexistent/artifact/dir")
        assert biases == {key: 0.0 for key in VITERBI_BIAS_KEYS}

    def test_load_calibration_reads_shipped_file(self, tmp_path):
        import json

        from armor_inference.runners._viterbi import load_calibration

        calibration = {
            "operating_points": {
                "default": {
                    "biases": {
                        "transition_bias_background_stay": 0.0,
                        "transition_bias_background_to_start": 1.5,
                        "transition_bias_inside_to_continue": 0.0,
                        "transition_bias_inside_to_end": 0.0,
                        "transition_bias_end_to_background": 0.0,
                        "transition_bias_end_to_start": -2.0,
                    }
                }
            }
        }
        (tmp_path / "viterbi_calibration.json").write_text(json.dumps(calibration))
        biases = load_calibration(str(tmp_path))
        assert biases["transition_bias_background_to_start"] == 1.5
        assert biases["transition_bias_end_to_start"] == -2.0


class TestNerPostprocessViterbi:
    """`_postprocess_single` end to end with a Viterbi decoder attached —
    same orphaned-boundary scenario as TestViterbiDecoder, but through the
    runner's actual entity-counting/decision path."""

    _ID2LABEL = {0: "O", 1: "B-ADDR", 2: "I-ADDR", 3: "E-ADDR", 4: "S-ADDR"}

    def _runner(self, with_viterbi: bool):
        from armor_inference.runners._viterbi import build_decoder_for_label_space
        from armor_inference.runners.ner import NerRunner

        runner = NerRunner("pii_ner", {"threshold": 0.5})
        runner._id2label = dict(self._ID2LABEL)
        if with_viterbi:
            class_names = [self._ID2LABEL[i] for i in sorted(self._ID2LABEL)]
            runner._viterbi = build_decoder_for_label_space(
                class_names, artifact_dir="/nonexistent"
            )
            assert runner._viterbi is not None
        return runner

    _LOGITS = [
        [-5.0, 8.0, 9.0, -5.0, -5.0],
        [-5.0, -5.0, -5.0, 10.0, -5.0],
    ]

    def test_argmax_alone_misses_the_orphaned_span(self):
        pytest.importorskip("numpy")
        out = self._runner(with_viterbi=False)._postprocess_single(self._LOGITS)
        assert out.label_scores["entities"] == 0

    def test_viterbi_recovers_the_span_argmax_misses(self):
        pytest.importorskip("numpy")
        out = self._runner(with_viterbi=True)._postprocess_single(self._LOGITS)
        assert out.label_scores["entities"] == 1
        assert out.decision == "BLOCK"

    def test_viterbi_still_correct_on_the_existing_bioes_regression_cases(self):
        pytest.importorskip("numpy")
        # Same two scenarios TestNerPostprocessBioes covers, now decoded
        # through Viterbi instead of argmax — the fix must not regress the
        # already-fixed BIOES counting behavior.
        single_token = [
            [10.0, -5.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, -5.0, 10.0],  # S-ADDR
            [10.0, -5.0, -5.0, -5.0, -5.0],
        ]
        out = self._runner(with_viterbi=True)._postprocess_single(single_token)
        assert out.label_scores["entities"] == 1
        assert out.decision == "BLOCK"

        multi_token = [
            [-5.0, 10.0, -5.0, -5.0, -5.0],  # B-ADDR
            [-5.0, -5.0, 10.0, -5.0, -5.0],  # I-ADDR
            [-5.0, -5.0, -5.0, 10.0, -5.0],  # E-ADDR
        ]
        out = self._runner(with_viterbi=True)._postprocess_single(multi_token)
        assert out.label_scores["entities"] == 1
        assert out.decision == "BLOCK"


# ── embedding helpers ─────────────────────────────────────────────────


class TestEmbeddingCosineSimilarity:
    def test_identical_vectors(self):
        from armor_inference.runners.embedding import _cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        from armor_inference.runners.embedding import _cosine_similarity

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        from armor_inference.runners.embedding import _cosine_similarity

        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        from armor_inference.runners.embedding import _cosine_similarity

        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


class TestEmbeddingPostprocessDecisionBoundary:
    """Same regression as TestClassifierPostprocessDecisionBoundary, for the
    topic_intent runner: an unrelated embedding's cosine similarity to a
    label vector is essentially never exactly 0.0 (sentence embeddings are
    anisotropic — unrelated pairs commonly land at 0.05-0.3, not ~0), so the
    old `best_score > 0.0` boundary forced WARN on nearly every comparison.
    """

    def _runner(self, threshold: float = 0.7):
        from armor_inference.runners.embedding import EmbeddingRunner

        runner = EmbeddingRunner("topic_intent", {"threshold": threshold})
        # Pre-seed the label-embedding cache so scoring never calls
        # `_embed` (which needs a loaded ONNX session) — this method's
        # `import numpy` happens unconditionally, so numpy still has to be
        # importable, just not onnxruntime/tokenizers.
        runner._label_embeddings["competitor"] = [1.0, 0.0, 0.0]
        return runner

    def test_an_unrelated_embedding_allows_despite_nonzero_cosine_similarity(self):
        pytest.importorskip("numpy")
        # A small component along the label vector's own axis plus a large
        # orthogonal one — cosine similarity to [1, 0, 0] works out to
        # ~0.05: nonzero, but nowhere near enough to be "about" the label.
        # This is the embedding-space equivalent of the classifier's
        # near-zero-but-nonzero softmax tail.
        embedding = [0.05, 0.05, 0.999]
        out = self._runner()._score_with_embeddings(embedding, {"topic_labels": ["competitor"]})
        assert out.decision == "ALLOW"

    def test_a_genuinely_borderline_similarity_still_warns(self):
        pytest.importorskip("numpy")
        embedding = [0.5, 0.866, 0.0]  # cosine similarity to [1,0,0] is 0.5
        out = self._runner()._score_with_embeddings(embedding, {"topic_labels": ["competitor"]})
        assert out.decision == "WARN"


# ── factory construction (no load) ────────────────────────────────────


class TestFactoryConstruction:
    """Verify that `make_runner` creates the right runner type without
    calling `load()` — the factories are importable without onnxruntime."""

    def test_classifier_factory(self):
        from armor_inference.runners.classifier import ClassifierRunner, make_runner

        runner = make_runner("prompt_injection", {"threshold": 0.5})
        assert isinstance(runner, ClassifierRunner)
        assert runner.task == "prompt_injection"
        assert runner.runner_kind == "classifier"

    def test_ner_factory(self):
        from armor_inference.runners.ner import NerRunner, make_runner

        runner = make_runner("pii_ner", {"threshold": 0.5})
        assert isinstance(runner, NerRunner)
        assert runner.task == "pii_ner"
        assert runner.runner_kind == "ner"

    def test_embedding_factory(self):
        from armor_inference.runners.embedding import EmbeddingRunner, make_runner

        runner = make_runner("topic_intent", {"threshold": 0.5})
        assert isinstance(runner, EmbeddingRunner)
        assert runner.task == "topic_intent"
        assert runner.runner_kind == "embedding"

    def test_nli_factory(self):
        from armor_inference.runners.nli import NliRunner, make_runner

        runner = make_runner("some_nli_task", {"threshold": 0.5})
        assert isinstance(runner, NliRunner)
        assert runner.task == "some_nli_task"
        assert runner.runner_kind == "nli"


# ── registry integration ──────────────────────────────────────────────


class TestRegistryHeavyKinds:
    """Verify that the registry can now resolve the heavy runner kinds.
    They will fail at `load()` (no onnxruntime), but the factory should
    be importable and the registry should mark them unavailable, not
    unknown."""

    def test_classifier_is_recognized(self):
        from armor_inference.config import InferenceConfig
        from armor_inference.registry import RunnerRegistry

        reg = RunnerRegistry(
            InferenceConfig(
                task_specs={
                    "prompt_injection": {
                        "runner": "classifier",
                        "model_id": "org/model",
                        "revision": "main",
                    }
                }
            )
        )
        info = {m.task: m for m in reg.list_models()}["prompt_injection"]
        assert not info.available  # onnxruntime not installed
        assert info.runner == "classifier"
        # The detail should name the missing dependency, not "unknown runner kind"
        assert "unknown runner kind" not in (info.detail or "")
        assert "onnxruntime" in (info.detail or "")

    def test_ner_is_recognized(self):
        from armor_inference.config import InferenceConfig
        from armor_inference.registry import RunnerRegistry

        reg = RunnerRegistry(
            InferenceConfig(
                task_specs={
                    "pii_ner": {
                        "runner": "ner",
                        "model_id": "org/model",
                        "revision": "main",
                    }
                }
            )
        )
        info = {m.task: m for m in reg.list_models()}["pii_ner"]
        assert not info.available
        assert info.runner == "ner"

    def test_embedding_is_recognized(self):
        from armor_inference.config import InferenceConfig
        from armor_inference.registry import RunnerRegistry

        reg = RunnerRegistry(
            InferenceConfig(
                task_specs={
                    "topic_intent": {
                        "runner": "embedding",
                        "model_id": "org/model",
                        "revision": "main",
                    }
                }
            )
        )
        info = {m.task: m for m in reg.list_models()}["topic_intent"]
        assert not info.available
        assert info.runner == "embedding"

    def test_nli_is_recognized(self):
        from armor_inference.config import InferenceConfig
        from armor_inference.registry import RunnerRegistry

        reg = RunnerRegistry(
            InferenceConfig(
                task_specs={
                    "some_task": {
                        "runner": "nli",
                        "model_id": "org/model",
                        "revision": "main",
                    }
                }
            )
        )
        info = {m.task: m for m in reg.list_models()}["some_task"]
        assert not info.available
        assert info.runner == "nli"


# ── onnx helper functions ─────────────────────────────────────────────


class TestOnnxHelpers:
    def test_find_onnx_missing_dir(self):
        from armor_inference.runners._heavy import _find_onnx
        from armor_inference.runners.base import RunnerUnavailable

        with pytest.raises(RunnerUnavailable, match="no .onnx file"):
            _find_onnx("/nonexistent/path")

    def test_find_tokenizer_missing(self):
        from armor_inference.runners._heavy import _find_tokenizer
        from armor_inference.runners.base import RunnerUnavailable

        with pytest.raises(RunnerUnavailable, match="tokenizer.json not found"):
            _find_tokenizer("/nonexistent/path")

    def test_ort_type_to_python(self):
        from armor_inference.runners._heavy import _ort_type_to_python

        assert _ort_type_to_python("tensor(int64)") is int
        assert _ort_type_to_python("tensor(int32)") is int
        assert _ort_type_to_python("tensor(float)") is float
        assert _ort_type_to_python("tensor(float16)") is float
        # Unknown type defaults to int
        assert _ort_type_to_python("tensor(string)") is int


# ── OnnxTextRunner batching/chunking ────────────────────────────────────
#
# Exercises `_tokenize_batch`/`infer_batch`'s pure logic with a fake
# tokenizer/session (no real `tokenizers`/`onnxruntime`) — real ONNX
# artifacts aren't available in this test env, but `numpy` (an `[onnx]`
# extra, not installed by CI's `pip install "./inference[dev]"`) is needed
# for the array plumbing these methods share with the real thing, so these
# skip cleanly rather than failing when it's absent.


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _FakeTokenizer:
    """Deliberately does *not* truncate/pad in `encode()` — matching a real
    `Tokenizer` after `no_truncation()`/`no_padding()` (see `_heavy.py`'s
    `load()`), so these tests exercise the same raw-token-count path
    `_tokenize_batch` relies on to decide whether to chunk."""

    def __init__(self, id_map):
        self._id_map = id_map

    def encode(self, text):
        return _FakeEncoding(self._id_map[text])

    def token_to_id(self, token):
        return 0


def _make_runner(tokenizer, max_length=8, overlap=2):
    from armor_inference.runners._heavy import OnnxTextRunner
    from armor_inference.runners.base import InferOutput

    class _Runner(OnnxTextRunner):
        def _postprocess_single(self, logits, params=None):
            return InferOutput(decision="allow", risk_score=0)

    runner = _Runner(task="test", spec={})
    runner.max_length = max_length
    runner.overlap = overlap
    runner._tokenizer = tokenizer
    runner._input_names = ["input_ids", "attention_mask"]
    runner._input_types = {"input_ids": int, "attention_mask": int}
    return runner


class TestTokenizeBatch:
    def test_short_input_is_padded_to_max_length_not_chunked(self):
        pytest.importorskip("numpy")
        tokenizer = _FakeTokenizer({"hi": [1, 2, 3]})
        runner = _make_runner(tokenizer, max_length=8)

        inputs, chunk_map, n_original = runner._tokenize_batch(["hi"])

        assert n_original == 1
        assert chunk_map == [(0, 0)]
        assert len(inputs["input_ids"][0]) == 8
        assert list(inputs["attention_mask"][0]) == [1, 1, 1, 0, 0, 0, 0, 0]

    def test_input_longer_than_max_length_is_chunked_with_overlap(self):
        # Regression test: `load()` used to call `enable_truncation`, which
        # caps every `Tokenizer.encode()` at `max_length` — making this
        # branch dead code (see `_heavy.py`'s `load()` comment). A tokenizer
        # that returns the true, un-truncated id count (as a real one does
        # post-fix) must still land in the chunking branch, not silently
        # get treated as a single short sequence.
        pytest.importorskip("numpy")
        long_ids = list(range(20))
        tokenizer = _FakeTokenizer({"long": long_ids})
        runner = _make_runner(tokenizer, max_length=8, overlap=2)

        inputs, chunk_map, n_original = runner._tokenize_batch(["long"])

        assert n_original == 1
        # step = max_length - overlap = 6; ceil(20 / 6) = 4 chunks
        assert [c[1] for c in chunk_map] == [0, 1, 2, 3]
        assert len(inputs["input_ids"]) == 4
        assert all(len(row) == 8 for row in inputs["input_ids"])


class TestInferBatchSingleItem:
    def test_single_item_batch_returns_a_list_not_a_bare_output(self):
        # Regression test: a bare (non-list) return here previously made
        # the batcher's `len(outputs) != len(texts)` check
        # (`batching.py::_run`) raise `TypeError` outside its `try/except`,
        # permanently killing the batcher's worker task.
        np = pytest.importorskip("numpy")
        from armor_inference.runners.base import InferOutput

        tokenizer = _FakeTokenizer({"hi": [1, 2, 3]})
        runner = _make_runner(tokenizer, max_length=8)

        class _FakeSession:
            def run(self, output_names, inputs):
                return [np.zeros((1, 2), dtype=np.float32)]

        runner._session = _FakeSession()
        runner._output_names = ["logits"]

        outputs = runner.infer_batch(["hi"])

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert isinstance(outputs[0], InferOutput)
