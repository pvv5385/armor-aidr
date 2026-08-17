"""ONNX-backed text runner — the shared base for classifier, NER, NLI,
and embedding runners.

Heavy deps (`onnxruntime`, `tokenizers`, `numpy`) are imported inside
`load()`, never at module scope, so the registry can decide a task is
unavailable without the import cost or the crash. This is the contract
that lets the whole service boot with no ML stack.

Design constraints:
- **INT8 preferred**: prefers `model_quantized.onnx` over `model.onnx`.
  DeBERTa-v3 is the exception — `_should_skip_quantization` in `fetch.py`
  keeps fp32, so this runner serves whichever graph it finds.
- **256-token cap**: sequences longer than `max_length` are chunked into
  overlapping windows; the runner aggregates (max-pool over chunks for
  classification, mean-pool for embeddings, first-chunk for NER). This is
  the hard cap the gate measures, not a suggestion.
- **Dynamic batching**: the batcher upstream already groups concurrent
  calls; this runner processes the whole batch in one `session.run()`.
- **dtype from graph**: each input tensor's type is read from the ONNX
  graph metadata; the caller never hardcodes float32 vs int64.
- **Device auto-detected, not hardcoded**: `load()` asks `_device.py` for
  the best execution provider this `onnxruntime` build and
  `ARMOR_INFERENCE_DEVICE` agree on (CUDA/TensorRT/ROCm if present, CPU
  otherwise) and falls back to CPU if that provider fails to initialize —
  see inference/README.md's "GPU acceleration" section.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from armor_inference.runners._device import provider_to_device, select_providers
from armor_inference.runners.base import Runner, RunnerUnavailable

logger = logging.getLogger(__name__)

_DEFAULT_MAX_LENGTH = 256
_DEFAULT_OVERLAP = 64


def _find_onnx(dirpath: str) -> str:
    """Pick the best ONNX graph in `dirpath`. Quantized first, then fp32."""
    d = Path(dirpath)
    for name in ("model_quantized.onnx", "model.onnx"):
        p = d / name
        if p.is_file():
            return str(p)
    onnx_files = sorted(d.glob("*.onnx"))
    if onnx_files:
        return str(onnx_files[0])
    raise RunnerUnavailable(f"no .onnx file found in {dirpath}")


def _find_tokenizer(dirpath: str) -> str:
    """Locate `tokenizer.json` — the `tokenizers` Rust crate's native format."""
    p = Path(dirpath) / "tokenizer.json"
    if p.is_file():
        return str(p)
    raise RunnerUnavailable(f"tokenizer.json not found in {dirpath}")


class OnnxTextRunner(Runner):
    """ONNX-backed base for text classification, NER, NLI, and embedding
    runners. Subclasses implement `_postprocess()` to turn raw logits into
    `InferOutput`.

    Attributes set by subclasses before calling `super().load()`:
        task, model_version, runner_kind, max_length, overlap,
        _label_names, _unsafe_pattern, _threshold
    """

    max_length: int = _DEFAULT_MAX_LENGTH
    overlap: int = _DEFAULT_OVERLAP
    _label_names: List[str] = []
    _unsafe_pattern: Optional[re.Pattern] = None
    _threshold: float = 0.5

    def __init__(self, task: str, spec: Dict[str, Any]):
        super().__init__()
        self.task = task
        self._spec = spec
        self._threshold = float(spec.get("threshold", 0.5))
        self._session = None
        self._tokenizer = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_types: Dict[str, type] = {}

    def load(self) -> None:
        """Lazy-import heavy deps, resolve artifacts, load ONNX graph +
        tokenizer. Raises `RunnerUnavailable` on any failure."""
        try:
            import numpy as np  # noqa: F401
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RunnerUnavailable(
                f"onnxruntime/tokenizers/numpy not installed: {exc}. "
                f"Install the [onnx] extra: pip install './inference[onnx]'"
            ) from exc

        from armor_inference.runners._artifacts import (
            resolve_artifact_dir,
            verify_pinned,
        )

        artifact_dir = resolve_artifact_dir(self._spec)
        verify_pinned(artifact_dir, self._spec.get("sha256"))

        onnx_path = _find_onnx(artifact_dir)
        tok_path = _find_tokenizer(artifact_dir)

        logger.info("task '%s': loading %s", self.task, onnx_path)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # "auto" by default: whichever accelerator this onnxruntime build was
        # installed with (onnxruntime-gpu -> CUDA/TensorRT, a ROCm build ->
        # ROCm) wins, CPU otherwise. A task-level `device` in the catalog
        # spec overrides the service-wide ARMOR_INFERENCE_DEVICE, so one
        # sidecar can pin a single heavy model to GPU without moving every
        # other (already CPU-fast) task onto it too.
        requested_device = self._spec.get("device") or os.getenv(
            "ARMOR_INFERENCE_DEVICE", "auto"
        )
        providers = select_providers(requested_device, ort.get_available_providers())
        try:
            self._session = ort.InferenceSession(onnx_path, opts, providers=providers)
        except Exception as exc:  # noqa: BLE001 — provider init failure, not our bug
            if providers == ["CPUExecutionProvider"]:
                raise
            logger.warning(
                "task '%s': failed to initialize %s (%s); falling back to CPU",
                self.task,
                providers,
                exc,
            )
            self._session = ort.InferenceSession(
                onnx_path, opts, providers=["CPUExecutionProvider"]
            )
        self.device = provider_to_device(self._session.get_providers()[0])

        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._input_types = {}
        for inp in self._session.get_inputs():
            self._input_types[inp.name] = _ort_type_to_python(inp.type)

        self._tokenizer = Tokenizer.from_file(tok_path)
        # No truncation/padding: `_tokenize_batch` needs each text's raw,
        # un-truncated token count to decide whether it needs chunking.
        # `enable_truncation` would silently cap every `encode()` at
        # `max_length`, making `len(ids) <= self.max_length` always true and
        # the chunking branch below unreachable — long inputs would be
        # silently truncated instead of chunked. Padding is applied manually
        # per-chunk in `_tokenize_batch` instead. Explicitly disabled (not
        # just "never enabled") since `tokenizer.json` can bake in its own
        # truncation/padding defaults that `from_file` would otherwise honor.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

        model_version = self._spec.get("model_version")
        if not model_version:
            model_id = self._spec.get("model_id", "unknown")
            revision = self._spec.get("revision", "main")
            model_version = f"{model_id}@{revision}"
        self.model_version = model_version
        logger.info(
            "task '%s': ready (%s) on %s", self.task, self.model_version, self.device
        )

    def _pad_to_max_length(
        self, ids: List[int], mask: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Right-pads one (already truncation-free, <= max_length) sequence
        so every row handed to `_cast_array` has the same length — the
        tokenizer no longer does this itself (see `load()`)."""
        pad_len = self.max_length - len(ids)
        if pad_len <= 0:
            return ids, mask
        pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        return ids + [pad_id] * pad_len, mask + [0] * pad_len

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Tokenize a batch of texts into model inputs, chunking long texts."""
        all_input_ids: List[List[int]] = []
        all_attention_mask: List[List[int]] = []
        chunk_map: List[Tuple[int, int]] = []  # (original_index, chunk_index)

        for i, text in enumerate(texts):
            tokens = self._tokenizer.encode(text)
            ids = tokens.ids
            mask = tokens.attention_mask

            if len(ids) <= self.max_length:
                pad_ids, pad_mask = self._pad_to_max_length(ids, mask)
                all_input_ids.append(pad_ids)
                all_attention_mask.append(pad_mask)
                chunk_map.append((i, 0))
            else:
                # Chunk with overlap
                step = self.max_length - self.overlap
                chunks = []
                for start in range(0, len(ids), step):
                    chunk_ids, chunk_mask = self._pad_to_max_length(
                        ids[start : start + self.max_length],
                        mask[start : start + self.max_length],
                    )
                    chunks.append((chunk_ids, chunk_mask))
                    chunk_map.append((i, len(chunks) - 1))
                all_input_ids.extend(c[0] for c in chunks)
                all_attention_mask.extend(c[1] for c in chunks)

        batch_input_ids = all_input_ids
        batch_attention_mask = all_attention_mask

        inputs: Dict[str, Any] = {}
        for name in self._input_names:
            lower = name.lower()
            if "token_type" in lower:
                # Single-segment classification/NER: every position belongs
                # to segment 0. Must precede the "token" check below, which
                # would otherwise also match this name and feed it real
                # vocabulary ids — those exceed the embedding table's size-2
                # range and ONNX Runtime's Gather rejects them.
                zeros = [[0] * len(row) for row in batch_input_ids]
                inputs[name] = _cast_array(zeros, self._input_types.get(name, int))
            elif "input_ids" in lower or "token" in lower:
                inputs[name] = _cast_array(batch_input_ids, self._input_types.get(name, int))
            elif "attention" in lower or "mask" in lower:
                inputs[name] = _cast_array(batch_attention_mask, self._input_types.get(name, int))
            else:
                # Fallback: try attention mask shape
                inputs[name] = _cast_array(batch_attention_mask, self._input_types.get(name, int))

        return inputs, chunk_map, len(texts)

    def infer_batch(
        self, texts: List[str], params: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        if self._session is None or self._tokenizer is None:
            raise RunnerUnavailable(f"task '{self.task}' not loaded")

        inputs, chunk_map, n_original = self._tokenize_batch(texts)
        raw_outputs = self._session.run(self._output_names, inputs)

        # raw_outputs is a list of numpy arrays, one per output name
        logits = raw_outputs[0]  # shape: (batch, num_labels) or similar

        # Aggregate chunks back to original texts. `infer_batch` must always
        # return one output per input text (the batcher's whole contract,
        # `batching.py`'s `len(outputs) != len(texts)` check) — a bare
        # `InferOutput` here instead of a 1-element list fails that `len()`
        # check with a TypeError raised *outside* the batcher's try/except,
        # which kills its `_run` worker task permanently.
        if len(texts) == 1 and logits.shape[0] == 1:
            return [self._postprocess_single(logits[0], params)]

        # Multi-chunk aggregation: group by original index
        chunk_outputs: Dict[int, List] = {}
        for idx, (orig_idx, _chunk_idx) in enumerate(chunk_map):
            chunk_outputs.setdefault(orig_idx, []).append(logits[idx])

        results = []
        for i in range(n_original):
            chunks = chunk_outputs.get(i, [logits[i]])
            results.append(self._aggregate_chunks(chunks, params))
        return results

    def _aggregate_chunks(self, chunks: List, params: Optional[Dict[str, Any]] = None):
        """Override per runner kind. Default: max-pool over chunks."""
        import numpy as np

        stacked = np.array(chunks)
        maxed = stacked.max(axis=0)
        return self._postprocess_single(maxed, params)

    def _postprocess_single(self, logits, params: Optional[Dict[str, Any]] = None):
        """Override per runner kind. Turn raw logits into InferOutput."""
        raise NotImplementedError


def _ort_type_to_python(type_str: str) -> type:
    """Map ONNX type strings to Python/numpy types."""
    t = type_str.lower()
    if "int64" in t:
        return int
    if "int32" in t:
        return int
    if "float" in t:
        return float
    return int  # default to int (for token ids)


def _cast_array(data: List, target_type: type):
    """Cast a list-of-lists to a numpy array with the right dtype."""
    import numpy as np

    if target_type is float:
        return np.array(data, dtype=np.float32)
    return np.array(data, dtype=np.int64)
