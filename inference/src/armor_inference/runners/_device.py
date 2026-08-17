"""Select ONNX Runtime execution providers for the requested device.

The available providers reflect the installed ONNX Runtime build and the
current host. This module turns a requested device into the provider order
used by `InferenceSession`, while keeping CPU as a fallback.
"""

from __future__ import annotations

from typing import List, Sequence

# Preferred provider order for each device family. TensorRT is tried first
# when available because it is often faster, but CPU remains the fallback
# if GPU initialization fails.

_DEVICE_PROVIDERS = {
    "cuda": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    "rocm": ["ROCMExecutionProvider"],
}

VALID_DEVICES = ("auto", "cpu", "cuda", "rocm")


def select_providers(requested: str, available: Sequence[str]) -> List[str]:
    """Build the provider order for `InferenceSession(providers=...)`.

    `requested` is the requested device setting ("auto", "cpu", "cuda",
    or "rocm"), and `available` is the list returned by
    `onnxruntime.get_available_providers()`.

    Raises `ValueError` when a requested device cannot be satisfied by this
    build, so misconfiguration fails loudly instead of silently falling back
    to CPU.
    """
    requested = (requested or "auto").strip().lower()
    available = list(available)

    if requested == "cpu":
        return ["CPUExecutionProvider"]

    if requested in _DEVICE_PROVIDERS:
        wanted = [p for p in _DEVICE_PROVIDERS[requested] if p in available]
        if not wanted:
            raise ValueError(
                f"ARMOR_INFERENCE_DEVICE={requested!r} but this onnxruntime "
                f"build reports no matching execution provider (available: "
                f"{available}). Install the matching extra (e.g. "
                f"'./inference[cuda]' for onnxruntime-gpu) or set "
                f"ARMOR_INFERENCE_DEVICE=auto to fall back to whatever this "
                f"host actually has."
            )
        return wanted + ["CPUExecutionProvider"]

    if requested != "auto":
        raise ValueError(
            f"ARMOR_INFERENCE_DEVICE={requested!r} is not one of {VALID_DEVICES}"
        )

    # auto: offer every accelerator this build/host combination exposes,
    # CPU always last as the safety net `_heavy.py` relies on when the
    # accelerated EP is compiled in but fails to actually initialize.
    
    ordered: List[str] = []
    for family in ("cuda", "rocm"):
        for provider in _DEVICE_PROVIDERS[family]:
            if provider in available and provider not in ordered:
                ordered.append(provider)
    ordered.append("CPUExecutionProvider")
    return ordered


def provider_to_device(provider: str) -> str:
    """Map an ONNX Runtime execution provider to a short device label."""
    if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        return "cuda"
    if provider == "ROCMExecutionProvider":
        return "rocm"
    if provider == "CPUExecutionProvider":
        return "cpu"
    return provider
