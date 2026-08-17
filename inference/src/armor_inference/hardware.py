"""Best-effort host hardware inventory: CPU, RAM, GPU.

Standard library only, plus whatever ML stack this build already has
(`onnxruntime`, if the `onnx`/`cuda` extra is installed) — matching the base
image's zero-new-dependency ethos (see `main.py`'s module docstring). GPU
detection shells out to `nvidia-smi` when present; a host with no GPU (or no
`nvidia-smi` on PATH) just reports an empty `gpus` list rather than failing
the request.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import List, Optional, Tuple

from pydantic import BaseModel


class CpuInfo(BaseModel):
    model: Optional[str] = None
    architecture: str
    logical_cores: Optional[int] = None
    physical_cores: Optional[int] = None


class MemoryInfo(BaseModel):
    total_bytes: Optional[int] = None


class GpuInfo(BaseModel):
    name: str
    memory_total_mb: Optional[int] = None
    driver_version: Optional[str] = None


class HardwareInfo(BaseModel):
    cpu: CpuInfo
    memory: MemoryInfo
    gpus: List[GpuInfo]
    os: str
    python_version: str
    onnxruntime_version: Optional[str] = None
    onnxruntime_providers: List[str] = []


def _linux_cpuinfo() -> Tuple[Optional[str], Optional[int]]:
    """`(model name, physical core count)` from `/proc/cpuinfo`, best-effort.

    Physical cores is the count of distinct (physical id, core id) pairs
    across logical processors — plain `os.cpu_count()` counts hyperthreads
    too, which overstates cores on any SMT-enabled host.
    """
    try:
        text = open("/proc/cpuinfo").read()
    except OSError:
        return None, None

    model: Optional[str] = None
    pairs = set()
    phys_id = core_id = None
    for line in text.splitlines():
        if not line.strip():
            if phys_id is not None and core_id is not None:
                pairs.add((phys_id, core_id))
            phys_id = core_id = None
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "model name" and model is None:
            model = value
        elif key == "physical id":
            phys_id = value
        elif key == "core id":
            core_id = value
    if phys_id is not None and core_id is not None:
        pairs.add((phys_id, core_id))
    return model, (len(pairs) or None)


def _linux_mem_total_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _nvidia_gpus() -> List[GpuInfo]:
    """Queries `nvidia-smi` if it's on PATH. Absent binary, no GPU, or a
    timed-out/failed call all just mean "no GPUs to report" — this must
    never turn a CPU-only host into a 500."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            memory_mb: Optional[int] = int(float(parts[1]))
        except ValueError:
            memory_mb = None
        gpus.append(
            GpuInfo(
                name=parts[0],
                memory_total_mb=memory_mb,
                driver_version=parts[2] if len(parts) > 2 and parts[2] else None,
            )
        )
    return gpus


def _onnxruntime_info() -> Tuple[Optional[str], List[str]]:
    try:
        import onnxruntime as ort
    except ImportError:
        return None, []
    return ort.__version__, list(ort.get_available_providers())


def get_hardware_info() -> HardwareInfo:
    model, physical_cores = (None, None)
    if platform.system() == "Linux":
        model, physical_cores = _linux_cpuinfo()
    if model is None:
        model = platform.processor() or None

    mem_total = _linux_mem_total_bytes() if platform.system() == "Linux" else None
    ort_version, ort_providers = _onnxruntime_info()

    return HardwareInfo(
        cpu=CpuInfo(
            model=model,
            architecture=platform.machine(),
            logical_cores=os.cpu_count(),
            physical_cores=physical_cores,
        ),
        memory=MemoryInfo(total_bytes=mem_total),
        gpus=_nvidia_gpus(),
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        onnxruntime_version=ort_version,
        onnxruntime_providers=ort_providers,
    )
