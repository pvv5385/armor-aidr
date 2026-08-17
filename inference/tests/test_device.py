import pytest

from armor_inference.runners._device import provider_to_device, select_providers

CPU_ONLY = ["CPUExecutionProvider"]
CUDA_BUILD = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
ROCM_BUILD = ["ROCMExecutionProvider", "CPUExecutionProvider"]


class TestSelectProviders:
    def test_auto_on_cpu_only_build_stays_cpu(self):
        assert select_providers("auto", CPU_ONLY) == ["CPUExecutionProvider"]

    def test_auto_on_cuda_build_prefers_tensorrt_then_cuda_then_cpu(self):
        assert select_providers("auto", CUDA_BUILD) == [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_auto_on_rocm_build_uses_rocm_then_cpu(self):
        assert select_providers("auto", ROCM_BUILD) == [
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_explicit_cpu_ignores_available_accelerators(self):
        assert select_providers("cpu", CUDA_BUILD) == ["CPUExecutionProvider"]

    def test_explicit_cuda_on_matching_build(self):
        assert select_providers("cuda", CUDA_BUILD) == [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_explicit_cuda_on_cpu_only_build_raises(self):
        with pytest.raises(ValueError, match="cuda"):
            select_providers("cuda", CPU_ONLY)

    def test_explicit_rocm_on_cpu_only_build_raises(self):
        with pytest.raises(ValueError, match="rocm"):
            select_providers("rocm", CPU_ONLY)

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="mlu"):
            select_providers("mlu", CPU_ONLY)

    def test_none_defaults_to_auto(self):
        assert select_providers(None, CPU_ONLY) == ["CPUExecutionProvider"]

    def test_case_and_whitespace_insensitive(self):
        assert select_providers("  CUDA  ", CUDA_BUILD)[0] == "TensorrtExecutionProvider"


class TestProviderToDevice:
    def test_cuda_provider(self):
        assert provider_to_device("CUDAExecutionProvider") == "cuda"

    def test_tensorrt_provider_maps_to_cuda(self):
        assert provider_to_device("TensorrtExecutionProvider") == "cuda"

    def test_rocm_provider(self):
        assert provider_to_device("ROCMExecutionProvider") == "rocm"

    def test_cpu_provider(self):
        assert provider_to_device("CPUExecutionProvider") == "cpu"
