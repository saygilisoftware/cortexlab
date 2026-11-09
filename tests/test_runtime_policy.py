from types import SimpleNamespace

import pytest

from backend.runtime_policy import GIB, MemoryBudgetError, ensure_cpu_capacity, resource_policy


def test_default_is_cpu_even_when_gpu_exists(monkeypatch):
    monkeypatch.delenv("CORTEXLAB_DEVICE", raising=False)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=12 * GIB))
    assert ensure_cpu_capacity()["model_device"] == "cpu"
    assert resource_policy()["automatic_cuda"] is False
    assert resource_policy()["renderer"]["max_fps"] == 15


def test_low_ram_blocks_large_cpu_model_without_cuda_fallback(monkeypatch):
    monkeypatch.delenv("CORTEXLAB_DEVICE", raising=False)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=4 * GIB))
    with pytest.raises(MemoryBudgetError, match="No CUDA fallback"):
        ensure_cpu_capacity()


def test_auto_device_is_rejected(monkeypatch):
    monkeypatch.setenv("CORTEXLAB_DEVICE", "auto")
    with pytest.raises(ValueError, match="automatic CUDA"):
        resource_policy()
