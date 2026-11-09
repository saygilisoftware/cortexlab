"""Conservative, explicit resource policy. Checking CPU policy never touches CUDA."""
import os

import psutil

GIB = 1024 ** 3
MIN_CPU_AVAILABLE = 10 * GIB


class MemoryBudgetError(RuntimeError):
    pass


def resource_policy():
    device = os.environ.get("CORTEXLAB_DEVICE", "cpu").lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("CORTEXLAB_DEVICE must be cpu or cuda; automatic CUDA loading is disabled")
    available = psutil.virtual_memory().available
    return {
        "model_device": device, "automatic_cuda": False,
        "available_ram_bytes": available, "minimum_cpu_available_bytes": MIN_CPU_AVAILABLE,
        "inference_allowed": device == "cuda" or available >= MIN_CPU_AVAILABLE,
        "renderer": {"profile": "eco", "max_fps": 15, "pixel_ratio": 1, "default_connections": 50000, "render_on_demand": True},
    }


def ensure_cpu_capacity():
    policy = resource_policy()
    if policy["model_device"] == "cpu" and not policy["inference_allowed"]:
        raise MemoryBudgetError(
            f"CPU Qwen loading is blocked by the memory budget: at least {MIN_CPU_AVAILABLE / GIB:.0f} GiB available RAM required; "
            f"currently {policy['available_ram_bytes'] / GIB:.1f} GiB. Recorded replay and CPU weight inspection remain available. "
            "No CUDA fallback will be attempted."
        )
    return policy
