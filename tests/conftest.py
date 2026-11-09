"""CPU-only development tests on the installed CUDA-enabled Torch 2.6 build."""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def masked_accelerator_compatibility():
    # Torch 2.6+cu124 raises instead of returning None when all CUDA devices are
    # explicitly masked; Transformers 5.19 expects None. This test-process-only
    # shim matches that CPU-only contract and never queries/initializes CUDA.
    patch = pytest.MonkeyPatch()
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        import torch
        if torch.__version__.startswith("2.6.") and "+cu" in torch.__version__:
            patch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    yield
    patch.undo()
