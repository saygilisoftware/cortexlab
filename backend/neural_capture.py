"""Real dense activation capture and CPU-only partial safetensors weight reads."""
import base64
import json
import math
import struct
import time

import torch

from .qwen_adapter import _model_path, _read_local_artifacts

CONTEXT_LIMIT = 128
MAX_NEW_TOKENS = 16
DEFAULT_WEIGHT_BUDGET = 50_000
MAX_WEIGHT_BUDGET = 4_000_000
MAX_ACTIVATIONS = 229_376
WEIGHT_MAGIC = b"CTXWGT01"


def architecture(model):
    config = model.config
    result = {"layers": config.num_hidden_layers, "mlp_width": config.intermediate_size,
              "residual_width": config.hidden_size, "intermediate_size": config.intermediate_size,
              "d_model": config.hidden_size}
    result["units_per_layer"] = result["mlp_width"] + result["residual_width"]
    result["activation_count"] = result["layers"] * result["units_per_layer"]
    if not 0 < result["activation_count"] <= MAX_ACTIVATIONS:
        raise ValueError("Dense capture exceeds the supported activation budget")
    return result


def prepare_input(inspector, prompt, max_new_tokens):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 400:
        raise ValueError("Prompt must contain 1–400 characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError("Dense generation requires 1–16 tokens")
    text = inspector.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = inspector.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] or ids.shape[1] + max_new_tokens > CONTEXT_LIMIT:
        raise ValueError("Chat-templated input plus generation exceeds the 128-token capture context")
    return ids.to(inspector.device)


def capture_dense_forward(model, ids):
    """Capture only the last input position; no attention tensors or KV cache retained."""
    dimensions = architecture(model)
    if model.training or model.config._attn_implementation != "eager":
        raise ValueError("Dense inspection requires eval mode and eager attention")
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= CONTEXT_LIMIT:
        raise ValueError("Expected one bounded input sequence")
    captures, handles = {}, []
    for index, block in enumerate(model.model.layers):
        def mlp_hook(module, args, index=index):
            value = args[0]
            if value.shape != (1, ids.shape[1], dimensions["mlp_width"]):
                raise RuntimeError("Unexpected SwiGLU product shape")
            captures[(index, "mlp")] = value[0, -1].detach().clone()
        def residual_hook(module, args, result, index=index):
            value = result[0] if isinstance(result, tuple) else result
            if value.shape != (1, ids.shape[1], dimensions["residual_width"]):
                raise RuntimeError("Unexpected decoder residual shape")
            captures[(index, "residual")] = value[0, -1].detach().clone()
        handles.append(block.mlp.down_proj.register_forward_pre_hook(mlp_hook))
        handles.append(block.register_forward_hook(residual_hook))
    try:
        if ids.device.type == "cuda": torch.cuda.synchronize(ids.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           output_attentions=False, output_hidden_states=False,
                           return_dict=True, logits_to_keep=1)
        if ids.device.type == "cuda": torch.cuda.synchronize(ids.device)
        inference_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        values = torch.cat([captures[(index, kind)] for index in range(dimensions["layers"])
                            for kind in ("mlp", "residual")]).float().cpu().contiguous()
        logits = result.logits[0, -1].detach().float().cpu()
        if values.numel() != dimensions["activation_count"] or not torch.isfinite(values).all() or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid or non-finite dense capture")
        payload = values.numpy().astype("<f4", copy=False).tobytes()
        encoded = base64.b64encode(payload).decode("ascii")
        return {"logits": logits, "activation_b64": encoded, "inference_ms": inference_ms,
                "capture_ms": (time.perf_counter() - started) * 1000}
    finally:
        for handle in handles: handle.remove()


def sample_weight_matrix(weight, layer, budget=DEFAULT_WEIGHT_BUDGET):
    """Generic arithmetic helper; CPU fixture tests and real partial reads share this path."""
    if type(layer) is not int or not 0 <= layer <= 27:
        raise ValueError("Layer must be an integer from 0 to 27")
    if type(budget) is not int or not 1 <= budget <= MAX_WEIGHT_BUDGET:
        raise ValueError("Connection budget must be an integer from 1 to 4,000,000")
    if weight.device.type != "cpu" or weight.ndim != 2:
        raise ValueError("Weight sampling requires a CPU matrix")
    target_count, source_count = weight.shape
    total = target_count * source_count
    if not total: raise ValueError("Empty weight matrix")
    count = min(budget, total)
    stride = (total + count - 1) // count
    while math.gcd(stride, total) != 1: stride += 1
    indices = (torch.arange(count, dtype=torch.int64) * stride) % total
    selected = weight.reshape(-1)[indices].float().contiguous()
    if not torch.isfinite(selected).all(): raise ValueError("Non-finite selected weights")
    header = {"version": 1, "layer": layer, "source_count": source_count, "target_count": target_count,
              "total_connections": total, "count": count, "budget": budget, "stride": stride,
              "dtype": "float32", "sample_policy": "coprime-modular-v1", "shape": [target_count, source_count],
              "weight_abs_max": float(selected.abs().max())}
    header_bytes = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    raw = selected.numpy().astype("<f4", copy=False).tobytes()
    return WEIGHT_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + raw


def read_local_weight_sample(layer, budget=DEFAULT_WEIGHT_BUDGET):
    """Read one ~25MB CPU matrix only, without constructing/loading Qwen or touching CUDA."""
    if type(layer) is not int or not 0 <= layer <= 27:
        raise ValueError("Layer must be an integer from 0 to 27")
    if type(budget) is not int or not 1 <= budget <= MAX_WEIGHT_BUDGET:
        raise ValueError("Invalid connection budget")
    from safetensors import safe_open
    path = _model_path()
    _, artifacts = _read_local_artifacts(path)
    key = f"model.layers.{layer}.mlp.down_proj.weight"
    index = path / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        filename = mapping.get(key)
        if not filename: raise ValueError("Requested down_proj weight is absent from safetensors index")
        candidate = path / filename
        if candidate not in artifacts: raise ValueError("Weight shard is outside the validated local artifact set")
    else:
        candidate = path / "model.safetensors"
    with safe_open(str(candidate), framework="pt", device="cpu") as handle:
        if key not in handle.keys(): raise ValueError("Requested down_proj weight is absent")
        if handle.get_slice(key).get_shape() != [2048, 6144]:
            raise ValueError("Unexpected Qwen3-1.7B down_proj matrix shape")
        weight = handle.get_tensor(key)
        return sample_weight_matrix(weight, layer, budget)
