"""Local-only Qwen3-1.7B adapter with bounded, selective real tensor capture.

No model download or fallback model is performed. The v2 record distinguishes
sampled residual channels from full-width post-block L2 norms. Implementation
was checked against installed Transformers 5.19 Qwen3 source: direct decoder
hooks precede the final RMSNorm, unlike the last output_hidden_states value.
"""
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from .runtime_policy import ensure_cpu_capacity

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "cortexlab.session/v2"
MODEL_ID = "Qwen/Qwen3-1.7B"
CONTEXT_LIMIT = 64
EXPECTED_CONFIG = {
    "model_type": "qwen3", "num_hidden_layers": 28,
    "num_attention_heads": 16, "hidden_size": 2048, "vocab_size": 151936,
}


def _model_path():
    return Path(os.environ.get("CORTEXLAB_QWEN_PATH", str(ROOT / "models" / "Qwen3-1.7B"))).expanduser().resolve()


def _read_local_artifacts(path):
    """Cheap completeness checks, not a claim that untrusted local files are valid weights."""
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Local Qwen3-1.7B config.json is missing")
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("Model config exceeds the supported size")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"Expected Qwen3-1.7B {key}={expected}, found {config.get(key)!r}")
    artifacts = [config_path]
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        candidate = path / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Local {filename} is missing")
        artifacts.append(candidate)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        if index.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Safetensors index exceeds the supported size")
        manifest = json.loads(index.read_text(encoding="utf-8"))
        weight_map = manifest.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index has no weight map")
        names = sorted(set(weight_map.values()))
        artifacts.append(index)
    else:
        names = ["model.safetensors"]
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Only local safetensors shards in the configured directory are supported")
        candidate = path / name
        if candidate.resolve().parent != path or not candidate.is_file() or candidate.stat().st_size < 8:
            raise FileNotFoundError(f"Local safetensors shard is missing or incomplete: {name}")
        artifacts.append(candidate)
    generation_config = path / "generation_config.json"
    if generation_config.is_file():
        artifacts.append(generation_config)
    return config, artifacts


def local_model_status():
    """Inspect only local filenames/config; never import/load/download Transformers weights."""
    path = _model_path()
    try:
        _, artifacts = _read_local_artifacts(path)
        return {"model": "qwen3", "model_id": MODEL_ID, "path": str(path), "weights_available": True,
                "reason": "Local safetensors/tokenizer files found; model has not been loaded by this status check",
                "artifact_files": [artifact.name for artifact in artifacts]}
    except (OSError, ValueError, TypeError) as error:
        return {"model": "qwen3", "model_id": MODEL_ID, "path": str(path), "weights_available": False, "reason": str(error)}


def _hash_artifacts(artifacts):
    records = []
    for artifact in sorted(artifacts, key=lambda item: item.name):
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        records.append({"file": artifact.name, "bytes": artifact.stat().st_size, "sha256": digest.hexdigest()})
    return {"algorithm": "sha256", "files": records,
            "scope": "Hashes of supplied local artifacts; these identify bytes, not independently attested download provenance"}


@dataclass(frozen=True)
class CaptureSelection:
    layer_indices: tuple = (0, 13, 27)
    head_indices: tuple = (0, 5, 10, 15)
    channel_indices: tuple = tuple(range(64))

    def validate(self, config):
        for name, upper_bound, count_limit in (
            ("layer_indices", config.num_hidden_layers, 3),
            ("head_indices", config.num_attention_heads, 4),
            ("channel_indices", config.hidden_size, 64),
        ):
            values = getattr(self, name)
            if not 1 <= len(values) <= count_limit or any(type(value) is not int or not 0 <= value < upper_bound for value in values):
                raise ValueError(f"Invalid capture {name}")
            if list(values) != sorted(set(values)):
                raise ValueError(f"Capture {name} must be unique and increasing")


def capture_qwen_forward(model, input_ids, selection=CaptureSelection()):
    """Capture one real eager Qwen3 forward; generic size permits explicitly tiny unit fixtures.

    Returns the last-position logits plus CPU float32 sampled tensor records.
    This helper makes no pretrained identity/competence claim. The production
    QwenInspector loader separately enforces the complete 1.7B configuration.
    """
    selection.validate(model.config)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 1 <= input_ids.shape[1] <= CONTEXT_LIMIT:
        raise ValueError("Capture requires one input sequence of 1–64 tokens")
    if model.training:
        raise ValueError("Inspection requires model.eval()")
    if model.config._attn_implementation != "eager":
        raise ValueError("Genuine full attention capture requires the eager attention backend")
    residuals = {}
    handles = []
    for layer_index in selection.layer_indices:
        def hook(module, args, result, index=layer_index):
            # Current Qwen3DecoderLayer returns a Tensor. Explicit tuple handling
            # also supports older HF decoder interfaces without assuming RMSNorm semantics.
            residual = result[0] if isinstance(result, tuple) else result
            if not isinstance(residual, torch.Tensor) or residual.shape != (1, input_ids.shape[1], model.config.hidden_size):
                raise RuntimeError("Unexpected Qwen decoder hook output")
            residuals[index] = residual.detach()
        handles.append(model.model.layers[layer_index].register_forward_hook(hook))
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, output_attentions=True, output_hidden_states=False,
                           return_dict=True, logits_to_keep=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_ms = (time.perf_counter() - started) * 1000
        capture_start = time.perf_counter()
        if output.attentions is None or len(output.attentions) != model.config.num_hidden_layers:
            raise RuntimeError("Qwen backend did not return all real layer attention tensors")
        layers = []
        for layer_index in selection.layer_indices:
            attention = output.attentions[layer_index]
            if attention is None or attention.shape != (1, model.config.num_attention_heads, input_ids.shape[1], input_ids.shape[1]):
                raise RuntimeError("Unexpected Qwen attention tensor shape")
            full_residual = residuals[layer_index][0].float()
            sampled = full_residual[:, list(selection.channel_indices)]
            weights = attention[0, list(selection.head_indices)].float()
            norms = full_residual.norm(dim=-1)
            if not all(torch.isfinite(item).all() for item in (sampled, weights, norms)):
                raise RuntimeError("Non-finite model capture")
            layers.append({"index": layer_index, "attention": weights.cpu().tolist(),
                           "residual": sampled.cpu().tolist(), "norms": norms.cpu().tolist()})
        logits = output.logits[0, -1].detach().float().cpu()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite model logits")
        capture_ms = (time.perf_counter() - capture_start) * 1000
        return logits, layers, {"inference_ms": inference_ms, "capture_ms": capture_ms}
    finally:
        for handle in handles:
            handle.remove()


def _validate_request(prompt, max_new_tokens, ablate):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 400:
        raise ValueError("Prompt must contain 1–400 characters")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4:
        raise ValueError("Qwen3 max_new_tokens must be an integer from 1 to 4")
    if ablate is not None:
        raise ValueError("Head ablation is not implemented for the pretrained Qwen3 adapter")


def _chat_ids(tokenizer, prompt, max_new_tokens, device):
    # Do not silently truncate: the template's actual tokens count toward the budget.
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] or ids.shape[1] + max_new_tokens > CONTEXT_LIMIT:
        raise ValueError("Chat-templated prompt plus generation exceeds the 64-token capture context")
    return ids.to(device)


def _token_text(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


class QwenInspector:
    """Real local pretrained model only; construction fails when required weights are absent."""

    def __init__(self):
        path = _model_path()
        _, artifacts = _read_local_artifacts(path)
        # Lazy imports keep tiny-model startup/status independent of this optional stack.
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        for key, expected in EXPECTED_CONFIG.items():
            if getattr(config, key, None) != expected:
                raise ValueError(f"Unsupported pretrained Qwen configuration: {key}")
        if getattr(config, "num_key_value_heads", None) != 8 or getattr(config, "head_dim", None) != 128:
            raise ValueError("Expected Qwen3-1.7B grouped-query attention with 8 KV heads and head_dim 128")
        if any(kind != "full_attention" for kind in getattr(config, "layer_types", [])):
            raise ValueError("Only the standard full-attention Qwen3-1.7B configuration is supported")
        policy = ensure_cpu_capacity()
        # CUDA requires an explicit process configuration; never select it just
        # because hardware is present. Default development/runtime stays on CPU.
        use_cuda = policy["model_device"] == "cuda"
        if use_cuda and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise RuntimeError("Explicit CUDA mode requires a BF16-capable available GPU; no automatic device fallback")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        # Omitting device_map avoids an unnecessary Accelerate dependency. Only
        # safetensors may be loaded; no .bin/pickle fallback or remote code.
        loaded_model, loading_info = AutoModelForCausalLM.from_pretrained(
            str(path), config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=dtype, attn_implementation="eager", output_loading_info=True,
        )
        if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Pretrained checkpoint did not completely and exactly load; refusing partially initialized model")
        self.model = loaded_model.to(self.device).eval()
        self.selection = CaptureSelection()
        self.selection.validate(self.model.config)
        self.transformers_version = transformers.__version__
        self.dtype = str(dtype).removeprefix("torch.")
        self.metadata = {
            "id": MODEL_ID, "name": "Qwen3-1.7B · local pretrained weights", "layers": 28,
            "heads": 16, "d_model": 2048, "context_limit": CONTEXT_LIMIT,
            "native_context_limit": self.model.config.max_position_embeddings,
            "vocab_size": 151936, "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "trained": True, "seed": None,
            "tokenizer": {"id": MODEL_ID, "kind": "huggingface", "vocab_size": 151936,
                          "chat_template": "User chat, add_generation_prompt=True, enable_thinking=False",
                          "token_labels": "Individual token decode; byte fragments may render replacement characters"},
            "training": {"source": MODEL_ID, "license": "Apache-2.0", "scope": "Locally supplied pretrained Qwen3-1.7B; CortexLab did not train or evaluate this model"},
            "artifacts": _hash_artifacts(artifacts),
            "loading": {"complete": True, "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
        }

    def inspect(self, prompt, max_new_tokens=4, ablate=None):
        _validate_request(prompt, max_new_tokens, ablate)
        ids = _chat_ids(self.tokenizer, prompt, max_new_tokens, self.device)
        self.selection.validate(self.model.config)
        generated_ids = []
        frames = []
        model_ms = capture_ms = 0.0
        started = time.perf_counter()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        for index in range(max_new_tokens):
            logits, layers, timings = capture_qwen_forward(self.model, ids, self.selection)
            model_ms += timings["inference_ms"]
            capture_ms += timings["capture_ms"]
            serialization_start = time.perf_counter()
            probabilities = logits.softmax(dim=-1)
            predicted_id = int(probabilities.argmax())
            values, top_ids = probabilities.topk(5)
            input_ids = ids[0].detach().cpu().tolist()
            frame = {
                "index": index, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "inference_ms": timings["inference_ms"], "input_ids": input_ids,
                "tokens": [_token_text(self.tokenizer, token) for token in input_ids],
                "predicted_id": predicted_id, "predicted_token": _token_text(self.tokenizer, predicted_id),
                "top_predictions": [{"id": int(token), "token": _token_text(self.tokenizer, token), "probability": float(value)} for value, token in zip(values, top_ids)],
                "layers": layers,
            }
            frames.append(frame)
            generated_ids.append(predicted_id)
            capture_ms += (time.perf_counter() - serialization_start) * 1000
            if predicted_id in eos_ids:
                break
            ids = torch.cat((ids, torch.tensor([[predicted_id]], device=self.device)), dim=1)
        return {
            "schema_version": SCHEMA_VERSION, "id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.metadata, "request": {"prompt": prompt, "max_new_tokens": max_new_tokens, "ablate": None, "model": "qwen3"},
            "runtime": {"torch": torch.__version__, "transformers": self.transformers_version,
                        "device": str(self.device), "dtype": self.dtype,
                        "attention_backend": "eager", "use_cache": False,
                        "timing": "Synchronized CUDA forward if CUDA; capture includes CPU transfers and tensor serialization"},
            "capture": {
                "layer_indices": list(self.selection.layer_indices), "head_indices": list(self.selection.head_indices),
                "channel_indices": list(self.selection.channel_indices), "dtype": "float32", "source_dtype": self.dtype,
                "attention": "[captured_head][query][key]: genuine causal post-softmax attention; original model query-head indices in head_indices",
                "activation": "[token][sampled_channel]: sampled post-block residual before final RMSNorm; original channel indices in channel_indices",
                "norms": "L2 norm of full 2048-channel post-block residual, computed in float32; NOT the norm of the channel sample",
                "playback": "Recorded autoregressive generation steps; animation is not execution profiling or model reasoning",
            },
            "frames": frames, "metrics": {"total_ms": (time.perf_counter() - started) * 1000, "model_ms": model_ms, "capture_ms": capture_ms},
            "output": self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        }
