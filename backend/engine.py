"""Checkpoint loading, bounded inference, immutable recording and v1 validation."""
import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from .model import Config, TinyTransformer, Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cortexlab.session/v1"


class Inspector:
    def __init__(self, artifact_dir=None):
        directory = Path(artifact_dir) if artifact_dir else ROOT / "examples"
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        checkpoint = directory / "tiny-transformer.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash differs from model metadata")
        config = Config()
        for name in ("layers", "heads", "d_model", "context_limit", "seed"):
            if metadata[name] != getattr(config, name):
                raise ValueError(f"Unsupported model configuration: {name}")
        self.tokenizer = Tokenizer(metadata["tokenizer"]["vocabulary"])
        if self.tokenizer.vocabulary[:4] != Tokenizer.specials or len(set(self.tokenizer.vocabulary)) != metadata["vocab_size"]:
            raise ValueError("Invalid tokenizer metadata")
        self.model = TinyTransformer(metadata["vocab_size"], config)
        if metadata["parameter_count"] != sum(parameter.numel() for parameter in self.model.parameters()):
            raise ValueError("Parameter count differs from metadata")
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("Non-finite checkpoint")
        self.model.eval()
        self.metadata = metadata

    def validate_request(self, prompt, max_new_tokens, ablate=None):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 400:
            raise ValueError("Prompt must contain 1–400 characters")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8:
            raise ValueError("max_new_tokens must be an integer from 1 to 8")
        ids = self.tokenizer.encode(prompt)
        if len(ids) + max_new_tokens > self.metadata["context_limit"]:
            raise ValueError("Prompt plus generation exceeds the 48-token context")
        if ablate is not None:
            if set(ablate) != {"layer", "head"} or type(ablate["layer"]) is not int or type(ablate["head"]) is not int:
                raise ValueError("Ablation requires integer layer and head")
            if not 0 <= ablate["layer"] < 3 or not 0 <= ablate["head"] < 4:
                raise ValueError("Ablation outside model bounds")
        return ids

    def inspect(self, prompt, max_new_tokens=4, ablate=None):
        ids = self.validate_request(prompt, max_new_tokens, ablate)
        original_ids = ids.copy()
        frames = []
        model_ms = capture_ms = 0.0
        started = time.perf_counter()
        ablation = (ablate["layer"], ablate["head"]) if ablate else None
        with torch.inference_mode():
            for index in range(max_new_tokens):
                forward_start = time.perf_counter()
                logits, captured = self.model(torch.tensor([ids]), capture=True, ablate=ablation)
                inference_ms = (time.perf_counter() - forward_start) * 1000
                model_ms += inference_ms
                capture_start = time.perf_counter()
                probabilities = logits[0, -1].softmax(-1)
                # Stable, greedy choice; suppress no classes so recorded logits retain their exact semantics.
                predicted_id = int(probabilities.argmax())
                values, indices = probabilities.topk(min(5, len(probabilities)))
                layers = [{"index": layer_index, "attention": attention[0].tolist(), "residual": residual[0].tolist(), "norms": residual[0].norm(dim=-1).tolist()} for layer_index, (attention, residual) in enumerate(captured)]
                frames.append({
                    "index": index, "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "inference_ms": inference_ms, "input_ids": ids.copy(),
                    "tokens": [self.tokenizer.vocabulary[token] for token in ids],
                    "predicted_id": predicted_id, "predicted_token": self.tokenizer.vocabulary[predicted_id],
                    "top_predictions": [{"token": self.tokenizer.vocabulary[int(token)], "probability": float(value)} for value, token in zip(values, indices)],
                    "layers": layers,
                })
                capture_ms += (time.perf_counter() - capture_start) * 1000
                ids.append(predicted_id)
                if predicted_id == 2:
                    break
        result = {
            "schema_version": SCHEMA, "id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.metadata, "request": {"prompt": prompt, "max_new_tokens": max_new_tokens, "ablate": ablate},
            "runtime": {"torch": torch.__version__, "device": "cpu"},
            "capture": {"attention": "[head][query][key]: causal post-softmax probability, before optional head-output ablation", "activation": "[token][channel]: post-block residual; norms are L2 across channels", "dtype": "float32", "playback": "Recorded generation steps; animation does not represent execution timing"},
            "frames": frames, "metrics": {"total_ms": (time.perf_counter() - started) * 1000, "model_ms": model_ms, "capture_ms": capture_ms},
            "output": self.tokenizer.decode(ids[len(original_ids):]),
        }
        return result


def validate_session(session):
    """Strict numerical/shape gate for the bounded v1 export (raises ValueError)."""
    try:
        if session["schema_version"] != SCHEMA:
            raise ValueError("Unsupported session version")
        model = session["model"]
        if (model["layers"], model["heads"], model["d_model"], model["context_limit"]) != (3, 4, 64, 48):
            raise ValueError("Unsupported model shape")
        frames = session["frames"]
        if not 1 <= len(frames) <= 8:
            raise ValueError("Invalid frame count")
        vocabulary = model["tokenizer"]["vocabulary"]
        if len(vocabulary) != model["vocab_size"]:
            raise ValueError("Invalid vocabulary")
        previous = None
        elapsed = -1
        for index, frame in enumerate(frames):
            ids = frame["input_ids"]
            length = len(ids)
            if frame["index"] != index or not 1 <= length <= 48 or len(frame["tokens"]) != length:
                raise ValueError("Invalid frame axes")
            if any(type(token) is not int or not 0 <= token < len(vocabulary) for token in ids):
                raise ValueError("Invalid token id")
            if frame["tokens"] != [vocabulary[token] for token in ids]:
                raise ValueError("Token labels differ from IDs")
            if previous is not None and ids != previous["input_ids"] + [previous["predicted_id"]]:
                raise ValueError("Nonsequential frames")
            if type(frame["predicted_id"]) is not int or not 0 <= frame["predicted_id"] < len(vocabulary) or frame["predicted_token"] != vocabulary[frame["predicted_id"]]:
                raise ValueError("Invalid prediction")
            if frame["elapsed_ms"] < elapsed or frame["inference_ms"] < 0:
                raise ValueError("Invalid timing")
            elapsed = frame["elapsed_ms"]
            if len(frame["layers"]) != 3:
                raise ValueError("Invalid layer count")
            for layer_index, layer in enumerate(frame["layers"]):
                attention = torch.tensor(layer["attention"], dtype=torch.float64)
                residual = torch.tensor(layer["residual"], dtype=torch.float64)
                norms = torch.tensor(layer["norms"], dtype=torch.float64)
                if layer["index"] != layer_index or attention.shape != (4, length, length) or residual.shape != (length, 64) or norms.shape != (length,):
                    raise ValueError("Invalid tensor shape")
                if not all(torch.isfinite(tensor).all() for tensor in (attention, residual, norms)):
                    raise ValueError("Non-finite tensor")
                if (attention < 0).any() or (attention > 1).any() or not torch.allclose(attention.sum(-1), torch.ones(4, length, dtype=torch.float64), atol=2e-5):
                    raise ValueError("Invalid attention probabilities")
                if attention.triu(1).abs().max() > 1e-8 or not torch.allclose(residual.norm(dim=-1), norms, atol=2e-5, rtol=2e-5):
                    raise ValueError("Invalid causal weights or residual norms")
            for prediction in frame["top_predictions"]:
                if prediction["token"] not in vocabulary or not math.isfinite(prediction["probability"]) or not 0 <= prediction["probability"] <= 1:
                    raise ValueError("Invalid top prediction")
            previous = frame
        def finite_tree(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite session value")
            if isinstance(value, dict):
                for child in value.values(): finite_tree(child)
            elif isinstance(value, list):
                for child in value: finite_tree(child)
        finite_tree(session)
    except (KeyError, TypeError, RuntimeError, IndexError) as error:
        raise ValueError("Malformed session") from error
    return session
