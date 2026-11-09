"""Tests use RANDOM tiny Qwen3 fixtures; they do not execute pretrained 1.7B weights."""
import json
from types import SimpleNamespace

import pytest
import torch

from backend.qwen_adapter import (
    CONTEXT_LIMIT, EXPECTED_CONFIG, CaptureSelection, QwenInspector,
    _chat_ids, _validate_request, capture_qwen_forward, local_model_status,
)


@pytest.fixture(scope="module")
def random_qwen_fixture():
    # Optional dependency: tiny-model release tests do not need Transformers.
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(37)
        config = Qwen3Config(vocab_size=71, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                             eos_token_id=2, bos_token_id=1, pad_token_id=0)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
    return model


def test_random_fixture_capture_matches_direct_hooks_and_attention(random_qwen_fixture):
    model = random_qwen_fixture
    ids = torch.tensor([[1, 8, 17, 23, 9]])
    selection = CaptureSelection((0, 2), (0, 3), (0, 7, 11))
    references = {}
    handles = [model.model.layers[index].register_forward_hook(
        lambda module, args, result, index=index: references.__setitem__(index, result.detach().clone())) for index in selection.layer_indices]
    try:
        with torch.inference_mode():
            direct = model(ids, use_cache=False, output_attentions=True, output_hidden_states=True, return_dict=True, logits_to_keep=1)
    finally:
        for handle in handles: handle.remove()
    original_hook_counts = [len(layer._forward_hooks) for layer in model.model.layers]
    logits, records, timings = capture_qwen_forward(model, ids, selection)
    assert [len(layer._forward_hooks) for layer in model.model.layers] == original_hook_counts
    torch.testing.assert_close(logits, direct.logits[0, -1], atol=0, rtol=0)
    assert [record["index"] for record in records] == [0, 2]
    for record in records:
        index = record["index"]
        weights = torch.tensor(record["attention"])
        residual = torch.tensor(record["residual"])
        norms = torch.tensor(record["norms"])
        assert weights.shape == (2, 5, 5)
        assert residual.shape == (5, 3)
        torch.testing.assert_close(weights, direct.attentions[index][0, [0, 3]], atol=0, rtol=0)
        torch.testing.assert_close(residual, references[index][0, :, [0, 7, 11]], atol=0, rtol=0)
        torch.testing.assert_close(norms, references[index][0].float().norm(dim=-1), atol=0, rtol=0)
        assert torch.count_nonzero(weights.triu(1)) == 0
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 5), atol=1e-6, rtol=1e-6)
        assert not torch.allclose(residual.norm(dim=-1), norms)
    assert not torch.allclose(references[2], direct.hidden_states[-1])  # Final RMSNorm is different.
    assert timings["inference_ms"] > 0 and timings["capture_ms"] > 0


def test_random_fixture_prompt_change_causality_and_repeat(random_qwen_fixture):
    model = random_qwen_fixture
    selection = CaptureSelection((0, 2), (0, 3), (0, 7, 11))
    ids = torch.tensor([[1, 8, 17, 23]])
    first, first_records, _ = capture_qwen_forward(model, ids, selection)
    repeat, repeat_records, _ = capture_qwen_forward(model, ids, selection)
    changed = ids.clone()
    changed[0, -1] = 31
    changed_logits, changed_records, _ = capture_qwen_forward(model, changed, selection)
    torch.testing.assert_close(first, repeat, atol=0, rtol=0)
    assert first_records == repeat_records
    assert not torch.allclose(first, changed_logits)
    for original, altered in zip(first_records, changed_records):
        torch.testing.assert_close(torch.tensor(original["residual"][:-1]), torch.tensor(altered["residual"][:-1]), atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(original["attention"])[:, :-1, :-1], torch.tensor(altered["attention"])[:, :-1, :-1], atol=0, rtol=0)


def test_qwen_eager_arithmetic_matches_independent_torch_sdpa(random_qwen_fixture):
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    with torch.random.fork_rng():
        torch.manual_seed(91)
        query = torch.randn(1, 4, 5, 8)
        key, value = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    mask = torch.full((5, 5), float("-inf")).triu(1)
    output, probabilities = eager_attention_forward(SimpleNamespace(num_key_value_groups=2, training=False), query, key, value, mask, scaling=8 ** -0.5)
    reference = torch.nn.functional.scaled_dot_product_attention(query, key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1), is_causal=True)
    torch.testing.assert_close(output.transpose(1, 2), reference, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(probabilities.triu(1)) == 0
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, 4, 5), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("selection", [
    CaptureSelection((0, 0), (0,), (0,)), CaptureSelection((3,), (0,), (0,)),
    CaptureSelection((0,), (4,), (0,)), CaptureSelection((0,), (0,), (32,)),
    CaptureSelection((0,), (0,), ()), CaptureSelection((2, 0), (0,), (0,)),
])
def test_capture_selection_is_bounded(random_qwen_fixture, selection):
    with pytest.raises(ValueError): capture_qwen_forward(random_qwen_fixture, torch.tensor([[1, 2]]), selection)


def test_context_and_ablation_limits():
    class TokenizerFixture:
        """TEST-ONLY tokenizer stand-in verifies API/template control; no model identity claim."""
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "test"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "template"
        def __call__(self, text, **kwargs):
            assert text == "template"
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return {"input_ids": torch.ones(1, self.length, dtype=torch.long)}
    tokenizer = TokenizerFixture()
    tokenizer.length = 60
    assert _chat_ids(tokenizer, "test", 4, torch.device("cpu")).shape[1] == 60
    tokenizer.length = 61
    with pytest.raises(ValueError, match="64-token"): _chat_ids(tokenizer, "test", 4, torch.device("cpu"))
    for prompt, count, ablate in [("", 1, None), (" ", 1, None), ("x" * 401, 1, None), ("test", 5, None), ("test", True, None), ("test", 1, {"layer": 0, "head": 0})]:
        with pytest.raises(ValueError): _validate_request(prompt, count, ablate)


def test_local_status_never_substitutes_tiny_model_or_fetches(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEXLAB_QWEN_PATH", str(tmp_path))
    status = local_model_status()
    assert status["weights_available"] is False
    assert "missing" in status["reason"]
    with pytest.raises(FileNotFoundError): QwenInspector()
    (tmp_path / "config.json").write_text(json.dumps({**EXPECTED_CONFIG, "hidden_size": 32}), encoding="utf-8")
    assert local_model_status()["weights_available"] is False
    assert "hidden_size=2048" in local_model_status()["reason"]


def test_local_status_checks_all_shards_and_rejects_path_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEXLAB_QWEN_PATH", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(EXPECTED_CONFIG), encoding="utf-8")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        (tmp_path / filename).write_text("{}", encoding="utf-8")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}), encoding="utf-8")
    (tmp_path / "part1.safetensors").write_bytes(b"TESTONLY")
    assert local_model_status()["weights_available"] is False
    (tmp_path / "part2.safetensors").write_bytes(b"TESTONLY")
    # This cheap status only confirms filenames/config; invalid dummy bytes are never loaded.
    assert local_model_status()["weights_available"] is True
    index.write_text(json.dumps({"weight_map": {"a": "../part1.safetensors"}}), encoding="utf-8")
    assert local_model_status()["weights_available"] is False


def test_random_fixture_hooks_removed_on_forward_error(random_qwen_fixture, monkeypatch):
    model = random_qwen_fixture
    before = [len(layer._forward_hooks) for layer in model.model.layers]
    def fail(*args, **kwargs): raise RuntimeError("TEST-ONLY intentional failure")
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        capture_qwen_forward(model, torch.tensor([[1, 2]]), CaptureSelection((0, 2), (0,), (0,)))
    assert [len(layer._forward_hooks) for layer in model.model.layers] == before


def test_loader_local_safe_flags_and_partial_checkpoint_rejection(random_qwen_fixture, tmp_path, monkeypatch):
    # This test uses a tiny random CPU fixture, not a full model allocation.
    monkeypatch.setattr("backend.qwen_adapter.ensure_cpu_capacity", lambda: {"model_device": "cpu"})
    """Mocks only the loader contract; dummy files are never deserialized or labeled pretrained."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    monkeypatch.setenv("CORTEXLAB_QWEN_PATH", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(EXPECTED_CONFIG), encoding="utf-8")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        (tmp_path / filename).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"TESTONLY")
    config = SimpleNamespace(**EXPECTED_CONFIG, num_key_value_heads=8, head_dim=128, layer_types=["full_attention"] * 28)
    calls = {}
    def load_config(path, **kwargs):
        calls["config"] = kwargs
        return config
    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = kwargs
        return object()
    def load_model(path, **kwargs):
        calls["model"] = kwargs
        return object(), {"missing_keys": ["TEST-ONLY missing key"], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    monkeypatch.setattr(AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    with pytest.raises(ValueError, match="partially initialized"):
        QwenInspector()
    for kind in ("config", "tokenizer", "model"):
        assert calls[kind]["local_files_only"] is True
        assert calls[kind]["trust_remote_code"] is False
    assert calls["model"]["use_safetensors"] is True
    assert calls["model"]["output_loading_info"] is True
    assert calls["model"]["attn_implementation"] == "eager"
    assert "device_map" not in calls["model"]
