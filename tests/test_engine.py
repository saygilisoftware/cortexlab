import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from backend.engine import Inspector, ROOT, validate_session
from backend.model import Attention, Config

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def inspector():
    return Inspector()


def test_attention_against_independent_numpy_oracle():
    torch.manual_seed(19)
    attention = Attention(Config()).eval()
    x = torch.randn(1, 5, 64)
    with torch.inference_mode():
        output, weights = attention(x)
        projected = x.numpy() @ attention.qkv.weight.numpy().T + attention.qkv.bias.numpy()
        q, k, v = [array.reshape(1, 5, 4, 16).transpose(0, 2, 1, 3) for array in np.split(projected, 3, axis=-1)]
        scores = np.einsum("bhid,bhjd->bhij", q, k) / 4
        scores = np.where(np.triu(np.ones((5, 5)), 1).astype(bool), -np.inf, scores)
        probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        mixed = np.einsum("bhij,bhjd->bhid", probabilities, v).transpose(0, 2, 1, 3).reshape(1, 5, 64)
        reference = mixed @ attention.proj.weight.numpy().T + attention.proj.bias.numpy()
    np.testing.assert_allclose(weights.numpy(), probabilities, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(output.numpy(), reference, atol=2e-6, rtol=2e-6)
    assert not np.allclose(weights[0, 0], weights[0, 1])


def test_causal_invariance_and_capture_fidelity(inspector):
    ids = torch.tensor([inspector.tokenizer.encode("the cyan star moves .")])
    changed = ids.clone()
    changed[0, -2:] = torch.tensor([3, 3])
    with torch.inference_mode():
        logits, layers = inspector.model(ids, capture=True)
        ordinary, _ = inspector.model(ids)
        altered, altered_layers = inspector.model(changed, capture=True)
    torch.testing.assert_close(logits, ordinary, rtol=0, atol=0)
    torch.testing.assert_close(logits[:, :-2], altered[:, :-2], atol=2e-6, rtol=2e-6)
    for (weights, residual), (_, altered_residual) in zip(layers, altered_layers):
        assert weights.shape == (1, 4, ids.shape[1], ids.shape[1])
        assert torch.count_nonzero(weights.triu(1)) == 0
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        torch.testing.assert_close(residual[:, :-2], altered_residual[:, :-2], atol=2e-6, rtol=2e-6)
    recorded = inspector.inspect("the cyan star moves .", 1)["frames"][0]
    for layer, (weights, residual) in zip(recorded["layers"], layers):
        torch.testing.assert_close(torch.tensor(layer["attention"]), weights[0], atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(layer["residual"]), residual[0], atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(layer["norms"]), residual[0].norm(dim=-1), atol=0, rtol=0)


def test_determinism_prompt_dependence_and_real_ablation(inspector):
    first = inspector.inspect("the cyan star", 3)
    repeat = inspector.inspect("the cyan star", 3)
    second = inspector.inspect("the blue planet", 3)
    assert first["frames"][0]["layers"] == repeat["frames"][0]["layers"]
    assert first["output"] == repeat["output"]
    assert first["frames"][0]["layers"] != second["frames"][0]["layers"]
    ablated = inspector.inspect("the cyan star", 1, {"layer": 0, "head": 1})
    assert first["frames"][0]["layers"][0]["attention"] == ablated["frames"][0]["layers"][0]["attention"]
    assert first["frames"][0]["layers"][0]["residual"] != ablated["frames"][0]["layers"][0]["residual"]
    assert first["frames"][0]["top_predictions"] != ablated["frames"][0]["top_predictions"]


def test_round_trip_and_reject_bad_recordings(inspector):
    session = inspector.inspect("observe the planet", 3)
    assert validate_session(json.loads(json.dumps(session))) == session
    for mutate in [
        lambda value: value.update(schema_version="bad"),
        lambda value: value["frames"][0]["layers"][0]["attention"][0][0].__setitem__(0, 0.5),
        lambda value: value["frames"][0]["layers"][0]["norms"].__setitem__(0, float("nan")),
        lambda value: value["frames"][0]["layers"][0]["residual"].pop(),
        lambda value: value["frames"][0]["tokens"].__setitem__(0, "wrong"),
    ]:
        invalid = copy.deepcopy(session)
        mutate(invalid)
        with pytest.raises(ValueError): validate_session(invalid)


def test_tokenizer_unknowns_and_context(inspector):
    assert inspector.tokenizer.encode("你好 unseenword") == [1, 3, 3]
    for prompt, count, ablate in [("", 1, None), (" " * 2, 1, None), ("a" * 401, 1, None), ("the " * 45, 8, None), ("the", True, None), ("the", 9, None), ("the", 1, {"layer": 3, "head": 0})]:
        with pytest.raises(ValueError): inspector.inspect(prompt, count, ablate)


def test_checkpoint_learning_metrics_and_demo():
    metadata = json.loads((ROOT / "examples" / "model.json").read_text(encoding="utf-8"))
    training = metadata["training"]
    assert training["final_heldout"]["loss"] < training["initial_heldout"]["loss"]
    assert training["final_heldout"]["accuracy"] > training["baseline"]["heldout_accuracy"]
    # Recompute held-out metrics from the checked-in checkpoint, not metadata alone.
    import random
    from backend.corpus import sentences
    texts = sorted(set(sentences()))
    random.Random(Config().seed).shuffle(texts)
    inspector = Inspector()
    losses, correct, total = [], 0, 0
    with torch.inference_mode():
        for text in texts[::7]:
            sequence = inspector.tokenizer.encode(text, eos=True)
            targets = torch.tensor(sequence[1:])
            logits, _ = inspector.model(torch.tensor([sequence[:-1]]))
            losses.append(float(F.cross_entropy(logits[0], targets, reduction="sum")))
            correct += int((logits[0].argmax(-1) == targets).sum())
            total += len(targets)
    assert correct / total == pytest.approx(training["final_heldout"]["accuracy"], abs=1e-6)
    assert sum(losses) / total == pytest.approx(training["final_heldout"]["loss"], abs=1e-5)
    demo = json.loads((ROOT / "examples" / "demo-session.json").read_text(encoding="utf-8"))
    assert validate_session(demo)
    fresh = inspector.inspect(**demo["request"])
    assert fresh["output"] == demo["output"]
    # Stored Windows/CPU values replay exactly, while re-inference on another
    # supported CPU/BLAS build may differ by floating-point roundoff.
    for actual, recorded in zip(fresh["frames"][0]["layers"], demo["frames"][0]["layers"]):
        for field in ("attention", "residual", "norms"):
            torch.testing.assert_close(torch.tensor(actual[field]), torch.tensor(recorded[field]), atol=2e-5, rtol=2e-5)
