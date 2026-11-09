"""CPU-only tiny RANDOM Qwen fixtures; no full model, CUDA, server or browser startup."""
import asyncio
import base64
import json
import math
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import backend.neural_api as api
import backend.neural_capture as capture


@pytest.fixture(scope="module")
def fixture_inspector():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(91)
        config = Qwen3Config(vocab_size=37, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval().to("cpu")
    class Tokenizer:
        length = 4
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return messages[0]["content"]
        def __call__(self, text, **kwargs):
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return {"input_ids": torch.tensor([[1, 3, 5, 7][:self.length]]) if self.length <= 4 else torch.ones(1, self.length, dtype=torch.long)}
        def decode(self, ids, **kwargs): return "".join(f"[{int(token)}]" for token in ids)
    return SimpleNamespace(model=model, tokenizer=Tokenizer(), device=torch.device("cpu"), metadata={"id": "TEST-ONLY random tiny Qwen", "trained": False})


def wait_terminal(service, run_id, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = service.inspect_run(run_id)
        if result["status"] in api.TERMINAL: return result
        time.sleep(0.005)
    pytest.fail("Fixture job did not reach terminal status")


def ready(): return {"weights_available": True, "reason": "TEST-ONLY fixture"}


def make_service(inspector, **kwargs):
    return api.NeuralService(lambda: inspector, threading.Lock(), status_reader=ready, **kwargs)


def unpack_weights(payload):
    assert payload[:8] == b"CTXWGT01"
    length = struct.unpack("<I", payload[8:12])[0]
    header = json.loads(payload[12:12 + length])
    weights = np.frombuffer(payload[12 + length:], dtype="<f4")
    assert len(weights) == header["count"]
    return header, weights


def test_dense_last_token_hook_fidelity_and_cleanup(fixture_inspector):
    model = fixture_inspector.model
    ids = capture.prepare_input(fixture_inspector, "test", 2)
    observed, handles = {}, []
    for index, block in enumerate(model.model.layers):
        handles.append(block.mlp.down_proj.register_forward_pre_hook(lambda module, args, index=index: observed.__setitem__((index, "mlp"), args[0][0, -1].detach().clone())))
        handles.append(block.register_forward_hook(lambda module, args, result, index=index: observed.__setitem__((index, "residual"), result[0, -1].detach().clone())))
    try:
        with torch.inference_mode(): reference = model(ids, use_cache=False, output_attentions=False, return_dict=True, logits_to_keep=1)
    finally:
        for handle in handles: handle.remove()
    result = capture.capture_dense_forward(model, ids)
    values = np.frombuffer(base64.b64decode(result["activation_b64"], validate=True), dtype="<f4")
    expected = torch.cat([observed[(index, kind)] for index in range(2) for kind in ("mlp", "residual")]).numpy()
    np.testing.assert_array_equal(values, expected)
    assert len(values) == 2 * (24 + 16)
    torch.testing.assert_close(result["logits"], reference.logits[0, -1], atol=0, rtol=0)
    assert all(not block._forward_hooks and not block.mlp.down_proj._forward_pre_hooks for block in model.model.layers)
    assert result["inference_ms"] > 0 and result["capture_ms"] > 0


def test_dense_hooks_removed_after_failure(fixture_inspector, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError("TEST-ONLY forward failure")
    monkeypatch.setattr(fixture_inspector.model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward failure"):
        capture.capture_dense_forward(fixture_inspector.model, torch.tensor([[1, 2]]))
    assert all(not block._forward_hooks and not block.mlp.down_proj._forward_pre_hooks for block in fixture_inspector.model.model.layers)


@pytest.mark.parametrize("budget", [1, 7, 24, 50_000])
def test_binary_exact_coprime_modular_samples(budget):
    matrix = torch.arange(24, dtype=torch.float32).reshape(3, 8) - 11
    header, weights = unpack_weights(capture.sample_weight_matrix(matrix, 13, budget))
    total, count = 24, min(24, budget)
    stride = math.ceil(total / count)
    while math.gcd(stride, total) != 1: stride += 1
    indices = [(index * stride) % total for index in range(count)]
    assert len(set(indices)) == count
    assert header == {"version": 1, "layer": 13, "source_count": 8, "target_count": 3,
                      "total_connections": 24, "count": count, "budget": budget, "stride": stride,
                      "dtype": "float32", "sample_policy": "coprime-modular-v1", "shape": [3, 8],
                      "weight_abs_max": float(matrix.flatten()[indices].abs().max())}
    np.testing.assert_array_equal(weights, matrix.flatten()[indices].numpy())
    if count == total:
        assert {index % 8 for index in indices} == set(range(8))
        assert {index // 8 for index in indices} == set(range(3))


@pytest.mark.parametrize("layer,budget", [(-1, 1), (28, 1), (True, 1), (0, 0), (0, 4_000_001), (0, True)])
def test_weight_bounds(layer, budget):
    with pytest.raises(ValueError): capture.sample_weight_matrix(torch.zeros(2, 3), layer, budget)


def test_direct_safetensor_reader_never_constructs_model(tmp_path, monkeypatch):
    """Fake safe_open validates selected-key-only CPU access; no real model/shard opened."""
    safetensors = pytest.importorskip("safetensors")
    key = "model.layers.13.mlp.down_proj.weight"
    filename = tmp_path / "part.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: filename.name}}))
    monkeypatch.setattr(capture, "_model_path", lambda: tmp_path)
    monkeypatch.setattr(capture, "_read_local_artifacts", lambda path: ({}, [filename]))
    calls = []
    class Handle:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self): return [key]
        def get_slice(self, name):
            assert name == key
            return SimpleNamespace(get_shape=lambda: [2048, 6144])
        def get_tensor(self, name):
            assert name == key
            calls.append("selected matrix")
            return "TEST-ONLY matrix stand-in"
    def safe_open(path, framework, device):
        assert path == str(filename) and framework == "pt" and device == "cpu"
        return Handle()
    monkeypatch.setattr(safetensors, "safe_open", safe_open)
    monkeypatch.setattr(capture, "sample_weight_matrix", lambda value, layer, budget: (value, layer, budget))
    assert capture.read_local_weight_sample(13, 7) == ("TEST-ONLY matrix stand-in", 13, 7)
    assert calls == ["selected matrix"]


def test_lifecycle_events_reconnect_and_exact_frame_count(fixture_inspector):
    service = make_service(fixture_inspector)
    try:
        run = service.start(api.RunRequest(prompt="test", max_new_tokens=3))
        result = wait_terminal(service, run["id"])
        assert result["status"] == "done" and result["frame_count"] == 3
        events = service.runs[run["id"]].events
        assert [event[0] for event in events] == [1, 2, 3, 4, 5]
        assert [event[1] for event in events] == ["start", "frame", "frame", "frame", "done"]
        frames = [json.loads(event[2]) for event in events if event[1] == "frame"]
        assert [frame["input_token_count"] for frame in frames] == [4, 5, 6]
        assert [frame["index"] for frame in frames] == [0, 1, 2]
        assert len(base64.b64decode(frames[0]["activation_b64"])) == 80 * 4
        async def subscribe():
            token, _ = service.subscribe(run["id"], 2, asyncio.get_running_loop())
            try:
                event, terminal = service.next_event(run["id"], 2)
                assert terminal and event[0] == 3 and event[1] == "frame"
            finally: service.unsubscribe(token)
        asyncio.run(subscribe())
        assert service.status()["subscribers"] == 0
        with pytest.raises(HTTPException): service.subscribe(run["id"], 99, None)
        assert service.execution_lock.acquire(blocking=False)
        service.execution_lock.release()
    finally: service.close()


def test_cancel_during_loading_releases_lock_and_no_forward(fixture_inspector):
    entered, release = threading.Event(), threading.Event()
    def loader():
        entered.set()
        assert release.wait(3)
        return fixture_inspector
    lock = threading.Lock()
    service = api.NeuralService(loader, lock, status_reader=ready)
    try:
        run = service.start(api.RunRequest(prompt="test", max_new_tokens=4))
        assert entered.wait(1)
        with pytest.raises(HTTPException) as busy: service.start(api.RunRequest(prompt="test"))
        assert busy.value.status_code == 503
        assert service.cancel(run["id"])["status"] == "cancelling"
        release.set()
        result = wait_terminal(service, run["id"])
        assert result["status"] == "cancelled" and result["frame_count"] == 0
        assert lock.acquire(blocking=False)
        lock.release()
    finally:
        release.set()
        service.close()


def test_ram_guard_error_is_terminal_no_fallback_and_retention_ttl(fixture_inspector):
    clock = [1000.0]
    def guarded(): raise RuntimeError("Insufficient available RAM: TEST-ONLY guard")
    service = api.NeuralService(guarded, threading.Lock(), status_reader=ready, clock=lambda: clock[0])
    try:
        ids = []
        for _ in range(4):
            run = service.start(api.RunRequest(prompt="test"))
            ids.append(run["id"])
            assert wait_terminal(service, run["id"])["status"] == "error"
        assert len(service.runs) == 2 and list(service.runs) == ids[-2:]
        event = service.runs[ids[-1]].events[-1]
        assert event[1] == "error" and "RAM" in json.loads(event[2])["message"]
        assert service.status()["retained_bytes"] <= api.MAX_RETAINED_BYTES
        clock[0] += 601
        assert service.status()["retained_runs"] == 0
    finally: service.close()


def test_weight_cache_is_lru_bounded_and_uses_no_inspector(monkeypatch):
    monkeypatch.setattr(api, "MAX_CACHE_BYTES", 20)
    reads = []
    def read(layer, budget):
        reads.append((layer, budget))
        return bytes([layer]) * 9
    def forbidden(): raise AssertionError("Weight inspection must not load the model")
    service = api.NeuralService(forbidden, threading.Lock(), weight_reader=read, status_reader=ready)
    try:
        service.weights(0, 1)
        service.weights(1, 1)
        service.weights(0, 1)
        service.weights(2, 1)
        assert list(service.cache) == [(0, 1), (2, 1)]
        assert service.cache_bytes == 18 and len(reads) == 3
        service.execution_lock.acquire()
        try:
            assert service.weights(0, 1) == bytes(9)  # Cache hit needs no computation lock.
            with pytest.raises(HTTPException) as busy: service.weights(3, 1)
            assert busy.value.status_code == 503
        finally: service.execution_lock.release()
    finally: service.close()
    assert service.cache_bytes == 0


def test_router_body_bounds_sse_resume_and_shutdown(fixture_inspector, monkeypatch):
    monkeypatch.setattr(api, "local_model_status", ready)
    # Default function values bind at definition; explicitly replace the service's reader.
    app = FastAPI()
    router = api.create_neural_router(lambda: fixture_inspector, threading.Lock())
    service = router.neural_service
    service.status_reader = ready
    service.weight_reader = lambda layer, budget: capture.sample_weight_matrix(torch.arange(24).reshape(3, 8).float(), layer, budget)
    app.include_router(router)
    with TestClient(app) as client:
        for body in [b"[]", b"null", b"{}", b'{"prompt":" "}', b'{"prompt":"x","max_new_tokens":true}', b'{"prompt":"x","max_new_tokens":17}']:
            assert client.post("/api/neural/runs", content=body, headers={"content-type": "application/json"}).status_code == 422
        assert client.post("/api/neural/runs", content=iter([b" " * 8192, b" " * 8192, b"x"]), headers={"content-type": "application/json"}).status_code == 413
        assert client.post("/api/neural/runs", content=b'{"prompt":"test"}', headers={"content-type": "text/plain"}).status_code == 415
        assert client.post("/api/neural/runs", content=b'{"prompt":"test"}', headers={"content-type": "text/plain", "origin": "https://unrelated.invalid"}).status_code == 403
        assert client.post("/api/neural/runs", json={"prompt": "test"}, headers={"origin": "https://unrelated.invalid"}).status_code == 403
        response = client.post("/api/neural/runs", json={"prompt": "test", "max_new_tokens": 2})
        assert response.status_code == 202
        run = response.json()
        assert client.post(f"/api/neural/runs/{run['id']}/cancel", headers={"origin": "https://unrelated.invalid"}).status_code == 403
        wait_terminal(service, run["id"])
        stream = client.get(run["events_url"], headers={"Last-Event-ID": "2"})
        assert stream.status_code == 200 and stream.headers["content-type"].startswith("text/event-stream")
        assert "id: 3\nevent: frame" in stream.text and "id: 1\n" not in stream.text
        assert "event: done" in stream.text
        assert client.get(run["events_url"], headers={"Last-Event-ID": "bad"}).status_code == 400
        assert client.get(run["events_url"], headers={"Last-Event-ID": "999"}).status_code == 400
        assert client.get("/api/neural/runs/missing").status_code == 404
        assert client.get("/api/neural/weights?budget=0").status_code == 422
        assert client.get("/api/neural/weights?layer=28").status_code == 422
        assert client.get("/api/neural/weights?layer=1&budget=7").status_code == 200
        assert client.get("/api/neural/status").json()["default_weight_budget"] == 50_000
    assert service.closed and service.cache_bytes == 0


def test_close_during_weight_read_cannot_repopulate_cache():
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    def reader(layer, budget):
        entered.set()
        assert release.wait(3)
        return b"TEST-ONLY bytes"
    service = api.NeuralService(lambda: None, lock, weight_reader=reader, status_reader=ready)
    result = []
    thread = threading.Thread(target=lambda: result.append(service.weights(0, 1)))
    thread.start()
    try:
        assert entered.wait(1)
        service.close()
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive() and result == [b"TEST-ONLY bytes"]
        assert service.cache_bytes == 0 and not service.cache
        assert lock.acquire(timeout=1)
        lock.release()
    finally:
        release.set()
        thread.join(timeout=2)


def test_shutdown_preserves_pending_job_cleanup_lease():
    """A deferred executor proves close cannot cancel the job that owns the lock."""
    service = api.NeuralService(lambda: pytest.fail("Cancelled pending job must not load model"), threading.Lock(), status_reader=ready)
    service.executor.shutdown()
    class DeferredExecutor:
        def submit(self, callback, run): self.callback, self.run = callback, run
        def shutdown(self, wait, cancel_futures):
            assert wait is False and cancel_futures is False
            self.callback(self.run)
    service.executor = DeferredExecutor()
    service.start(api.RunRequest(prompt="test"))
    assert service.execution_lock.locked()
    service.close()
    assert not service.execution_lock.locked() and not service.runs


def test_subscriber_cap_and_release(fixture_inspector):
    service = make_service(fixture_inspector)
    try:
        run = service.start(api.RunRequest(prompt="test", max_new_tokens=1))
        wait_terminal(service, run["id"])
        async def subscribe():
            loop = asyncio.get_running_loop()
            tokens = [service.subscribe(run["id"], 0, loop)[0] for _ in range(api.MAX_SUBSCRIBERS)]
            try:
                with pytest.raises(HTTPException) as busy: service.subscribe(run["id"], 0, loop)
                assert busy.value.status_code == 429
            finally:
                for token in tokens: service.unsubscribe(token)
        asyncio.run(subscribe())
        assert not service.subscribers
    finally: service.close()


def test_record_budget_reserves_an_explicit_terminal_error(fixture_inspector, monkeypatch):
    monkeypatch.setattr(api, "MAX_RUN_BYTES", 20_000)
    service = make_service(fixture_inspector)
    try:
        run = service.start(api.RunRequest(prompt="test", max_new_tokens=16))
        result = wait_terminal(service, run["id"])
        assert result["status"] == "error"
        assert result["retained_bytes"] <= 20_000
        events = service.runs[run["id"]].events
        assert events[-1][1] == "error" and "budget" in json.loads(events[-1][2])["message"]
        assert service.execution_lock.acquire(timeout=1)
        service.execution_lock.release()
    finally: service.close()


def test_context_limit_errors_without_a_forward(fixture_inspector, monkeypatch):
    monkeypatch.setattr(fixture_inspector.tokenizer, "length", 125)
    service = make_service(fixture_inspector)
    try:
        run = service.start(api.RunRequest(prompt="test", max_new_tokens=4))
        result = wait_terminal(service, run["id"])
        assert result["status"] == "error" and result["frame_count"] == 0
        assert "128-token" in json.loads(service.runs[run["id"]].events[-1][2])["message"]
    finally: service.close()


def test_submit_failure_leaves_no_orphan_run_or_lock():
    service = api.NeuralService(lambda: None, threading.Lock(), status_reader=ready)
    service.executor.shutdown()
    try:
        with pytest.raises(HTTPException) as failure: service.start(api.RunRequest(prompt="test"))
        assert failure.value.status_code == 503
        assert not service.runs and not service.execution_lock.locked()
    finally: service.close()
