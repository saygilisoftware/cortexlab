import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from backend.app import app, execution_lock, MAX_INFERENCE_BODY_BYTES
from backend.engine import validate_session

client = TestClient(app)


def test_health_model_demo_and_inference():
    assert client.get("/api/health").json()["model_ready"] is True
    model = client.get("/api/model").json()
    assert (model["layers"], model["heads"], model["d_model"]) == (3, 4, 64)
    assert validate_session(client.get("/api/demo").json())
    result = client.post("/api/infer", json={"prompt": "the violet crystal", "max_new_tokens": 3})
    assert result.status_code == 200
    assert validate_session(result.json())
    ablated = client.post("/api/infer", json={"prompt": "the violet crystal", "max_new_tokens": 1, "ablate": {"layer": 2, "head": 3}})
    assert ablated.status_code == 200


@pytest.mark.parametrize("payload", [
    {}, {"prompt": ""}, {"prompt": "   "}, {"prompt": "x" * 401},
    {"prompt": "the " * 47, "max_new_tokens": 8},
    {"prompt": "the", "max_new_tokens": 0}, {"prompt": "the", "max_new_tokens": 9},
    {"prompt": "the", "max_new_tokens": True}, {"prompt": "the", "max_new_tokens": 1.5},
    {"prompt": "the", "max_new_tokens": "2"}, {"prompt": "the", "bad": 1},
    {"prompt": "the", "ablate": {"layer": 3, "head": 0}},
    {"prompt": "the", "ablate": {"layer": 1, "head": -1}},
    {"prompt": "the", "ablate": {"layer": True, "head": 0}},
])
def test_api_rejects_unbounded_or_malformed_requests(payload):
    assert client.post("/api/infer", json=payload).status_code == 422


def test_busy_execution_is_bounded():
    execution_lock.acquire()
    try:
        response = client.post("/api/infer", json={"prompt": "the cyan star"})
        assert response.status_code == 503
    finally:
        execution_lock.release()


def test_model_profiles_and_real_qwen_replay_without_loading_weights():
    profiles = client.get("/api/models").json()["models"]
    assert [profile["id"] for profile in profiles] == ["tiny", "qwen3"]
    assert profiles[1]["max_new_tokens"] == 4
    recording = client.get("/api/demo?model=qwen3").json()
    assert recording["schema_version"] == "cortexlab.session/v2"
    assert recording["model"]["parameter_count"] == 1720574976
    assert recording["capture"]["layer_indices"] == [0, 13, 27]


def test_qwen_unavailable_is_explicit_and_tiny_still_works(monkeypatch):
    import backend.app as api
    monkeypatch.setattr(api, "qwen_status", lambda: {"weights_available": False, "reason": "Test missing local weights"})
    response = client.post("/api/infer", json={"model": "qwen3", "prompt": "hello", "max_new_tokens": 4})
    assert response.status_code == 503
    assert "missing local weights" in response.json()["detail"]
    assert client.get("/api/model?model=qwen3").status_code == 503
    assert client.get("/api/model").json()["id"] == "cortexlab-tiny-v1"


@pytest.mark.parametrize("payload", [
    {"model": "qwen3", "prompt": "hello", "max_new_tokens": 8},
    {"model": "qwen3", "prompt": "hello", "ablate": {"layer": 0, "head": 0}},
    {"model": "unknown", "prompt": "hello"},
])
def test_qwen_rejects_unsupported_capture_requests(payload):
    assert client.post("/api/infer", json=payload).status_code == 422


def test_qwen_metadata_loading_obeys_same_execution_lock(monkeypatch):
    import backend.app as api
    monkeypatch.setattr(api, "qwen_status", lambda: {"weights_available": True})
    execution_lock.acquire()
    try:
        assert client.get("/api/model?model=qwen3").status_code == 503
    finally:
        execution_lock.release()


def test_content_length_rejects_large_body_before_json_validation():
    response = client.post("/api/infer", content=b" " * (MAX_INFERENCE_BODY_BYTES + 1), headers={"content-type": "application/json"})
    assert int(response.request.headers["content-length"]) > MAX_INFERENCE_BODY_BYTES
    assert response.status_code == 413
    assert "16 KiB" in response.json()["detail"]


def test_chunked_body_limit_without_content_length():
    response = client.post("/api/infer", content=iter([b" " * 8192, b" " * 8192, b"x"]), headers={"content-type": "application/json"})
    assert "content-length" not in response.request.headers
    assert response.request.headers["transfer-encoding"] == "chunked"
    assert response.status_code == 413


def test_actual_multichunk_bytes_override_missing_or_inaccurate_length():
    # Drive ASGI receive directly so the test guarantees distinct incoming chunks.
    async def call(headers):
        messages = iter([
            {"type": "http.request", "body": b" " * 8192, "more_body": True},
            {"type": "http.request", "body": b" " * 8192, "more_body": True},
            {"type": "http.request", "body": b"x", "more_body": False},
        ])
        responses = []
        consumed = 0
        async def receive():
            nonlocal consumed
            consumed += 1
            return next(messages)
        async def send(message):
            responses.append(message)
        await app({"type": "http", "path": "/api/infer", "method": "POST", "headers": headers}, receive, send)
        assert consumed == 3
        assert responses[0]["status"] == 413
        assert json.loads(responses[1]["body"])["detail"] == "Inference request exceeds 16 KiB"
    asyncio.run(call([(b"transfer-encoding", b"chunked")]))
    asyncio.run(call([(b"content-length", b"1")]))


def test_exact_byte_limit_and_small_chunked_json_reach_real_validation():
    body = json.dumps({"prompt": "the cyan star", "max_new_tokens": 1}).encode()
    exact = body + b" " * (MAX_INFERENCE_BODY_BYTES - len(body))
    response = client.post("/api/infer", content=exact, headers={"content-type": "application/json"})
    assert response.status_code == 200
    assert validate_session(response.json())
    chunked = client.post("/api/infer", content=iter([body[:7], body[7:]]), headers={"content-type": "application/json"})
    assert "content-length" not in chunked.request.headers
    assert chunked.status_code == 200
    assert validate_session(chunked.json())


@pytest.mark.parametrize("body", [b"[]", b"null", b"true", b"123", b'"the cyan star"', b'{"prompt":'])
def test_small_invalid_json_shapes_are_rejected(body):
    assert client.post("/api/infer", content=body, headers={"content-type": "application/json"}).status_code == 422
