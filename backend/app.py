"""Loopback-first synchronous recording API; no token streaming claim."""
import json
import threading
from functools import lru_cache
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from .engine import Inspector, ROOT
from .runtime_policy import resource_policy
from .neural_api import create_neural_router

torch.set_num_threads(1)
app = FastAPI(title="CortexLab", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver", "[::1]"])
execution_lock = threading.Lock()
MAX_INFERENCE_BODY_BYTES = 16 * 1024


class InferenceBodyLimit:
    """Bound raw request bytes before JSON parsing, including chunked requests.

    Buffer at most the small accepted body. Content-Length is an early rejection
    hint; actual received bytes enforce the limit even when the header is absent
    or inaccurate. No general-purpose request logging or persistence is added.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/infer":
            await self.app(scope, receive, send)
            return

        async def reject():
            await JSONResponse({"detail": "Inference request exceeds 16 KiB"}, status_code=413)(scope, receive, send)

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    if int(value) > MAX_INFERENCE_BODY_BYTES:
                        await reject()
                        return
                except ValueError:
                    pass  # The received byte count remains authoritative.
        chunks = []
        received_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            received_bytes += len(chunk)
            if received_bytes > MAX_INFERENCE_BODY_BYTES:
                await reject()
                return
            if chunk:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


app.add_middleware(InferenceBodyLimit)


@lru_cache(maxsize=1)
def inspector():
    return Inspector()


@lru_cache(maxsize=1)
def qwen_inspector():
    from .qwen_adapter import QwenInspector
    return QwenInspector()


def qwen_status():
    try:
        from .qwen_adapter import local_model_status
        return {**local_model_status(), **resource_policy()}
    except ImportError:
        return {"weights_available": False, "reason": "Optional Qwen dependencies are missing; install requirements-qwen.txt"}


class Ablation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer: StrictInt = Field(ge=0, le=2)
    head: StrictInt = Field(ge=0, le=3)


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=400)
    max_new_tokens: StrictInt = Field(default=4, ge=1, le=8)
    ablate: Ablation | None = None
    model: Literal["tiny", "qwen3"] = "tiny"

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Prompt cannot be blank")
        return value


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": inspector().metadata["trained"], "device": "cpu", "qwen_loaded": bool(qwen_inspector.cache_info().currsize), "resource_policy": resource_policy()}


@app.get("/api/model")
def model_info(model: Literal["tiny", "qwen3"] = "tiny"):
    if model == "qwen3":
        status = qwen_status()
        if not status["weights_available"]:
            raise HTTPException(503, status["reason"])
        if not execution_lock.acquire(blocking=False):
            raise HTTPException(503, "Model execution or loading is busy; retry after it completes")
        try:
            return qwen_inspector().metadata
        except (ValueError, RuntimeError, ImportError, OSError) as error:
            raise HTTPException(503, str(error)) from error
        finally:
            execution_lock.release()
    return inspector().metadata


@app.get("/api/models")
def models():
    status = qwen_status()
    loaded = bool(qwen_inspector.cache_info().currsize)
    return {"models": [
        {"id": "tiny", "name": "Cortex Tiny · offline fixture", "weights_available": True, "max_new_tokens": 8, "recording_available": True},
        {"id": "qwen3", "name": "Qwen3-1.7B · pretrained", "max_new_tokens": 4,
         "recording_available": (ROOT / "examples" / "qwen-demo-session.json").is_file(), "loaded": loaded,
         "device": str(qwen_inspector().device) if loaded else None, **status},
    ]}


@app.get("/api/demo")
def demo(model: Literal["tiny", "qwen3"] = "tiny"):
    filename = "qwen-demo-session.json" if model == "qwen3" else "demo-session.json"
    path = ROOT / "examples" / filename
    if not path.is_file():
        raise HTTPException(404, "No Qwen recording yet. Run python scripts/qwen_demo.py after downloading the official weights.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/infer")
def infer(request: InferRequest):
    if not execution_lock.acquire(blocking=False):
        raise HTTPException(503, "Inference busy; retry after the current recording completes")
    try:
        if request.model == "qwen3":
            if request.max_new_tokens > 4:
                raise ValueError("Qwen demo captures at most 4 generated tokens per recording")
            if request.ablate is not None:
                raise ValueError("Head-output ablation is not implemented for the Qwen adapter")
            status = qwen_status()
            if not status["weights_available"]:
                raise HTTPException(503, status["reason"])
            try:
                return qwen_inspector().inspect(request.prompt, request.max_new_tokens)
            except (ImportError, OSError, RuntimeError) as error:
                raise HTTPException(503, str(error)) from error
        return inspector().inspect(request.prompt, request.max_new_tokens, request.ablate.model_dump() if request.ablate else None)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        execution_lock.release()


frontend = ROOT / "frontend"
app.include_router(create_neural_router(qwen_inspector, execution_lock))
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
