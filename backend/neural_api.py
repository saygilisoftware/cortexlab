"""Bounded explicit-run lifecycle and resumable SSE, sharing the application's execution lock."""
import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

from .engine import ROOT
from .neural_capture import (DEFAULT_WEIGHT_BUDGET, MAX_WEIGHT_BUDGET, architecture,
                             capture_dense_forward, prepare_input, read_local_weight_sample)
from .qwen_adapter import local_model_status

TERMINAL = {"done", "error", "cancelled"}
MAX_RETAINED_BYTES = 96 * 1024 * 1024
MAX_CACHE_BYTES = 32 * 1024 * 1024
MAX_RUN_BYTES = 30 * 1024 * 1024
MAX_EVENT_BYTES = 2 * 1024 * 1024
TTL_SECONDS = 600
MAX_SUBSCRIBERS = 4


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=400)
    max_new_tokens: StrictInt = Field(default=12, ge=1, le=16)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError("Prompt cannot be blank")
        return value


@dataclass
class Run:
    id: str
    request: dict
    created: float
    updated: float
    status: str = "loading"
    events: list = field(default_factory=list)
    retained_bytes: int = 0
    frame_count: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)


class NeuralService:
    def __init__(self, get_inspector, execution_lock, *, clock=time.monotonic,
                 weight_reader=read_local_weight_sample, status_reader=local_model_status):
        self.get_inspector, self.execution_lock = get_inspector, execution_lock
        self.clock, self.weight_reader, self.status_reader = clock, weight_reader, status_reader
        self.lock = threading.RLock()
        self.runs = OrderedDict()
        self.subscribers = {}
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cortex-neural")

    def _prune(self):
        now = self.clock()
        completed = [run for run in self.runs.values() if run.status in TERMINAL]
        remove = {run.id for run in completed if now - run.updated >= TTL_SECONDS}
        remove.update(run.id for run in completed[:-2])
        for run_id in remove: self.runs.pop(run_id, None)

    def _run(self, run_id):
        self._prune()
        if run_id not in self.runs: raise HTTPException(404, "Run is unknown or its retained history expired")
        return self.runs[run_id]

    def status(self):
        with self.lock:
            self._prune()
            local = self.status_reader()
            return {"readiness": local, "inference_starts": "explicit POST only; model/RAM checks occur in worker",
                    "demo_available": (ROOT / "examples" / "neural-demo.json").is_file(),
                    "active_run": next((run.id for run in self.runs.values() if run.status not in TERMINAL), None),
                    "retained_runs": len(self.runs), "retained_bytes": sum(run.retained_bytes for run in self.runs.values()),
                    "retention_limit_bytes": MAX_RETAINED_BYTES, "ttl_seconds": TTL_SECONDS,
                    "history_runs": 2, "subscribers": len(self.subscribers), "max_subscribers": MAX_SUBSCRIBERS,
                    "weight_cache_bytes": self.cache_bytes, "weight_cache_limit_bytes": MAX_CACHE_BYTES,
                    "default_weight_budget": DEFAULT_WEIGHT_BUDGET, "max_weight_budget": MAX_WEIGHT_BUDGET,
                    "context_limit": 128, "max_new_tokens": 16}

    def start(self, request):
        local = self.status_reader()
        if not local.get("weights_available"): raise HTTPException(503, local.get("reason", "Local model weights unavailable"))
        if not self.execution_lock.acquire(blocking=False): raise HTTPException(503, "Model execution or weight extraction is busy")
        run = None
        try:
            with self.lock:
                if self.closed: raise HTTPException(503, "Neural service is shutting down")
                self._prune()
                now = self.clock()
                run = Run(str(uuid.uuid4()), request.model_dump(), now, now)
                self.runs[run.id] = run
                self.executor.submit(self._worker, run)
                return {"id": run.id, "status": run.status, "events_url": f"/api/neural/runs/{run.id}/events"}
        except Exception as error:
            with self.lock:
                if run is not None: self.runs.pop(run.id, None)
            self.execution_lock.release()
            if isinstance(error, HTTPException): raise
            raise HTTPException(503, "Neural worker is unavailable") from error

    def _publish(self, run, event, data):
        encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        size = len(encoded.encode("utf-8")) + 80
        if size > MAX_EVENT_BYTES: raise RuntimeError("Event exceeds bounded transport size")
        with self.lock:
            reserve = 0 if event in TERMINAL else 16 * 1024
            if run.retained_bytes + size > MAX_RUN_BYTES - reserve or sum(item.retained_bytes for item in self.runs.values()) + size > MAX_RETAINED_BYTES - reserve:
                raise RuntimeError("Run recording exceeded retained-memory budget")
            run.events.append((len(run.events) + 1, event, encoded))
            run.retained_bytes += size
            run.updated = self.clock()
            if event == "frame": run.frame_count += 1
            if event in TERMINAL: run.status = event
            notifications = [(loop, signal) for subscribed, loop, signal in self.subscribers.values() if subscribed == run.id]
        for loop, signal in notifications:
            try: loop.call_soon_threadsafe(signal.set)
            except RuntimeError: pass  # Disconnected event loop; subscriber finally removes it.

    def _worker(self, run):
        started = time.perf_counter()
        output, generated, model_ms, capture_ms, load_ms = "", [], 0.0, 0.0, 0.0
        try:
            if run.cancel.is_set():
                self._publish(run, "cancelled", {"output": output, "generated_tokens": 0})
                return
            load_start = time.perf_counter()
            inspector = self.get_inspector()  # Loading/RAM guards happen off the API event loop.
            load_ms = (time.perf_counter() - load_start) * 1000
            if run.cancel.is_set():
                self._publish(run, "cancelled", {"output": output, "generated_tokens": 0, "load_ms": load_ms})
                return
            ids = prepare_input(inspector, **run.request)
            dims = architecture(inspector.model)
            self._publish(run, "start", {
                "version": 1, "architecture": dims, "model": inspector.metadata,
                "request": run.request, "input_token_count": ids.shape[1], "load_ms": load_ms,
                "capture": {"order": "layer-major: mlp then residual", "dtype": "float32", "byte_order": "little",
                            "position": "last input token", "source_dtype": str(next(inspector.model.parameters()).dtype).removeprefix("torch.")},
            })
            with self.lock: run.status = "cancelling" if run.cancel.is_set() else "running"
            eos = inspector.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
            for index in range(run.request["max_new_tokens"]):
                if run.cancel.is_set(): break
                captured = capture_dense_forward(inspector.model, ids)
                token_id = int(captured["logits"].argmax())
                generated.append(token_id)
                token_text = inspector.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                output = inspector.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                model_ms += captured["inference_ms"]
                capture_ms += captured["capture_ms"]
                self._publish(run, "frame", {
                    "index": index, "token_id": token_id, "token_text": token_text, "output": output,
                    "input_token_count": ids.shape[1], "input_token_id": int(ids[0, -1]),
                    "input_token_text": inspector.tokenizer.decode([int(ids[0, -1])], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "inference_ms": captured["inference_ms"], "capture_ms": captured["capture_ms"],
                    "activation_b64": captured["activation_b64"],
                })
                if token_id in eos_ids: break
                ids = torch.cat((ids, torch.tensor([[token_id]], device=ids.device)), dim=1)
            terminal = "cancelled" if run.cancel.is_set() else "done"
            self._publish(run, terminal, {"output": output, "generated_tokens": len(generated),
                "metrics": {"total_ms": (time.perf_counter() - started) * 1000,
                            "model_ms": model_ms, "capture_ms": capture_ms, "load_ms": load_ms}})
        except Exception as error:
            try: self._publish(run, "error", {"message": str(error)[:2000], "output": output, "generated_tokens": len(generated)})
            except Exception:
                with self.lock: run.status = "error"  # Never retain an unbounded failure payload.
        finally:
            self.execution_lock.release()
            with self.lock:
                if self.closed: self.runs.clear()
                else: self._prune()

    def inspect_run(self, run_id):
        with self.lock:
            run = self._run(run_id)
            return {"id": run.id, "status": run.status, "request": run.request,
                    "frame_count": run.frame_count, "last_event_id": len(run.events),
                    "retained_bytes": run.retained_bytes}

    def cancel(self, run_id):
        with self.lock:
            run = self._run(run_id)
            if run.status not in TERMINAL:
                run.cancel.set()
                run.status = "cancelling"
            return {"id": run.id, "status": run.status}

    def subscribe(self, run_id, after, loop):
        with self.lock:
            run = self._run(run_id)
            if after < 0 or after > len(run.events): raise HTTPException(400, "Last-Event-ID is outside retained run history")
            if len(self.subscribers) >= MAX_SUBSCRIBERS: raise HTTPException(429, "Too many active event streams")
            token, signal = str(uuid.uuid4()), asyncio.Event()
            self.subscribers[token] = (run_id, loop, signal)
            return token, signal

    def next_event(self, run_id, after):
        with self.lock:
            run = self._run(run_id)
            return (run.events[after] if after < len(run.events) else None), run.status in TERMINAL

    def unsubscribe(self, token):
        with self.lock: self.subscribers.pop(token, None)

    def weights(self, layer, budget):
        key = (layer, budget)
        with self.lock:
            if self.closed: raise HTTPException(503, "Neural service is shutting down")
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        if not self.execution_lock.acquire(blocking=False): raise HTTPException(503, "Model execution or weight extraction is busy")
        try:
            payload = self.weight_reader(layer, budget)  # Direct safetensors CPU read, never get_inspector().
            with self.lock:
                while self.cache and self.cache_bytes + len(payload) > MAX_CACHE_BYTES:
                    _, evicted = self.cache.popitem(last=False)
                    self.cache_bytes -= len(evicted)
                if not self.closed and len(payload) <= MAX_CACHE_BYTES:
                    self.cache[key] = payload
                    self.cache_bytes += len(payload)
            return payload
        except (ValueError, ImportError, OSError, RuntimeError) as error:
            raise HTTPException(503, str(error)) from error
        finally: self.execution_lock.release()

    def close(self):
        with self.lock:
            self.closed = True
            for run in self.runs.values():
                if run.status not in TERMINAL: run.cancel.set()
            self.cache.clear()
            self.cache_bytes = 0
            self.runs = OrderedDict((key, run) for key, run in self.runs.items() if run.status not in TERMINAL)
            notifications = [(loop, signal) for _, loop, signal in self.subscribers.values()]
        for loop, signal in notifications:
            try: loop.call_soon_threadsafe(signal.set)
            except RuntimeError: pass
        # A running forward cannot safely be killed. Its finally releases the
        # shared lock; no further forward starts after its cancellation boundary.
        self.executor.shutdown(wait=False, cancel_futures=False)


def create_neural_router(get_inspector, execution_lock):
    service = NeuralService(get_inspector, execution_lock)
    router = APIRouter(prefix="/api/neural", tags=["neural"])
    router.neural_service = service
    router.add_event_handler("shutdown", service.close)

    def check_origin(request):
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if origin is not None and origin != expected:
            raise HTTPException(403, "Cross-origin neural control is not permitted")

    @router.get("/status")
    def status(): return service.status()

    @router.get("/demo")
    def demo():
        path = ROOT / "examples" / "neural-demo.json"
        if not path.is_file(): raise HTTPException(404, "No real dense recording is available; inference requires an explicit run")
        if path.stat().st_size > MAX_RUN_BYTES: raise HTTPException(503, "Dense demo exceeds recording budget")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "cortexlab.neural.demo/v1": raise HTTPException(503, "Unsupported dense demo schema")
        return data

    @router.post("/runs", status_code=202)
    async def start(request: Request):
        check_origin(request)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(415, "Neural runs require application/json")
        chunks, count = [], 0
        async for chunk in request.stream():
            count += len(chunk)
            if count > 16 * 1024: raise HTTPException(413, "Neural request exceeds 16 KiB")
            if chunk: chunks.append(chunk)
        try: parsed = RunRequest.model_validate_json(b"".join(chunks))
        except ValidationError as error: raise HTTPException(422, str(error)) from error
        return service.start(parsed)

    @router.get("/runs/{run_id}")
    def inspect_run(run_id: str): return service.inspect_run(run_id)

    @router.post("/runs/{run_id}/cancel")
    def cancel(run_id: str, request: Request):
        check_origin(request)
        return service.cancel(run_id)

    @router.get("/runs/{run_id}/events")
    async def events(run_id: str, request: Request):
        raw = request.headers.get("last-event-id", "0")
        if not raw.isascii() or not raw.isdigit() or len(raw) > 10: raise HTTPException(400, "Invalid Last-Event-ID")
        after = int(raw)
        token, signal = service.subscribe(run_id, after, asyncio.get_running_loop())
        async def generate():
            cursor = after
            try:
                while True:
                    signal.clear()
                    try: item, terminal = service.next_event(run_id, cursor)
                    except HTTPException:
                        yield f'id: {cursor + 1}\nevent: error\ndata: {{"message":"Retained run history expired"}}\n\n'
                        break
                    if item:
                        event_id, event, data = item
                        yield f"id: {event_id}\nevent: {event}\ndata: {data}\n\n"
                        cursor = event_id
                        continue
                    if terminal or await request.is_disconnected(): break
                    try: await asyncio.wait_for(signal.wait(), timeout=10)
                    except asyncio.TimeoutError: yield ": keep-alive\n\n"
            finally: service.unsubscribe(token)
        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @router.get("/weights")
    def weights(layer: int = Query(default=13, ge=0, le=27), budget: int = Query(default=DEFAULT_WEIGHT_BUDGET, ge=1, le=MAX_WEIGHT_BUDGET)):
        return Response(service.weights(layer, budget), media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    return router
