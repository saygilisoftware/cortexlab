# Stack and compatibility

## Accepted

- Python 3.10+; installed/test target 3.10.0. PyTorch 2.6.0 (environment CUDA build; default computation CPU float32). Explicit small decoder implementation makes Q/K/V and residual capture auditable. No Transformers dependency required.
- FastAPI 0.115.12 / Pydantic 2.11.4 / Uvicorn 0.34.2 available locally. Bounded request/response and same-origin static delivery. pytest/HTTPX available for integration.
- Direct Three.js **0.170.0**, WebGL2, native accessible DOM controls. Registry research found 0.187.0 current, but complete tagged r170 source was retrievable under network restrictions. Version pin and vendor license/provenance are checked in. Real WebGL2 shader/render tests passed. Exact byte identity with a released upstream artifact could not be independently verified; delivered hashes identify the files used.
- Native JavaScript ES modules plus a strict runtime tensor contract and Node contract tests. This deliberately avoids a frontend build/package-install step. TypeScript/React add no necessary functionality for the bounded instrument; static TypeScript checking is not claimed.
- Python Playwright 1.59.0 and actual system Chrome 156.0.8078.80 validated desktop/mobile viewports, node picking, replay and integration. CI is configured for Playwright Chromium; remote CI has not run.

## Rejected for MVP

R3F 10 is documented alpha, R3F9 pairs React19; neither is needed. WebGPURenderer has documented fallback, but no computation justifies its extra surface here. JAX/ONNX/Rust/MuJoCo add unrelated environments. TransformerLens is a future adapter; current circuit-tracer and TransformerLens main manifests disagree on Transformers compatibility, so do not combine them casually. External pretrained weights add download and license requirements. WebSocket streaming adds ordering/cancellation/backpressure concerns without improving short complete recordings.

## Constraints

Shell network is disabled. Do not claim a clean online installation has been tested. Never modify other desktop projects or use their application code. Reuse installed toolchains read-only where available, and preserve original third-party license notices for any vendored dependency.
