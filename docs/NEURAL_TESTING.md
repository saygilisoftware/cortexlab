# Dense observatory validation — current boundary

The latest CPU verification was completed on 2026-11-06. GPU performance, dense browser rendering and the full pretrained neuron-frame flow await verification.

## Executed checks

- `$env:CUDA_VISIBLE_DEVICES='-1'; python -m pytest -q --tb=short`: **79 passed in11.57s**. Includes the original engine/API, random tiny Qwen fixtures, runtime-policy tests and24 dense capture/lifecycle tests. No full pretrained model is loaded by this suite.
- `node --test tests/*.test.mjs`: **69 passed**, including v1/v2 recordings, binary weights, signed activation transport, exact coprime coordinates, reconnect deduplication, invalid-event rejection, memory/terminal state and autoregressive continuity.
- `npm run check`: all seven browser modules parse.
- Direct CPU safetensors sample: **2,000,000 real block13 down-projection weights**,8,000,252-byte envelope,stride7,~91.28ms single local sample;5 independent source-coordinate probes match exactly. Full matrix12,582,912. Scope:CPU selected-matrix read only, not a full-model load or a renderer benchmark. Evidence:`examples/neural-weight-validation.json`.
- The actual50K measured sample was parsed by the browser's Node-compatible protocol parser, preserving raw signed float32 values and indices. It is stored as `examples/neural-weight-sample.bin` with the Qwen license.
- Local port8000 remains closed. No GPU/browser/server was started for this revision.

The CUDA-enabled Torch2.6 build has a masked-device import incompatibility with Transformers5.19. `tests/conftest.py` installs a test-process-only `current_accelerator()->None` shim when CUDA_VISIBLE_DEVICES=-1; it does not modify installed packages, initialize CUDA, or make a pretrained execution claim. CPU tensors and tests remain genuine computations.

## Product boundaries exercised by tests

The CPU suite checks actual SwiGLU `down_proj` input and pre-final-norm residual hooks against direct module values; hook cleanup; context/generation bounds; selected-matrix arithmetic; raw-body cap; content-type/origin rejection; shared execution lock; cancellation between loading/forward boundaries; exactly terminal state; resumable SSE sequence IDs; subscriber cap; retained byte budgets/TTL/history count;32MiB LRU weights; shutdown pending jobs; failed worker submission; and cache insertion after close. Root app adds a loopback TrustedHost allowlist.

Renderer source requests low-power WebGL, pixel ratio1,15Hz maximum in every quality profile and on-demand scheduling. Eco configures28,672 sampled sites and up to50K edges; higher settings are explicit. Stats distinguish model size/configured instances/last actual submission, raw captures/unmeasured structure, and CPU submit timing/GPU measurement gaps. Correct source/protocol arithmetic is not proof of successful shader compilation or visual quality.

## Not yet verified

- A genuine **full229,376-value neural frame** from the pretrained model; the earlier genuine Qwen session captures only selected blocks/channels and is never padded/faked into a dense recording.
- Dense GLSL compilation, Eco visual quality, unit picking, browser context-loss recovery and live controller behavior in an actual browser after this revision.
- GPU submission/performance at2M or4M connections; physical display FPS, power, thermals and graphics memory.
- Fresh online installation, remote CI, browser/OS matrix, multi-user/hosted security and production-readiness certification.

The underlying APIs, controller and renderer are implemented. The complete dense visual flow still needs validation on suitable hardware. The source archive is a development snapshot.
