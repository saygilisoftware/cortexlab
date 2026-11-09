# Performance and measurement limits

Measured2026-11-06. Separate computation, recording conversion, HTTP and display timing. Targets below are planning budgets, not universal guarantees. Raw JSON files preserve the observed samples/metadata.

## Engine

Command:`python scripts/record_demo.py --benchmark`. Torch2.6.0+cu124, CPU float32, one thread; Windows; checkpoint158,912params,3 blocks/4 heads/64width. Prompt `the cyan`, max_new_tokens8, EOS after4 frames;3 warmups followed by30 recordings.

| Measurement | Result |
|---|---:|
| Warm engine total median | 4.664ms |
| Warm engine total p95 (nearest rank) | 5.301ms |
| Forward time median | 4.000ms |
| Capture/list conversion median | 0.606ms |
| Compact recording median | 86,244bytes |

Source:`examples/backend-benchmark.json`. `model_ms` includes instrumented forward/capture detach work inside the model; `capture_ms` includes softmax/top5, tensor-to-list conversion and norm calculation. These are coarse wall-time regions, not an operation-level profiler. Total excludes JSON encoding/HTTP/browser. Timing jitter and concurrent desktop activity affect results. The richer bundled demo's single cold-ish recording was58.26ms total (51.88ms forward,5.86ms capture); it is not a warm percentile benchmark.

## Browser

Command:`python scripts/e2e.py --channel chrome`. Chrome156.0.8078.80 headless,1600×1050,devicePixelRatio1; WebGL2 via ANGLE/NVIDIA RTX4060Ti/D3D11. Samples are `requestAnimationFrame` timestamp intervals in a600-sample ring after warm-up and real interaction. They measure browser scheduling, not GPU execution time or a physical display's FPS.

Final raw report:`examples/browser-validation.json`:

- Normal24-node inspection:25 draw calls,12 geometries,12 textures; mean frame interval6.00ms, p95 5.60ms. A p95 below the mean is possible with a small number of slow scheduling outliers; do not interpret this as a constant frame rate.
- Maximum supported request:40 input tokens +8 predictions,47 input tokens in the final forward frame,141 residual nodes,64 draw calls,26,082 triangles,51 textures. Compact exported JSON3,430,366bytes; successful browser reimport. The8th prediction is not an input node until a subsequent forward pass, which is not recorded.
- Thirty scene replacements preserved12 geometries/12 textures on the same-size24-node scene; resource counts are not byte-accurate GPU memory.
- 60FPS is an aspiration; these headless intervals do not certify60FPS across hardware. Visible-browser refresh behavior, software rendering and mobile hardware remain unbenchmarked.

## Budgets and changes driven by evidence

Initial budgets:context<=48,3blocks/4heads,width64;<=144 residual nodes;<=48 selected-query arcs;<100 draw calls;warm engine<2s;recording<5MiB;p95 visible-frame interval<=25ms. Local bounded workloads meet engine/payload/draw-call targets; visible-display timing remains an independent validation item.

The largest context test exposed a real round-trip bug: pretty-printed JSON exceeded5MiB although compact data fit. Export now writes compact JSON, and the maximum-budget browser flow verifies reimport. Instancing, a batched attention polyline/arrow mesh, bounded pixel ratio1.5, and resource disposal keep rendering bounded. Geometry is rebuilt on discrete selection/frame changes; the animation loop reuses matrix/color scratch objects. This is sufficient at current scale; do not claim large-model scalability.

Attention storage is O(frames × layers × heads × tokens²); full residual storage is O(frames × layers × tokens × width). Prefix recomputation intentionally trades inference throughput for simple complete capture semantics. No KV-cache optimization, binary transport, LOD or GPU compute is implemented.

CPU utilization, GPU utilization, process RSS and exact GPU memory were not instrumented. No transport-streaming latency exists:HTTP returns a complete session. Request-to-display latency is not isolated from frontend validation/render work by the current benchmark. Future profiling should add those distinct measurements before scaling.
