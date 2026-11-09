# Dense neural observatory — accepted revision

User correction: the token/layer MVP is too sparse. The primary experience must show large-scale real neuron activations, millions of measured weight connections, and live token input/output. Product-quality resource/lifecycle boundaries are part of this revision; no claim of broad production certification precedes testing.

## Scientific mapping

Qwen3 has28 decoder blocks, hidden width2048 and MLP intermediate width6144. Capture the LAST INPUT TOKEN at each autoregressive forward pass. For every block capture the real `down_proj` input (`SiLU(gate_proj(x))*up_proj(x)`,6144 MLP units) and post-block residual (2048 channels). This yields172,032 MLP units plus57,344 residual channels =229,376 displayed computational units. They are not1.7B individual parameters/neuron sites. The full model executes every step; positions are schematic.

Connections in the dense lens are actual `down_proj.weight[residual_channel,mlp_unit]` values for the selected block. A block contains12,582,912 such weights. Default GPU rendering uses2,000,000 deterministic strided connections, adjustable within a tested bound. Report sampled/rendered count separately from full matrix size. This is not an attention graph. Signs/magnitudes use real weights. Activation-dependent emphasis uses captured endpoints. No invented traffic or particles as computation. Recorded generation frames can be replayed; source execution timing and presentation transitions are distinct.

## Module ownership and contracts

- Backend specialist: `backend/neural_capture.py`, `backend/neural_api.py`, `tests/test_neural.py`. Engine hooks and binary datasets are independent of HTTP. Existing QwenInspector/model remains the single shared loaded resource.
- Graphics specialist: `frontend/src/neural-scene.js`, `frontend/src/neural-protocol.js`. GPU-buffered points/lines consume data only. No inference or fake values in renderer.
- Orchestrator: app router integration, DOM/controller, tests/browser verification, docs. No concurrent edits to another owner's modules.

### Binary weights v1

Little-endian envelope:8-byte ASCII `CTXWGT01`, uint32 JSON header length, UTF-8 JSON header, packed float32 weights. Header:version1,layer,source_count6144,target_count2048,total_connections12582912,count,budget,stride,dtype `float32`,sample_policy `coprime-modular-v1`,shape `[target_count,source_count]`. Stride is the smallest integer >=ceil(total/count) coprime with total; flat index=(i*stride)%total; source=flat%source_count,target=floor(flat/source_count). Use uint shader arithmetic, not float indexing. Preserve signed raw values. Cap budget4,000,000; **Eco default50,000**, higher densities explicit controls. Two-million capacity is an implementation target, not a verified GPU result. Header also includes weight_abs_max of the selected sample for display normalization. Geometry and texture encoding must agree exactly.

### Neural run v1

`POST /api/neural/runs` accepts prompt<=400characters,max_new_tokens1–16 (default12),chat-templated input+generation<=128. Returns202 `{id,status,events_url}`. One GPU job at a time; reject concurrent work with503/429 rather than queue indefinitely. Reuse the existing execution lock across old and new APIs.

`GET /api/neural/runs/{id}/events`: resumable SSE with monotonic integer event IDs and JSON data. Event types `start`, `frame`, `done`, `error`, `cancelled`. `start` defines actual architecture,capture order,dtype,request,model provenance. `frame`:index,token_id,token_text,cumulative output,input_token_count,elapsed_ms,inference_ms,capture_ms,activation_b64. Activation payload is little-endian float32 in layer-major order:6144 MLP values then2048 residual values per layer. Source BF16 converted to float32 for transport, not normalized measurements. One frame=one completed real autoregressive forward, before the prediction becomes input. `done` reports actual output/metrics. Preserve raw values; normalization belongs to presentation.

`POST /api/neural/runs/{id}/cancel` signals cancellation between bounded forwards. `GET /api/neural/runs/{id}` exposes lifecycle/progress. `GET /api/neural/weights?layer=13&budget=2000000` returns binary envelope. `GET /api/neural/status` exposes readiness/budgets. Reconnect resumes after Last-Event-ID; bounded retained runs/TTL. No model paths from HTTP. Explicit errors; no toy fallback.

## Product boundaries and budgets

### User hardware constraint (latest directive)

Default model device is CPU, with no automatic CUDA selection. Large CPU model loads require an available-RAM guard. The default renderer uses a low-power context, pixel ratio at most one, a 15 Hz cap and on-demand rendering. Eco mode uses a 50K connection budget with disclosed point subsampling. The 2M and 4M modes are opt-in capacities whose GPU performance remains unverified. Weight inspection reads a selected safetensors matrix without loading the full model. Inference starts only on an explicit run request.

Bound all inputs,recorded frames,concurrent work,retained runs,event history and weight cache. Target active+two history runs with total retention<=96MiB and10-minute TTL; weight-cache<=32MiB. Cancellation must release locks and terminate worker within the current forward plus bounded cleanup. Resource counters must remain stable through replacement/dispose. Cap scene pixel ratio1.5. Target<100 draw calls; test2M edges/229K units independently from inference latency. A physical60FPS guarantee is not assumed; profile and expose budget controls. Handle context loss/unsupported WebGL and reduced motion. No arbitrary upload/checkpoint execution or remote hosting.

## Acceptance

Independent hook fidelity against real Qwen module values; exact binary sample-index/weight tests; malformed/truncated protocol rejection; run lifecycle/concurrency/cancel/reconnect/retention tests. Actual2M draw workload and229,376 units must be observed, not just a label. Browser tests cover live generated input/output, pause/replay, real selection/filter/budget controls, shader errors and resource replacement. Keep the prior tensor lens as a secondary precise inspection view. Existing contracts/tests must continue passing.
