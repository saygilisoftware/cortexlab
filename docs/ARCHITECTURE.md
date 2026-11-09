# Architecture accepted before implementation

The original tensor-lens contract below remains supported. Dense neural architecture is separately versioned in [NEURAL_ARCHITECTURE.md](NEURAL_ARCHITECTURE.md): independent capture/binary transport/job registry, resumable SSE, bounded history/cache and resource-aware controller/renderer. Latest policy is CPU default with10GiB available-RAM gating and Eco15Hz/on-demand graphics. Dense GPU validation remains pending; see NEURAL_TESTING.md.

`prompt -> tokenizer -> PyTorch causal decoder -> capture/serializer -> session JSON -> HTTP -> replay store -> Three.js renderer + numerical inspector`

The algorithm never imports a rendering dependency. The renderer never generates measurements. The session is the extension seam for future pretrained/TransformerLens adapters. Replay uses captured values without running Torch.

## Contract v1

Top-level `schema_version: "cortexlab.session/v1"`, `id`, `created_at`, `model`, `request`, `runtime`, `capture`, `frames`, `metrics`, `output`.

`model`: `id`, `name`, `layers`, `heads`, `d_model`, `context_limit`, `vocab_size`, `parameter_count`, `trained`, `seed`, `checkpoint_sha256`, `tokenizer`, `training` provenance/metrics object.

`request`: `prompt`, `max_new_tokens` (1–8), optional `ablate: {layer,head}|null`. Greedy decoding, no sampling-temperature claim. Prompt <=400 characters; token count plus generation must fit context 48; unsupported words become explicit `<unk>` tokens.

`runtime`: `torch`, `device`. `capture`: text descriptions of attention/activation semantics and dtype.

Each frame: `index`, `elapsed_ms`, `inference_ms`, `input_ids`, `tokens`, `predicted_id`, `predicted_token`, `top_predictions: [{token,probability}]`, `layers: [{index, attention, residual, norms}]`.

Axes: attention `[head][query][key]`, residual `[token][channel]`, norms `[token]` = L2 norm of **post-block residual**. Attention = causal masked post-softmax probabilities, before optional head-output ablation. Ablation zeros selected head's weighted value output before concatenation/projection, not its attention matrix. `metrics`: measured `total_ms`, `model_ms`, `capture_ms`; any additional metric must be named. The predicted token is not yet an input activation in that frame. Frames retain complete bounded tensors; display filtering never rewrites them.

`POST /api/infer`: validated request -> complete recording. `GET /api/health`: operational status. `GET /api/model`: model metadata. `GET /api/demo`: checked-in real recording. `/` serves local frontend and dependencies. No token-streaming claim. API requests serialized with a bounded execution lock. No arbitrary checkpoint upload or code execution. Bind loopback by default. Import is bounded JSON and must validate shape, finite values, probability semantics and version before changing state.

## Scene mapping

Schematic token axis and model-depth axis; each node is a token residual vector, not a neuron. Hue encodes block depth (cyan to violet); brightness and unselected sphere scale encode L2 norm divided by the maximum norm across all layers/tokens in the current frame. Gold/enlarged selection overrides that encoding. Thin vertical lines identify the same token across blocks; they are schematic identity traces, not attention or identified circuits. Selected layer/head/query produces actual query-to-key attention arcs, with direction arrowheads and weight-dependent color/intensity. Exact row weights and residual channels remain in a numerical inspector. Scene camera/grid are presentation. Node transforms/colors ease for 220ms between selections/frames; numeric inspection always uses exact selected-frame values. Reduced-motion preference disables transitions.

Camera orbit/pan/zoom, focus, presets; token/layer/head selection; attention visibility/threshold; timeline, play/pause/speed; prompt/run; import/export are implemented and browser-tested. Baseline comparison uses the first recorded frame only, with matching input IDs and checkpoint hash; later generated prefixes may diverge, so they are not compared as equivalent states. Compact JSON export is necessary to keep the maximum capture inside the 5MiB import budget. ASGI input is limited to 16KiB before JSON parsing, including chunked bodies; concurrent inference fails clearly with503.

## Budgets and risks

Initial targets: <=48 context, 3 blocks, 4 heads, width64; <=144 residual nodes; <=48 selected-query arcs; <100 draw calls; warm inference <2s; session <5MB; visible-tab p95 frame interval <=25ms, 60FPS aspiration. Targets are not measured results. CPU/GPU utilization and full GPU memory require separate instrumentation; do not infer them from renderer counts.

Full prefix recomputation simplifies truthful attention capture, at cost O(generation × layers × heads × context²) recorded attention. No KV cache. Exact rerun depends on runtime; JSON playback does not. See `DECISIONS.md` for tradeoffs and `TESTING.md` for gates.
