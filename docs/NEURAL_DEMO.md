# Dense neural observatory demonstration

This lens is designed to show real **last-input-token** Qwen activations and a measured subset of a selected MLP down-projection matrix. It uses the same locally loaded Qwen model as the tensor inspector. The exact implementation and verification status is tracked in [NEURAL_ARCHITECTURE.md](NEURAL_ARCHITECTURE.md), [TESTING.md](TESTING.md), and [the independent review](agents/neural-review.md).

**Checkpoint status:** dense-module implementation and validation are in progress. Full neural-frame capture and GPU rendering of 229,376 units/two million connections are unverified. This is a planned demonstration procedure, not evidence that every control already works. Development review is CPU-only; the server/panel stays stopped.

## Current hardware boundary

The default is CPU execution with an available-RAM guard. The Eco renderer uses 50,000 connections, disclosed point subsampling, a pixel ratio at most one and a 15 Hz cap with on-demand rendering. Higher-density modes are opt-in and await performance validation. A selected safetensors matrix can be inspected without loading the full model.

## Prerequisites and launch

Install the base/optional Qwen dependencies and download the official safetensors snapshot as described in [QWEN_DEMO.md](QWEN_DEMO.md). Keep weights in `models/Qwen3-1.7B` or set `CORTEXLAB_QWEN_PATH` to an existing local snapshot before starting the application. Use the main README's local-server launch command. Model weights remain outside the source release.

## Walkthrough

1. When explicitly starting a demonstration on suitable hardware, open the dense neural lens. Check model identity, readiness, and displayed capture semantics; opening the page must not preload inference weights or choose CUDA automatically.
2. Enter a short prompt and run. The chat-template tokens plus generation must fit the 128-token observatory limit. The run captures at most 16 generated tokens; this cap is not the model's full capability.
3. Observe token output as completed forwards arrive. Each frame contains 6144 MLP intermediate values and 2048 residual values per block for the **last input position**. Generated text and activation frames must come from the same run ID.
4. Select a block and inspect its down-projection weight lens. Begin at the Eco 50,000-connection budget. Higher budgets require an explicit choice on suitable hardware. Read sampled, submitted, and displayed counts separately from the 12,582,912 full-matrix weights; an uploaded buffer or capacity label does not prove successful rendering.
5. Inspect activation values and sign/magnitude encodings. Visibility and threshold controls change presentation, not the captured tensors or model weights.
6. Pause/replay captured generation steps. The recorded timings remain execution measurements; playback speed and smooth transitions are viewing choices.
7. Cancel a running job or disconnect/reconnect. Cancellation is cooperative between forwards; an in-progress forward or initial model load can finish first. Cancelling generation does not itself unload the cached model. Reconnection should resume retained event IDs rather than rerun inference.

Only demonstrate controls that exist and have passed the current validation build. The earlier precise token/layer tensor lens remains useful for attention and selected residual-channel inspection; the dense weight lens is a different measurement.

## Reading the scene correctly

There are 172,032 MLP intermediate units and 57,344 residual channels across 28 blocks: 229,376 activation sites in total. These are not 1.72 billion neurons. The model's parameter count includes scalar learned weights that do not each correspond to an independently displayed unit.

The MLP capture is the input to `down_proj`: `SiLU(gate_proj(x)) * up_proj(x)`. A connection's signed value is `down_proj.weight[target_residual_channel, source_mlp_unit]` from one selected block. It is not an attention probability. The residual endpoint also includes the residual stream carried into the MLP, so a connection alone does not explain the full residual activation. Spatial positions are schematic.

The revised coprime modular policy uses `flat=(edge_index*stride)%total_connections`, with stride the smallest integer at least `ceil(total/count)` that is coprime to the matrix size. A two-million sample uses stride seven and covers all source units and target channels; this is no longer the default. Coverage does not imply uniform target-row counts or statistical representativeness. The binary header, CPU sampler, and shader must agree on this policy. Thresholding or activation-dependent emphasis must remain clearly identified as presentation derived from measured values.

## Performance and limits

Eco 50,000 connections is the default. Two/four-million modes are optional implementation capacities whose GPU workload has not been tested on this machine. Browser frame time, inference latency, capture transfer time, and model/GPU memory are separate metrics. A disconnected browser does not necessarily cancel a bounded job; use the explicit cancel action. Completed run history is local, bounded, and temporary rather than durable experiment storage unless an export feature is explicitly provided.

No screenshot, workload count, latency, cancellation guarantee, or production-readiness claim should be inferred from this walkthrough alone. Use the measured validation report for the tested hardware/build and remaining limitations.
