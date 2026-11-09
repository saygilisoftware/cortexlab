# Roadmap and quality gates

The original tensor-lens gates below record historical MVP verification. The dense Neural Atlas has implemented APIs, controller, renderer and CPU tests. Dense visual integration, full pretrained neuron-frame validation and GPU performance checks remain pending. See [Neural Testing](NEURAL_TESTING.md).

## Delivered local MVP

| Gate | Status | Evidence |
|---|---|---|
| 1 Research validated | PASS within available access | Six shared research docs; eight role reports; actual source inspection |
| 2 Product selected | PASS | Fixed CortexLab directive; eight comparative concepts evaluated in PROJECT_SELECTION |
| 3 Architecture accepted | PASS | Versioned tensor contract, bounded HTTP, exact capture/replay semantics and budgets |
| 4 Algorithm proven | PASS | Trained checkpoint; independent attention oracle, causality/capture/learning tests |
| 5 Visualization integrated | PASS | Real tensors drive interactive WebGL2; exact numerical links and actual node click tested |
| 6 Product usable | PASS locally | Source startup, prompt, controls, replay, import/export, intervention;16 browser acceptance checks |
| 7 Release quality assessed | PASS for local MVP, broader release pending |32 Python +11JS tests, screenshots, raw benchmarks, license/provenance and explicit limits |

The repository has not been pushed or publicly deployed. CI configuration exists but remote CI/clean online installation have not run.

## Next milestones, in order

1. **Clean release validation:** run fresh CPU installation and configured Ubuntu3.10/3.12 CI; verify upstream vendor byte identity or replace with official integrity-checked npm distribution on a network-enabled machine. Test Firefox/Safari and real mobile GPU. Owner:maintainer; acceptance:repeatable clean setup and documented matrix.
2. **Pretrained adapter:** choose a pinned, licensed safetensors model; verify tokenizer/eager-attention capture, capture fidelity and memory caps. Keep the original offline fixture as the default. Owner:AI/backend; depends on recording conformance; acceptance:independent tensors match adapter output, no remote code/checkpoint execution.
3. **Comparison workspace:** store named baselines and align tokens explicitly; show attention/residual/logit deltas for equivalent inputs and controlled interventions. Current comparison is first-frame only. Owner:UX/AI; acceptance:alignment and interpretation tests.
4. **Longer execution:** cancellable, sequence-numbered streaming using the same session frame schema, bounded storage/backpressure and disconnect behavior. Owner:systems; acceptance:recorded replay matches streamed events exactly.
5. **Usability and accessibility:** compare3D versus matrix on inspection tasks; keyboard scene navigation, screen-reader descriptions, larger text/density presets and reduced-vision contrast review. Owner:product; acceptance:user-task data, not aesthetics alone.
6. **Scale based on profiling:** measure request-to-display, CPU/GPU utilization/memory, JSON parse costs and large-model capture. Investigate sparse views/binary recording/KV capture only when measurements justify complexity. Owner:graphics/systems; acceptance:budgets on named supported hardware.
