# Research synthesis

Investigation: 2026-11-06. The eight specialist-role reports in `agents/` were reviewed before production implementation. Three real subagents covered A/H, B/C, and D/E/F/G. No upstream benchmarks or tests were run.

## Verified findings

0xLane source separates observations, processing, schema-versioned world state and an independently scheduled Three.js viewer. Its local checkout was read at commit `ee07d93025e4a2741c932cfcd482a628bdb960f0`; traceon-replay source adds bounded JSON validation and explicit replay semantics. Their code is not copied. See `agents/context-report.md` for source-level evidence.

Netron inspects serialized model graphs. TransformerLens instruments real activations. BertViz inspects head/model/neuron attention. Transformer Explainer already executes GPT-2 with user text in the browser. circuit-tracer supports feature attribution graphs and interventions. Therefore neither live attention nor model inspection is a novelty claim. Evidence and licenses: `COMPETITORS.md`, `RESEARCH_SOURCES.md`, and `agents/research-report.md`.

The transformer attention operation and causal masking are well established. Captured attention is a routing distribution, not proof of causal importance or reasoning. Full per-head attention and named residual hook points are required for an honest display. Independent arithmetic and causal-invariance tests will verify capture.

## Accepted engineering deductions

- Default to a small, locally trained causal transformer and original corpus. This avoids model downloads, ambiguous weight licenses and silently random language-model output. Its competence must be measured and clearly limited.
- Record complete generation-step snapshots. HTTP completion plus browser replay is sufficient for bounded sessions; do not call this token streaming or layer execution profiling.
- Map token position and model depth to schematic 3D axes. Link each mark to exact tensor indices and numeric inspection. Geometry is not an embedding projection.
- Prefer direct Three.js/WebGL2 to React/R3F/WebGPU for this bounded MVP. Preserve a renderer boundary and measure before optimizing.

## Hypotheses, not established results

Spatial token/layer inspection plus portable, provenance-bearing replay may be useful for learning and debugging. It is not proven unique, scientifically novel, easier than BertViz, or faster than competing tools. A future usability study should compare task accuracy and completion time against a matrix view.

## Phase-zero gate

PASS for available context: workspace/runtime inventory, source investigation, competitor landscape, capture semantics, rendering tradeoffs and risks are documented. Shell network access is denied; GPU is verified through Torch as NVIDIA GeForce RTX 4060 Ti. Browser UI automation ended because its policy host could not establish the current URL, so no further native UI actions are attempted. Dependency and browser-test availability remain implementation risks, not verified capabilities.
