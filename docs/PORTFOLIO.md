# CortexLab portfolio notes

CortexLab is an interactive 3D observatory for a small PyTorch transformer. Its engineering objective is to make captured attention and residual activations directly inspectable across layers and generation steps.

## Technical story

- A computation-to-state boundary keeps model instrumentation independent of graphics.
- Versioned recordings preserve tensor values, capture semantics, and model provenance for replay.
- A token-by-layer Three.js scene provides spatial context, while numerical inspection preserves precision.
- Correctness checks should establish attention masking/normalization, tensor capture fidelity, deterministic controlled runs, and API/recording consistency.

These boundaries are implemented and verified locally. The current validation includes 32 Python tests, 11 JavaScript contract tests and 16 real-browser acceptance checks. See TESTING for methods and limitations; remote CI and clean installation remain pending.

## Positioning

Netron, BertViz, Transformer Explainer, TransformerLens, and circuit-tracer already cover important visualization or interpretability workflows. CortexLab's proposed contribution is a compact local workflow combining a real capture adapter, spatial inspection, and portable replay. This is a product-design hypothesis, not a claim of an unprecedented algorithm or scientific discovery. See [competitor research](COMPETITORS.md).

Present actual screenshots from the running application and describe their prompt, model/checkpoint, selected layer/head/query, and replay step. Link reproducible commands alongside measured performance. Do not promise general language understanding, arbitrary-model support, causal explanation, or a universal rendering frame rate.

## Suggested repository description

Interactive 3D inspection and replay of genuine PyTorch transformer attention and activations.

## CV entry (English)

**CortexLab — Interactive 3D AI Model Observatory | Python, PyTorch, Three.js, WebGL2, FastAPI**

- Built an end-to-end transformer inspection platform capturing per-head attention and residual activations into versioned recordings, with direct 3D picking, exact tensor inspection, timeline replay and JSON round trips.
- Implemented a trained 158,912-parameter offline fixture and controlled head-output interventions; verified attention arithmetic, causal masking and capture fidelity with 32 Python tests, 11 JavaScript contract tests and 16 browser acceptance checks.

Optional measured detail if space permits: “Validated a 141-node scene at 64 draw calls on the tested RTX 4060 Ti browser environment.” Avoid quoting headless scheduling as universal FPS or representing the educational fixture as a general LLM.

## CV özeti (Türkçe)

PyTorch transformer modellerinden gerçek attention ve aktivasyon verilerini yakalayan; Three.js ile etkileşimli3D inceleme, kayıtlı çıkarım oynatma ve kontrollü attention-head müdahalesi sunan uçtan uca açık kaynak araç geliştirdim. Matematiksel doğruluk, API ve arayüz akışlarını32 Python testi, 11 JavaScript sözleşme testi ve 16 tarayıcı kabul kontrolüyle doğruladım.

Do not imply publication until an actual remote repository exists. Add its real URL after publishing. The repository includes actual screenshots, a browser capture script and reproducible test commands for reviewers.
