# Third-party provenance

CortexLab's original source, authored demonstration corpus, and locally trained demonstration checkpoint are provided under [MIT](LICENSE). External software and model artifacts retain their own licenses. Installed packages include their own notices; preserve these when redistributing packages or binaries.

## Architectural references

- [0xLane](https://github.com/YourAnonELF/0xLane), inspected at local commit `ee07d93025e4a2741c932cfcd482a628bdb960f0`, declares AGPL-3.0-or-later. Its domain-state boundary and independently scheduled renderer informed conceptual design. No implementation code or assets were copied.
- [traceon-replay](https://github.com/YourAnonELF/traceon-replay) declares Apache-2.0. Its versioned recording and safe plain-data replay informed conceptual design. No implementation code or assets were copied.

See [repository research](docs/agents/context-report.md) for inspected files. Reference licenses do not replace the licenses of actual CortexLab dependencies.

## Runtime dependencies

| Dependency | Pinned version | License / source |
| --- | --- | --- |
| Three.js and OrbitControls | 0.170.0 (r170) | MIT; copyright 2010-2024 Three.js authors. Full [vendored license](frontend/vendor/THREE-LICENSE.txt) and [provenance/hashes](frontend/vendor/PROVENANCE.md). |
| PyTorch | 2.6.0 | BSD-3-Clause; [official license and copyright holders](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE). The validated environment contains 2.6.0+cu124; computation defaults to CPU. |
| FastAPI | 0.115.12 | MIT; [source](https://github.com/fastapi/fastapi). |
| Pydantic | 2.11.4 | MIT; [source](https://github.com/pydantic/pydantic). |
| Uvicorn | 0.34.2 | BSD-3-Clause; [source](https://github.com/encode/uvicorn). |
| pytest (validation) | 9.0.3 | MIT; [source](https://github.com/pytest-dev/pytest). |
| HTTPX (validation) | 0.28.1 | BSD-3-Clause; [source](https://github.com/encode/httpx). |

Python package versions and license fields/notices were checked against installed distribution metadata on 2026-11-06. The full Three.js notice is included because its source is vendored. Other packages are installed separately and carry their own full notices and transitive dependencies. Preserve all relevant package notices if redistributing a combined binary/environment. Do not interpret CortexLab's MIT license as relicensing those dependencies. Browser automation also uses Playwright and a browser installation; retain their distribution notices when bundling them.

## Model and data

`examples/neural-weight-sample.bin` contains50,000 measured scalar weights from block13's MLP down projection of the user's locally downloaded Qwen/Qwen3-1.7B snapshot. This is a deterministic coprime sample, not original MIT model data. The model is Apache-2.0; its full supplied license is preserved as [examples/QWEN-LICENSE.txt](examples/QWEN-LICENSE.txt). `examples/neural-weight-validation.json` records a separate2M CPU extraction check. No complete Qwen checkpoint is included in the source archive. Genuine Qwen tensor recordings retain model provenance and derived-data interpretation limits. Preserve the model license when redistributing those artifacts.

The original Cortex Tiny demonstration is a small task-specific model. The optional Qwen profile uses third-party pretrained weights and keeps their Apache-2.0 license distinct. Check each recording's model metadata and artifact hashes. User-imported recordings and other future checkpoints require separate provenance review.
