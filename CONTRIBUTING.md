# Contributing

CortexLab makes captured model state inspectable. Contributions should preserve the distinction between measured tensors and presentation animation.

1. Read [the architecture](docs/ARCHITECTURE.md) and [decisions](docs/DECISIONS.md).
2. Follow the README installation procedure, then run `python -m pytest`.
3. Keep model adapters, session serialization, and rendering independent. Preserve the versioned contract and document migrations.
4. Add meaningful validation for changed computation: tensor shapes, attention masks and normalization, capture fidelity, and recording roundtrips. Inspect graphics changes in the running browser.
5. Describe the problem, resulting behavior, commands run, and remaining limitations in your pull request. Include actual screenshots for visual changes.

New model adapters must identify model/checkpoint/tokenizer versions, licenses, tensor axes, capture points, device/dtype, and unsupported operations. Missing capture data must remain distinguishable from zero values. Do not label attention as causal importance or fabricate intermediate execution states.

Use small fixtures and deterministic seeds. Do not commit credentials, private prompts, downloaded model caches, or unrelated generated artifacts. Keep dependency notices current. Submissions are licensed under the repository's MIT license; only submit work you have permission to contribute.
