# Security

CortexLab is a local research instrument. Bind it to `127.0.0.1`. Public hosting, authentication, tenant isolation, and hostile multi-user workloads are outside the initial MVP scope.

Recordings contain prompts, token labels, model outputs, and internal activations. Review them before sharing. Local export does not redact prompt content automatically. Do not enter secrets into shared demos.

Treat imported recordings as untrusted data. Accept only the documented, bounded JSON session format with schema, shape, and finite-number validation. Never load uploaded Python, pickle objects, or arbitrary checkpoints. Model adapters should use known local artifacts and explicit provenance.

For a vulnerability, use the repository's private GitHub security advisory channel if enabled. Otherwise contact the maintainer through a private channel before publishing exploit details. Do not include credentials, private prompts, or sensitive recordings in public issues. No response-time commitment or hosted security service is implied.
