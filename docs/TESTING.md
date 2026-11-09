# Verification evidence

The latest CPU-only revision passed 79 Python tests and 69 JavaScript tests. All seven browser modules parse. Historical GPU tensor-lens results preceded the current CPU, RAM and Eco resource policy. See [Neural Testing](NEURAL_TESTING.md) for the scope of current verification.

Local validation:2026-11-06. Windows, Python3.10.0, Torch2.6.0+cu124 running CPU float32/one thread; Node24.13.1; Python Playwright1.59.0; Chrome156.0.8078.80. Real WebGL2 through ANGLE/NVIDIA RTX4060Ti/D3D11. No upstream projects were modified or executed.

## Core commands

| Command | Observed result |
|---|---|
| `python scripts/train.py` | 500 updates; held-out loss4.08848→0.866016; accuracy57.84% vs18.65% majority baseline |
| `python scripts/record_demo.py` | Genuine richer demo:3 frames,8/9/10 input tokens, continuation `pulses.` |
| `python scripts/record_demo.py --benchmark` | 3 warmups +30 measured recordings; raw `examples/backend-benchmark.json` |
| `python -m pytest -q` | 32 tests pass (6 engine +26 API cases); final duration logged by run |
| `node --test tests/contracts.test.mjs` | 11 contract tests pass |
| `npm run check` | Native frontend modules parse; no npm dependencies/install required |
| `python scripts/e2e.py --channel chrome` | 16 real-browser acceptance checks pass; own server started and stopped |
| `python scripts/capture_demo.py --channel chrome` |38 actual screenshots,22 encoded frames saved as5,048,389-byte GIF; no page errors |
| `python scripts/package_release.py` | Clean source archive created; required model/vendor license included; ZIP CRC passes |

An additional extracted-archive smoke check started the packaged server from a separate workspace staging directory, loaded the checkpoint, received200 from UI/JavaScript/vendor/demo routes, and executed real inference. Functional assertions passed. Windows cleanup of the generated staging directory failed, and automatic approval rejected a recursive removal because sandbox approval is disabled. The leftover `cortexlab-release-check-*` directory is ignored and excluded from release packaging; it is unrelated to the working application or any other project.

Core tests independently implement attention in NumPy, checking both probabilities and projected output against Torch. They verify future-token causal invariance, capture/noncapture logit equality, per-head shapes, exact capture fidelity within one run, same-runtime repeat determinism, prompt dependence, actual head ablation, held-out recomputation from weights, checked-in demo consistency with numeric tolerance, JSON round trip and bad inputs. API tests cover malformed types, 48-token context, generation bounds, busy503, 16KiB raw-body limit413 including missing/incorrect length and chunked ASGI messages, plus the exact body-size boundary.

Browser contract tests reject unsupported version, bad masks/sums, wrong shapes/norms/token labels, non-finite values, changed context and invalid intervention indices. Import is bounded to5MiB before parsing; file text is rendered as text, not HTML. Browser E2E also confirms an invalid import preserves the existing session.

## Real browser flows

Raw report: `examples/browser-validation.json`. Verified initial real recording/visible WebGL canvas, visible prompt submission through HTTP/PyTorch, exact numeric attention/norm inspection, actual projected-node mouse click, orbit drag/presets/focus, attention matrix and threshold/visibility, timeline growth/play/pause/speed, real file download/reimport, bad causal import rejection, controlled ablation with matching baseline, guide, stable GPU resource counts over30 replacements, full48-token request budget (40 input +8 requested predictions,141 rendered residual nodes at final47-token prefix), compact export/reimport under5MiB, reduced motion,390px no horizontal overflow and no-WebGL tensor-inspector fallback. Normal supported-mode flow had no page/console/shader/failed-asset errors. The no-WebGL test intentionally removes WebGL access and expects the fallback.

Screenshots: `docs/images/cortexlab.png` (actual1600px desktop), `cortexlab-mobile.png` (actual390px responsive stress state). Animated capture:`cortexlab-demo.gif`,38 sampled screenshots resized to1000×750, preserving measured sample pacing. These are application captures, not generated mockups. Model and selected state are visible in the image. The earlier exploratory screenshot is not release evidence. Playwright's initial WebM attempt required an unavailable Playwright-managed encoder; the final verified capture uses installed Pillow to assemble actual screenshots, without downloading or changing system software.

## Limits

Configured GitHub Actions targets Python3.10/3.12 on Ubuntu with CPU Torch, Node22 contracts and Playwright Chromium. It has **not** run remotely. Fresh package installation was blocked by shell/browser network policy in the development session, so clean online installation is not claimed. The implementation was validated against installed pinned direct dependencies. No Safari/Firefox/real mobile hardware, accessibility user study, wheel distribution, remote service penetration test, exhaustive GPU-memory/utilization profile or cross-platform retraining experiment was performed.

Initial task scope is a verified local vertical slice. Follow `ROADMAP.md` before a broad public stable release. `SECURITY.md` describes local-only boundaries.
