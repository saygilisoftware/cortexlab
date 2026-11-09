# Reproducible demonstration

Follow the repository README's installation instructions, then run `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000` from the checkout. Open http://127.0.0.1:8000. The Python API serves native browser modules; Node is not required at runtime. The included real recording loads automatically: `a comet is cyan . the comet`, requested6 tokens, EOS after3 captured frames, continuation `pulses.`.

Actual screenshots: [desktop](images/cortexlab.png), [390px maximum-context layout](images/cortexlab-mobile.png). [Sampled animated walkthrough](images/cortexlab-demo.gif) was sampled from38 actual browser screenshots (22 encoded frames after identical-frame merging) with measured sampling intervals, not a rendering-FPS demonstration or generated mockup. Recreate it with `python scripts/capture_demo.py --channel chrome` (or Playwright Chromium without `--channel`).

## Walkthrough

1. Open the local application and read the displayed model identity and training limitations.
2. Load the bundled recording. It contains captured inference, so replay does not require rerunning PyTorch.
3. Choose a recorded generation step. Select a layer, attention head, and query token. Compare the displayed links with the exact attention row in the inspector.
4. Select a token residual node. Inspect its post-block vector and L2 norm. Its position encodes token and layer, not a learned embedding coordinate or a single neuron.
5. Scrub the timeline, then play and pause. Playback speed changes viewing time; recorded inference timings remain separate measurements.
6. Enter a short prompt supported by the demonstration vocabulary and run inference. Unsupported tokens must be visible as `<unk>` rather than silently interpreted as known words.
7. Export the recording and import it again. Confirm that the selected step's values remain the same.

## Interpretation

Attention links encode post-softmax query-to-key probabilities for the chosen layer and head. Residual-node color/scale encode a named activation statistic. Camera movement, grid, lighting, and eased transitions are presentation. Attention alone does not establish causal attribution or model reasoning.

Node hue encodes block depth; size/brightness encode frame-relative L2 norm. Selected gold overrides that mapping. Vertical traces identify the same token across blocks, not attention. Exact numeric values snap to the recorded step while presentation transitions ease for220ms; reduced motion disables those transitions.

Controlled intervention is implemented and browser-tested: choose block01/head02, retain the same prompt, check “Zero selected head output” and run. The comparison uses the first frame with matching input IDs/checkpoint; downstream generated prefixes can diverge. Attention in the intervention block is captured before the output is zeroed, so unchanged weights plus changed residuals are expected. This is not an automatic causal circuit explanation.

See [the architecture](ARCHITECTURE.md), [testing](TESTING.md), [model card](MODEL_CARD.md), and [performance](PERFORMANCE.md) for the actual data and its limits.
