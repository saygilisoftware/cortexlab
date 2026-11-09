# Qwen3-1.7B demonstration

CortexLab's optional Qwen profile is intended to inspect a locally downloaded **Qwen/Qwen3-1.7B** checkpoint. The small bundled CortexLab model remains a separate demonstration. A successful small-model or random tiny-Qwen test does not establish successful pretrained 1.7B execution.

## Setup

Use Python 3.10+ and install the optional dependencies:

```powershell
python -m pip install -r requirements-qwen.txt
```

The integration targets installed Transformers 5.19.0, tokenizers 0.23.1, safetensors 0.8.0, and huggingface-hub 1.33.0. Direct single-device loading omits `device_map`, so Accelerate is not required; the automatic/offload placement examples on the model card use a different loading path.

Download the official snapshot in a terminal with network access:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-1.7B', local_dir='models/Qwen3-1.7B', allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer*','merges.txt','vocab.json','LICENSE','README.md'])"
```

This downloads weights and tokenizer/configuration data without Python source or pickle `.bin` checkpoints. For reproducible research, add `revision='<full reviewed repository commit>'` and retain the resolved revision and file hashes. `main` can change. Keep upstream LICENSE and any NOTICE alongside the snapshot. The model uses [Apache-2.0](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE); CortexLab's MIT license does not relicense it.

The default path is `models/Qwen3-1.7B`. An alternate local snapshot can be selected before launch:

```powershell
$env:CORTEXLAB_QWEN_PATH = 'D:\models\Qwen3-1.7B'
python scripts/qwen_demo.py
```

Without the environment override, run `python scripts/qwen_demo.py` from the repository root. The recorded GPU experiment produced Paris/Tokyo. The current default is CPU with an available-RAM guard. Select the Qwen profile after loading completes, enter a short prompt, and run capture. Downloaded pretrained weights are excluded from the source release. Keep `models/` ignored by version control.

## What the capture represents

The official [model configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json) specifies 28 decoder blocks, 16 query attention heads, 8 key/value heads, hidden width 2048, and attention head width 128. The observatory records a **selection**, not a smaller replacement model:

| Measurement | Captured selection |
| --- | --- |
| Original decoder-layer IDs, zero based | 0, 13, 27 |
| Original query-head IDs, zero based | 0, 5, 10, 15 |
| Residual channels | 64 explicitly identified raw channels |
| Residual norm | L2 over the full 2048-channel post-block vector |
| Session limit | At most 64 context tokens and 4 generated tokens |

All 28 layers still execute. The 64-token cap is an observatory capture limit, not the model's advertised context capacity. Head IDs refer to query heads, not the eight grouped key/value heads. The recorded channel IDs and full-width norm must remain explicit in session metadata; a 64-channel panel is not the full residual vector.

Eager attention returns actual masked, post-softmax query-to-key weights. In evaluation mode dropout is disabled. Post-block residual hooks capture after the block's MLP residual addition and before the model's final RMSNorm. Playback changes presentation time, not measured inference time. Attention is an observation, not a causal explanation or a visualized chain of thought. Qwen head ablation is unsupported in this integration.

## Resources and current limitations

Weights alone require approximately 3.4 GB at two bytes per parameter, or 6.8 GB at float32, using the rounded 1.7B name as an estimate. Runtime overhead, loading conversions, activations, and temporary buffers add memory. These are estimates, not observed peak memory or a hardware guarantee. Eager full-prefix capture favors inspectability over generation throughput. CPU runs can be slow; avoid inferring pretrained-model latency from the tiny bundled model.

Loading should use local files, safetensors only, `trust_remote_code=False`, evaluation mode, and inference mode. Preserve the actual dtype/device, tokenizer/chat-template settings, model revision or file hashes, and selected capture IDs in recordings. If a chat template is used, count its added tokens against the context limit and record whether thinking is enabled. A four-token generation budget usually cannot produce a complete conversational answer.

Recorded Qwen inference and browser validation results are in `examples/qwen-validation.json` and `examples/qwen-browser-validation.json`. The current default is CPU, with a 10 GiB available-RAM guard. Full dense capture and rendering are tracked separately in [Neural Testing](NEURAL_TESTING.md).
