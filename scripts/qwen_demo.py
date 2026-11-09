"""Record real local Qwen3-1.7B inference; no download or toy fallback.

Stop any GPU-loaded CortexLab server before running a separate recording process
on an 8 GiB GPU. For normal use, select Qwen in the running web interface.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.qwen_adapter import QwenInspector, local_model_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Check local files without loading weights")
    parser.add_argument("--prompt", default="Name the capital of France. Answer with one word.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    status = local_model_status()
    if args.status:
        print(json.dumps(status, indent=2)); return
    if not status["weights_available"]:
        raise SystemExit(status["reason"] + "; see docs/QWEN_DEMO.md")
    inspector = QwenInspector()
    recording = inspector.inspect(args.prompt, args.max_new_tokens)
    target = ROOT / "examples" / "qwen-demo-session.json"
    encoded = json.dumps(recording, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 5 * 1024 * 1024:
        raise SystemExit("Capture exceeds the 5 MiB browser import budget; shorten the prompt")
    target.write_text(encoded, encoding="utf-8")
    print(json.dumps({"recording": str(target), "model": recording["model"]["id"], "parameters": recording["model"]["parameter_count"], "device": recording["runtime"]["device"], "output": recording["output"], "frames": len(recording["frames"]), "metrics": recording["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
