"""Record real trained-model inference, independently of the frontend."""
import json
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from backend.engine import Inspector, ROOT, validate_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Run 3 warmups and 30 measured engine recordings")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.benchmark:
        inspector = Inspector()
        for _ in range(3): inspector.inspect("the cyan", 8)
        sessions = [inspector.inspect("the cyan", 8) for _ in range(30)]
        result = {
            "scenario": {"prompt": "the cyan", "max_new_tokens": 8},
            "device": "cpu", "torch": torch.__version__, "threads": torch.get_num_threads(),
            "samples": 30, "frames_per_session": [len(session["frames"]) for session in sessions],
            "total_ms_median": statistics.median(session["metrics"]["total_ms"] for session in sessions),
            "total_ms_p95": sorted(session["metrics"]["total_ms"] for session in sessions)[28],
            "model_ms_median": statistics.median(session["metrics"]["model_ms"] for session in sessions),
            "capture_ms_median": statistics.median(session["metrics"]["capture_ms"] for session in sessions),
            "bytes_median": statistics.median(len(json.dumps(session, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for session in sessions),
            "scope": "Engine timing only; excludes HTTP transfer and browser rendering. Warm CPU float32, 3 warmups. P95 nearest-rank.",
        }
        (ROOT / "examples" / "backend-benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(0)
    config = json.loads((ROOT / "examples" / "demo-config.json").read_text(encoding="utf-8"))
    recording = validate_session(Inspector().inspect(**config))
    target = ROOT / "examples" / "demo-session.json"
    target.write_text(json.dumps(recording, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"path": str(target), "bytes": target.stat().st_size, "frames": len(recording["frames"]), "output": recording["output"], "metrics": recording["metrics"]}, indent=2))
