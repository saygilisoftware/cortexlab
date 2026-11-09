"""Real pretrained Qwen browser test; opt-in, requires local official weights.

Uses an existing server so the verified model remains warm for interactive use.
"""
import argparse
import json
import math
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from e2e import ROOT, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--channel")
    args = parser.parse_args()
    url = args.url.rstrip("/"); wait_ready(url)
    # Do not turn a verification command into an unsafe automatic device load.
    import urllib.request
    with urllib.request.urlopen(url + "/api/models") as response:
        qwen = next(model for model in json.load(response)["models"] if model["id"] == "qwen3")
    if not qwen["weights_available"] or not qwen.get("loaded") and qwen.get("inference_allowed") is False:
        raise SystemExit("Qwen live verification is blocked by the local weights/RAM resource policy; no inference was requested.")
    errors, checks = [], []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, **({"channel": args.channel} if args.channel else {}))
        except Exception:
            if args.channel: raise
            browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1200}, device_scale_factor=1, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto(url + "/?model=qwen3", wait_until="networkidle")
        page.wait_for_function("cortexlab.getSession()?.schema_version==='cortexlab.session/v2'")
        session = page.evaluate("cortexlab.getSession()")
        assert session["model"]["parameter_count"] == 1720574976
        assert session["model"]["layers"] == 28
        assert page.locator("#ablate").is_disabled()
        assert page.locator("#model-select").input_value() == "qwen3"
        checks.append("Real 1.7B recorded session loads through the Qwen profile; unsupported ablation disabled")
        page.locator('[data-layer="2"]').click(); page.locator('[data-head="3"]').click()
        assert page.locator("#selected-layer").inner_text() == "BLOCK 28"
        assert page.evaluate("cortexlab.getSourceSelection()") == {"layer": 27, "head": 15, "token": len(session["frames"][0]["tokens"]) - 1}
        selection = page.evaluate("cortexlab.getSelection()")
        probability = float(page.locator('#attention-list [data-key="1"] .weight').get_attribute("data-weight"))
        assert math.isclose(probability, session["frames"][0]["layers"][2]["attention"][3][selection["token"]][1], abs_tol=1e-12)
        assert "64/2048" in page.locator("#channel-count").inner_text()
        checks.append("Original block/head indices and exact BF16-source measurements map correctly; sample disclosed")

        # This goes through actual UI/HTTP/model execution, not the replay endpoint.
        first_id = session["id"]
        started = time.perf_counter(); page.locator("#run").click()
        page.wait_for_function("id => cortexlab.getSession().id !== id", arg=first_id, timeout=180000)
        first_request_ms = (time.perf_counter() - started) * 1000
        first = page.evaluate("cortexlab.getSession()")
        assert first["runtime"]["device"] in {"cpu", "cuda"}
        assert first["output"].strip() == "Paris"
        checks.append("Fresh real CUDA inference via visible Run button produces Paris")

        page.locator("#prompt").fill("Name the capital of Japan. Answer with one word.")
        started = time.perf_counter(); page.locator("#run").click()
        page.wait_for_function("id => cortexlab.getSession().id !== id", arg=first["id"], timeout=60000)
        warm_request_ms = (time.perf_counter() - started) * 1000
        second = page.evaluate("cortexlab.getSession()")
        assert "Tokyo" in second["output"]
        assert first["frames"][0]["input_ids"] != second["frames"][0]["input_ids"]
        assert first["frames"][0]["layers"][0]["residual"] != second["frames"][0]["layers"][0]["residual"]
        checks.append("Changed prompt produces Tokyo and different genuine captured residuals")

        page.locator("#timeline").evaluate("e => {e.value='1';e.dispatchEvent(new Event('input',{bubbles:true}));}")
        assert page.evaluate("cortexlab.getSelection().frame") == 1
        assert page.locator("#token-list > button").count() == len(second["frames"][1]["tokens"])
        page.locator("#matrix-toggle").click(); assert page.locator("#matrix").is_visible(); page.locator("#matrix-close").click()
        checks.append("Real pretrained generation-step replay and attention matrix work")

        with tempfile.TemporaryDirectory(prefix="cortexlab-qwen-roundtrip-") as directory:
            with page.expect_download() as info: page.locator("#export-session").click()
            target = Path(directory) / "qwen.json"; info.value.save_as(target)
            assert json.loads(target.read_text(encoding="utf-8")) == second
            page.locator("#session-file").set_input_files(str(target))
            page.wait_for_function("document.getElementById('view-mode').textContent==='IMPORTED'")
            assert page.evaluate("cortexlab.getSession()") == second
        checks.append("Selective pretrained recording exports/reimports without changing measured values")
        page.locator('[data-layer="1"]').click(); page.locator('[data-head="0"]').click()
        page.locator('[data-camera="overview"]').click(); page.wait_for_timeout(1000)
        assert page.locator("#scene-fallback").is_hidden()
        stats = page.evaluate("cortexlab.getStats()")
        assert stats["nodes"] > 0 and stats["drawCalls"] < 100
        page.screenshot(path=str(ROOT / "docs" / "images" / "qwen3-live-demo.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844}); page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
        page.screenshot(path=str(ROOT / "docs" / "images" / "qwen3-mobile-demo.png"), full_page=True)
        checks.append("Qwen 3D scene and responsive controls render without horizontal overflow")
        assert not errors, errors
        checks.append("No browser page, console or shader errors")
        report = {"pretrained_weights_executed": True, "passed": len(checks), "checks": checks, "browser": browser.version, "first_ui_request_ms": first_request_ms, "warm_ui_request_ms": warm_request_ms, "scope": "Two actual local browser requests, not a latency percentile benchmark; first is not guaranteed cold and includes loading/hashing only if the server model cache is empty.", "first_output": first["output"], "second_output": second["output"], "first_metrics": first["metrics"], "second_metrics": second["metrics"], "scene": stats, "errors": errors}
        (ROOT / "examples" / "qwen-browser-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)); context.close(); browser.close()


if __name__ == "__main__": main()
