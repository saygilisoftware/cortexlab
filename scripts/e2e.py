"""Real-browser acceptance test. Starts its own loopback server by default.

python scripts/e2e.py [--url http://127.0.0.1:8000] [--channel chrome]
Chromium installed by Playwright is preferred; system Chrome is a fallback.
"""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "images"


def wait_ready(url):
    for _ in range(100):
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("CortexLab server did not become ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--channel", default=os.environ.get("CORTEXLAB_BROWSER_CHANNEL"))
    args = parser.parse_args()
    owned = None
    if args.url:
        url = args.url.rstrip("/")
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        owned = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    checks = []
    errors = []
    browser = None
    try:
        wait_ready(url)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True, **({"channel": args.channel} if args.channel else {}))
            except Exception:
                if args.channel:
                    raise
                browser = playwright.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(viewport={"width": 1600, "height": 1050}, device_scale_factor=1, accept_downloads=True)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            page.on("response", lambda response: errors.append(f"HTTP {response.status}: {response.url}") if response.status >= 400 else None)
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("window.cortexlab?.getSession() && window.cortexlab?.getStats()?.frames > 10")
            assert page.locator("#scene-fallback").is_hidden()
            assert page.locator("#viewport > canvas").is_visible()
            assert page.evaluate("cortexlab.getStats().nodes") > 0
            checks.append("Initial real recording and WebGL scene loaded without fallback")

            # Execute a fresh real inference through the visible UI.
            initial_id = page.evaluate("cortexlab.getSession().id")
            page.locator("#prompt").fill("a comet is cyan . the comet")
            page.locator("#new-tokens").select_option("6")
            page.locator("#run").click()
            page.wait_for_function("old => cortexlab.getSession().id !== old", arg=initial_id)
            session = page.evaluate("cortexlab.getSession()")
            assert session["request"]["prompt"] == "a comet is cyan . the comet"
            assert session["frames"][0]["tokens"][1] == "a"
            checks.append("Prompt -> HTTP -> PyTorch tensors -> rendered session")

            page.locator('[data-layer="0"]').click()
            page.locator('[data-head="2"]').click()
            page.locator('#token-list [data-token="3"]').click()
            selected = page.evaluate("cortexlab.getSelection()")
            assert selected == {"frame": 0, "layer": 0, "head": 2, "token": 3}
            weight = float(page.locator('#attention-list [data-key="1"] .weight').get_attribute("data-weight"))
            expected = session["frames"][0]["layers"][0]["attention"][2][3][1]
            assert math.isclose(weight, expected, abs_tol=1e-12)
            expected_norm = session["frames"][0]["layers"][0]["norms"][3]
            assert f"{expected_norm:.4f}" in page.locator("#norm-value").inner_text()
            checks.append("Layer/head/query controls display exact recorded attention and residual norm")

            # Click an actual projected, raycast-visible instance in the WebGL canvas.
            page.locator('[data-camera="overview"]').click()
            page.wait_for_timeout(1100)
            target = page.evaluate("""() => {
              const s=cortexlab.getScene(), r=s.renderer.domElement.getBoundingClientRect(), selected=cortexlab.getSelection();
              for (let layer=s.positions.length-1;layer>=0;layer--) for(let token=0;token<s.positions[layer].length;token++) {
                if(layer===selected.layer && token===selected.token) continue;
                const p=s.positions[layer][token].clone().project(s.camera);
                const x=r.left+(p.x+1)*r.width/2,y=r.top+(1-p.y)*r.height/2;
                if(p.z>1||p.z< -1||x<r.left||x>r.right||y<r.top||y>r.bottom) continue;
                const hit=s.pick({clientX:x,clientY:y});
                if(hit?.layer===layer && hit?.token===token) return {x,y,layer,token};
              }
              return null;
            }""")
            assert target, "No visible selectable residual node"
            page.mouse.click(target["x"], target["y"])
            assert page.evaluate("cortexlab.getSelection().token") == target["token"]
            assert page.evaluate("cortexlab.getSelection().layer") == target["layer"]
            checks.append("Actual WebGL node click updates query/layer inspector")

            before_camera = page.evaluate("cortexlab.getScene().camera.position.toArray()")
            bounds = page.locator("#viewport > canvas").bounding_box()
            page.mouse.move(bounds["x"] + bounds["width"] * .6, bounds["y"] + bounds["height"] * .6)
            page.mouse.down(); page.mouse.move(bounds["x"] + bounds["width"] * .6 + 80, bounds["y"] + bounds["height"] * .6 + 20, steps=12); page.mouse.up()
            page.wait_for_timeout(200)
            after_camera = page.evaluate("cortexlab.getScene().camera.position.toArray()")
            assert before_camera != after_camera
            page.locator('[data-camera="front"]').click()
            page.locator("#focus").click()
            checks.append("Orbit drag, camera preset and selection focus work")

            page.locator("#matrix-toggle").click()
            assert page.locator("#matrix").is_visible()
            page.locator("#matrix-close").click()
            page.locator("#threshold").evaluate("e => {e.value='0.5';e.dispatchEvent(new Event('input',{bubbles:true}));}")
            assert page.locator("#threshold-value").inner_text() == "0.50"
            page.locator("#show-attention").uncheck(); page.locator("#show-attention").check()
            checks.append("Attention matrix, threshold and visibility controls work")

            assert len(session["frames"]) >= 2
            page.locator("#timeline").evaluate("e => {e.value='1';e.dispatchEvent(new Event('input',{bubbles:true}));}")
            assert page.evaluate("cortexlab.getSelection().frame") == 1
            assert page.locator("#token-list > button").count() == len(session["frames"][1]["tokens"])
            page.locator("#speed").select_option("4")
            page.locator("#timeline").evaluate("e => {e.value='0';e.dispatchEvent(new Event('input',{bubbles:true}));}")
            page.locator("#play").click()
            page.wait_for_function("cortexlab.getSelection().frame > 0")
            if page.locator("#play").get_attribute("aria-label") == "Pause recording":
                page.locator("#play").click()
            checks.append("Timeline, context growth, play/pause and playback speed use recorded steps")

            # Exact data round-trip through the actual browser download/file-input flow.
            with tempfile.TemporaryDirectory(prefix="cortexlab-e2e-") as directory:
                with page.expect_download() as info:
                    page.locator("#export-session").click()
                exported = Path(directory) / "session.json"
                info.value.save_as(exported)
                assert json.loads(exported.read_text(encoding="utf-8")) == session
                page.locator("#session-file").set_input_files(str(exported))
                page.wait_for_function("document.getElementById('view-mode').textContent === 'IMPORTED'")
                assert page.evaluate("cortexlab.getSession()") == session
                invalid = Path(directory) / "invalid.json"
                bad = json.loads(exported.read_text(encoding="utf-8")); bad["frames"][0]["layers"][0]["attention"][0][0][1] = .9
                invalid.write_text(json.dumps(bad), encoding="utf-8")
                page.locator("#session-file").set_input_files(str(invalid))
                page.wait_for_function("document.getElementById('run-status').classList.contains('error')")
                assert page.evaluate("cortexlab.getSession().id") == session["id"]
            checks.append("Export/import preserves all tensors; malformed causal weights rejected without replacing session")

            baseline = page.evaluate("cortexlab.getSession()")
            page.locator('[data-layer="0"]').click(); page.locator('[data-head="1"]').click()
            page.locator("#ablate").check()
            page.locator("#run").click()
            page.wait_for_function("old => cortexlab.getSession().id !== old", arg=baseline["id"])
            ablated = page.evaluate("cortexlab.getSession()")
            assert ablated["request"]["ablate"] == {"layer": 0, "head": 1}
            assert ablated["frames"][0]["layers"][0]["attention"] == baseline["frames"][0]["layers"][0]["attention"]
            assert ablated["frames"][0]["layers"][0]["residual"] != baseline["frames"][0]["layers"][0]["residual"]
            assert page.locator("#comparison").is_visible()
            checks.append("Controlled head-output ablation changes real residuals and shows same-input baseline comparison")

            page.locator("#help-toggle").click(); assert page.locator("#help").is_visible(); page.locator("#help-close").click()
            checks.append("Interpretation guide opens and closes")

            # Representative portfolio screenshot: fresh baseline with wider attention view.
            page.locator("#ablate").uncheck(); page.locator("#run").click()
            page.wait_for_function("old => cortexlab.getSession().id !== old", arg=ablated["id"])
            page.locator('[data-layer="1"]').click(); page.locator('[data-head="0"]').click()
            page.locator("#threshold").evaluate("e => {e.value='0';e.dispatchEvent(new Event('input',{bubbles:true}));}")
            page.locator('[data-camera="overview"]').click(); page.wait_for_timeout(1300)
            page.screenshot(path=str(ARTIFACTS / "cortexlab.png"), full_page=True)
            stats_before = page.evaluate("cortexlab.getStats()")
            # Repeated selection/frame replacement must not leak GPU resource counts.
            page.evaluate("""() => {const s=cortexlab.getScene(), session=cortexlab.getSession(); for(let i=0;i<30;i++) s.setFrame(session.frames[0],{layer:i%3,head:i%4,token:2,threshold:0,showAttention:true});}""")
            page.wait_for_timeout(300)
            stats_after = page.evaluate("cortexlab.getStats()")
            assert stats_after["geometries"] <= stats_before["geometries"] + 1
            assert stats_after["textures"] <= stats_before["textures"] + 1
            checks.append("Thirty scene replacements have stable geometry/texture resource counts")

            # Sample requestAnimationFrame intervals only after warm-up. These are not GPU timings.
            page.wait_for_timeout(1800)
            stats = page.evaluate("cortexlab.getStats()")
            assert stats["sampleCount"] > 30 and stats["drawCalls"] < 100
            context_info = page.evaluate("""() => {
              const gl=cortexlab.getScene().renderer.getContext(), debug=gl.getExtension('WEBGL_debug_renderer_info');
              return {user_agent:navigator.userAgent, webgl_version:gl.getParameter(gl.VERSION),
                gpu_renderer:debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : 'unavailable',
                gpu_vendor:debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : 'unavailable'};
            }""")
            # Exercise the largest supported prompt+generation budget and its real round trip.
            previous_id = page.evaluate("cortexlab.getSession().id")
            page.locator("#prompt").fill("the " * 39)
            page.locator("#new-tokens").select_option("8")
            page.locator("#run").click()
            page.wait_for_function("old => cortexlab.getSession().id !== old", arg=previous_id, timeout=20000)
            maximum = page.evaluate("({id:cortexlab.getSession().id,frames:cortexlab.getSession().frames.length,context:cortexlab.getSession().frames[0].tokens.length})")
            assert maximum["context"] == 40
            page.locator("#timeline").evaluate("e => {e.value=e.max;e.dispatchEvent(new Event('input',{bubbles:true}));}")
            page.wait_for_timeout(400)
            maximum_stats = page.evaluate("cortexlab.getStats()")
            assert maximum_stats["nodes"] <= 144 and maximum_stats["drawCalls"] < 100
            with tempfile.TemporaryDirectory(prefix="cortexlab-max-") as directory:
                with page.expect_download() as info:
                    page.locator("#export-session").click()
                exported = Path(directory) / "maximum.json"; info.value.save_as(exported)
                maximum["export_bytes"] = exported.stat().st_size
                assert maximum["export_bytes"] < 5 * 1024 * 1024
                page.locator("#session-file").set_input_files(str(exported))
                page.wait_for_function("document.getElementById('view-mode').textContent === 'IMPORTED'")
                assert page.evaluate("cortexlab.getSession().id") == maximum["id"]
            checks.append("Maximum 48-token budget renders below 100 draw calls and exports/reimports below 5 MiB")
            page.emulate_media(reduced_motion="reduce")
            page.locator('[data-layer="2"]').click()
            page.wait_for_timeout(100)
            assert page.evaluate("cortexlab.getScene().transitions.length") == 0
            assert page.evaluate("cortexlab.getScene().motionPreference.matches")
            checks.append("Reduced-motion preference disables presentation transitions")
            # Responsive native controls remain accessible and there is no horizontal overflow.
            page.set_viewport_size({"width": 390, "height": 844}); page.wait_for_timeout(300)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            assert page.locator("#run").is_visible()
            page.screenshot(path=str(ARTIFACTS / "cortexlab-mobile.png"), full_page=True)
            checks.append("390px responsive layout has no horizontal overflow")
            fallback = context.new_page()
            fallback.add_init_script("""const original=HTMLCanvasElement.prototype.getContext;
              HTMLCanvasElement.prototype.getContext=function(kind,...args) {
                return String(kind).startsWith('webgl') ? null : original.call(this,kind,...args);
              };""")
            fallback.goto(url, wait_until="networkidle")
            fallback.wait_for_function("window.cortexlab?.getSession() != null")
            assert fallback.locator("#scene-fallback").is_visible()
            assert fallback.locator("#token-list > button").count() > 0
            fallback.close()
            checks.append("No-WebGL fallback retains functional tensor inspection and recording")
            assert not errors, "Browser errors: " + repr(errors)
            checks.append("No page, console, shader or failed-asset errors")
            report = {"checks": checks, "passed": len(checks), "browser": browser.version, "environment": context_info, "viewport": "1600x1050 desktop / 390x844 mobile", "rendering_sample": stats, "maximum_budget": {**maximum, "scene": maximum_stats}, "resource_counts_before": stats_before, "resource_counts_after": stats_after, "errors": errors, "measurement_note": "Headless Chromium requestAnimationFrame intervals, not GPU timings or visible-display FPS; includes interaction workload. See PERFORMANCE.md."}
            (ROOT / "examples" / "browser-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            context.close(); browser.close(); browser = None
    finally:
        if browser:
            browser.close()
        if owned:
            owned.terminate()
            try:
                owned.wait(timeout=5)
            except subprocess.TimeoutExpired:
                owned.kill(); owned.wait()


if __name__ == "__main__":
    main()
