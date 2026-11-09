"""Capture an actual browser walkthrough as GIF; no generated/modelled UI imagery.

Requires the same Playwright installation as scripts/e2e.py.
"""
import argparse
import socket
import subprocess
import sys
import io
import time
import json

from playwright.sync_api import sync_playwright
from PIL import Image
from e2e import ROOT, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(url)
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True, **({"channel": args.channel} if args.channel else {}))
            except Exception:
                if args.channel:
                    raise
                browser = p.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
            page = context.new_page(); errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("window.cortexlab?.getStats()?.frames > 10")
            frames, timestamps = [], []
            def capture(count=3):
                for _ in range(count):
                    page.wait_for_timeout(160)
                    timestamps.append(time.perf_counter())
                    image = Image.open(io.BytesIO(page.screenshot())).convert('RGB')
                    frames.append(image.resize((1000, 750), Image.Resampling.LANCZOS))
            capture(4)
            page.locator('[data-head="2"]').click(); capture()
            page.locator('#token-list [data-token="2"]').click(); capture()
            page.locator('[data-camera="front"]').click(); capture(5)
            page.locator('[data-layer="2"]').click(); capture()
            page.locator('[data-camera="overview"]').click(); capture(5)
            page.locator('#token-list [data-token="7"]').click()
            page.locator("#timeline").evaluate("e => {e.value='1';e.dispatchEvent(new Event('input',{bubbles:true}));}"); capture()
            page.locator("#timeline").evaluate("e => {e.value='2';e.dispatchEvent(new Event('input',{bubbles:true}));}"); capture()
            page.locator("#matrix-toggle").click(); capture(4); page.locator("#matrix-close").click()
            page.locator("#prompt").fill("the violet crystal")
            previous = page.evaluate("cortexlab.getSession().id")
            page.locator("#run").click(); page.wait_for_function("id => cortexlab.getSession().id !== id", arg=previous)
            capture(5)
            assert not errors, errors
            target = ROOT / "docs" / "images" / "cortexlab-demo.gif"
            # Preserve actual sampling intervals; this is a sampled walkthrough,
            # not a rendering-performance recording or a synthetic mockup.
            durations = [max(20, round((b-a)*1000)) for a,b in zip(timestamps,timestamps[1:])] + [1500]
            frames[0].save(target, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
            with Image.open(target) as encoded:
                encoded_durations = []
                for i in range(encoded.n_frames):
                    encoded.seek(i); encoded_durations.append(encoded.info.get('duration', 0))
                report = {"captured_screenshots": len(frames), "encoded_frames": encoded.n_frames, "duration_ms": sum(encoded_durations), "bytes": target.stat().st_size, "source_viewport": [1600, 1200], "output_dimensions": [1000, 750], "note": "Actual browser screenshots; GIF encoder may merge identical frames. Sampling intervals preserved to GIF precision; not a rendering FPS benchmark."}
            (ROOT / 'examples' / 'demo-capture.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            context.close(); browser.close()
            print(f"Captured {len(frames)} actual screenshots, {report['encoded_frames']} encoded GIF frames: {target} ({target.stat().st_size:,} bytes)")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()


if __name__ == "__main__":
    main()
