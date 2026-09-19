"""Real Streamlit + Chromium acceptance, including complete server restart.

Run explicitly: RUN_BROWSER_TESTS=1 python -m pytest tests/test_browser.py -q
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Opt-in real browser acceptance")
REPO = Path(__file__).resolve().parents[1]


def test_create_tag_restart(tmp_path):
    from playwright.sync_api import sync_playwright, expect
    from test_dxf_parser import sample
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    environment = dict(os.environ, CAD_WORKSPACE=str(tmp_path / "workspace"))
    def start():
        log = (tmp_path / "streamlit.log").open("a")
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true",
            f"--server.port={port}", "--server.address=127.0.0.1"], cwd=REPO, env=environment, stdout=log, stderr=log)
        log.close()
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1)
                return process
            except OSError:
                time.sleep(.1)
        process.terminate()
        raise RuntimeError((tmp_path / "streamlit.log").read_text())
    process = start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 1680, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            page.get_by_role("button", name="+ Add Layout", exact=True).click()
            page.get_by_role("textbox", name="Layout name", exact=True).fill("Living Room Test 01")
            page.locator('input[type=file]').set_input_files({"name": "floorplan.dxf", "mimeType": "application/octet-stream", "buffer": sample()})
            page.get_by_role("button", name="Create Layout", exact=True).click()
            expect(page.get_by_role("heading", name="Living Room Test 01")).to_be_visible(timeout=20000)
            frame = page.frame_locator('iframe[title="app.ui.viewer.cad_viewer"]')
            canvas = frame.locator('#viewport canvas')
            expect(canvas).to_be_visible(timeout=20000)
            expect(frame.locator("#status")).to_contain_text("7 entities")
            # Fit is 1.25 times bounds: x=[0,6000], y=[-500,4000] (bulge curve).
            # Obtain actual bounds from the backend and apply the documented orthographic fit.
            from app.cad.entities import load_for_editor
            root = next((tmp_path / "workspace").glob("Layout_*"))
            state, geometry, _ = load_for_editor(root)
            points = [v for e in geometry for path in e["paths"] for v in path]
            xmin, xmax = min(v[0] for v in points), max(v[0] for v in points)
            ymin, ymax = min(v[1] for v in points), max(v[1] for v in points)
            box = canvas.bounding_box()
            height = max(ymax-ymin, (xmax-xmin)*box["height"]/box["width"], 1)*1.25
            def pixel(x, y):
                return {"x": ((x-(xmin+xmax)/2)/(height*box["width"]/box["height"])+.5)*box["width"],
                        "y": (.5-(y-(ymin+ymax)/2)/height)*box["height"]}
            canvas.click(position=pixel(3000, 0))
            expect(page.get_by_text("1 selected · LINE", exact=False)).to_be_visible()
            page.get_by_role("button", name="Apply & autosave", exact=True).click()
            expect(page.get_by_text("1 / 7 tagged", exact=True)).to_be_visible()
            # Shift-select the floor using another edge and verify browser→Python multi-selection.
            canvas.click(position=pixel(6000, 2000), modifiers=["Shift"])
            expect(frame.locator("#status")).to_contain_text("2 selected")
            expect(page.get_by_text("2 selected · LINE", exact=False)).to_be_visible()
            # Select floor alone; semantic change is outside the form so contextual fields update.
            canvas.click(position=pixel(6000, 2000))
            expect(page.get_by_text("1 selected · LWPOLYLINE", exact=False)).to_be_visible()
            page.get_by_role("combobox", name="Semantic", exact=True).click()
            page.get_by_role("option", name="Floor", exact=True).click()
            expect(page.get_by_role("spinbutton", name="Thickness (mm)", exact=True)).to_be_visible()
            page.get_by_role("button", name="Apply & autosave", exact=True).click()
            expect(page.get_by_text("2 / 7 tagged", exact=True)).to_be_visible()
            canvas.click(position=pixel(2232.05, 1500))
            expect(page.get_by_text("1 selected · INSERT", exact=False)).to_be_visible()
            page.get_by_role("combobox", name="Semantic", exact=True).click()
            page.get_by_role("option", name="Furniture", exact=True).click()
            expect(page.get_by_role("combobox", name="Category", exact=True)).to_be_visible()
            page.get_by_role("textbox", name="Material", exact=True).fill("fabric")
            import cv2
            import numpy as np
            _, reference = cv2.imencode(".png", np.full((24, 24, 3), 180, dtype=np.uint8))
            page.locator('input[type=file]').set_input_files({"name":"sofa.png", "mimeType":"image/png", "buffer":reference.tobytes()})
            page.get_by_role("button", name="Apply & autosave", exact=True).click()
            expect(page.get_by_text("3 / 7 tagged", exact=True)).to_be_visible()
            saved = json.loads((root / "layout.json").read_text())
            assert saved["entities"]["entity_00001"]["height"] == 2800
            assert saved["entities"]["entity_00002"]["thickness"] == 150
            assert saved["entities"]["entity_00003"]["height"] == 820
            assert saved["entities"]["entity_00003"]["category"] == "sofa"
            reference_path = saved["entities"]["entity_00003"]["reference_image"]
            assert (root / reference_path).read_bytes() == reference.tobytes()
            # Exercise zoom and pan, then reset before taking the visual QA artifact.
            original_status = frame.locator("#status").inner_text()
            canvas.hover(position={"x":box["width"]/2,"y":box["height"]/2})
            page.mouse.wheel(0, -300)
            expect(frame.locator("#status")).not_to_have_text(original_status)
            bounds = canvas.bounding_box()
            page.mouse.move(bounds["x"]+100, bounds["y"]+100)
            page.mouse.down()
            page.mouse.move(bounds["x"]+170, bounds["y"]+140, steps=5)
            page.mouse.up()
            frame.get_by_role("button", name="Fit to view", exact=True).click()
            expect(frame.locator("#status")).to_have_text(original_status)
            screenshots = REPO / "test-results"
            screenshots.mkdir(exist_ok=True)
            page.locator('[data-testid="stMain"]').evaluate('(el) => el.scrollTop = 0')
            page.screenshot(path=str(screenshots / "editor.png"), full_page=True)
            process.terminate()
            process.wait(timeout=10)
            process = start()
            page.close()
            page = browser.new_page(viewport={"width":1680, "height":1100})
            page.goto(url)
            page.get_by_role("button", name="Open / Edit Layout", exact=True).click()
            expect(page.get_by_text("3 / 7 tagged", exact=True)).to_be_visible(timeout=20000)
            assert json.loads((root / "layout.json").read_text())["entities"] == saved["entities"]
            frame = page.frame_locator('iframe[title="app.ui.viewer.cad_viewer"]')
            canvas = frame.locator('#viewport canvas')
            expect(frame.locator("#status")).to_contain_text("7 entities")
            canvas.click(position=pixel(2232.05, 1500))
            expect(page.get_by_role("spinbutton", name="Height (mm)", exact=True)).to_have_value("820.000")
            expect(page.get_by_role("textbox", name="Material", exact=True)).to_have_value("fabric")
            expect(page.get_by_text("Saved furniture reference", exact=True)).to_be_visible()
            page.get_by_role("combobox", name="Semantic", exact=True).click()
            page.get_by_role("option", name="Void", exact=True).click()
            opening = page.get_by_role("combobox", name="Opening type", exact=True)
            expect(opening).to_be_visible()
            expect(page.get_by_role("spinbutton", name="Opening Height (mm)", exact=True)).to_have_value("2100.000")
            opening.click()
            page.get_by_role("option", name="Window", exact=True).click()
            page.get_by_role("spinbutton", name="Opening Height (mm)", exact=True).fill("1200")
            page.get_by_role("spinbutton", name="Sill Height (mm)", exact=True).fill("900")
            page.get_by_role("button", name="Apply & autosave", exact=True).click()
            # A fresh browser session must recover the Void properties from disk.
            expect(page.get_by_text("Saved furniture reference", exact=True)).not_to_be_visible()
            page.reload()
            page.get_by_role("button", name="Open / Edit Layout", exact=True).click()
            frame = page.frame_locator('iframe[title="app.ui.viewer.cad_viewer"]').last
            expect(frame.locator("#status")).to_contain_text("7 entities")
            frame.locator('#viewport canvas').click(position=pixel(2232.05, 1500))
            expect(page.get_by_role("combobox", name="Semantic", exact=True)).to_have_value("Void")
            expect(page.get_by_role("combobox", name="Opening type", exact=True)).to_have_value("Window")
            expect(page.get_by_role("spinbutton", name="Opening Height (mm)", exact=True)).to_have_value("1200.000")
            expect(page.get_by_role("spinbutton", name="Sill Height (mm)", exact=True)).to_have_value("900.000")
            void = json.loads((root / "layout.json").read_text())["entities"]["entity_00003"]
            assert void["semantic"] == "void" and void["category"] == "window"
            assert "reference_image" not in void
            # Unapplied annotation drafts survive a round-trip through Camera Mode.
            material=page.get_by_role("textbox",name="Material",exact=True)
            material.fill("draft glazing");material.press("Tab")
            page.get_by_text("Camera Mode",exact=True).click()
            expect(frame.locator('#camera-workspace')).to_be_visible()
            page.get_by_text("Annotation Mode",exact=True).click()
            expect(page.get_by_role("textbox",name="Material",exact=True)).to_have_value("draft glazing")
            assert json.loads((root / "layout.json").read_text())["entities"]["entity_00003"]["material"] != "draft glazing"
            assert not errors
            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=10)
