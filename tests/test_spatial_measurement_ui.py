"""UI tests for Part 7B Spatial Measurement dashboard."""
from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest

from app.geometry.camera import new_camera
from test_spatial_correspondence import evaluation
from test_automatic_correspondence import Detector, candidate, context


def page(root, state):
    from pathlib import Path
    from app.ui.spatial_validation import render
    render(Path(root), state)


def test_spatial_measurement_ui_flow(evaluation, monkeypatch):
    from app.ui import spatial_validation as ui
    root, state = evaluation
    state["cameras"] = {
        "camera_001": new_camera(5.0, 2.0, "m", heading=180.0, output={"width": 80, "height": 40})
    }
    state["entities"]["bed"]["height"] = 0.50

    c = context(evaluation)
    detector = Detector([candidate(c["ids"] == 1)])
    monkeypatch.setattr(ui, "get_backend", lambda config: detector)

    app = AppTest.from_function(page, kwargs={"root": str(root), "state": state}).run(timeout=15)
    assert not app.exception

    # 1. Part 7A: select render, run auto correspondence, confirm
    next(w for w in app.selectbox if w.label == "Generated Render").select("render_a.png").run()
    next(w for w in app.button if w.label == "Run Automatic Correspondence").click().run()
    next(w for w in app.checkbox if w.label == "Show cases needing attention only").uncheck().run()
    next(w for w in app.button if w.label == "Confirm").click().run()
    assert not app.exception

    # 2. Part 7B: Run Spatial Measurements button is now active
    run_btn = next((w for w in app.button if w.label == "Run Spatial Measurements"), None)
    assert run_btn is not None
    run_btn.click().run()
    assert not app.exception

    # 3. Check measurements.json file on disk
    meas_file = root / "generated/camera_001/renders/evaluation/render_a/sam3/measurements.json"
    assert meas_file.is_file()

    meas_data = json.loads(meas_file.read_text())
    assert meas_data["stage"] == "Part 7B — Spatial Measurement and Metric Reconstruction"
    assert len(meas_data["entities"]) == 1
    ent_res = meas_data["entities"][0]
    assert ent_res["entity_id"] == "bed"
    assert ent_res["measurement_status"] == "evaluated"
    assert ent_res["planar"]["status"] == "evaluated"
    assert "position_error_m" in ent_res["planar"]
    assert ent_res["image_space"]["mask_iou"] > 0
    assert ent_res["vertical"]["status"] == "evaluated"
    assert ent_res["vertical"]["cad_height_m"] == 0.50
