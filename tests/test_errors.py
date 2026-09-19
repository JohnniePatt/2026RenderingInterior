import json
from copy import deepcopy
import cv2
import numpy as np
import pytest
from test_dxf_parser import sample
from app.layout.manager import create, open_layout
from app.layout.persistence import read, save, resolve
from app.cad.entities import load_for_editor
from app.annotation.tagging import apply, store_reference


def test_missing_source_and_path_escape(tmp_path):
    root, _ = create(tmp_path, "Test", "test.dxf", sample())
    (root / "source/test.dxf").unlink()
    with pytest.raises(ValueError, match="missing"):
        open_layout(root)
    with pytest.raises(ValueError):
        resolve(root, "../escape.png")


def test_reference_and_missing_reference_retained(tmp_path):
    root, _ = create(tmp_path, "Test", "test.dxf", sample())
    state, geometry, _ = load_for_editor(root)
    _, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    relative = store_reference(root, "../sofa.png", encoded.tobytes())
    state = apply(root, state, [geometry[2]["id"]], "furniture", {"category": "custom_sofa", "height": 820, "reference_image": relative})
    assert not relative.startswith("/") and (root / relative).is_file()
    (root / relative).unlink()
    assert read(root)["entities"][geometry[2]["id"]]["reference_image"] == relative
    with pytest.raises(ValueError, match="Invalid reference"):
        store_reference(root, "bad.png", b"not an image")


def test_unknown_units_and_invalid_camera(tmp_path):
    root, _ = create(tmp_path, "Test", "test.dxf", sample())
    state, geometry, _ = load_for_editor(root)
    state["source"]["units"] = "unknown"
    with pytest.raises(ValueError, match="Confirm source units"):
        apply(root, state, [geometry[0]["id"]], "wall", {"height": 2800})
    state["cameras"] = {"bad": {"position": [0, 0, 1500], "heading_deg": 0, "pitch_deg": 0, "fov_deg": 190}}
    with pytest.raises(ValueError, match="FOV"):
        save(root, state)


def test_atomic_replace_failure_keeps_old_json(tmp_path, monkeypatch):
    root, state = create(tmp_path, "Test", "test.dxf", sample())
    original = (root / "layout.json").read_bytes()
    def fail(*args):
        raise OSError("simulated disk error")
    monkeypatch.setattr("app.layout.persistence.os.replace", fail)
    with pytest.raises(OSError, match="disk error"):
        save(root, state)
    assert (root / "layout.json").read_bytes() == original
    assert not list(root.glob("*.tmp"))
