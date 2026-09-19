import base64
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from app.geometry.camera import new_camera, save_cameras
from app.geometry.conditioning import (
    SEMANTIC_CLASSES,
    SEMANTIC_NAME_TO_ID,
    build_instance_mapping,
    build_conditioning_metadata,
    decode_data_url,
    save_conditioning_outputs,
)
from app.cad.entities import load_for_editor
from app.layout.manager import create
from app.layout.persistence import read
from test_dxf_parser import sample


def test_instance_mapping():
    state = {
        "entities": {
            "ent_b": {"semantic": "furniture", "orphaned": False},
            "ent_a": {"semantic": "wall", "orphaned": False},
            "ent_c": {"semantic": None, "orphaned": False},
            "ent_d": {"semantic": "floor", "orphaned": True},
        },
        "proxy_settings": {"auto_ceiling": True},
    }
    mapping = build_instance_mapping(state)
    assert mapping["0"] == "background"
    assert mapping["1"] == "ent_a"
    assert mapping["2"] == "ent_b"
    assert mapping["3"] == "auto_ceiling"


def test_build_conditioning_metadata(tmp_path):
    root, _ = create(tmp_path, "TestLayout", "room.dxf", sample())
    state, geometry, _ = load_for_editor(root)
    state["source"]["units"] = "mm"
    camera = new_camera(2000, 3000, "mm", heading=45, name="Cam1")
    cameras = {"camera_001": camera}
    state = save_cameras(root, state, cameras, "camera_001")

    ent_ids = list(state["entities"].keys())
    state["entities"][ent_ids[0]]["semantic"] = "wall"
    state["entities"][ent_ids[0]]["height"] = 2.8
    window_id = ent_ids[1]
    state["entities"][window_id].update({
        "semantic": "void",
        "category": "window",
        "sill_height": 0.5,
        "opening_height": 2.0,
        "description": "Double hung window",
        "material": "Aluminium"
    })

    metadata = build_conditioning_metadata(state, "camera_001", {"width": 1920, "height": 1080})
    assert metadata["layout_id"] == state["layout"]["id"]
    assert metadata["camera_id"] == "camera_001"
    assert metadata["resolution"] == {"width": 1920, "height": 1080}
    assert "K" in metadata["matrices"] and "R" in metadata["matrices"]
    assert "t" in metadata["matrices"] and "P" in metadata["matrices"]
    assert metadata["depth"]["unit"] == "m"
    assert metadata["depth"]["background"] == "NaN"
    assert metadata["entity_details"][window_id]["category"] == "window"
    assert metadata["entity_details"][window_id]["description"] == "Double hung window"


def encode_png_data_url(img_array):
    _, buf = cv2.imencode(".png", img_array)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_save_conditioning_outputs(tmp_path):
    root, _ = create(tmp_path, "ConditioningLayout", "room.dxf", sample())
    state, geometry, _ = load_for_editor(root)
    state["source"]["units"] = "mm"
    camera = new_camera(1000, 2000, "mm", heading=90, name="MainCam")
    cameras = {"camera_001": camera}
    state = save_cameras(root, state, cameras, "camera_001")

    ent_id = sorted(state["entities"].keys())[0]
    state["entities"][ent_id]["semantic"] = "wall"

    # Synthetic 4x4 RGBA test images
    # Depth raw: forward depth = 2500 mm (2.5 m)
    depth_raw = np.zeros((4, 4, 4), dtype=np.uint8)
    depth_raw[2, 2] = [0, 9, 196, 255]

    # Instance raw: instance ID 1 -> R = 1, G = 0, A = 255
    inst_raw = np.zeros((4, 4, 4), dtype=np.uint8)
    inst_raw[2, 2] = [0, 0, 1, 255]

    # Semantic raw: wall = 1 -> R = 1, A = 255
    sem_raw = np.zeros((4, 4, 4), dtype=np.uint8)
    sem_raw[2, 2] = [0, 0, 1, 255]

    proxy_img = np.full((4, 4, 3), 128, dtype=np.uint8)
    depth_vis = np.full((4, 4, 3), 200, dtype=np.uint8)
    inst_vis = np.full((4, 4, 3), 150, dtype=np.uint8)
    sem_vis = np.full((4, 4, 3), 100, dtype=np.uint8)

    payload = {
        "camera_id": "camera_001",
        "resolution": {"width": 4, "height": 4},
        "proxy_png": encode_png_data_url(proxy_img),
        "depth_vis_png": encode_png_data_url(depth_vis),
        "depth_raw_png": encode_png_data_url(depth_raw),
        "instance_vis_png": encode_png_data_url(inst_vis),
        "instance_raw_png": encode_png_data_url(inst_raw),
        "semantic_vis_png": encode_png_data_url(sem_vis),
        "semantic_raw_png": encode_png_data_url(sem_raw),
    }

    res = save_conditioning_outputs(root, state, payload)
    output_dir = Path(res["output_dir"])
    assert output_dir.exists()

    expected_files = [
        "proxy.png", "depth.png", "depth.npy",
        "instance.png", "instance.npy",
        "semantic.png", "semantic.npy",
        "metadata.json"
    ]
    for f in expected_files:
        assert (output_dir / f).exists(), f"Missing file: {f}"

    # Verify depth.npy
    depth_arr = np.load(output_dir / "depth.npy")
    assert depth_arr.dtype == np.float32
    assert depth_arr.shape == (4, 4)
    assert np.isnan(depth_arr[0, 0])
    assert pytest.approx(depth_arr[2, 2], abs=1e-3) == 2.5

    # Verify instance.npy
    inst_arr = np.load(output_dir / "instance.npy")
    assert inst_arr.dtype == np.int32
    assert inst_arr.shape == (4, 4)
    assert inst_arr[0, 0] == 0
    assert inst_arr[2, 2] == 1

    # Verify semantic.npy
    sem_arr = np.load(output_dir / "semantic.npy")
    assert sem_arr.dtype == np.uint8
    assert sem_arr.shape == (4, 4)
    assert sem_arr[0, 0] == 0
    assert sem_arr[2, 2] == 1

    # Verify metadata.json
    meta = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["camera_id"] == "camera_001"
    assert meta["resolution"] == {"width": 4, "height": 4}
    assert "K" in meta["matrices"]

    # Verify layout.json outputs registration and deduplication
    updated_layout = read(root)
    assert "outputs" in updated_layout
    assert "generated/camera_001/proxy.png" in updated_layout["outputs"]["proxy"]
    assert "generated/camera_001/depth.npy" in updated_layout["outputs"]["depth"]

    save_conditioning_outputs(root, updated_layout, payload)
    layout_reloaded = read(root)
    assert layout_reloaded["outputs"]["proxy"].count("generated/camera_001/proxy.png") == 1
    assert layout_reloaded["outputs"]["depth"].count("generated/camera_001/depth.npy") == 1

