import math
import numpy as np
import pytest
from app.annotation.tagging import CATEGORIES, apply
from app.cad.entities import load_for_editor, migrate_entities_and_cameras
from app.geometry.camera import (
    calculate_orientation,
    calculate_target,
    camera_aspect,
    camera_matrices,
    new_camera,
    save_cameras,
    set_camera_orientation,
    set_camera_target,
    validate_camera
)
from app.geometry.proxy import build_proxy, normalize_opening
from app.layout.manager import create, open_layout
from app.layout.persistence import read, save
from app.layout.schema import validate
from tests.test_dxf_parser import sample


def test_1_old_window():
    old_record = {
        "semantic": "void",
        "category": "window",
        "height": 1.2,
        "base_elevation": 0.9
    }
    norm = normalize_opening(old_record)
    assert norm["sill_height"] == pytest.approx(0.9)
    assert norm["opening_height"] == pytest.approx(1.2)
    assert norm["bottom_z"] == pytest.approx(0.9)
    assert norm["top_z"] == pytest.approx(2.1)

    state = {
        "source": {"units": "m"},
        "entities": {"entity_1": old_record}
    }
    geometry = [{"id": "entity_1", "paths": [[[0, 0], [2, 0], [2, 0.2], [0, 0.2], [0, 0]]]}]
    proxy = build_proxy(state, geometry)
    assert len(proxy["meshes"]) == 1
    mesh = proxy["meshes"][0]
    assert mesh["semantic"] == "void"
    assert mesh["z_min"] == pytest.approx(0.9)
    assert mesh["z_max"] == pytest.approx(2.1)


def test_2_new_window():
    new_record = {
        "semantic": "void",
        "category": "window",
        "sill_height": 0.9,
        "opening_height": 1.2,
        "description": "condo window big screen"
    }
    norm = normalize_opening(new_record)
    assert norm["bottom_z"] == pytest.approx(0.9)
    assert norm["top_z"] == pytest.approx(2.1)

    state = {
        "source": {"units": "m"},
        "entities": {"entity_1": new_record}
    }
    geometry = [{"id": "entity_1", "paths": [[[1, 1], [3, 1], [3, 1.2], [1, 1.2], [1, 1]]]}]
    proxy = build_proxy(state, geometry)
    assert len(proxy["meshes"]) == 1
    mesh = proxy["meshes"][0]
    assert mesh["z_min"] == pytest.approx(0.9)
    assert mesh["z_max"] == pytest.approx(2.1)


def test_3_door():
    door_record = {
        "semantic": "void",
        "category": "door",
        "base_elevation": 0.0,
        "opening_height": 2.1,
        "description": "modern door white color"
    }
    norm = normalize_opening(door_record)
    assert norm["bottom_z"] == pytest.approx(0.0)
    assert norm["top_z"] == pytest.approx(2.1)

    state = {
        "source": {"units": "m"},
        "entities": {"entity_1": door_record}
    }
    geometry = [{"id": "entity_1", "paths": [[[0, 0], [1, 0], [1, 0.15], [0, 0.15], [0, 0]]]}]
    proxy = build_proxy(state, geometry)
    assert len(proxy["meshes"]) == 1
    mesh = proxy["meshes"][0]
    assert mesh["z_min"] == pytest.approx(0.0)
    assert mesh["z_max"] == pytest.approx(2.1)


def test_4_camera_backward_compatibility(tmp_path):
    root, state = create(tmp_path, "LegacyCamera", "room.dxf", sample())
    old_camera = {
        "name": "Legacy Cam",
        "position": [5.0, 2.0, 1.5],
        "heading_deg": 45.0,
        "pitch_deg": 0.0,
        "fov_deg": 60.0
    }
    state["cameras"] = {"camera_001": old_camera}
    state["active_camera"] = "camera_001"
    save(root, state)

    reloaded, _ = open_layout(root)
    migrate_entities_and_cameras(reloaded)
    cam = reloaded["cameras"]["camera_001"]
    assert cam["output"] == {"width": 1024, "height": 1024}

    matrices = camera_matrices(cam)
    assert matrices["K"][0, 2] == pytest.approx(512)
    assert matrices["K"][1, 2] == pytest.approx(512)


def test_5_camera_aspect_ratio():
    cam = {
        "name": "Widescreen Cam",
        "position": [0, 0, 1.5],
        "heading_deg": 0.0,
        "pitch_deg": 0.0,
        "fov_deg": 60.0,
        "output": {"width": 1920, "height": 1080}
    }
    validate_camera(cam)
    aspect = camera_aspect(cam)
    assert aspect == pytest.approx(16.0 / 9.0)

    matrices = camera_matrices(cam)
    assert matrices["K"][0, 2] == pytest.approx(960)
    assert matrices["K"][1, 2] == pytest.approx(540)


def test_6_target_synchronization():
    pos = [1.0, 2.0, 1.5]
    h, p = calculate_orientation(pos, [11.0, 2.0, 1.5])
    assert h == pytest.approx(0.0)
    assert p == pytest.approx(0.0)

    h, p = calculate_orientation(pos, [1.0, 12.0, 1.5])
    assert h == pytest.approx(90.0)
    assert p == pytest.approx(0.0)

    h, p = calculate_orientation(pos, [-9.0, 2.0, 1.5])
    assert h == pytest.approx(180.0)
    assert p == pytest.approx(0.0)

    h, p = calculate_orientation(pos, [1.0 + 10.0, 2.0, 1.5 + 10.0])
    assert h == pytest.approx(0.0)
    assert p == pytest.approx(45.0)

    tgt = calculate_target(pos, heading_deg=90.0, pitch_deg=0.0, distance=10.0)
    assert tgt[0] == pytest.approx(1.0)
    assert tgt[1] == pytest.approx(12.0)
    assert tgt[2] == pytest.approx(1.5)

    cam = new_camera(1.0, 2.0, "m", heading=0.0)
    set_camera_target(cam, [1.0, 11.0, 1.5])
    assert cam["heading_deg"] == pytest.approx(90.0)
    assert cam["pitch_deg"] == pytest.approx(0.0)


def test_7_camera_matrices():
    cam = new_camera(0.0, 0.0, "m", heading=0.0)
    m_base = camera_matrices(cam)
    P_base = m_base["P"]
    assert P_base.shape == (3, 4)
    np.testing.assert_allclose(P_base, m_base["K"] @ np.column_stack([m_base["R"], m_base["t"]]))

    cam_fov = dict(cam, fov_deg=90.0)
    m_fov = camera_matrices(cam_fov)
    assert m_fov["K"][0, 0] != m_base["K"][0, 0]
    assert not np.allclose(m_fov["P"], P_base)

    cam_res = dict(cam, output={"width": 1920, "height": 1080})
    m_res = camera_matrices(cam_res)
    assert m_res["K"][0, 2] == 960
    assert not np.allclose(m_res["P"], P_base)

    cam_pos = dict(cam, position=[5.0, 3.0, 1.5])
    m_pos = camera_matrices(cam_pos)
    assert not np.allclose(m_pos["t"], m_base["t"])
    assert not np.allclose(m_pos["P"], P_base)

    cam_tgt = set_camera_target(dict(cam), [0.0, 10.0, 1.5])
    m_tgt = camera_matrices(cam_tgt)
    assert not np.allclose(m_tgt["R"], m_base["R"])
    assert not np.allclose(m_tgt["P"], P_base)


def test_8_persistence(tmp_path):
    root, _ = create(tmp_path, "PersistenceTest", "room.dxf", sample())
    state, _, _ = load_for_editor(root)
    state["source"]["units"] = "m"
    state["source"]["units_confirmed"] = True

    entity_id = list(state["entities"].keys())[0]
    props = {
        "category": "window",
        "sill_height": 0.9,
        "opening_height": 1.2,
        "description": "condo window big screen",
        "notes": "south facing"
    }
    state = apply(root, state, [entity_id], "void", props)

    cam = new_camera(2.0, 3.0, "m", heading=45.0, name="Living Room")
    cam["output"] = {"width": 1920, "height": 1080}
    set_camera_target(cam, [5.0, 6.0, 1.5])
    save_cameras(root, state, {"camera_001": cam}, "camera_001")

    reloaded, _ = open_layout(root)
    saved_entity = reloaded["entities"][entity_id]
    assert saved_entity["category"] == "window"
    assert saved_entity["sill_height"] == 0.9
    assert saved_entity["opening_height"] == 1.2
    assert saved_entity["description"] == "condo window big screen"

    saved_cam = reloaded["cameras"]["camera_001"]
    assert saved_cam["output"] == {"width": 1920, "height": 1080}
    assert saved_cam["target"] == [5.0, 6.0, 1.5]
    assert saved_cam["heading_deg"] == pytest.approx(45.0)


def test_9_furniture_categories_and_refrigerator(tmp_path):
    assert "refrigerator" in CATEGORIES
    root, _ = create(tmp_path, "FridgeTest", "room.dxf", sample())
    state, _, _ = load_for_editor(root)
    state["source"]["units"] = "m"
    state["source"]["units_confirmed"] = True

    entity_id = list(state["entities"].keys())[0]
    props = {
        "category": "refrigerator",
        "height": 1.8,
        "description": "stainless steel double door"
    }
    updated = apply(root, state, [entity_id], "furniture", props)
    rec = updated["entities"][entity_id]
    assert rec["semantic"] == "furniture"
    assert rec["category"] == "refrigerator"
    assert rec["height"] == 1.8
    assert rec["description"] == "stainless steel double door"
