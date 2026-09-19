import json
import pytest
from tests.test_dxf_parser import sample
from app.layout.manager import create, open_layout
from app.layout.persistence import read, save
from app.geometry.proxy import build_proxy


def test_1_old_window_json_loads_correctly(tmp_path):
    root, state = create(tmp_path, "LegacyWindowTest", "room.dxf", sample())
    entity_id = "entity_00001"

    # Write raw old Window JSON with legacy height and base_elevation
    state["entities"][entity_id] = {
        "semantic": "void",
        "category": "window",
        "height": 1.5,
        "base_elevation": 0.8,
        "material": "glass",
        "notes": "legacy window notes",
        "dxf_handle": "2FF",
        "orphaned": False
    }
    (root / "layout.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    # 1. Old Window JSON loads correctly: height -> opening_height, base_elevation -> sill_height
    loaded = read(root)
    win = loaded["entities"][entity_id]
    assert win["semantic"] == "void"
    assert win["category"] == "window"
    assert win["sill_height"] == pytest.approx(0.8)
    assert win["opening_height"] == pytest.approx(1.5)
    assert "height" not in win
    assert "base_elevation" not in win
    assert win["material"] == "glass"
    assert win["notes"] == "legacy window notes"


def test_2_saving_window_removes_height_and_base_elevation(tmp_path):
    root, state = create(tmp_path, "LegacyWindowSaveTest", "room.dxf", sample())
    entity_id = "entity_00001"
    state["entities"][entity_id] = {
        "semantic": "void",
        "category": "window",
        "height": 1.5,
        "base_elevation": 0.8,
        "material": "aluminum",
        "notes": "south facing",
        "dxf_handle": "2FF",
        "orphaned": False
    }
    (root / "layout.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    loaded = read(root)
    # 2. Saving it removes height and base_elevation and writes sill_height + opening_height
    save(root, loaded)

    raw_disk = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    disk_win = raw_disk["entities"][entity_id]
    assert disk_win["semantic"] == "void"
    assert disk_win["category"] == "window"
    assert disk_win["sill_height"] == pytest.approx(0.8)
    assert disk_win["opening_height"] == pytest.approx(1.5)
    assert "height" not in disk_win
    assert "base_elevation" not in disk_win
    assert disk_win["material"] == "aluminum"
    assert disk_win["notes"] == "south facing"


def test_3_old_door_json_loads_correctly(tmp_path):
    root, state = create(tmp_path, "LegacyDoorTest", "room.dxf", sample())
    entity_id = "entity_00001"

    # Write raw old Door JSON with legacy height and base_elevation
    state["entities"][entity_id] = {
        "semantic": "void",
        "category": "door",
        "height": 2.1,
        "base_elevation": 0.0,
        "material": "wood",
        "notes": "entry door",
        "dxf_handle": "38C",
        "orphaned": False
    }
    (root / "layout.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    # 3. Old Door JSON loads correctly: height -> opening_height, base_elevation remains
    loaded = read(root)
    door = loaded["entities"][entity_id]
    assert door["semantic"] == "void"
    assert door["category"] == "door"
    assert door["base_elevation"] == pytest.approx(0.0)
    assert door["opening_height"] == pytest.approx(2.1)
    assert "height" not in door
    assert "sill_height" not in door
    assert door["material"] == "wood"
    assert door["notes"] == "entry door"


def test_4_saving_door_removes_height_and_writes_base_elevation_and_opening_height(tmp_path):
    root, state = create(tmp_path, "LegacyDoorSaveTest", "room.dxf", sample())
    entity_id = "entity_00001"
    state["entities"][entity_id] = {
        "semantic": "void",
        "category": "door",
        "height": 2.1,
        "base_elevation": 0.0,
        "material": "oak",
        "notes": "bedroom door",
        "dxf_handle": "38C",
        "orphaned": False
    }
    (root / "layout.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    loaded = read(root)
    # 4. Saving it removes height and writes base_elevation + opening_height
    save(root, loaded)

    raw_disk = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    disk_door = raw_disk["entities"][entity_id]
    assert disk_door["semantic"] == "void"
    assert disk_door["category"] == "door"
    assert disk_door["base_elevation"] == pytest.approx(0.0)
    assert disk_door["opening_height"] == pytest.approx(2.1)
    assert "height" not in disk_door
    assert "sill_height" not in disk_door
    assert disk_door["material"] == "oak"
    assert disk_door["notes"] == "bedroom door"


def test_5_reopening_newly_saved_layout_produces_identical_proxy_geometry(tmp_path):
    root, state = create(tmp_path, "ProxyIdenticalTest", "room.dxf", sample())
    state["source"]["units"] = "m"
    state["source"]["units_confirmed"] = True

    # Old style window and door
    old_window = {
        "semantic": "void",
        "category": "window",
        "height": 1.2,
        "base_elevation": 0.9,
        "dxf_handle": "win_1",
        "orphaned": False
    }
    old_door = {
        "semantic": "void",
        "category": "door",
        "height": 2.1,
        "base_elevation": 0.0,
        "dxf_handle": "door_1",
        "orphaned": False
    }
    state["entities"] = {"win_ent": old_window, "door_ent": old_door}
    geometry = [
        {"id": "win_ent", "paths": [[[0, 0], [2, 0], [2, 0.2], [0, 0.2], [0, 0]]]},
        {"id": "door_ent", "paths": [[[3, 0], [4, 0], [4, 0.2], [3, 0.2], [3, 0]]]}
    ]

    # Build proxy before migration/save
    proxy_before = build_proxy(state, geometry)

    # Save (migrating to canonical schema)
    save(root, state)

    # Reopen using open_layout
    reopened, _ = open_layout(root)
    proxy_after = build_proxy(reopened, geometry)

    # 5. Reopening the newly saved Layout produces identical proxy geometry
    assert len(proxy_before["meshes"]) == len(proxy_after["meshes"]) == 2
    for mb, ma in zip(proxy_before["meshes"], proxy_after["meshes"]):
        assert mb["semantic"] == ma["semantic"] == "void"
        assert mb["z_min"] == pytest.approx(ma["z_min"])
        assert mb["z_max"] == pytest.approx(ma["z_max"])
        assert mb["outer"] == ma["outer"]
    assert proxy_before["bounds"] == proxy_after["bounds"]
