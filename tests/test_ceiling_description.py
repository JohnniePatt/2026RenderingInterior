import json
import pytest
from app.layout.manager import create
from app.layout.persistence import read, save
from app.annotation.tagging import apply
from app.geometry.proxy import build_proxy
from app.layout.schema import validate


def test_ceiling_persistence_in_layout_json(tmp_path):
    root, state = create(tmp_path, "Ceiling Test", "room.dxf", b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF")
    # Add wall and floor entities
    state["entities"] = {
        "wall_1": {
            "semantic": "wall",
            "dxf_handle": "W1",
            "dxf_type": "LINE",
            "height": 2.8,
            "base_elevation": 0.0,
            "description": "Concrete accent wall",
            "material": "concrete",
            "notes": ""
        },
        "floor_1": {
            "semantic": "floor",
            "dxf_handle": "F1",
            "dxf_type": "LWPOLYLINE",
            "elevation": 0.0,
            "thickness": 0.15,
            "description": "Light oak hardwood flooring",
            "material": "oak",
            "notes": ""
        }
    }
    state["source"]["units"] = "m"
    state["proxy_settings"] = {
        "auto_ceiling": True,
        "ceiling_description": "White gypsum board ceiling with cove lighting",
        "ceiling_material": "gypsum"
    }

    save(root, state)

    # 1. Verify layout.json directly on disk
    raw_json = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    assert "ceiling" in raw_json
    ceiling = raw_json["ceiling"]
    assert ceiling["enabled"] is True
    assert ceiling["elevation"] == 2.8
    assert ceiling["thickness"] == 0.05
    assert ceiling["description"] == "White gypsum board ceiling with cove lighting"
    assert ceiling["material"] == "gypsum"
    assert ceiling["source"] == "auto_wall_top"

    # 2. Verify read() reloads correctly
    reopened = read(root)
    assert reopened["ceiling"]["description"] == "White gypsum board ceiling with cove lighting"
    assert reopened["ceiling"]["elevation"] == 2.8


def test_ceiling_custom_elevation_override(tmp_path):
    root, state = create(tmp_path, "Ceiling Override Test", "room.dxf", b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF")
    state["entities"] = {
        "wall_1": {
            "semantic": "wall",
            "dxf_handle": "W1",
            "dxf_type": "LINE",
            "height": 2.8,
            "base_elevation": 0.0,
        }
    }
    state["source"]["units"] = "m"
    state["proxy_settings"] = {
        "auto_ceiling": True,
        "ceiling_height": 3.4,
        "ceiling_description": "High exposed industrial ceiling",
        "ceiling_material": "concrete"
    }

    save(root, state)

    reopened = read(root)
    assert reopened["ceiling"]["elevation"] == 3.4
    assert reopened["ceiling"]["source"] == "explicit_ceiling_height"
    assert reopened["ceiling"]["description"] == "High exposed industrial ceiling"


def test_build_proxy_attaches_ceiling_description():
    floor_poly = [[0, 0], [5, 0], [5, 4], [0, 4], [0, 0]]
    wall_poly = [[0, 0], [5, 0], [5, 0.2], [0, 0.2], [0, 0]]
    state = {
        "source": {"units": "m"},
        "entities": {
            "floor_1": {"semantic": "floor", "elevation": 0, "thickness": 0.15, "dxf_handle": "F1"},
            "wall_1": {"semantic": "wall", "height": 3.0, "base_elevation": 0, "dxf_handle": "W1"}
        },
        "proxy_settings": {
            "auto_ceiling": True,
            "ceiling_description": "Modern minimalist white ceiling with recessed LED downlights",
            "ceiling_material": "matte paint"
        }
    }
    geometry = [
        {"id": "floor_1", "paths": [floor_poly]},
        {"id": "wall_1", "paths": [wall_poly]}
    ]

    proxy = build_proxy(state, geometry)
    ceiling_meshes = [m for m in proxy["meshes"] if m["semantic"] == "ceiling"]
    assert len(ceiling_meshes) == 1
    assert ceiling_meshes[0]["description"] == "Modern minimalist white ceiling with recessed LED downlights"
    assert ceiling_meshes[0]["material"] == "matte paint"
    assert ceiling_meshes[0]["z_min"] == 3.0


def test_wall_and_floor_description_persistence(tmp_path):
    root, state = create(tmp_path, "Wall Floor Desc Test", "room.dxf", b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF")
    state["entities"] = {
        "wall_1": {"dxf_handle": "W1", "dxf_type": "LINE", "semantic": None},
        "floor_1": {"dxf_handle": "F1", "dxf_type": "LWPOLYLINE", "semantic": None}
    }
    state["source"]["units"] = "m"

    state = apply(root, state, ["wall_1"], "wall", {
        "height": 2.8, "base_elevation": 0.0, "description": "Textured plaster wall grey tone", "material": "plaster"
    })
    state = apply(root, state, ["floor_1"], "floor", {
        "elevation": 0.0, "thickness": 0.15, "description": "Seamless polished concrete floor", "material": "concrete"
    })

    save(root, state)
    reopened = read(root)
    assert reopened["entities"]["wall_1"]["description"] == "Textured plaster wall grey tone"
    assert reopened["entities"]["floor_1"]["description"] == "Seamless polished concrete floor"


def test_ceiling_schema_validation():
    # Invalid ceiling type
    bad_state_1 = {
        "schema_version": "0.1",
        "layout": {"id": "Layout_20260919_120000", "name": "T", "status": "new", "created_at": "2026-09-19T00:00:00+00:00", "updated_at": "2026-09-19T00:00:00+00:00"},
        "source": {"cad_file": "source.dxf", "format": "DXF", "units": "m"},
        "entities": {}, "cameras": {}, "active_camera": None,
        "outputs": {k: [] for k in ("proxy", "depth", "instance", "semantic", "renders")},
        "evaluation": {},
        "ceiling": "not an object"
    }
    with pytest.raises(ValueError, match="Ceiling must be an object"):
        validate(bad_state_1)

    # Invalid ceiling elevation
    bad_state_2 = {
        "schema_version": "0.1",
        "layout": {"id": "Layout_20260919_120000", "name": "T", "status": "new", "created_at": "2026-09-19T00:00:00+00:00", "updated_at": "2026-09-19T00:00:00+00:00"},
        "source": {"cad_file": "source.dxf", "format": "DXF", "units": "m"},
        "entities": {}, "cameras": {}, "active_camera": None,
        "outputs": {k: [] for k in ("proxy", "depth", "instance", "semantic", "renders")},
        "evaluation": {},
        "ceiling": {"elevation": -5.0}
    }
    with pytest.raises(ValueError, match="Ceiling elevation must be a nonnegative number"):
        validate(bad_state_2)
