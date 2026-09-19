import pytest
from app.geometry.proxy import (
    FURNITURE_PALETTE,
    get_block_key,
    compute_furniture_colors,
    build_proxy,
)


def test_get_block_key_priority():
    # 1. Description takes top priority
    rec1 = {
        "description": "Dining Table Set",
        "category": "table",
        "block_name": "BLK_TABLE_01",
    }
    assert get_block_key(rec1, "ent_1") == "desc:dining table set"

    # 2. Category if no description
    rec2 = {
        "description": "",
        "category": "chair",
        "block_name": "BLK_CHAIR_01",
    }
    assert get_block_key(rec2, "ent_2") == "cat:chair"

    # 3. Block name if no description and category is 'other' or empty
    rec3 = {
        "description": None,
        "category": "other",
        "block_name": "BLK_SOFA_CUSTOM",
    }
    assert get_block_key(rec3, "ent_3") == "block:BLK_SOFA_CUSTOM"

    # 4. Entity ID fallback
    rec4 = {
        "description": "",
        "category": "",
        "block_name": "",
    }
    assert get_block_key(rec4, "ent_4") == "id:ent_4"


def test_compute_furniture_colors_grouping_and_distinction():
    state = {
        "entities": {
            "ent_1": {
                "semantic": "furniture",
                "description": "Dining Table",
            },
            "ent_2": {
                "semantic": "furniture",
                "description": "Dining Table",  # Same block set / description
            },
            "ent_3": {
                "semantic": "furniture",
                "description": "Bed King Size",  # Different block set
            },
            "ent_4": {
                "semantic": "furniture",
                "description": "Refrigerator",   # Different block set
            },
            "ent_wall": {
                "semantic": "wall",
                "description": "Concrete Wall",
            },
            "ent_orphaned": {
                "semantic": "furniture",
                "description": "Ghost Chair",
                "orphaned": True,
            }
        }
    }

    colors = compute_furniture_colors(state)

    # ent_1 and ent_2 have the exact same description -> must have identical color
    assert colors["ent_1"] == colors["ent_2"]

    # ent_1, ent_3, ent_4 have different descriptions -> must have distinct colors
    assert colors["ent_1"] != colors["ent_3"]
    assert colors["ent_3"] != colors["ent_4"]
    assert colors["ent_1"] != colors["ent_4"]

    # All assigned colors must come from FURNITURE_PALETTE
    for color in colors.values():
        assert color in FURNITURE_PALETTE

    # Non-furniture or orphaned entities should not be in furniture_colors
    assert "ent_wall" not in colors
    assert "ent_orphaned" not in colors


def test_build_proxy_attaches_furniture_colors():
    # Simple rectangle path: 0,0 to 1,1
    rect_paths = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]]
    geometry = [
        {"id": "ent_table", "paths": rect_paths},
        {"id": "ent_chair", "paths": rect_paths},
        {"id": "ent_wall", "paths": rect_paths},
    ]

    state = {
        "source": {"units": "m"},
        "entities": {
            "ent_table": {
                "semantic": "furniture",
                "description": "Dining Table",
                "height": 0.75,
                "base_elevation": 0.0,
            },
            "ent_chair": {
                "semantic": "furniture",
                "description": "Office Chair",
                "height": 0.9,
                "base_elevation": 0.0,
            },
            "ent_wall": {
                "semantic": "wall",
                "height": 2.8,
                "base_elevation": 0.0,
            },
        },
        "proxy_settings": {
            "auto_ceiling": False,
            "furniture_style": "outer_boundary",
        }
    }

    result = build_proxy(state, geometry)
    meshes = result["meshes"]

    table_mesh = next(m for m in meshes if m["entity_id"] == "ent_table")
    chair_mesh = next(m for m in meshes if m["entity_id"] == "ent_chair")
    wall_mesh = next(m for m in meshes if m["entity_id"] == "ent_wall")

    assert "color" in table_mesh
    assert "color" in chair_mesh
    assert table_mesh["color"] != chair_mesh["color"]
    assert table_mesh["color"] in FURNITURE_PALETTE
    assert chair_mesh["color"] in FURNITURE_PALETTE

    # Wall mesh should NOT have a furniture color attached
    assert "color" not in wall_mesh
    assert wall_mesh["semantic"] == "wall"
