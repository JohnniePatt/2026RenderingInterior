import pytest
from test_dxf_parser import sample
from app.layout.manager import create
from app.cad.entities import load_for_editor
from app.annotation.tagging import apply, clear


@pytest.mark.parametrize("category,height,base", [("door", 2100, 0), ("window", 1200, 900), ("opening", 2400, 0)])
def test_void_reopen_and_semantic_change(tmp_path, category, height, base):
    root, _ = create(tmp_path, "Openings", "plan.dxf", sample())
    state, geometry, _ = load_for_editor(root)
    ids = [e["id"] for e in geometry[:2]]
    state = apply(root, state, ids, "void", {"category": category, "height": height,
        "base_elevation": base, "notes": "Opening test", "material": "glass"})
    restored, _, _ = load_for_editor(root)
    for entity_id in ids:
        record = restored["entities"][entity_id]
        assert record["semantic"] == "void"
        assert "height" not in record
        if category == "window":
            assert (record["category"], record["sill_height"], record["opening_height"]) == (category, base, height)
            assert "base_elevation" not in record
        else:
            assert (record["category"], record["base_elevation"], record["opening_height"]) == (category, base, height)
            assert "sill_height" not in record
    state = apply(root, restored, ids, "wall", {"height": 2800, "base_elevation": 0})
    assert "category" not in state["entities"][ids[0]]
    state = clear(root, state, ids)
    assert state["entities"][ids[0]]["semantic"] is None


@pytest.mark.parametrize("props", [
    {"category": "sofa", "height": 2100, "base_elevation": 0},
    {"category": "door", "height": 0, "base_elevation": 0},
    {"category": "window", "height": 1200},
    {"category": "window", "height": 1200, "base_elevation": float("nan")},
])
def test_invalid_void_does_not_overwrite_layout(tmp_path, props):
    root, _ = create(tmp_path, "Openings", "plan.dxf", sample())
    state, geometry, _ = load_for_editor(root)
    previous = (root / "layout.json").read_bytes()
    with pytest.raises(ValueError):
        apply(root, state, [geometry[0]["id"]], "void", props)
    assert (root / "layout.json").read_bytes() == previous
