from copy import deepcopy
import io
import ezdxf
from app.layout.manager import create
from app.layout.persistence import read
from app.cad.entities import load_for_editor, reconcile
from app.cad.parser import parse
from app.annotation.tagging import apply


def sample():
    doc = ezdxf.new("R2010")
    doc.units = 4
    model = doc.modelspace()
    model.add_line((0, 0), (6000, 0), dxfattribs={"layer": "A-WALL"})
    model.add_lwpolyline([(0, 0), (6000, 0), (6000, 4000), (0, 4000)], close=True)
    block = doc.blocks.new("SOFA_A01")
    block.add_lwpolyline([(0, 0), (2000, 0), (2000, 800), (0, 800)], close=True)
    model.add_blockref("SOFA_A01", (500, 500), dxfattribs={"rotation": 30, "xscale": 2})
    model.add_arc((3000, 2000), 500, 0, 90)
    model.add_circle((4000, 2000), 350)
    model.add_polyline2d([(10, 10), (20, 10), (20, 20)])
    model.add_lwpolyline([(0, 0, 1), (1000, 0, 0)], format="xyb")
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode()


def test_types_blocks_curves_and_no_inferred_semantics(tmp_path):
    root, _ = create(tmp_path, "Test", "test.dxf", sample())
    state, viewport, warnings = load_for_editor(root)
    assert len(viewport) == 7
    assert all(e["semantic"] is None for e in state["entities"].values())
    assert {e["dxf_type"] for e in state["entities"].values()} == {"LINE", "LWPOLYLINE", "POLYLINE", "INSERT", "ARC", "CIRCLE"}
    block = viewport[2]["paths"][0]
    assert block[0][:2] == [500, 500]
    assert block[1][0] > 3900 and block[1][1] > 2400
    assert len(viewport[-1]["paths"][0]) > 4  # bulge is curved, not a straight chord


def test_acceptance_annotations_reopen_and_orphans(tmp_path):
    root, _ = create(tmp_path, "Living Room Test 01", "floorplan.dxf", sample())
    state, viewport, _ = load_for_editor(root)
    annotations = [("wall", {"height": 2800, "base_elevation": 0}),
                   ("floor", {"thickness": 150, "elevation": 0}),
                   ("furniture", {"height": 820, "category": "sofa"})]
    for item, (semantic, props) in zip(viewport, annotations):
        state = apply(root, state, [item["id"]], semantic, props)
    expected = deepcopy(state["entities"])
    restored, _, _ = load_for_editor(root)
    assert restored["entities"] == expected
    source = root / "source/floorplan.dxf"
    doc = ezdxf.readfile(source)
    victim = list(doc.modelspace())[0]
    doc.modelspace().delete_entity(victim)
    doc.saveas(source)
    restored, _, warnings = load_for_editor(root)
    first = restored["entities"][viewport[0]["id"]]
    assert first["orphaned"] and first["height"] == 2800
    assert any("changed" in w for w in warnings)
    parsed = parse(source)
    parsed.entities.reverse()
    reconcile(restored, parsed)
    assert restored["entities"][viewport[2]["id"]]["height"] == 820
    assert read(root)["entities"][viewport[0]["id"]]["orphaned"]
