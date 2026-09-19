import io
import json
import pytest
import ezdxf
from app.layout.manager import create, open_layout, DIRECTORIES
from app.layout.persistence import read, save
from app.layout.discovery import discover


def drawing():
    doc = ezdxf.new("R2010")
    doc.modelspace().add_line((0, 0), (100, 100))
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode()


def test_create_reopen_rename_and_duplicates(tmp_path):
    content = drawing()
    root, state = create(tmp_path, "Living Room Test 01", "../../floorplan.dxf", content)
    assert all((root / p).is_dir() for p in DIRECTORIES)
    reopened, source = open_layout(root)
    assert source.read_bytes() == content
    assert reopened == state
    create(tmp_path, "Living Room Test 01", "floorplan.dxf", content)
    layouts, labels, errors = discover(tmp_path)
    assert len(layouts) == 2 and len(set(labels.values())) == 2 and not errors
    state["layout"]["name"] = "Renamed"
    save(root, state)
    assert read(root)["layout"]["name"] == "Renamed"
    assert root.name == state["layout"]["id"]


def test_bad_save_preserves_previous_json(tmp_path):
    root, state = create(tmp_path, "Test", "test.dxf", drawing())
    before = (root / "layout.json").read_bytes()
    state["layout"]["status"] = "wrong"
    with pytest.raises(ValueError):
        save(root, state)
    assert (root / "layout.json").read_bytes() == before
    assert not (root / ".save.lock").exists()


def test_stale_write_and_corrupt_discovery(tmp_path):
    root, state = create(tmp_path, "Test", "test.dxf", drawing())
    stale = read(root)
    save(root, state)
    with pytest.raises(ValueError, match="another session"):
        save(root, stale)
    (root / "layout.json").write_text("{bad json")
    assert len(discover(tmp_path)[2]) == 1


@pytest.mark.parametrize("name,data", [("test.dwg", b"bad"), ("test.dxf", b"bad"), ("test.dxf", b"")])
def test_invalid_import(tmp_path, name, data):
    with pytest.raises(ValueError):
        create(tmp_path, "Test", name, data)
    assert not list(tmp_path.glob("Layout_*"))
