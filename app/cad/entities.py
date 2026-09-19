"""Match by DXF handle, never by modelspace order; retain missing annotations."""
from copy import deepcopy
import hashlib
from app.cad.parser import parse
from app.layout.manager import open_layout
from app.layout.persistence import save
from app.layout.schema import migrate_entities_and_cameras


def reconcile(state, parsed):
    records = state["entities"]
    handles = {}
    for entity_id, record in records.items():
        handle = record["dxf_handle"]
        if handle in handles:
            raise ValueError(f"Duplicate persisted DXF handle: {handle}")
        handles[handle] = entity_id
        record["orphaned"] = True
    counter = max([int(k.removeprefix("entity_")) for k in records if k.startswith("entity_") and k[7:].isdigit()] or [0])
    viewport = []
    for item in parsed.entities:
        handle = item["dxf_handle"]
        entity_id = handles.get(handle)
        if entity_id is None:
            counter += 1
            entity_id = f"entity_{counter:05d}"
            records[entity_id] = {"semantic": None}
        records[entity_id].update({k: v for k, v in item.items() if k != "paths"})
        records[entity_id]["orphaned"] = False
        viewport.append({"id": entity_id, "paths": item["paths"]})
    migrate_entities_and_cameras(state)
    state["source"]["dxf_units"] = parsed.units
    if not state["source"].get("units_confirmed"):
        state["source"]["units"] = parsed.units
    if state["layout"]["status"] == "new":
        state["layout"]["status"] = "annotating"
    return viewport


def load_for_editor(root):
    state, source = open_layout(root)
    before = deepcopy(state)
    parsed = parse(source)
    viewport = reconcile(state, parsed)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != state["source"].get("sha256"):
        parsed.warnings.append("Canonical DXF has changed since import. Handle matches were restored; review geometry and orphaned annotations.")
    orphaned = sum(bool(e.get("orphaned")) for e in state["entities"].values())
    if orphaned:
        parsed.warnings.append(f"{orphaned} missing entities retained as orphaned annotations.")
    if state != before:
        save(root, state)
    return state, viewport, parsed.warnings
