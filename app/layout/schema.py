"""JSON schema and validation; all dimensions use source drawing units."""
import math
import re
from datetime import datetime
from pathlib import PurePosixPath

STATUSES = {"new", "annotating", "ready", "generated", "evaluated"}
SEMANTICS = {None, "wall", "floor", "furniture", "void"}
VOID_CATEGORIES = ("door", "window", "opening")


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Invalid layout-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Path must remain inside the Layout")
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(state):
    if not isinstance(state, dict) or state.get("schema_version") != "0.1":
        raise ValueError("Unsupported or missing layout schema version")
    try:
        meta = state["layout"]
        if not re.fullmatch(r"Layout_\d{8}_\d{6}", meta["id"]):
            raise ValueError("Invalid Layout ID")
        if not isinstance(meta["name"], str) or not meta["name"].strip():
            raise ValueError("Layout name is required")
        if meta["status"] not in STATUSES:
            raise ValueError("Invalid Layout status")
        for key in ("created_at", "updated_at"):
            datetime.fromisoformat(meta[key])
        relative_path(state["source"]["cad_file"])
        if state["source"]["format"] != "DXF" or not isinstance(state["source"]["units"], str):
            raise ValueError("Invalid source metadata")
        if not isinstance(state["entities"], dict) or not isinstance(state["cameras"], dict):
            raise ValueError("Entities and cameras must be objects")
        for entity in state["entities"].values():
            if not isinstance(entity, dict) or entity.get("semantic") not in SEMANTICS:
                raise ValueError("Invalid entity semantic")
            if not isinstance(entity.get("dxf_handle"), str) or not entity["dxf_handle"]:
                raise ValueError("Entity DXF handle is required")
            for key in ("height", "thickness", "base_elevation", "elevation", "sill_height", "opening_height"):
                if key in entity and not finite(entity[key]):
                    raise ValueError(f"Invalid {key}")
            for key in ("height", "thickness", "opening_height"):
                if key in entity and entity[key] < 0:
                    raise ValueError(f"{key} cannot be negative")
            if entity.get("semantic") == "void":
                if entity.get("category") not in VOID_CATEGORIES:
                    raise ValueError("Void category must be door, window or opening")
                cat = entity.get("category")
                if cat == "window":
                    has_new = finite(entity.get("sill_height")) and finite(entity.get("opening_height")) and entity["opening_height"] > 0
                    has_old = finite(entity.get("base_elevation")) and finite(entity.get("height")) and entity["height"] > 0
                    if not (has_new or has_old):
                        raise ValueError("Window requires sill_height and positive opening_height")
                else:
                    has_new = finite(entity.get("base_elevation")) and finite(entity.get("opening_height")) and entity["opening_height"] > 0
                    has_old = finite(entity.get("base_elevation")) and finite(entity.get("height")) and entity["height"] > 0
                    if not (has_new or has_old):
                        raise ValueError("Door/opening requires base_elevation and positive opening_height")
            if entity.get("reference_image"):
                relative_path(entity["reference_image"])
        for camera in state["cameras"].values():
            from app.geometry.camera import validate_camera
            validate_camera(camera)
        thickness = state.get("proxy_settings", {}).get("line_wall_thickness", 0)
        if not finite(thickness) or thickness < 0:
            raise ValueError("Proxy wall thickness must be a nonnegative number")
        style = state.get("proxy_settings", {}).get("furniture_style", "outer_boundary")
        if style not in ("outer_boundary", "box"):
            raise ValueError("Invalid furniture proxy style")
        auto_c = state.get("proxy_settings", {}).get("auto_ceiling", True)
        if not isinstance(auto_c, bool):
            raise ValueError("auto_ceiling must be a boolean")
        c_h = state.get("proxy_settings", {}).get("ceiling_height")
        if c_h is not None and (not finite(c_h) or c_h <= 0):
            raise ValueError("Ceiling height must be a positive number")
        ceil = state.get("ceiling")
        if ceil is not None:
            if not isinstance(ceil, dict):
                raise ValueError("Ceiling must be an object")
            if "enabled" in ceil and not isinstance(ceil["enabled"], bool):
                raise ValueError("Ceiling enabled must be a boolean")
            if "elevation" in ceil and ceil["elevation"] is not None and (not finite(ceil["elevation"]) or ceil["elevation"] < 0):
                raise ValueError("Ceiling elevation must be a nonnegative number")
            if "thickness" in ceil and ceil["thickness"] is not None and (not finite(ceil["thickness"]) or ceil["thickness"] < 0):
                raise ValueError("Ceiling thickness must be a nonnegative number")
            if "description" in ceil and ceil["description"] is not None and not isinstance(ceil["description"], str):
                raise ValueError("Ceiling description must be a string")
            if "material" in ceil and ceil["material"] is not None and not isinstance(ceil["material"], str):
                raise ValueError("Ceiling material must be a string")
        if state["active_camera"] is not None and state["active_camera"] not in state["cameras"]:
            raise ValueError("Active camera does not exist")
        if not isinstance(state["outputs"], dict) or not isinstance(state["evaluation"], dict):
            raise ValueError("Invalid research state")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed layout.json: {exc}") from exc


def new_state(layout_id, name, source):
    now = timestamp()
    return {
        "schema_version": "0.1",
        "layout": {"id": layout_id, "name": name.strip(), "status": "new", "created_at": now, "updated_at": now},
        "source": {"cad_file": source, "format": "DXF", "units": "unknown"},
        "entities": {}, "cameras": {}, "active_camera": None,
        "outputs": {key: [] for key in ("proxy", "depth", "instance", "semantic", "planar", "renders")},
        "evaluation": {},
    }


def migrate_entities_and_cameras(state):
    state.setdefault("outputs", {}).setdefault("planar", [])
    for entity in state.get("entities", {}).values():
        if entity.get("semantic") == "void":
            cat = entity.get("category", "door")
            if cat == "window":
                if "opening_height" not in entity and "height" in entity:
                    entity["opening_height"] = entity["height"]
                if "sill_height" not in entity and "base_elevation" in entity:
                    entity["sill_height"] = entity["base_elevation"]
                entity.pop("height", None)
                entity.pop("base_elevation", None)
            else:
                if "opening_height" not in entity and "height" in entity:
                    entity["opening_height"] = entity["height"]
                if "base_elevation" not in entity and "sill_height" in entity:
                    entity["base_elevation"] = entity["sill_height"]
                entity.pop("height", None)
                entity.pop("sill_height", None)
    for camera in state.get("cameras", {}).values():
        if "output" not in camera or not camera["output"]:
            camera["output"] = {"width": 1024, "height": 1024}
        if "target" not in camera or not isinstance(camera["target"], list) or len(camera["target"]) != 3:
            from app.geometry.camera import calculate_target
            camera["target"] = calculate_target(
                camera.get("position", [0, 0, 1.5]),
                camera.get("heading_deg", 0),
                camera.get("pitch_deg", 0)
            )
    ps = state.get("proxy_settings")
    if ps is not None:
        if "ceiling" in state and isinstance(state["ceiling"], dict):
            ceil = state["ceiling"]
            if "enabled" in ceil and "auto_ceiling" not in ps:
                ps["auto_ceiling"] = ceil["enabled"]
            if "description" in ceil and "ceiling_description" not in ps:
                ps["ceiling_description"] = ceil["description"]
            if "material" in ceil and "ceiling_material" not in ps:
                ps["ceiling_material"] = ceil["material"]
            if "elevation" in ceil and "ceiling_height" not in ps and ceil.get("source") == "explicit_ceiling_height":
                ps["ceiling_height"] = ceil["elevation"]
        elif "auto_ceiling" in ps or ps.get("ceiling_description") or ps.get("ceiling_height"):
            state["ceiling"] = {
                "enabled": bool(ps.get("auto_ceiling", True)),
                "elevation": ps.get("ceiling_height"),
                "description": ps.get("ceiling_description", ""),
                "material": ps.get("ceiling_material"),
                "source": "explicit_ceiling_height" if ps.get("ceiling_height") is not None else "auto_wall_top"
            }

