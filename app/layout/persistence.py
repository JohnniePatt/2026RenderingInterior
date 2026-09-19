"""Validated, atomic JSON persistence with stale-write protection."""
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from app.layout.schema import migrate_entities_and_cameras, relative_path, timestamp, validate


def resolve(root, relative):
    root = Path(root).resolve()
    target = (root / relative_path(relative)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path escapes Layout directory")
    return target


def read(root):
    try:
        state = json.loads((Path(root) / "layout.json").read_text(encoding="utf-8"))
        validate(state)
        if state["layout"]["id"] != Path(root).name:
            raise ValueError("Layout ID does not match its directory")
        migrate_entities_and_cameras(state)
        return state
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot open {Path(root).name}: {exc}") from exc


def save(root, state):
    root = Path(root)
    if state["layout"]["id"] != root.name:
        raise ValueError("Layout ID is immutable")
    target = root / "layout.json"
    # A short exclusive lock prevents simultaneous writers from racing the revision check.
    lock = root / ".save.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("Layout is being saved. Retry; if the app crashed, remove .save.lock after stopping it.") from exc
    temp = None
    try:
        os.close(fd)
        if target.exists() and read(root)["layout"]["updated_at"] != state["layout"]["updated_at"]:
            raise ValueError("Layout changed in another session. Reopen it before saving.")
        migrate_entities_and_cameras(state)
        candidate = deepcopy(state)
        candidate["layout"]["updated_at"] = timestamp()
        # Compute and persist canonical ceiling details
        wall_tops = [
            float(rec.get("base_elevation", 0)) + float(rec.get("height", 0))
            for rec in candidate.get("entities", {}).values()
            if rec.get("semantic") == "wall" and not rec.get("orphaned") and float(rec.get("height", 0)) > 0
        ]
        has_ceiling_config = "proxy_settings" in candidate or "ceiling" in candidate or wall_tops
        if has_ceiling_config:
            unit = candidate.get("source", {}).get("units", "m")
            from app.cad.units import from_mm
            c_thickness = from_mm(50, unit) if not unit.startswith("unknown") else 0.05
            ps = candidate.setdefault("proxy_settings", {})
            auto_ceiling = ps.get("auto_ceiling", True)
            custom_h = ps.get("ceiling_height")
            if custom_h is not None:
                c_elev = float(custom_h)
                c_source = "explicit_ceiling_height"
            elif wall_tops:
                c_elev = max(wall_tops)
                c_source = "auto_wall_top"
            else:
                c_elev = None
                c_source = "pending_wall_tags"

            c_desc = ps.get("ceiling_description") or candidate.get("ceiling", {}).get("description") or ""
            c_mat = ps.get("ceiling_material") or candidate.get("ceiling", {}).get("material")

            candidate["ceiling"] = {
                "enabled": bool(auto_ceiling),
                "elevation": round(c_elev, 4) if c_elev is not None else None,
                "thickness": round(c_thickness, 4),
                "description": c_desc,
                "material": c_mat or None,
                "source": c_source
            }
            state["ceiling"] = deepcopy(candidate["ceiling"])
            state["proxy_settings"] = deepcopy(candidate["proxy_settings"])

        validate(candidate)
        # Ensure no legacy void fields are emitted by the serializer
        for entity in candidate.get("entities", {}).values():
            if entity.get("semantic") == "void":
                cat = entity.get("category")
                if cat == "window":
                    entity.pop("height", None)
                    entity.pop("base_elevation", None)
                else:
                    entity.pop("height", None)
                    entity.pop("sill_height", None)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(candidate, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        if os.name == "posix":
            directory = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        state["layout"]["updated_at"] = candidate["layout"]["updated_at"]
    finally:
        if temp and temp.exists():
            temp.unlink()
        lock.unlink(missing_ok=True)
