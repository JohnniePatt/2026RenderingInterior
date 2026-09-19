"""Source-only CAD geometry. INSERT children remain owned by their top-level handle."""
from dataclasses import dataclass
from collections import Counter
import ezdxf
import numpy as np
from ezdxf.disassemble import recursive_decompose
from ezdxf.path import make_path
from app.cad.units import drawing_units, from_mm

SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE", "INSERT", "ARC", "CIRCLE"}


@dataclass
class ParsedCAD:
    entities: list
    units: str
    warnings: list


def parse(source):
    try:
        doc = ezdxf.readfile(source)
        audit = doc.audit()
        if audit.has_errors or audit.has_fixes:
            raise ValueError("DXF needs repair; source was not modified")
    except (ezdxf.DXFError, OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot parse DXF: {exc}") from exc
    unit = drawing_units(doc)
    tolerance = from_mm(1.0, unit)
    entities, warnings, skipped = [], [], Counter()
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind not in SUPPORTED:
            skipped[kind] += 1
            continue
        handle = entity.dxf.handle
        if not handle:
            warnings.append(f"{kind} without a stable handle was skipped")
            continue
        item = {"dxf_handle": handle, "dxf_type": kind, "layer": entity.dxf.layer, "paths": []}
        if kind == "INSERT":
            item.update(block_name=entity.dxf.name, insertion_point=list(entity.dxf.insert),
                        rotation=float(entity.dxf.rotation),
                        scale=[float(entity.dxf.get(axis + "scale", 1)) for axis in "xyz"])
        try:
            parts = recursive_decompose([entity]) if kind == "INSERT" else [entity]
            for part in parts:
                if part.dxftype() not in SUPPORTED - {"INSERT"} | {"ELLIPSE", "SPLINE"}:
                    skipped[f"block child {part.dxftype()}"] += 1
                    continue
                try:
                    path = make_path(part)
                    points = np.asarray([list(v) for v in path.flattening(distance=tolerance)], dtype=float)
                    if len(points) >= 2 and np.isfinite(points).all():
                        item["paths"].append(points.tolist())
                        if np.max(np.abs(points[:, 2])) > tolerance:
                            warnings.append(f"{handle}: non-zero Z projected into the XY plan")
                except (ValueError, TypeError, ezdxf.DXFError) as exc:
                    warnings.append(f"{handle}: cannot display child geometry ({exc})")
        except (ValueError, TypeError, ezdxf.DXFError, RecursionError) as exc:
            warnings.append(f"{handle}: geometry could not be fully resolved ({exc})")
        if not item["paths"]:
            warnings.append(f"{handle} ({kind}) has no displayable plan geometry")
        entities.append(item)
    if skipped:
        warnings.append("Unsupported geometry omitted: " + ", ".join(f"{k} × {v}" for k, v in skipped.items()))
    if unit.startswith("unknown"):
        warnings.append("CAD units are unknown. Confirm drawing units before entering metric annotations.")
    return ParsedCAD(entities, unit, list(dict.fromkeys(warnings)))
