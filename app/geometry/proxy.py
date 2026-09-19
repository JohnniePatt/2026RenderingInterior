"""CAD-derived metric 3D proxy descriptors shared by all viewports.

No bounding-box substitutions. Closed nested rings use even/odd containment;
open walls are surfaces unless an explicitly persisted proxy thickness is set.
The original tessellated CAD paths are never mutated.
"""
import math
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, polygonize
from shapely.validation import explain_validity
from app.cad.units import from_mm


def closed_regions(paths, warnings, entity_id):
    rings, opened = [], []
    for path in paths:
        points = [(p[0], p[1]) for p in path]
        if len(points) < 2:
            continue
        if len(points) >= 4 and math.dist(points[0], points[-1]) < 1e-8:
            polygon = Polygon(points)
            if not polygon.is_valid or polygon.area <= 0:
                warnings.append(f"{entity_id}: invalid closed footprint skipped ({explain_validity(polygon)})")
            elif not any(polygon.equals(existing) for existing in rings):
                rings.append(polygon)
        else:
            opened.append(points)
    # Parent is the smallest enclosing ring, independent of DXF winding/order.
    rings.sort(key=lambda p: p.area, reverse=True)
    parents, depths = [], []
    for index, ring in enumerate(rings):
        enclosing = [i for i in range(index) if rings[i].contains(ring)]
        parent = enclosing[-1] if enclosing else None
        parents.append(parent)
        depths.append(depths[parent] + 1 if parent is not None else 0)
    regions = []
    for i, ring in enumerate(rings):
        if depths[i] % 2 == 0:
            holes = [list(rings[j].exterior.coords) for j in range(len(rings)) if parents[j] == i]
            regions.append(Polygon(ring.exterior.coords, holes))
    return regions, opened


def cluster_paths(lines, buffer_dist=0.03):
    """Cluster lines based on spatial proximity."""
    if not lines:
        return []
    buffered = [l.buffer(buffer_dist) for l in lines]
    unified = unary_union(buffered)
    clusters = [unified] if unified.geom_type == "Polygon" else list(unified.geoms)
    grouped = []
    for cluster in clusters:
        cl = [l for l in lines if l.intersects(cluster)]
        if cl:
            grouped.append(cl)
    return grouped


def process_furniture(paths, warnings, entity_id, unit="m", style="outer_boundary"):
    simple_rings, opened = [], []
    for p in paths:
        points = [(pt[0], pt[1]) for pt in p]
        if len(points) < 2:
            continue
        if len(points) >= 4 and math.dist(points[0], points[-1]) < 1e-8:
            poly = Polygon(points)
            if poly.is_valid and poly.area > 0:
                if not any(poly.equals(existing) for existing in simple_rings):
                    simple_rings.append(poly)
            else:
                warnings.append(f"{entity_id}: invalid closed footprint skipped ({explain_validity(poly)})")
        else:
            opened.append(points)

    # If all paths were already closed rings, use standard nested ring processing
    if simple_rings and not opened:
        simple_rings.sort(key=lambda p: p.area, reverse=True)
        parents, depths = [], []
        for index, ring in enumerate(simple_rings):
            enclosing = [i for i in range(index) if simple_rings[i].contains(ring)]
            parent = enclosing[-1] if enclosing else None
            parents.append(parent)
            depths.append(depths[parent] + 1 if parent is not None else 0)
        regions = []
        for i, ring in enumerate(simple_rings):
            if depths[i] % 2 == 0:
                holes = [list(simple_rings[j].exterior.coords) for j in range(len(simple_rings)) if parents[j] == i]
                regions.append(Polygon(ring.exterior.coords, holes))
        return regions, []

    # Single open path without closure and not in box mode: omit per strict metric rules
    if len(paths) <= 1 and not simple_rings and style != "box":
        return [], opened

    buffer_dist = from_mm(30, unit)
    dec = 6 if unit in ("m", "ft") else 3
    lines = [LineString([(round(p[0], dec), round(p[1], dec)) for p in path]) for path in paths if len(path) >= 2]
    if not lines:
        return [], paths

    all_polys = list(polygonize(unary_union(lines)))
    valid_polys = [p for p in all_polys if p.is_valid and p.area > 1e-8]

    if not valid_polys and len(paths) <= 1 and style != "box":
        return [], paths

    clusters = cluster_paths(lines, buffer_dist=buffer_dist)
    footprints = []

    for cl_lines in clusters:
        cl_union = unary_union(cl_lines)
        if style == "box":
            obb = cl_union.minimum_rotated_rectangle
            if obb.geom_type == "Polygon" and obb.area > 1e-8:
                footprints.append(obb)
            continue

        cl_polys = [p for p in valid_polys if cl_union.intersects(p) or p.intersects(cl_union)]
        if cl_polys:
            merged = unary_union(cl_polys)
            if merged.geom_type == "Polygon":
                footprints.append(merged)
            elif merged.geom_type == "MultiPolygon":
                footprints.extend(list(merged.geoms))
        elif len(cl_lines) >= 2 and cl_union.convex_hull.area > 1e-6:
            hull = cl_union.convex_hull
            if hull.geom_type == "Polygon":
                footprints.append(hull)

    return footprints, [] if footprints else opened


def normalize_opening(record):
    cat = record.get("category", "door")
    if cat == "window":
        sill = float(record["sill_height"]) if "sill_height" in record else float(record.get("base_elevation", 0.0))
        h = float(record["opening_height"]) if "opening_height" in record else float(record.get("height", 0.0))
        bottom_z = sill
        top_z = sill + h
        return {
            "category": cat,
            "sill_height": sill,
            "opening_height": h,
            "bottom_z": bottom_z,
            "top_z": top_z
        }
    else:
        base = float(record["base_elevation"]) if "base_elevation" in record else float(record.get("sill_height", 0.0))
        h = float(record["opening_height"]) if "opening_height" in record else float(record.get("height", 0.0))
        bottom_z = base
        top_z = base + h
        return {
            "category": cat,
            "base_elevation": base,
            "opening_height": h,
            "bottom_z": bottom_z,
            "top_z": top_z
        }


FURNITURE_PALETTE = [
    "#b99af7",  # Soft Lavender / Purple
    "#f59e0b",  # Warm Amber / Orange
    "#38bdf8",  # Sky Blue
    "#10b981",  # Emerald Green
    "#ec4899",  # Rose Pink
    "#6366f1",  # Royal Indigo
    "#84cc16",  # Lime Green
    "#ef4444",  # Crimson Red
    "#14b8a6",  # Deep Teal
    "#eab308",  # Warm Gold
    "#a855f7",  # Vivid Purple
    "#f97316",  # Tangerine Coral
]


def get_block_key(record, entity_id):
    desc = (record.get("description") or "").strip().lower()
    if desc:
        return f"desc:{desc}"
    cat = (record.get("category") or "").strip().lower()
    if cat and cat not in ("other", ""):
        return f"cat:{cat}"
    block = (record.get("block_name") or "").strip()
    if block:
        return f"block:{block}"
    return f"id:{entity_id}"


def compute_furniture_colors(state):
    """Assign deterministic distinct colors to furniture block sets based on description/identity."""
    entities = state.get("entities", {})
    unique_keys = sorted({
        get_block_key(rec, entity_id)
        for entity_id, rec in entities.items()
        if rec.get("semantic") == "furniture" and not rec.get("orphaned")
    })
    key_to_color = {
        key: FURNITURE_PALETTE[i % len(FURNITURE_PALETTE)]
        for i, key in enumerate(unique_keys)
    }
    entity_colors = {}
    for entity_id, rec in entities.items():
        if rec.get("semantic") == "furniture" and not rec.get("orphaned"):
            key = get_block_key(rec, entity_id)
            entity_colors[entity_id] = key_to_color.get(key, FURNITURE_PALETTE[0])
    return entity_colors


def build_proxy(state, geometry):
    descriptors, warnings = [], []
    proxy_settings = state.get("proxy_settings", {})
    thickness = proxy_settings.get("line_wall_thickness", 0.0)
    furniture_style = proxy_settings.get("furniture_style", "outer_boundary")
    auto_ceiling = proxy_settings.get("auto_ceiling", True)
    custom_ceiling_height = proxy_settings.get("ceiling_height")
    unit = state.get("source", {}).get("units", "m")
    if unit.startswith("unknown"):
        return {"meshes": [], "warnings": ["Confirm drawing units before building a metric proxy."], "bounds": None}
    furniture_colors = compute_furniture_colors(state)
    for entity in geometry:
        entity_id = entity["id"]
        record = state["entities"].get(entity_id, {})
        semantic = record.get("semantic")
        if record.get("orphaned") or not semantic:
            continue
        if semantic == "void":
            norm = normalize_opening(record)
            bottom = norm["bottom_z"]
            top = norm["top_z"]
            if top <= bottom:
                warnings.append(f"{entity_id}: zero-height opening skipped")
                continue
            regions, opened = closed_regions(entity["paths"], warnings, entity_id)
            offset_dist = from_mm(5, unit)
            if opened:
                if thickness > 0:
                    regions += [LineString(path).buffer(thickness / 2 + offset_dist, cap_style="flat", join_style="mitre") for path in opened]
                else:
                    warnings.append(f"{entity_id}: {len(opened)} open paths omitted from {semantic} footprint; no box inferred.")
            if regions:
                merged = unary_union(regions)
                buffered = merged.buffer(offset_dist, join_style="mitre")
                polygons = [buffered] if buffered.geom_type == "Polygon" else list(buffered.geoms)
                for polygon in polygons:
                    if not polygon.is_empty:
                        descriptors.append({"entity_id": entity_id, "semantic": "void", "category": norm["category"],
                            "kind": "solid", "outer": [list(p) for p in polygon.exterior.coords],
                            "holes": [[list(p) for p in ring.coords] for ring in polygon.interiors],
                            "z_min": bottom, "z_max": top, "dimension_source": "cad_footprint"})
            continue
        if semantic not in {"wall", "floor", "furniture"}:
            continue
        if semantic == "floor":
            top = float(record.get("elevation", 0))
            bottom = top - float(record.get("thickness", 0))
        else:
            bottom = float(record.get("base_elevation", 0))
            top = bottom + float(record.get("height", 0))
        if top <= bottom and semantic != "floor":
            warnings.append(f"{entity_id}: zero-height proxy skipped")
            continue

        if semantic == "furniture":
            regions, opened = process_furniture(entity["paths"], warnings, entity_id, unit=unit, style=furniture_style)
            if not regions and opened:
                warnings.append(f"{entity_id}: {len(opened)} open paths omitted from {semantic} footprint; no box inferred.")
        else:
            regions, opened = closed_regions(entity["paths"], warnings, entity_id)

        if semantic == "wall" and opened:
            if thickness > 0:
                regions += [LineString(path).buffer(thickness/2, cap_style="flat", join_style="mitre") for path in opened]
            else:
                warnings.append(f"{entity_id}: line wall shown as a zero-thickness vertical surface (no CAD thickness).")
                for path in opened:
                    descriptors.append({"entity_id": entity_id, "semantic": semantic, "kind": "surface",
                        "path": path, "z_min": bottom, "z_max": top, "dimension_source": "cad_surface"})
        elif semantic != "furniture" and opened:
            warnings.append(f"{entity_id}: {len(opened)} open paths omitted from {semantic} footprint; no box inferred.")

        if regions:
            merged = unary_union(regions)
            polygons = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
            for polygon in polygons:
                mesh_desc = {"entity_id": entity_id, "semantic": semantic, "kind": "solid",
                    "outer": [list(p) for p in polygon.exterior.coords],
                    "holes": [[list(p) for p in ring.coords] for ring in polygon.interiors],
                    "z_min": bottom, "z_max": top,
                    "dimension_source": "explicit_proxy_thickness" if semantic == "wall" and opened and thickness > 0 else ("oriented_box" if furniture_style == "box" and semantic == "furniture" else "cad_footprint")}
                if semantic == "furniture" and entity_id in furniture_colors:
                    mesh_desc["color"] = furniture_colors[entity_id]
                descriptors.append(mesh_desc)

    wall_tops = [float(rec.get("base_elevation", 0)) + float(rec.get("height", 0))
                 for rec in state["entities"].values()
                 if rec.get("semantic") == "wall" and not rec.get("orphaned") and float(rec.get("height", 0)) > 0]
    if auto_ceiling and wall_tops:
        ceiling_elevation = float(custom_ceiling_height) if custom_ceiling_height is not None else max(wall_tops)
        ceiling_desc = state.get("ceiling", {}).get("description") or proxy_settings.get("ceiling_description", "")
        ceiling_mat = state.get("ceiling", {}).get("material") or proxy_settings.get("ceiling_material")
        floor_solids = [m for m in descriptors if m["semantic"] == "floor" and m["kind"] == "solid"]
        ceiling_thickness = from_mm(50, unit)
        for floor_mesh in floor_solids:
            descriptors.append({
                "entity_id": "auto_ceiling",
                "semantic": "ceiling",
                "kind": "solid",
                "outer": floor_mesh["outer"],
                "holes": floor_mesh["holes"],
                "z_min": ceiling_elevation,
                "z_max": ceiling_elevation + ceiling_thickness,
                "dimension_source": "auto_wall_top" if custom_ceiling_height is None else "explicit_ceiling_height",
                "description": ceiling_desc,
                "material": ceiling_mat
            })

    points = []
    for mesh in descriptors:
        for pt in mesh.get("outer", mesh.get("path", [])):
            points.extend([(pt[0], pt[1], mesh["z_min"]), (pt[0], pt[1], mesh["z_max"])])
    bounds = [[min(p[i] for p in points) for i in range(3)],
              [max(p[i] for p in points) for i in range(3)]] if points else None
    return {"meshes": descriptors, "warnings": list(dict.fromkeys(warnings)), "bounds": bounds}
