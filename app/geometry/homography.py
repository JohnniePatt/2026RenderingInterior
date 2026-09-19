"""Planar Homography and Conditioning for CAD floor plane.

Connects 2D CAD floor geometry to the existing calibrated virtual camera
using deterministic planar homography H:
    [u', v', w']^T = H_cad_to_image * [X, Y, 1]^T
    u = u' / w',  v = v' / w'   (for w' > 0)
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import cv2
import numpy as np
from app.geometry.camera import camera_matrices, project_points, validate_camera
from app.layout.schema import finite


def get_floor_elevation(state):
    """Determine the floor reference elevation Z0 from tagged entities or proxy settings."""
    entities = state.get("entities", {})
    # Look for explicit floor entity
    for rec in entities.values():
        if rec.get("semantic") == "floor" and not rec.get("orphaned"):
            elev = rec.get("elevation")
            if elev is not None and finite(elev):
                return float(elev)
    # Check proxy settings fallback
    proxy_settings = state.get("proxy_settings", {})
    if "floor_elevation" in proxy_settings and finite(proxy_settings["floor_elevation"]):
        return float(proxy_settings["floor_elevation"])
    return 0.0


def derive_homography(camera, plane_elevation=0.0, width=None, height=None):
    """Derive H_cad_to_image and H_image_to_cad deterministically from camera matrices.

    For a horizontal floor plane Z = Z0:
        X_world = [X, Y, Z0, 1]^T
        x_image ~ P * X_world = [p1, p2, p3, p4] * [X, Y, Z0, 1]^T
                              = p1*X + p2*Y + (p3*Z0 + p4)
                              = [p1, p2, p3*Z0 + p4] * [X, Y, 1]^T
    Therefore:
        H_cad_to_image = [p1, p2, p3*Z0 + p4]
    """
    validate_camera(camera)
    if width is None and height is None:
        out = camera.get("output") or {}
        width = float(out.get("width", 1024))
        height = float(out.get("height", 1024))
    elif width is None or height is None:
        raise ValueError("Both width and height must be provided if one is given")

    Z0 = float(plane_elevation)
    matrices = camera_matrices(camera, width, height)
    P = matrices["P"]

    # H = [p1, p2, p3*Z0 + p4]
    H_cad_to_image = np.column_stack([P[:, 0], P[:, 1], P[:, 2] * Z0 + P[:, 3]])

    # Check condition number and invertibility
    cond = float(np.linalg.cond(H_cad_to_image))
    if not np.isfinite(cond) or cond > 1e14:
        raise ValueError(f"Singular or poorly conditioned homography matrix (cond={cond:.2e})")

    H_image_to_cad = np.linalg.inv(H_cad_to_image)

    return {
        "H_cad_to_image": H_cad_to_image,
        "H_image_to_cad": H_image_to_cad,
        "plane_elevation": Z0,
        "width": int(width),
        "height": int(height),
        "condition_number": cond,
        "K": matrices["K"],
        "R": matrices["R"],
        "t": matrices["t"],
        "P": P
    }


def validate_homography(H_cad_to_image, camera, plane_elevation=0.0, width=None, height=None, sample_points=None):
    """Validate homography reprojection against full 3D camera projection.

    Returns dict with sample_count, mean_reprojection_error_px, and max_reprojection_error_px.
    """
    if width is None and height is None:
        out = camera.get("output") or {}
        width = float(out.get("width", 1024))
        height = float(out.get("height", 1024))

    Z0 = float(plane_elevation)

    if sample_points is None:
        # Generate grid of points around camera target or position
        pos = camera["position"]
        tgt = camera.get("target", [pos[0] + 5, pos[1], pos[2]])
        cx, cy = tgt[0], tgt[1]
        span = 4.0
        xs = np.linspace(cx - span, cx + span, 7)
        ys = np.linspace(cy - span, cy + span, 7)
        grid_x, grid_y = np.meshgrid(xs, ys)
        pts_2d = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    else:
        pts_2d = np.asarray(sample_points, dtype=float)[:, :2]

    # 3D points at plane elevation
    pts_3d = np.column_stack([pts_2d, np.full(len(pts_2d), Z0)])
    proj_3d, depths = project_points(pts_3d, camera, width, height)

    # Homography projection
    homo_pts = np.column_stack([pts_2d, np.ones(len(pts_2d))])
    proj_h_homo = (H_cad_to_image @ homo_pts.T).T
    w_prime = proj_h_homo[:, 2]

    # Only evaluate points in front of the camera (visible in both)
    valid = (w_prime > 0.05) & np.isfinite(proj_3d[:, 0]) & np.isfinite(proj_3d[:, 1])
    if not np.any(valid):
        return {
            "sample_count": 0,
            "mean_reprojection_error_px": 0.0,
            "max_reprojection_error_px": 0.0,
            "valid": True
        }

    proj_h = proj_h_homo[valid, :2] / proj_h_homo[valid, 2:3]
    proj_3d_valid = proj_3d[valid]

    errors = np.linalg.norm(proj_h - proj_3d_valid, axis=1)
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    return {
        "sample_count": int(np.sum(valid)),
        "mean_reprojection_error_px": mean_err,
        "max_reprojection_error_px": max_err,
        "valid": bool(max_err < 1.0)
    }


def clip_polygon_to_camera_front(poly_pts, H_cad_to_image, near_w=0.05):
    """Clip a 2D CAD polygon against the camera forward halfspace w'(X, Y) >= near_w using Sutherland-Hodgman.

    w'(X, Y) = H[2, 0]*X + H[2, 1]*Y + H[2, 2].
    """
    if len(poly_pts) < 3:
        return []
    a, b, c = H_cad_to_image[2]
    clipped = []
    n = len(poly_pts)
    for i in range(n):
        p1 = poly_pts[i]
        p2 = poly_pts[(i + 1) % n]
        val1 = a * p1[0] + b * p1[1] + c - near_w
        val2 = a * p2[0] + b * p2[1] + c - near_w
        if val1 >= 0:
            clipped.append(p1)
            if val2 < 0:
                t = val1 / (val1 - val2)
                ix = p1[0] + t * (p2[0] - p1[0])
                iy = p1[1] + t * (p2[1] - p1[1])
                clipped.append([ix, iy])
        elif val2 >= 0:
            t = val1 / (val1 - val2)
            ix = p1[0] + t * (p2[0] - p1[0])
            iy = p1[1] + t * (p2[1] - p1[1])
            clipped.append([ix, iy])
    return clipped


def project_cad_polygon(poly_pts, H_cad_to_image, near_w=0.05):
    """Project a 2D CAD polygon on the floor plane to image pixel coordinates using homography."""
    clipped = clip_polygon_to_camera_front(poly_pts, H_cad_to_image, near_w=near_w)
    if len(clipped) < 3:
        return []
    pts_homo = np.column_stack([clipped, np.ones(len(clipped))])
    proj = (H_cad_to_image @ pts_homo.T).T
    uvs = proj[:, :2] / proj[:, 2:3]
    return uvs.tolist()


def clip_line_segment_to_camera_front(p1, p2, H_cad_to_image, near_w=0.05):
    """Clip and project a 2D CAD line segment against the camera forward halfspace w'(X, Y) >= near_w."""
    a, b, c = H_cad_to_image[2]
    w1 = a * p1[0] + b * p1[1] + c
    w2 = a * p2[0] + b * p2[1] + c
    if w1 < near_w and w2 < near_w:
        return None

    pt1 = [float(p1[0]), float(p1[1])]
    pt2 = [float(p2[0]), float(p2[1])]

    if w1 >= near_w and w2 >= near_w:
        pass
    elif w1 >= near_w and w2 < near_w:
        t = (near_w - w1) / (w2 - w1)
        pt2 = [pt1[0] + t * (pt2[0] - pt1[0]), pt1[1] + t * (pt2[1] - pt1[1])]
    else:
        t = (near_w - w1) / (w2 - w1)
        pt1 = [pt1[0] + t * (pt2[0] - pt1[0]), pt1[1] + t * (pt2[1] - pt1[1])]

    def proj(p):
        u = H_cad_to_image[0, 0] * p[0] + H_cad_to_image[0, 1] * p[1] + H_cad_to_image[0, 2]
        v = H_cad_to_image[1, 0] * p[0] + H_cad_to_image[1, 1] * p[1] + H_cad_to_image[1, 2]
        w = H_cad_to_image[2, 0] * p[0] + H_cad_to_image[2, 1] * p[1] + H_cad_to_image[2, 2]
        return [int(round(u / w)), int(round(v / w))]

    return [proj(pt1), proj(pt2)]


def render_planar_map(geometry, state, camera_id_or_camera, width=None, height=None):
    """Render a deterministic, pixel-aligned Planar Conditioning Map in CAD Blueprint / X-ray wireframe style.

    Uses the planar homography H to project 2D CAD floor boundary, wall footprints, furniture footprints
    (including all interior CAD lines like pillows, folds, chair arcs), and void thresholds lying on the floor plane.
    Does NOT use 3D extrusion.
    """
    if isinstance(camera_id_or_camera, str):
        camera = state["cameras"][camera_id_or_camera]
    else:
        camera = camera_id_or_camera

    out = camera.get("output") or {}
    W = int(width or out.get("width", 1024))
    H = int(height or out.get("height", 1024))

    Z0 = get_floor_elevation(state)
    homo_data = derive_homography(camera, plane_elevation=Z0, width=W, height=H)
    H_mat = homo_data["H_cad_to_image"]

    # Initialize canvas with dark blueprint slate #0b131e (BGR: 30, 19, 11)
    img = np.full((H, W, 3), (30, 19, 11), dtype=np.uint8)
    overlay = img.copy()

    entities = state.get("entities", {})

    # 1. Floor subtle translucent fill
    floor_fill_bgr = (57, 38, 26)   # #1a2639
    floor_line_bgr = (183, 203, 101) # #65cbb7
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "floor" and not rec.get("orphaned"):
            for path in g.get("paths", []):
                uvs = project_cad_polygon(path, H_mat)
                if len(uvs) >= 3:
                    pts = np.array(uvs, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], floor_fill_bgr, lineType=cv2.LINE_AA)

    # 2. Wall subtle translucent fill
    wall_fill_bgr = (25, 43, 63)    # subtle warm orange fill
    wall_line_bgr = (101, 173, 251) # #fbad65
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "wall" and not rec.get("orphaned"):
            for path in g.get("paths", []):
                uvs = project_cad_polygon(path, H_mat)
                if len(uvs) >= 3:
                    pts = np.array(uvs, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], wall_fill_bgr, lineType=cv2.LINE_AA)

    # 3. Void subtle translucent fill
    void_fill_bgr = (64, 48, 27)    # subtle cyan fill
    void_line_bgr = (255, 191, 107) # #6bbfff
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "void" and not rec.get("orphaned"):
            base_elev = float(rec.get("base_elevation", rec.get("sill_height", 0.0)))
            if abs(base_elev - Z0) < 0.5:
                for path in g.get("paths", []):
                    uvs = project_cad_polygon(path, H_mat)
                    if len(uvs) >= 3:
                        pts = np.array(uvs, dtype=np.int32)
                        cv2.fillPoly(overlay, [pts], void_fill_bgr, lineType=cv2.LINE_AA)

    # 4. Furniture subtle translucent fill
    furn_fill_bgr = (62, 38, 46)    # subtle purple fill
    furn_line_bgr = (247, 154, 185) # #b99af7
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "furniture" and not rec.get("orphaned"):
            base_elev = float(rec.get("base_elevation", 0.0))
            if abs(base_elev - Z0) < 0.5:
                for path in g.get("paths", []):
                    uvs = project_cad_polygon(path, H_mat)
                    if len(uvs) >= 3:
                        pts = np.array(uvs, dtype=np.int32)
                        cv2.fillPoly(overlay, [pts], furn_fill_bgr, lineType=cv2.LINE_AA)

    # Blend subtle fills onto background (alpha = 0.5)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # Helper to draw all line segments of an entity with anti-aliasing
    def draw_entity_segments(g, color, thickness=2):
        for path in g.get("paths", []):
            for i in range(1, len(path)):
                seg = clip_line_segment_to_camera_front(path[i - 1], path[i], H_mat)
                if seg:
                    pt1, pt2 = seg
                    cv2.line(img, tuple(pt1), tuple(pt2), color, thickness, lineType=cv2.LINE_AA)

    # 5. Draw all CAD lines with full opacity (X-ray wireframe)
    # Floor boundary
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "floor" and not rec.get("orphaned"):
            draw_entity_segments(g, floor_line_bgr, thickness=2)

    # Wall footprints
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "wall" and not rec.get("orphaned"):
            draw_entity_segments(g, wall_line_bgr, thickness=2)

    # Void / door thresholds
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "void" and not rec.get("orphaned"):
            base_elev = float(rec.get("base_elevation", rec.get("sill_height", 0.0)))
            if abs(base_elev - Z0) < 0.5:
                draw_entity_segments(g, void_line_bgr, thickness=2)

    # Furniture: ALL CAD lines (interior details like pillows, folds, chair arcs, etc.)
    for g in geometry:
        eid = g["id"]
        rec = entities.get(eid, {})
        if rec.get("semantic") == "furniture" and not rec.get("orphaned"):
            base_elev = float(rec.get("base_elevation", 0.0))
            if abs(base_elev - Z0) < 0.5:
                draw_entity_segments(g, furn_line_bgr, thickness=2)

    return img


def build_homography_json(layout_id, camera_id, state, camera, width=None, height=None):
    """Construct authoritative homography.json document."""
    out = camera.get("output") or {}
    W = int(width or out.get("width", 1024))
    H = int(height or out.get("height", 1024))
    Z0 = get_floor_elevation(state)

    derived = derive_homography(camera, plane_elevation=Z0, width=W, height=H)
    val = validate_homography(derived["H_cad_to_image"], camera, plane_elevation=Z0, width=W, height=H)

    return {
        "schema_version": "0.1",
        "layout_id": layout_id,
        "camera_id": camera_id,
        "source_plane": {
            "semantic": "floor",
            "elevation_m": Z0
        },
        "image_coordinate_system": {
            "origin": "top_left",
            "u_axis": "right",
            "v_axis": "down",
            "width": W,
            "height": H
        },
        "H_cad_to_image": derived["H_cad_to_image"].tolist(),
        "H_image_to_cad": derived["H_image_to_cad"].tolist(),
        "validation": {
            "sample_count": val["sample_count"],
            "mean_reprojection_error_px": round(val["mean_reprojection_error_px"], 6),
            "max_reprojection_error_px": round(val["max_reprojection_error_px"], 6)
        }
    }
