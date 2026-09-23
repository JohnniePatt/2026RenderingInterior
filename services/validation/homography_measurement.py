"""Planar spatial measurement and reverse homography for Part 7B.

Uses calibrated camera planar homography to project floor-contact observations
from generated image space into CAD/world XY coordinates (meters).
"""
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon
from app.geometry.homography import derive_homography, validate_homography


def verify_camera_mapping_compatibility(
    camera: dict,
    render_size: Tuple[int, int],
    cad_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Verify that generated render preserves calibrated camera image geometry.

    Checks:
    - resolution matching
    - aspect ratio matching
    Returns status dict with 'status' in ('valid', 'invalid_or_requires_alignment')
    and diagnostic messages.
    """
    out = camera.get("output") or {}
    cam_w = int(out.get("width", 1024))
    cam_h = int(out.get("height", 1024))
    ren_w, ren_h = render_size

    reasons = []
    if (ren_w, ren_h) != (cam_w, cam_h):
        reasons.append(
            f"Render resolution ({ren_w}×{ren_h}) differs from camera output ({cam_w}×{cam_h})."
        )

    if cad_size is not None and (ren_w, ren_h) != cad_size:
        reasons.append(
            f"Render resolution ({ren_w}×{ren_h}) differs from CAD conditioning ({cad_size[0]}×{cad_size[1]})."
        )

    cam_aspect = cam_w / max(1, cam_h)
    ren_aspect = ren_w / max(1, ren_h)
    if abs(cam_aspect - ren_aspect) > 1e-3:
        reasons.append(
            f"Aspect ratio mismatch: render {ren_aspect:.4f} vs camera {cam_aspect:.4f}."
        )

    if reasons:
        return {
            "status": "invalid_or_requires_alignment",
            "reasons": reasons,
            "compatible": False,
        }

    return {
        "status": "valid",
        "reasons": [],
        "compatible": True,
    }


def pixel_to_cad_xy(
    u: float, v: float, H_image_to_cad: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Transform image pixel (u, v) to CAD ground plane (X, Y) in meters.

    Uses homogeneous reverse homography:
        [X', Y', W']^T = H_image_to_cad * [u, v, 1]^T
        X = X'/W', Y = Y'/W'
    Valid only for points on the reference plane with W' > 0.
    """
    pt_img = np.array([float(u), float(v), 1.0], dtype=float)
    pt_cad = H_image_to_cad @ pt_img

    w = pt_cad[2]
    if not np.isfinite(w) or abs(w) < 1e-9 or w <= 0:
        return None

    x = float(pt_cad[0] / w)
    y = float(pt_cad[1] / w)

    if not (np.isfinite(x) and np.isfinite(y)):
        return None

    return x, y


def extract_ground_contact_pixel(
    mask: np.ndarray, method: str = "mask_bottom_contact_centroid"
) -> Optional[Tuple[float, float]]:
    """Derive representative floor-contact observation pixel (u, v) from mask.

    Supported methods:
    - 'mask_bottom_contact_centroid': Centroid of bottom 10% vertical span of mask.
    - 'mask_bottom_midpoint': Midpoint of the lowest row containing mask pixels.
    - 'mask_footprint_proxy': Bottom center of the mask bounding box.
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return None

    r_min, r_max = int(np.min(rows)), int(np.max(rows))
    span = max(1, r_max - r_min)

    if method == "mask_bottom_contact_centroid":
        # Extract pixels in bottom 10% of mask height
        cutoff = r_max - max(1, int(0.10 * span))
        bottom_idx = rows >= cutoff
        u_contact = float(np.mean(cols[bottom_idx]))
        v_contact = float(np.mean(rows[bottom_idx]))
        return u_contact, v_contact

    elif method == "mask_bottom_midpoint":
        lowest_cols = cols[rows == r_max]
        u_contact = float(np.median(lowest_cols))
        v_contact = float(r_max)
        return u_contact, v_contact

    elif method == "mask_footprint_proxy":
        c_min, c_max = int(np.min(cols)), int(np.max(cols))
        u_contact = float((c_min + c_max) / 2.0)
        v_contact = float(r_max)
        return u_contact, v_contact

    raise ValueError(f"Unknown ground contact extraction method: '{method}'")


def calculate_position_error(
    gt_xy: Tuple[float, float], generated_xy: Tuple[float, float]
) -> float:
    """Calculate Euclidean position error in meters: E_position = sqrt((X_gen - X_gt)^2 + (Y_gen - Y_gt)^2)."""
    dx = generated_xy[0] - gt_xy[0]
    dy = generated_xy[1] - gt_xy[1]
    return float(math.hypot(dx, dy))


def calculate_footprint_metrics(
    gt_poly_coords: List[Tuple[float, float]],
    gen_poly_coords: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """Calculate metric floor-footprint error between reconstructed and CAD footprint polygons.

    Returns:
    - centroid_displacement_m
    - footprint_area_error_m2
    - footprint_area_error_percent
    - footprint_iou
    """
    if len(gt_poly_coords) < 3 or len(gen_poly_coords) < 3:
        return {
            "centroid_displacement_m": None,
            "footprint_area_error_m2": None,
            "footprint_area_error_percent": None,
            "footprint_iou": None,
        }

    try:
        poly_gt = Polygon(gt_poly_coords)
        poly_gen = Polygon(gen_poly_coords)

        if not poly_gt.is_valid:
            poly_gt = poly_gt.buffer(0)
        if not poly_gen.is_valid:
            poly_gen = poly_gen.buffer(0)

        c_gt = poly_gt.centroid
        c_gen = poly_gen.centroid
        disp_m = math.hypot(c_gen.x - c_gt.x, c_gen.y - c_gt.y)

        area_gt = poly_gt.area
        area_gen = poly_gen.area
        area_err_m2 = abs(area_gen - area_gt)
        area_err_pct = (area_err_m2 / max(1e-6, area_gt)) * 100.0 if area_gt > 0 else None

        inter_area = poly_gt.intersection(poly_gen).area
        union_area = poly_gt.union(poly_gen).area
        iou = float(inter_area / max(1e-6, union_area)) if union_area > 0 else 0.0

        return {
            "centroid_displacement_m": round(float(disp_m), 4),
            "footprint_area_error_m2": round(float(area_err_m2), 4),
            "footprint_area_error_percent": round(float(area_err_pct), 2) if area_err_pct is not None else None,
            "footprint_iou": round(float(iou), 4),
        }
    except Exception:
        return {
            "centroid_displacement_m": None,
            "footprint_area_error_m2": None,
            "footprint_area_error_percent": None,
            "footprint_iou": None,
        }
