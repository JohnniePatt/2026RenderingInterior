"""Single-view metrology for vertical height reconstruction (Part 7B).

Uses calibrated camera projection matrix and world vertical line geometry
to solve for 3D vertical dimensions from image-space base and top observations.
Does NOT use naive pixel-height ratios.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from app.geometry.camera import camera_matrices
from services.validation.homography_measurement import pixel_to_cad_xy


def propose_vertical_observations(
    mask: np.ndarray, method: str = "automatic"
) -> Optional[Dict[str, Any]]:
    """Propose base and top landmark pixels from generated mask.

    Finds:
    - base_pixel: ground-contact point at lowest vertical extent of the mask
    - top_pixel: highest vertical extent along the object centerline
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return None

    r_min, r_max = int(np.min(rows)), int(np.max(rows))
    span = max(1, r_max - r_min)

    # Top pixel: center of topmost 5% of mask
    top_cutoff = r_min + max(1, int(0.05 * span))
    top_idx = rows <= top_cutoff
    u_top = float(np.mean(cols[top_idx]))
    v_top = float(np.min(rows[top_idx]))

    # Base pixel: center of bottommost 5% of mask
    base_cutoff = r_max - max(1, int(0.05 * span))
    base_idx = rows >= base_cutoff
    u_base = float(np.mean(cols[base_idx]))
    v_base = float(np.max(rows[base_idx]))

    return {
        "base_pixel": [round(u_base, 2), round(v_base, 2)],
        "top_pixel": [round(u_top, 2), round(v_top, 2)],
        "observation_method": method,
    }


def reconstruct_vertical_height(
    base_pixel: Tuple[float, float],
    top_pixel: Tuple[float, float],
    camera: dict,
    H_image_to_cad: np.ndarray,
    base_elevation: float = 0.0,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Dict[str, Any]:
    """Reconstruct world height from image base and top observations using calibrated projection.

    Mathematical formulation:
    1. Base world coordinate:
       (X_base, Y_base) = pixel_to_cad_xy(u_base, v_base, H_image_to_cad)
       Z_base = base_elevation

    2. World vertical line through base:
       L(z) = [X_base, Y_base, z, 1]^T

    3. Calibrated projection matrix P (3x4):
       [u', v', w']^T = P * L(z) = a + b * z
       where a = P[:, 0]*X_base + P[:, 1]*Y_base + P[:, 3]
             b = P[:, 2]

    4. Linear system for top pixel (u_top, v_top):
       (u_top * b_2 - b_0) * z = a_0 - u_top * a_2
       (v_top * b_2 - b_1) * z = a_1 - v_top * a_2
       Solve via least squares for z_top.
    """
    u_base, v_base = float(base_pixel[0]), float(base_pixel[1])
    u_top, v_top = float(top_pixel[0]), float(top_pixel[1])

    # 1. Reverse homography to find base world coordinate on reference plane
    base_xy = pixel_to_cad_xy(u_base, v_base, H_image_to_cad)
    if base_xy is None:
        return {
            "status": "cannot_evaluate",
            "reason": "Base pixel could not be projected to floor plane (behind camera or invalid homography).",
            "reconstructed_height_m": None,
            "z_top": None,
            "z_base": float(base_elevation),
        }

    X_base, Y_base = base_xy
    Z_base = float(base_elevation)

    # 2. Camera projection matrix P
    out = camera.get("output") or {}
    w_cam = float(width if width is not None else out.get("width", 1024))
    h_cam = float(height if height is not None else out.get("height", 1024))

    matrices = camera_matrices(camera, w_cam, h_cam)
    P = matrices["P"]
    R = matrices["R"]
    t = matrices["t"]

    # 3. Formulate ray line L(z) projected through P
    # a = P[:, 0]*X_base + P[:, 1]*Y_base + P[:, 3]
    a = P[:, 0] * X_base + P[:, 1] * Y_base + P[:, 3]
    b = P[:, 2]  # effect of height z on homogeneous image coordinates

    # 4. Formulate linear least squares A * z = c
    A = np.array([
        [u_top * b[2] - b[0]],
        [v_top * b[2] - b[1]],
    ], dtype=float)

    c = np.array([
        a[0] - u_top * a[2],
        a[1] - v_top * a[2],
    ], dtype=float)

    # Check for degenerate vertical direction in camera projection
    norm_A = np.linalg.norm(A)
    if not np.isfinite(norm_A) or norm_A < 1e-7:
        return {
            "status": "cannot_evaluate",
            "reason": "Degenerate camera geometry: vertical line has near-zero image projection magnitude.",
            "reconstructed_height_m": None,
            "z_top": None,
            "z_base": Z_base,
        }

    # Solve least squares
    sol, residuals, rank, s = np.linalg.lstsq(A, c, rcond=None)
    z_top = float(sol[0])

    if not np.isfinite(z_top):
        return {
            "status": "cannot_evaluate",
            "reason": "Numerical singularity when solving for vertical elevation.",
            "reconstructed_height_m": None,
            "z_top": None,
            "z_base": Z_base,
        }

    # Verify that top point lies in front of camera
    top_world = np.array([X_base, Y_base, z_top], dtype=float)
    cam_top = R @ top_world + t
    if cam_top[2] <= 0:
        return {
            "status": "cannot_evaluate",
            "reason": "Reconstructed top point projects behind camera optical plane.",
            "reconstructed_height_m": None,
            "z_top": None,
            "z_base": Z_base,
        }

    reconstructed_h = z_top - Z_base
    if reconstructed_h <= 0:
        return {
            "status": "cannot_evaluate",
            "reason": f"Non-physical negative or zero height reconstructed ({reconstructed_h:.3f} m).",
            "reconstructed_height_m": None,
            "z_top": round(z_top, 4),
            "z_base": round(Z_base, 4),
        }

    return {
        "status": "evaluated",
        "reconstructed_height_m": round(float(reconstructed_h), 4),
        "z_top": round(float(z_top), 4),
        "z_base": round(float(Z_base), 4),
        "base_world_xy": [round(float(X_base), 4), round(float(Y_base), 4)],
    }


def calculate_height_errors(
    cad_height: float, reconstructed_height: float
) -> Dict[str, Any]:
    """Calculate absolute and relative height errors against CAD ground-truth height."""
    if cad_height is None or not np.isfinite(cad_height) or cad_height <= 0:
        return {
            "cad_height_m": None,
            "reconstructed_height_m": round(float(reconstructed_height), 4),
            "height_error_m": None,
            "height_error_percent": None,
        }

    abs_err = abs(reconstructed_height - cad_height)
    rel_err_pct = (abs_err / cad_height) * 100.0

    return {
        "cad_height_m": round(float(cad_height), 4),
        "reconstructed_height_m": round(float(reconstructed_height), 4),
        "height_error_m": round(float(abs_err), 4),
        "height_error_percent": round(float(rel_err_pct), 2),
    }
