"""Image-space mask metrics for Part 7B.

Pure image-space measurements between CAD projected ground-truth mask
and AI-generated object mask. These values are in pixel and percentage units;
they are NOT metric measurements in meters.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np


def compute_mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Compute (u_center, v_center) in image coordinates (col, row)."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return float(np.mean(cols)), float(np.mean(rows))


def compute_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Compute bounding box [min_col, min_row, max_col, max_row]."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return int(np.min(cols)), int(np.min(rows)), int(np.max(cols)), int(np.max(rows))


def mask_iou(gt_mask: np.ndarray, gen_mask: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) of two boolean masks."""
    gt_bool = gt_mask > 0
    gen_bool = gen_mask > 0
    intersection = np.logical_and(gt_bool, gen_bool).sum()
    union = np.logical_or(gt_bool, gen_bool).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def centroid_displacement_px(gt_mask: np.ndarray, gen_mask: np.ndarray) -> Optional[float]:
    """Calculate Euclidean distance between mask centroids in image space (pixels)."""
    c_gt = compute_mask_centroid(gt_mask)
    c_gen = compute_mask_centroid(gen_mask)
    if c_gt is None or c_gen is None:
        return None
    dx = c_gen[0] - c_gt[0]
    dy = c_gen[1] - c_gt[1]
    return float(np.hypot(dx, dy))


def bbox_dimension_errors(
    gt_mask: np.ndarray, gen_mask: np.ndarray
) -> Dict[str, Optional[float]]:
    """Calculate relative bounding-box width and height differences in percent."""
    bbox_gt = compute_mask_bbox(gt_mask)
    bbox_gen = compute_mask_bbox(gen_mask)
    if bbox_gt is None or bbox_gen is None:
        return {
            "bbox_width_error_percent": None,
            "bbox_height_error_percent": None,
        }

    w_gt = max(1, bbox_gt[2] - bbox_gt[0] + 1)
    h_gt = max(1, bbox_gt[3] - bbox_gt[1] + 1)
    w_gen = max(1, bbox_gen[2] - bbox_gen[0] + 1)
    h_gen = max(1, bbox_gen[3] - bbox_gen[1] + 1)

    w_err_pct = (abs(w_gen - w_gt) / w_gt) * 100.0
    h_err_pct = (abs(h_gen - h_gt) / h_gt) * 100.0

    return {
        "bbox_width_error_percent": round(float(w_err_pct), 3),
        "bbox_height_error_percent": round(float(h_err_pct), 3),
    }


def projected_area_error_percent(
    gt_mask: np.ndarray, gen_mask: np.ndarray
) -> Optional[float]:
    """Calculate relative projected mask area difference in percent."""
    area_gt = float(np.count_nonzero(gt_mask))
    area_gen = float(np.count_nonzero(gen_mask))
    if area_gt == 0:
        return None
    err_pct = (abs(area_gen - area_gt) / area_gt) * 100.0
    return round(float(err_pct), 3)


def calculate_image_space_metrics(
    gt_mask: np.ndarray, gen_mask: np.ndarray
) -> Dict[str, Any]:
    """Calculate all standard image-space metrics between ground-truth and prediction."""
    iou = mask_iou(gt_mask, gen_mask)
    disp_px = centroid_displacement_px(gt_mask, gen_mask)
    bbox_errs = bbox_dimension_errors(gt_mask, gen_mask)
    area_err = projected_area_error_percent(gt_mask, gen_mask)

    return {
        "mask_iou": round(float(iou), 4),
        "centroid_displacement_px": round(float(disp_px), 2) if disp_px is not None else None,
        "bbox_width_error_percent": bbox_errs["bbox_width_error_percent"],
        "bbox_height_error_percent": bbox_errs["bbox_height_error_percent"],
        "projected_area_error_percent": area_err,
    }
