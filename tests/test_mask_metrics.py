"""Unit tests for image-space mask metrics in Part 7B."""
import numpy as np
import pytest
from services.validation.mask_metrics import (
    bbox_dimension_errors,
    calculate_image_space_metrics,
    centroid_displacement_px,
    compute_mask_bbox,
    compute_mask_centroid,
    mask_iou,
    projected_area_error_percent,
)


def test_compute_mask_centroid_and_bbox():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:70] = True

    c = compute_mask_centroid(mask)
    assert c is not None
    assert c[0] == pytest.approx(49.5, abs=0.1)  # col center
    assert c[1] == pytest.approx(29.5, abs=0.1)  # row center

    bbox = compute_mask_bbox(mask)
    assert bbox == (30, 20, 69, 39)


def test_empty_mask_metrics():
    empty1 = np.zeros((50, 50), dtype=bool)
    empty2 = np.zeros((50, 50), dtype=bool)

    assert mask_iou(empty1, empty2) == 1.0
    assert centroid_displacement_px(empty1, empty2) is None
    assert projected_area_error_percent(empty1, empty2) is None


def test_identical_mask_metrics():
    m = np.zeros((80, 80), dtype=bool)
    m[10:30, 20:50] = True

    metrics = calculate_image_space_metrics(m, m)
    assert metrics["mask_iou"] == 1.0
    assert metrics["centroid_displacement_px"] == 0.0
    assert metrics["bbox_width_error_percent"] == 0.0
    assert metrics["bbox_height_error_percent"] == 0.0
    assert metrics["projected_area_error_percent"] == 0.0


def test_displaced_mask_metrics():
    m1 = np.zeros((100, 100), dtype=bool)
    m2 = np.zeros((100, 100), dtype=bool)

    m1[10:30, 10:30] = True  # 20x20
    m2[10:30, 20:40] = True  # 20x20, shifted 10px right

    # Intersection: 10x20 = 200, Union = 400 + 400 - 200 = 600 -> IoU = 1/3
    assert mask_iou(m1, m2) == pytest.approx(1.0 / 3.0, rel=1e-3)
    assert centroid_displacement_px(m1, m2) == pytest.approx(10.0, abs=1e-2)

    bbox_errs = bbox_dimension_errors(m1, m2)
    assert bbox_errs["bbox_width_error_percent"] == 0.0
    assert bbox_errs["bbox_height_error_percent"] == 0.0
