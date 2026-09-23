"""Unit tests for planar spatial measurement and homography in Part 7B."""
import numpy as np
import pytest
from app.geometry.camera import new_camera, project_points
from app.geometry.homography import derive_homography
from services.validation.homography_measurement import (
    calculate_footprint_metrics,
    calculate_position_error,
    extract_ground_contact_pixel,
    pixel_to_cad_xy,
    verify_camera_mapping_compatibility,
)


@pytest.fixture
def sample_cam():
    return new_camera(5.0, 2.0, "m", heading=180.0, name="Cam1", output={"width": 1024, "height": 1024})


def test_pixel_to_cad_xy_roundtrip(sample_cam):
    derived = derive_homography(sample_cam, plane_elevation=0.0, width=1024, height=1024)
    H_cad_to_image = derived["H_cad_to_image"]
    H_image_to_cad = derived["H_image_to_cad"]

    # Target point on floor in front of camera
    # Cam is at (5, 2, 1.5) looking heading 180 (towards -X: e.g. x=3.0, y=2.0)
    pt_world_xy = (3.0, 2.0)
    pt_world_3d = np.array([[3.0, 2.0, 0.0]])
    proj_px, depths = project_points(pt_world_3d, sample_cam, 1024, 1024)

    u, v = proj_px[0]
    assert depths[0] > 0
    assert np.isfinite(u) and np.isfinite(v)

    # Reverse homography
    recovered_xy = pixel_to_cad_xy(u, v, H_image_to_cad)
    assert recovered_xy is not None
    assert recovered_xy[0] == pytest.approx(pt_world_xy[0], abs=1e-4)
    assert recovered_xy[1] == pytest.approx(pt_world_xy[1], abs=1e-4)


def test_camera_mapping_compatibility_checks(sample_cam):
    # Compatible
    res_ok = verify_camera_mapping_compatibility(sample_cam, (1024, 1024), (1024, 1024))
    assert res_ok["compatible"]
    assert res_ok["status"] == "valid"

    # Incompatible render resolution
    res_bad = verify_camera_mapping_compatibility(sample_cam, (1376, 768), (1024, 1024))
    assert not res_bad["compatible"]
    assert res_bad["status"] == "invalid_or_requires_alignment"
    assert any("Render resolution" in r for r in res_bad["reasons"])


def test_extract_ground_contact_pixel():
    mask = np.zeros((100, 100), dtype=bool)
    # Rectangle from row 50 to 90, cols 30 to 70
    mask[50:91, 30:71] = True

    px_centroid = extract_ground_contact_pixel(mask, method="mask_bottom_contact_centroid")
    assert px_centroid is not None
    # Bottom 10% is rows ~86..90, cols 30..70
    assert px_centroid[0] == pytest.approx(50.0, abs=1.0)
    assert 85 <= px_centroid[1] <= 90

    px_midpoint = extract_ground_contact_pixel(mask, method="mask_bottom_midpoint")
    assert px_midpoint == (50.0, 90.0)

    px_proxy = extract_ground_contact_pixel(mask, method="mask_footprint_proxy")
    assert px_proxy == (50.0, 90.0)


def test_position_error():
    gt = (2.0, 3.0)
    gen = (2.3, 3.4)  # dx=0.3, dy=0.4 -> hypot=0.5
    err = calculate_position_error(gt, gen)
    assert err == pytest.approx(0.5, abs=1e-4)


def test_footprint_metrics():
    # 2x2 square vs identical shifted square
    sq1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
    sq2 = [(1, 0), (3, 0), (3, 2), (1, 2)]  # shifted 1m right, overlap is 1x2=2m^2, union is 3x2=6m^2

    metrics = calculate_footprint_metrics(sq1, sq2)
    assert metrics["centroid_displacement_m"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["footprint_area_error_m2"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["footprint_iou"] == pytest.approx(2.0 / 6.0, abs=1e-3)
