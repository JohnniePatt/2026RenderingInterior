"""Unit tests for single-view vertical metrology in Part 7B."""
import numpy as np
import pytest
from app.geometry.camera import new_camera, project_points
from app.geometry.homography import derive_homography
from services.validation.vertical_metrology import (
    calculate_height_errors,
    propose_vertical_observations,
    reconstruct_vertical_height,
)


@pytest.fixture
def sample_cam():
    # Camera at (6.0, 3.0, 1.6), pitch -10 degrees (looking slightly down), heading 180 (facing -X)
    cam = new_camera(6.0, 3.0, "m", heading=180.0, name="Cam1", output={"width": 1024, "height": 1024})
    cam["pitch_deg"] = -10.0
    return cam


def test_reconstruct_vertical_height_exact_known_box(sample_cam):
    # Object at X=3.0, Y=3.0, Base Z=0.0, Top Z=0.75m (height = 0.75m)
    derived = derive_homography(sample_cam, plane_elevation=0.0, width=1024, height=1024)
    H_img_to_cad = derived["H_image_to_cad"]

    base_3d = np.array([[3.0, 3.0, 0.0]])
    top_3d = np.array([[3.0, 3.0, 0.75]])

    base_px, base_depth = project_points(base_3d, sample_cam, 1024, 1024)
    top_px, top_depth = project_points(top_3d, sample_cam, 1024, 1024)

    assert base_depth[0] > 0 and top_depth[0] > 0

    res = reconstruct_vertical_height(
        base_px[0],
        top_px[0],
        sample_cam,
        H_img_to_cad,
        base_elevation=0.0,
        width=1024,
        height=1024,
    )

    assert res["status"] == "evaluated"
    assert res["reconstructed_height_m"] == pytest.approx(0.75, abs=1e-3)
    assert res["z_top"] == pytest.approx(0.75, abs=1e-3)


def test_calculate_height_errors():
    cad_h = 0.50
    recon_h = 0.55  # 0.05m error, 10%
    errs = calculate_height_errors(cad_h, recon_h)

    assert errs["cad_height_m"] == 0.50
    assert errs["reconstructed_height_m"] == 0.55
    assert errs["height_error_m"] == pytest.approx(0.05, abs=1e-4)
    assert errs["height_error_percent"] == pytest.approx(10.0, abs=1e-2)


def test_propose_vertical_observations():
    mask = np.zeros((100, 100), dtype=bool)
    # Box from row 20 to row 70, cols 40 to 60
    mask[20:71, 40:61] = True

    obs = propose_vertical_observations(mask)
    assert obs is not None
    assert obs["base_pixel"][1] == 70.0
    assert obs["top_pixel"][1] == 20.0
    assert obs["base_pixel"][0] == pytest.approx(50.0, abs=1.0)
    assert obs["top_pixel"][0] == pytest.approx(50.0, abs=1.0)


def test_reconstruct_vertical_height_invalid_geometry(sample_cam):
    derived = derive_homography(sample_cam, plane_elevation=0.0, width=1024, height=1024)
    H_img_to_cad = derived["H_image_to_cad"]

    # Inverted points (top lower than base in image, or point behind camera)
    res = reconstruct_vertical_height(
        [512, 100],  # base high up in sky
        [512, 900],  # top down near floor
        sample_cam,
        H_img_to_cad,
        base_elevation=0.0,
        width=1024,
        height=1024,
    )
    assert res["status"] == "cannot_evaluate"
