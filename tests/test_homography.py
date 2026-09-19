"""Automated validation for Planar Homography, conditioning map, and export."""
import json
from pathlib import Path
import numpy as np
import pytest
from app.geometry.camera import camera_matrices, project_points, new_camera
from app.geometry.homography import (
    derive_homography,
    validate_homography,
    get_floor_elevation,
    render_planar_map,
    build_homography_json,
    project_cad_polygon
)
from app.geometry.conditioning import save_conditioning_outputs
from app.layout.schema import new_state


@pytest.fixture
def sample_camera():
    return {
        "name": "Camera 001",
        "position": [9.642, 1.83, 0.8],
        "target": [4.773, 1.844, 0.8],
        "heading_deg": 179.83,
        "pitch_deg": 0.0,
        "fov_deg": 60.0,
        "output": {"width": 1920, "height": 1080}
    }


def test_1_homography_vs_3d_projection(sample_camera):
    """Test 1: Compare full 3D camera projection vs Planar Homography on floor plane."""
    Z0 = 0.0
    W, H = 1920, 1080
    derived = derive_homography(sample_camera, plane_elevation=Z0, width=W, height=H)
    H_mat = derived["H_cad_to_image"]

    # Sample points on the floor plane in front of camera (camera is at X=9.64 looking towards -X)
    sample_pts = np.array([
        [1.0, 1.0, Z0],
        [2.5, 3.0, Z0],
        [5.0, 1.8, Z0],
        [7.0, 1.0, Z0],
        [0.67, 0.47, Z0],
        [3.5, 2.2, Z0]
    ])

    proj_3d, _ = project_points(sample_pts, sample_camera, W, H)

    # Apply homography
    pts_2d_homo = np.column_stack([sample_pts[:, :2], np.ones(len(sample_pts))])
    proj_h_homo = (H_mat @ pts_2d_homo.T).T
    proj_h = proj_h_homo[:, :2] / proj_h_homo[:, 2:3]

    errors = np.linalg.norm(proj_3d - proj_h, axis=1)
    mean_err = np.mean(errors)
    max_err = np.max(errors)

    assert mean_err < 1.0, f"Mean error exceeds 1 px: {mean_err}"
    assert max_err < 1.0, f"Max error exceeds 1 px: {max_err}"
    # In practice, algebraic equivalence produces errors near machine precision
    assert max_err < 1e-6, f"Expected near-zero algebraic error, got {max_err}"


def test_2_inverse_homography_roundtrip(sample_camera):
    """Test 2: Roundtrip [X, Y] -> H -> [u, v] -> H^-1 -> [X', Y']."""
    Z0 = 0.0
    W, H = 1920, 1080
    derived = derive_homography(sample_camera, plane_elevation=Z0, width=W, height=H)
    H_cad_to_img = derived["H_cad_to_image"]
    H_img_to_cad = derived["H_image_to_cad"]

    pts_cad = np.array([
        [1.5, 1.2],
        [3.0, 2.5],
        [5.5, 1.9],
        [7.2, 0.8]
    ])

    # Forward
    homo_cad = np.column_stack([pts_cad, np.ones(len(pts_cad))])
    img_homo = (H_cad_to_img @ homo_cad.T).T
    uv = img_homo[:, :2] / img_homo[:, 2:3]

    # Inverse
    homo_uv = np.column_stack([uv, np.ones(len(uv))])
    cad_recon_homo = (H_img_to_cad @ homo_uv.T).T
    cad_recon = cad_recon_homo[:, :2] / cad_recon_homo[:, 2:3]

    diff = np.max(np.abs(pts_cad - cad_recon))
    assert diff < 1e-6, f"Inverse roundtrip error too large: {diff}"


def test_3_resolution_change(sample_camera):
    """Test 3: Changing resolution correctly adapts homography matrix."""
    H1 = derive_homography(sample_camera, width=1920, height=1080)["H_cad_to_image"]
    H2 = derive_homography(sample_camera, width=1024, height=1024)["H_cad_to_image"]

    assert not np.allclose(H1, H2), "H should differ across different resolutions / aspect ratios"

    # Both resolutions should still validate with near-zero error
    val1 = validate_homography(H1, sample_camera, width=1920, height=1080)
    val2 = validate_homography(H2, sample_camera, width=1024, height=1024)
    assert val1["max_reprojection_error_px"] < 1e-4
    assert val2["max_reprojection_error_px"] < 1e-4


def test_4_camera_change(sample_camera):
    """Test 4: Moving camera, changing FOV or target recalculates H deterministically."""
    H_orig = derive_homography(sample_camera, width=1024, height=1024)["H_cad_to_image"]

    # 1. Move camera position
    cam_moved = dict(sample_camera, position=[5.0, 2.0, 1.2])
    H_moved = derive_homography(cam_moved, width=1024, height=1024)["H_cad_to_image"]
    assert not np.allclose(H_orig, H_moved), "H must change when camera moves"

    # 2. Change FOV
    cam_fov = dict(sample_camera, fov_deg=45.0)
    H_fov = derive_homography(cam_fov, width=1024, height=1024)["H_cad_to_image"]
    assert not np.allclose(H_orig, H_fov), "H must change when FOV changes"

    # 3. Change target
    cam_tgt = dict(sample_camera, target=[2.0, 1.0, 0.5])
    H_tgt = derive_homography(cam_tgt, width=1024, height=1024)["H_cad_to_image"]
    assert not np.allclose(H_orig, H_tgt), "H must change when target changes"


def test_5_floor_elevation(sample_camera):
    """Test 5: Non-zero floor elevation is correctly incorporated."""
    H_z0 = derive_homography(sample_camera, plane_elevation=0.0, width=1024, height=1024)["H_cad_to_image"]
    H_z5 = derive_homography(sample_camera, plane_elevation=0.5, width=1024, height=1024)["H_cad_to_image"]

    assert not np.allclose(H_z0, H_z5), "H must differ for different plane elevations"

    # Points at Z=0.5 projected with H_z5 must match 3D projection at Z=0.5
    sample_pts_z5 = np.array([
        [2.0, 1.5, 0.5],
        [4.0, 2.0, 0.5]
    ])
    proj_3d, _ = project_points(sample_pts_z5, sample_camera, 1024, 1024)
    homo_pts = np.column_stack([sample_pts_z5[:, :2], np.ones(len(sample_pts_z5))])
    proj_h = (H_z5 @ homo_pts.T).T
    proj_h = proj_h[:, :2] / proj_h[:, 2:3]

    err = np.max(np.abs(proj_3d - proj_h))
    assert err < 1e-6, f"Z=0.5 projection error: {err}"


def test_6_planar_output_alignment(sample_camera):
    """Test 6: Projected planar floor boundary corners align with 3D projection."""
    floor_rect = [
        [0.67, 0.47],
        [9.0, 0.47],
        [9.0, 3.32],
        [0.67, 3.32]
    ]
    derived = derive_homography(sample_camera, plane_elevation=0.0, width=1920, height=1080)
    H_mat = derived["H_cad_to_image"]

    uvs = project_cad_polygon(floor_rect, H_mat)
    assert len(uvs) >= 3, "Floor polygon should project to at least 3 vertices"

    # Verify first corner against 3D projection
    pt_3d = np.array([[0.67, 0.47, 0.0]])
    p3d, _ = project_points(pt_3d, sample_camera, 1920, 1080)
    # The first vertex of floor_rect [0.67, 0.47] is in front of camera
    assert np.allclose(uvs[0], p3d[0], atol=1e-4)


def test_7_export_and_homography_json(tmp_path, sample_camera):
    """Test 7: Export saves planar.png, homography.json, and updates metadata and layout.json."""
    layout_id = "Layout_20260919_120000"
    root = tmp_path / layout_id
    root.mkdir()
    state = new_state(layout_id, "Test Layout", "source.dxf")
    state["source"]["units"] = "m"
    state["cameras"]["camera_001"] = sample_camera
    state["active_camera"] = "camera_001"
    state["entities"]["floor_01"] = {
        "semantic": "floor",
        "dxf_handle": "F01",
        "elevation": 0.0,
        "thickness": 0.15
    }
    (root / "layout.json").write_text(json.dumps(state))

    payload = {
        "camera_id": "camera_001",
        "resolution": {"width": 1920, "height": 1080}
    }

    result = save_conditioning_outputs(root, state, payload)
    assert "planar.png" in result["files"]
    assert "homography.json" in result["files"]

    out_dir = root / "generated" / "camera_001"
    assert (out_dir / "homography.json").exists()

    homo_json = json.loads((out_dir / "homography.json").read_text())
    assert homo_json["schema_version"] == "0.1"
    assert homo_json["layout_id"] == layout_id
    assert homo_json["camera_id"] == "camera_001"
    assert homo_json["source_plane"]["elevation_m"] == 0.0
    assert len(homo_json["H_cad_to_image"]) == 3
    assert len(homo_json["H_image_to_cad"]) == 3
    assert homo_json["validation"]["mean_reprojection_error_px"] < 1.0

    meta_json = json.loads((out_dir / "metadata.json").read_text())
    assert "planar" in meta_json
    assert meta_json["planar"]["image"] == "planar.png"
    assert meta_json["planar"]["homography"] == "homography.json"

    # Check layout.json outputs
    saved_state = json.loads((root / "layout.json").read_text())
    assert "planar" in saved_state["outputs"]
    assert "generated/camera_001/planar.png" in saved_state["outputs"]["planar"]


def test_8_line_segment_clipping_and_xray_render(sample_camera):
    """Test 8: Line segment clipping and X-ray CAD wireframe rendering."""
    from app.geometry.homography import clip_line_segment_to_camera_front, render_planar_map

    derived = derive_homography(sample_camera, plane_elevation=0.0, width=1920, height=1080)
    H_mat = derived["H_cad_to_image"]

    # Segment fully in front of camera
    seg_front = clip_line_segment_to_camera_front([1.0, 1.0], [2.0, 2.0], H_mat)
    assert seg_front is not None
    assert len(seg_front) == 2

    # Segment fully behind camera (camera is at X=9.5, looking towards -X)
    seg_behind = clip_line_segment_to_camera_front([20.0, 1.0], [25.0, 2.0], H_mat)
    assert seg_behind is None

    # Segment crossing camera plane
    seg_cross = clip_line_segment_to_camera_front([5.0, 1.0], [15.0, 1.0], H_mat)
    assert seg_cross is not None
    assert len(seg_cross) == 2

    # Test rendering with wall, furniture, void, and floor
    state = new_state("Layout_test", "Test Layout", "source.dxf")
    state["cameras"]["camera_001"] = sample_camera
    state["active_camera"] = "camera_001"
    state["entities"] = {
        "e_floor": {"semantic": "floor", "dxf_handle": "H1", "elevation": 0.0},
        "e_wall": {"semantic": "wall", "dxf_handle": "H2", "height": 2.8},
        "e_furn": {"semantic": "furniture", "dxf_handle": "H3", "category": "bed"},
        "e_void": {"semantic": "void", "dxf_handle": "H4", "category": "door", "base_elevation": 0.0}
    }
    geometry = [
        {"id": "e_floor", "paths": [[[0, 0], [10, 0], [10, 5], [0, 5], [0, 0]]]},
        {"id": "e_wall", "paths": [[[0, 0], [10, 0]], [[10, 0], [10, 5]]]},
        {"id": "e_furn", "paths": [[[2, 1], [4, 1], [4, 3], [2, 3], [2, 1]], [[2.2, 1.2], [3.8, 1.2]]]},
        {"id": "e_void", "paths": [[[5, 0], [6, 0]]]}
    ]
    img = render_planar_map(geometry, state, "camera_001", 1920, 1080)
    assert img.shape == (1080, 1920, 3)
    bg_pixels = np.all(img == [30, 19, 11], axis=-1)
    assert np.any(~bg_pixels), "Some pixels should be drawn in the planar map"

