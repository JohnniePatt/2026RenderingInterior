from copy import deepcopy
import numpy as np
import pytest
from app.geometry.camera import new_camera, save_cameras, camera_matrices, project_points
from app.layout.manager import create
from app.layout.persistence import read
from test_dxf_parser import sample


def test_camera_persistence_move_switch_delete(tmp_path):
    root, state = create(tmp_path, "Cameras", "room.dxf", sample())
    cameras = {"camera_001": new_camera(5420, 7310, "mm", 128, "Main"),
               "camera_002": new_camera(8210, 4250, "mm", 215, "Kitchen")}
    cameras["camera_001"].update(pitch_deg=-12.5, fov_deg=73)
    state = save_cameras(root, state, cameras, "camera_002")
    assert read(root)["cameras"] == cameras and read(root)["active_camera"] == "camera_002"
    cameras["camera_001"]["position"] = [1000, 2000, 1700]
    state = save_cameras(root, state, cameras, "camera_001")
    assert read(root)["cameras"]["camera_001"]["position"] == [1000, 2000, 1700]
    del cameras["camera_001"]
    save_cameras(root, state, cameras, "camera_002")
    assert read(root)["cameras"] == cameras


@pytest.mark.parametrize("unit,height", [("mm",1500),("m",1.5),("cm",150),("in",1500/25.4),("ft",1500/304.8)])
def test_camera_height_units(unit, height):
    assert new_camera(0, 0, unit)["position"][2] == pytest.approx(height)


def test_projection_orientation_and_intrinsics():
    camera = new_camera(0, 0, "m")
    pixels, depths = project_points([[10,0,1.5],[10,-1,1.5],[10,0,2.5],[-10,0,1.5]], camera, 800, 600)
    np.testing.assert_allclose(pixels[0], [400,300])
    assert pixels[1,0] > 400 and pixels[2,1] < 300
    assert np.isnan(pixels[3]).all() and depths[3] < 0
    matrices = camera_matrices(camera,800,600)
    np.testing.assert_allclose(matrices["R"] @ matrices["R"].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrices["R"]) == pytest.approx(1)
    assert matrices["K"][0,0] == pytest.approx(400/np.tan(np.pi/6))
    camera.update(heading_deg=90, pitch_deg=30)
    target = np.asarray(camera["position"]) + np.array([0, np.cos(np.pi/6), .5])*10
    pixels, _ = project_points([target], camera, 600, 800)
    np.testing.assert_allclose(pixels[0], [300,400], atol=1e-10)


def test_invalid_camera_and_stale_bridge(tmp_path):
    root, state = create(tmp_path, "Cameras", "room.dxf", sample())
    camera = new_camera(0,0,"mm")
    with pytest.raises(ValueError, match="changed"):
        save_cameras(root,state,{"camera_001":camera},"camera_001", "old")
    camera["pitch_deg"] = 90
    with pytest.raises(ValueError, match="pitch"):
        save_cameras(root,state,{"camera_001":camera},"camera_001")
    assert read(root)["cameras"] == {}
