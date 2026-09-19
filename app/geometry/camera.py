"""DXF world: X east, Y north, Z up; heading CCW from +X, pitch up.

Research extrinsics use OpenCV axes: image right, image down, forward.
fov_deg is HORIZONTAL. Three.js uses vertical FOV; convert using viewport aspect.
"""
from copy import deepcopy
import math
import numpy as np
from app.cad.units import from_mm, COMMON_UNITS
from app.layout.schema import finite
from app.layout.persistence import save


def validate_camera(camera):
    if not isinstance(camera, dict):
        raise ValueError("Camera must be an object")
    pos = camera.get("position")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3 or not all(finite(v) for v in pos):
        raise ValueError("Invalid camera position")
    if not all(finite(camera.get(k)) for k in ("heading_deg", "pitch_deg", "fov_deg")):
        raise ValueError("Invalid camera angles")
    if not 0 < camera["fov_deg"] < 180:
        raise ValueError("Invalid camera FOV")
    if not -90 < camera["pitch_deg"] < 90:
        raise ValueError("Camera pitch must be between -90 and 90 degrees")
    if not isinstance(camera.get("name"), str) or not camera["name"].strip():
        raise ValueError("Camera name is required")
    if "target" in camera and camera["target"] is not None:
        target = camera["target"]
        if not isinstance(target, (list, tuple)) or len(target) != 3 or not all(finite(v) for v in target):
            raise ValueError("Invalid camera target")
    if "output" in camera and camera["output"] is not None:
        output = camera["output"]
        if not isinstance(output, dict) or not finite(output.get("width")) or not finite(output.get("height")):
            raise ValueError("Camera output must contain numeric width and height")
        if output["width"] <= 0 or output["height"] <= 0:
            raise ValueError("Camera output dimensions must be positive")


def calculate_orientation(position, target):
    dx = float(target[0]) - float(position[0])
    dy = float(target[1]) - float(position[1])
    dz = float(target[2]) - float(position[2])
    dxy = math.hypot(dx, dy)
    if dxy < 1e-9 and abs(dz) < 1e-9:
        return 0.0, 0.0
    heading = (math.degrees(math.atan2(dy, dx))) % 360
    pitch = math.degrees(math.atan2(dz, dxy))
    pitch = max(-89.9, min(89.9, pitch))
    return heading, pitch


def calculate_target(position, heading_deg, pitch_deg, distance=5.0):
    h = math.radians(heading_deg)
    p = math.radians(pitch_deg)
    dx = distance * math.cos(p) * math.cos(h)
    dy = distance * math.cos(p) * math.sin(h)
    dz = distance * math.sin(p)
    return [float(position[0]) + dx, float(position[1]) + dy, float(position[2]) + dz]


def set_camera_target(camera, target):
    camera["target"] = [float(v) for v in target]
    h, p = calculate_orientation(camera["position"], camera["target"])
    camera["heading_deg"] = h
    camera["pitch_deg"] = p
    camera["_last_target"] = list(camera["target"])
    camera["_last_angles"] = (h, p)
    return camera


def set_camera_orientation(camera, heading_deg, pitch_deg, distance=None):
    camera["heading_deg"] = float(heading_deg) % 360
    camera["pitch_deg"] = float(pitch_deg)
    if "target" in camera and camera["target"] is not None:
        if distance is None:
            distance = math.dist(camera["position"], camera["target"])
            if distance < 1e-4:
                distance = 5.0
        camera["target"] = calculate_target(camera["position"], camera["heading_deg"], camera["pitch_deg"], distance)
        camera["_last_target"] = list(camera["target"])
    camera["_last_angles"] = (camera["heading_deg"], camera["pitch_deg"])
    return camera


def camera_aspect(camera_or_output):
    if isinstance(camera_or_output, dict):
        out = camera_or_output.get("output", camera_or_output)
        w = out.get("width", 1024)
        h = out.get("height", 1024)
        return float(w) / float(h)
    return 1.0


def new_camera(x, y, units, heading=0, name="Camera", output=None, target=None):
    if units not in COMMON_UNITS or units == "unknown":
        raise ValueError("Confirm drawing units before placing a camera")
    pos = [x, y, from_mm(1500, units)]
    out = dict(output) if output is not None else {"width": 1024, "height": 1024}
    if target is None:
        tgt = calculate_target(pos, heading, 0.0, distance=from_mm(5000, units))
    else:
        tgt = [float(v) for v in target]
    result = {"name": name, "position": pos, "target": tgt,
              "heading_deg": heading % 360, "pitch_deg": 0.0, "fov_deg": 60.0,
              "output": out}
    validate_camera(result)
    return result


def save_cameras(root, state, cameras, active, expected_revision=None):
    if expected_revision is not None and expected_revision != state["layout"]["updated_at"]:
        raise ValueError("Layout changed since this camera view loaded. Reopen the Layout before saving.")
    if not isinstance(cameras, dict):
        raise ValueError("Cameras must be an object")
    for camera_id, camera in cameras.items():
        if not isinstance(camera_id, str) or not camera_id.startswith("camera_"):
            raise ValueError("Invalid camera ID")
        if "output" not in camera or not camera["output"]:
            camera["output"] = {"width": 1024, "height": 1024}
        validate_camera(camera)
    if active is not None and active not in cameras:
        raise ValueError("Active camera does not exist")
    candidate = deepcopy(state)
    candidate.update(cameras=deepcopy(cameras), active_camera=active)
    save(root, candidate)
    return candidate


def camera_matrices(camera, width=None, height=None):
    validate_camera(camera)
    if width is None and height is None:
        out = camera.get("output") or {}
        width = float(out.get("width", 1024))
        height = float(out.get("height", 1024))
    elif width is None or height is None:
        raise ValueError("Both width and height must be provided if one is given")
    if not finite(width) or not finite(height) or width <= 0 or height <= 0:
        raise ValueError("Viewport dimensions must be positive")

    pos = camera["position"]
    tgt = camera.get("target")
    if tgt is not None:
        last_tgt = camera.get("_last_target")
        last_angles = camera.get("_last_angles")
        cur_angles = (camera.get("heading_deg"), camera.get("pitch_deg"))
        if last_tgt is not None and list(tgt) != list(last_tgt):
            h, p = calculate_orientation(pos, tgt)
            camera["heading_deg"] = h
            camera["pitch_deg"] = p
        elif last_angles is not None and cur_angles != last_angles:
            d = math.dist(pos, tgt)
            camera["target"] = calculate_target(pos, cur_angles[0], cur_angles[1], d if d > 1e-4 else 5.0)
        elif last_tgt is None and last_angles is None:
            # Sync orientation from target
            h, p = calculate_orientation(pos, tgt)
            if math.hypot(tgt[0] - pos[0], tgt[1] - pos[1]) > 1e-5:
                # If target was explicitly placed
                camera["heading_deg"] = h
                camera["pitch_deg"] = p
        camera["_last_target"] = list(camera["target"])
        camera["_last_angles"] = (camera["heading_deg"], camera["pitch_deg"])

    heading, pitch = np.radians([camera["heading_deg"], camera["pitch_deg"]])
    forward = np.array([np.cos(pitch)*np.cos(heading), np.cos(pitch)*np.sin(heading), np.sin(pitch)])
    right = np.array([np.sin(heading), -np.cos(heading), 0.0])
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    translation = -rotation @ np.asarray(camera["position"], dtype=float)
    focal = width / (2 * math.tan(math.radians(camera["fov_deg"]) / 2))
    intrinsic = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]])
    projection = intrinsic @ np.column_stack([rotation, translation])
    return {"K": intrinsic, "R": rotation, "t": translation, "P": projection}


def project_points(points, camera, width=None, height=None):
    matrices = camera_matrices(camera, width, height)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 world points")
    camera_points = points @ matrices["R"].T + matrices["t"]
    image = camera_points @ matrices["K"].T
    result = np.full((len(points), 2), np.nan)
    visible = camera_points[:, 2] > 0
    result[visible] = image[visible, :2] / image[visible, 2, None]
    return result, camera_points[:, 2]
