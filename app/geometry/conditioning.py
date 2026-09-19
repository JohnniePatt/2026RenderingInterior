"""Geometric conditioning outputs (Proxy RGB, Depth, Instance, Semantic).

Provides deterministic mapping, numeric array decoding, metadata serialization,
and registration with layout.json.
"""
import base64
import json
from pathlib import Path
import cv2
import numpy as np
from app.geometry.camera import camera_matrices
from app.layout.persistence import save

SEMANTIC_CLASSES = {
    0: "background",
    1: "wall",
    2: "floor",
    3: "furniture",
    4: "void",
    5: "ceiling",
}

SEMANTIC_NAME_TO_ID = {v: k for k, v in SEMANTIC_CLASSES.items()}


def build_instance_mapping(state):
    """Deterministically map integer instance IDs to CAD entity IDs and auto_ceiling."""
    entities = state.get("entities", {})
    sorted_ids = sorted(
        entity_id for entity_id, rec in entities.items()
        if not rec.get("orphaned") and rec.get("semantic")
    )
    mapping = {"0": "background"}
    for idx, entity_id in enumerate(sorted_ids, start=1):
        mapping[str(idx)] = entity_id

    # If ceiling is configured/active, assign next sequential instance ID
    c_info = state.get("ceiling") or {}
    auto_ceiling = state.get("proxy_settings", {}).get("auto_ceiling", True) or c_info.get("enabled", True)
    if auto_ceiling:
        mapping[str(len(mapping))] = "auto_ceiling"

    return mapping


def build_conditioning_metadata(state, camera_id, resolution):
    """Construct authoritative metadata dictionary for exported camera conditioning passes."""
    camera = state["cameras"][camera_id]
    w = int(resolution.get("width", 1024))
    h = int(resolution.get("height", 1024))
    matrices = camera_matrices(camera, width=w, height=h)

    inst_map = build_instance_mapping(state)
    sem_map = {str(k): v for k, v in SEMANTIC_CLASSES.items()}

    # Preserve door/window category details in metadata for downstream tasks
    categories = {}
    for entity_id, rec in state.get("entities", {}).items():
        if not rec.get("orphaned") and rec.get("semantic"):
            categories[entity_id] = {
                "semantic": rec.get("semantic"),
                "category": rec.get("category"),
                "description": rec.get("description", ""),
                "material": rec.get("material")
            }
    if "auto_ceiling" in inst_map.values():
        c_info = state.get("ceiling") or {}
        categories["auto_ceiling"] = {
            "semantic": "ceiling",
            "category": "ceiling",
            "description": c_info.get("description") or state.get("proxy_settings", {}).get("ceiling_description", ""),
            "material": c_info.get("material") or state.get("proxy_settings", {}).get("ceiling_material")
        }

    return {
        "schema_version": "0.1",
        "layout_id": state["layout"]["id"],
        "camera_id": camera_id,
        "resolution": {"width": w, "height": h},
        "camera": {
            "name": camera.get("name", camera_id),
            "position": [round(float(v), 6) for v in camera["position"]],
            "target": [round(float(v), 6) for v in camera.get("target", [])],
            "heading_deg": round(float(camera.get("heading_deg", 0)), 4),
            "pitch_deg": round(float(camera.get("pitch_deg", 0)), 4),
            "fov_deg": round(float(camera.get("fov_deg", 60)), 2)
        },
        "matrices": {
            "K": matrices["K"].tolist(),
            "R": matrices["R"].tolist(),
            "t": matrices["t"].tolist(),
            "P": matrices["P"].tolist()
        },
        "depth": {
            "unit": "m",
            "type": "camera_z",
            "background": "NaN"
        },
        "semantic_mapping": sem_map,
        "instance_mapping": inst_map,
        "entity_details": categories
    }


def decode_data_url(data_url):
    """Strip base64 data URL header and decode raw bytes."""
    if not isinstance(data_url, str):
        return b""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def save_conditioning_outputs(root, state, payload):
    """Decode and write all 4 conditioning passes, raw numpy arrays, and metadata to disk."""
    root = Path(root)
    camera_id = payload.get("camera_id")
    if not camera_id or camera_id not in state.get("cameras", {}):
        raise ValueError(f"Invalid or missing camera_id: {camera_id}")

    res = payload.get("resolution") or state["cameras"][camera_id].get("output") or {"width": 1024, "height": 1024}
    width = int(res["width"])
    height = int(res["height"])

    output_dir = root / "generated" / camera_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save visual PNGs
    for key, filename in [
        ("proxy_png", "proxy.png"),
        ("depth_vis_png", "depth.png"),
        ("instance_vis_png", "instance.png"),
        ("semantic_vis_png", "semantic.png"),
        ("planar_png", "planar.png"),
    ]:
        data = decode_data_url(payload.get(key, ""))
        if data:
            (output_dir / filename).write_bytes(data)

    # If planar.png was not supplied in payload, generate it authoritatively from CAD
    if not (output_dir / "planar.png").exists():
        from app.cad.entities import load_for_editor
        from app.geometry.homography import render_planar_map
        try:
            _, geometry, _ = load_for_editor(root)
            planar_img = render_planar_map(geometry, state, state["cameras"][camera_id], width=width, height=height)
            cv2.imwrite(str(output_dir / "planar.png"), planar_img)
        except Exception:
            pass

    # 2. Decode and save raw numeric Depth array (float32 camera_z in meters, NaN for bg)
    depth_raw_bytes = decode_data_url(payload.get("depth_raw_png", ""))
    if depth_raw_bytes:
        depth_img = cv2.imdecode(np.frombuffer(depth_raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            if depth_img.shape[2] == 4:
                b, g, r, a = cv2.split(depth_img)
            else:
                b, g, r = cv2.split(depth_img)
                a = np.full((depth_img.shape[0], depth_img.shape[1]), 255, dtype=np.uint8)
            # R is low byte (mm), G is mid byte, B is high byte
            depth_mm = r.astype(np.float32) + g.astype(np.float32) * 256.0 + b.astype(np.float32) * 65536.0
            depth_m = depth_mm / 1000.0
            depth_m[a == 0] = np.nan
            np.save(output_dir / "depth.npy", depth_m.astype(np.float32))

    # 3. Decode and save raw numeric Instance ID array (int32, 0 for bg)
    inst_raw_bytes = decode_data_url(payload.get("instance_raw_png", ""))
    if inst_raw_bytes:
        inst_img = cv2.imdecode(np.frombuffer(inst_raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if inst_img is not None:
            if inst_img.shape[2] == 4:
                b, g, r, a = cv2.split(inst_img)
            else:
                b, g, r = cv2.split(inst_img)
                a = np.full((inst_img.shape[0], inst_img.shape[1]), 255, dtype=np.uint8)
            inst_ids = r.astype(np.int32) + g.astype(np.int32) * 256
            inst_ids[a == 0] = 0
            np.save(output_dir / "instance.npy", inst_ids.astype(np.int32))

    # 4. Decode and save raw numeric Semantic class array (uint8, 0 for bg)
    sem_raw_bytes = decode_data_url(payload.get("semantic_raw_png", ""))
    if sem_raw_bytes:
        sem_img = cv2.imdecode(np.frombuffer(sem_raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if sem_img is not None:
            if sem_img.shape[2] == 4:
                b, g, r, a = cv2.split(sem_img)
            else:
                b, g, r = cv2.split(sem_img)
                a = np.full((sem_img.shape[0], sem_img.shape[1]), 255, dtype=np.uint8)
            sem_ids = r.astype(np.uint8)
            sem_ids[a == 0] = 0
            np.save(output_dir / "semantic.npy", sem_ids.astype(np.uint8))

    # 5. Build and save homography.json
    from app.geometry.homography import build_homography_json, get_floor_elevation
    homo_doc = build_homography_json(state["layout"]["id"], camera_id, state, state["cameras"][camera_id], width=width, height=height)
    (output_dir / "homography.json").write_text(json.dumps(homo_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 6. Build and save authoritative metadata.json
    metadata = build_conditioning_metadata(state, camera_id, {"width": width, "height": height})
    metadata["planar"] = {
        "image": "planar.png",
        "homography": "homography.json",
        "plane_elevation_m": get_floor_elevation(state)
    }
    # Merge any client-supplied metadata overrides
    if isinstance(payload.get("metadata"), dict):
        metadata.update(payload["metadata"])
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 7. Update layout.json outputs records cleanly without duplicates
    outputs = state.setdefault("outputs", {})
    for key, path in [
        ("proxy", f"generated/{camera_id}/proxy.png"),
        ("depth", f"generated/{camera_id}/depth.npy"),
        ("instance", f"generated/{camera_id}/instance.npy"),
        ("semantic", f"generated/{camera_id}/semantic.npy"),
        ("planar", f"generated/{camera_id}/planar.png"),
    ]:
        existing = outputs.setdefault(key, [])
        outputs[key] = list(dict.fromkeys([*existing, path]))

    save(root, state)
    return {
        "output_dir": str(output_dir),
        "files": [
            "proxy.png", "depth.png", "depth.npy",
            "instance.png", "instance.npy",
            "semantic.png", "semantic.npy",
            "planar.png", "homography.json",
            "metadata.json"
        ]
    }
