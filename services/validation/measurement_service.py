"""Measurement service orchestrating Part 7B spatial error computation and persistence.

Coordinates planar homography measurement, single-view metrology, image-space mask metrics,
applicability rules, visual debugging overlays, and measurements.json serialization.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.geometry.homography import derive_homography, get_floor_elevation
from services.spatial_correspondence import atomic_write
from services.validation.homography_measurement import (
    calculate_position_error,
    extract_ground_contact_pixel,
    pixel_to_cad_xy,
    verify_camera_mapping_compatibility,
)
from services.validation.mask_metrics import calculate_image_space_metrics
from services.validation.vertical_metrology import (
    calculate_height_errors,
    propose_vertical_observations,
    reconstruct_vertical_height,
)


def get_entity_applicability(semantic: str) -> Dict[str, bool]:
    """Define metric applicability based on entity semantic category."""
    sem = (semantic or "").lower()
    if sem == "furniture":
        return {
            "planar_position": True,
            "image_space_mask": True,
            "footprint": True,
            "vertical_height": True,
            "opening_metrics": False,
        }
    elif sem in ("void", "opening", "door"):
        return {
            "planar_position": True,
            "image_space_mask": True,
            "footprint": False,
            "vertical_height": True,
            "opening_metrics": True,
        }
    elif sem == "window":
        return {
            "planar_position": True,
            "image_space_mask": True,
            "footprint": False,
            "vertical_height": True,
            "opening_metrics": True,
        }
    elif sem == "floor":
        return {
            "planar_position": False,
            "image_space_mask": True,
            "footprint": True,
            "vertical_height": False,
            "opening_metrics": False,
        }
    elif sem in ("wall", "ceiling"):
        return {
            "planar_position": False,
            "image_space_mask": True,
            "footprint": False,
            "vertical_height": sem == "ceiling",
            "opening_metrics": False,
        }
    return {
        "planar_position": True,
        "image_space_mask": True,
        "footprint": False,
        "vertical_height": True,
        "opening_metrics": False,
    }


def get_cad_entity_elevation(entity: dict, floor_elevation: float = 0.0) -> float:
    """Retrieve base elevation of a CAD entity."""
    if "base_elevation" in entity and entity["base_elevation"] is not None:
        return float(entity["base_elevation"])
    if "elevation" in entity and entity["elevation"] is not None:
        return float(entity["elevation"])
    if "sill_height" in entity and entity["sill_height"] is not None:
        return float(floor_elevation) + float(entity["sill_height"])
    return float(floor_elevation)


def get_cad_entity_height(entity: dict) -> Optional[float]:
    """Retrieve metric CAD height from entity metadata."""
    if entity.get("height") is not None:
        return float(entity["height"])
    if entity.get("opening_height") is not None:
        return float(entity["opening_height"])
    return None


def get_cad_ground_truth_xy(
    entity: dict,
    cad_gt_mask: Optional[np.ndarray],
    H_image_to_cad: np.ndarray,
    method: str = "mask_bottom_contact_centroid",
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Retrieve ground-truth CAD XY coordinate matching the landmark definition."""
    # 1. Landmark consistency: If CAD GT mask is available, compute identical landmark definition
    if cad_gt_mask is not None and np.any(cad_gt_mask > 0):
        cad_contact_px = extract_ground_contact_pixel(cad_gt_mask, method=method)
        if cad_contact_px is not None:
            cad_xy = pixel_to_cad_xy(cad_contact_px[0], cad_contact_px[1], H_image_to_cad)
            if cad_xy is not None:
                return (round(cad_xy[0], 4), round(cad_xy[1], 4)), f"cad_gt_mask_{method}"

    # 2. Fallback to entity insertion point
    ins = entity.get("insertion_point")
    if ins and len(ins) >= 2:
        return (round(float(ins[0]), 4), round(float(ins[1]), 4)), "cad_entity_insertion_point"

    return None, "unavailable"


def measure_entity(
    context: dict,
    entity_id: str,
    correspondence_doc: dict,
    camera: dict,
    manual_observations: Optional[dict] = None,
) -> Dict[str, Any]:
    """Perform Part 7B spatial measurement for a single CAD entity."""
    # 1. Check correspondence record
    corr_record = next(
        (r for r in correspondence_doc.get("correspondences", []) if r.get("entity_id") == entity_id),
        None,
    )
    if not corr_record:
        return {
            "entity_id": entity_id,
            "measurement_status": "cannot_evaluate",
            "reason": "missing_correspondence: entity has no Part 7A correspondence record.",
        }

    status = corr_record.get("status")
    if status in ("rejected", "cannot_evaluate", "confirmed_missing", "segmentation_failure"):
        return {
            "entity_id": entity_id,
            "instance_id": corr_record.get("instance_id"),
            "measurement_status": "cannot_evaluate",
            "reason": f"Part 7A correspondence status is '{status}'.",
        }

    # 2. Check generated mask
    mask_rel = corr_record.get("generated_mask")
    if not mask_rel:
        return {
            "entity_id": entity_id,
            "instance_id": corr_record.get("instance_id"),
            "measurement_status": "cannot_evaluate",
            "reason": "No generated mask is associated with this correspondence record.",
        }

    mask_path = context["evaluation"] / mask_rel
    if not mask_path.is_file():
        return {
            "entity_id": entity_id,
            "instance_id": corr_record.get("instance_id"),
            "measurement_status": "cannot_evaluate",
            "reason": f"Generated mask file missing on disk: {mask_rel}.",
        }

    gen_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if not np.any(gen_mask):
        return {
            "entity_id": entity_id,
            "instance_id": corr_record.get("instance_id"),
            "measurement_status": "cannot_evaluate",
            "reason": "Generated mask is empty (0 foreground pixels).",
        }

    # 3. Check CAD ground truth mask
    instance_id = corr_record.get("instance_id")
    gt_mask = (context["ids"] == instance_id) if instance_id is not None else None
    if gt_mask is None or not np.any(gt_mask):
        return {
            "entity_id": entity_id,
            "instance_id": instance_id,
            "measurement_status": "cannot_evaluate",
            "reason": f"CAD ground truth instance {instance_id} is not visible in current camera view.",
        }

    # 4. Check Resolution / Crop Compatibility
    render_size = (context["image"].width, context["image"].height)
    cad_size = (context["ids"].shape[1], context["ids"].shape[0])
    compat = verify_camera_mapping_compatibility(camera, render_size, cad_size)
    if not compat["compatible"]:
        return {
            "entity_id": entity_id,
            "instance_id": instance_id,
            "measurement_status": "cannot_evaluate",
            "camera_mapping_status": "invalid_or_requires_alignment",
            "reason": f"Resolution or aspect ratio mismatch: {'; '.join(compat['reasons'])}",
        }

    # 5. Retrieve entity metadata and applicability
    entity_meta = context["entities"].get(entity_id, {})
    semantic = entity_meta.get("semantic", "furniture")
    appl = get_entity_applicability(semantic)
    floor_elev = get_floor_elevation({"entities": context["entities"]})
    base_elev = get_cad_entity_elevation(entity_meta, floor_elev)

    # 6. Derive Homography
    out = camera.get("output") or {}
    cam_w = float(out.get("width", render_size[0]))
    cam_h = float(out.get("height", render_size[1]))
    homo_derived = derive_homography(camera, plane_elevation=base_elev, width=cam_w, height=cam_h)
    H_image_to_cad = homo_derived["H_image_to_cad"]

    result: Dict[str, Any] = {
        "entity_id": entity_id,
        "instance_id": instance_id,
        "semantic": semantic,
        "measurement_status": "evaluated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "camera_id": context.get("camera_id"),
            "render_name": context.get("render", Path()).name,
            "render_resolution": list(render_size),
            "cad_resolution": list(cad_size),
            "homography_condition_number": round(homo_derived["condition_number"], 2),
            "reference_plane_elevation": round(base_elev, 4),
        },
    }

    # 7. Image-Space Mask Metrics
    if appl["image_space_mask"]:
        result["image_space"] = calculate_image_space_metrics(gt_mask, gen_mask)
    else:
        result["image_space"] = "not_applicable"

    # 8. Planar Spatial Measurement
    if appl["planar_position"]:
        contact_method = "mask_bottom_contact_centroid"
        gen_contact_px = extract_ground_contact_pixel(gen_mask, method=contact_method)
        if gen_contact_px is None:
            result["planar"] = {
                "status": "cannot_evaluate",
                "reason": "Failed to extract ground contact from generated mask.",
            }
        else:
            gen_xy = pixel_to_cad_xy(gen_contact_px[0], gen_contact_px[1], H_image_to_cad)
            gt_xy, gt_landmark_source = get_cad_ground_truth_xy(
                entity_meta, gt_mask, H_image_to_cad, method=contact_method
            )
            if gen_xy is None:
                result["planar"] = {
                    "status": "cannot_evaluate",
                    "reason": "Generated contact point projected behind camera or to infinity.",
                    "generated_contact_pixel": [round(gen_contact_px[0], 2), round(gen_contact_px[1], 2)],
                }
            elif gt_xy is None:
                result["planar"] = {
                    "status": "cannot_evaluate",
                    "reason": "CAD ground-truth landmark coordinates could not be resolved.",
                }
            else:
                pos_err = calculate_position_error(gt_xy, gen_xy)
                result["planar"] = {
                    "status": "evaluated",
                    "ground_contact_method": contact_method,
                    "gt_landmark_source": gt_landmark_source,
                    "generated_contact_pixel": [round(gen_contact_px[0], 2), round(gen_contact_px[1], 2)],
                    "ground_truth_xy": [round(gt_xy[0], 4), round(gt_xy[1], 4)],
                    "generated_xy": [round(gen_xy[0], 4), round(gen_xy[1], 4)],
                    "position_error_m": round(pos_err, 4),
                }
    else:
        result["planar"] = "not_applicable"

    # 9. Single-View Vertical Metrology
    if appl["vertical_height"]:
        # Check if human supplied/adjusted observations exist
        obs = manual_observations or propose_vertical_observations(gen_mask)
        cad_height = get_cad_entity_height(entity_meta)

        if not obs or not obs.get("base_pixel") or not obs.get("top_pixel"):
            result["vertical"] = {
                "status": "cannot_evaluate",
                "reason": "Unable to establish base and top observation points.",
            }
        elif cad_height is None:
            result["vertical"] = {
                "status": "cannot_evaluate",
                "reason": "CAD entity metadata has no height, opening_height, or metric dimension.",
            }
        else:
            v_recon = reconstruct_vertical_height(
                obs["base_pixel"],
                obs["top_pixel"],
                camera,
                H_image_to_cad,
                base_elevation=base_elev,
                width=cam_w,
                height=cam_h,
            )
            if v_recon["status"] != "evaluated":
                result["vertical"] = {
                    "status": "cannot_evaluate",
                    "reason": v_recon["reason"],
                    "observation_method": obs.get("observation_method", "automatic"),
                    "base_pixel": obs["base_pixel"],
                    "top_pixel": obs["top_pixel"],
                }
            else:
                h_recon = v_recon["reconstructed_height_m"]
                h_errs = calculate_height_errors(cad_height, h_recon)
                result["vertical"] = {
                    "status": "evaluated",
                    "observation_method": obs.get("observation_method", "automatic"),
                    "base_pixel": obs["base_pixel"],
                    "top_pixel": obs["top_pixel"],
                    "cad_height_m": h_errs["cad_height_m"],
                    "reconstructed_height_m": h_errs["reconstructed_height_m"],
                    "height_error_m": h_errs["height_error_m"],
                    "height_error_percent": h_errs["height_error_percent"],
                }
    else:
        result["vertical"] = "not_applicable"

    # 10. Openings (Door / Window) Metrics
    if appl["opening_metrics"]:
        sill_ht = entity_meta.get("sill_height")
        op_ht = entity_meta.get("opening_height")
        result["openings"] = {
            "opening_height_cad_m": round(float(op_ht), 4) if op_ht is not None else None,
            "sill_height_cad_m": round(float(sill_ht), 4) if sill_ht is not None else None,
            "evaluated": True,
        }
    else:
        result["openings"] = "not_applicable"

    return result


def measure_all_entities(
    context: dict,
    camera: dict,
    manual_observations_map: Optional[Dict[str, dict]] = None,
) -> Dict[str, Any]:
    """Execute Part 7B measurement across all visible entities with Part 7A correspondence."""
    doc = context.get("doc") or {}
    entities_results = []
    manual_map = manual_observations_map or {}

    for entity in context.get("visible", []):
        eid = entity["entity_id"]
        res = measure_entity(
            context,
            eid,
            doc,
            camera,
            manual_observations=manual_map.get(eid),
        )
        entities_results.append(res)

    summary = {
        "schema_version": "0.1",
        "stage": "Part 7B — Spatial Measurement and Metric Reconstruction",
        "render_id": context.get("render", Path()).name,
        "camera_id": context.get("camera_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_visible_entities": len(context.get("visible", [])),
        "evaluated_count": sum(1 for e in entities_results if e.get("measurement_status") == "evaluated"),
        "cannot_evaluate_count": sum(1 for e in entities_results if e.get("measurement_status") == "cannot_evaluate"),
        "entities": entities_results,
    }
    return summary


def save_measurements(context: dict, measurements_data: dict) -> Path:
    """Save measurements.json atomically under evaluation folder."""
    target_path = context["evaluation"] / "measurements.json"
    content = json.dumps(measurements_data, indent=2).encode("utf-8")
    atomic_write(target_path, content)
    return target_path


def load_measurements(context: dict) -> Optional[Dict[str, Any]]:
    """Load existing measurements.json if present."""
    target_path = context["evaluation"] / "measurements.json"
    if target_path.is_file():
        try:
            return json.loads(target_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def generate_debug_overlay(
    image: Image.Image,
    gt_mask: np.ndarray,
    gen_mask: np.ndarray,
    ground_contact_pixel: Optional[Tuple[float, float]] = None,
    top_pixel: Optional[Tuple[float, float]] = None,
    gt_contact_pixel: Optional[Tuple[float, float]] = None,
) -> Image.Image:
    """Generate dynamic diagnostic visual overlay for Part 7B validation.

    Overlays:
    - Base RGB image
    - CAD GT mask contour (Cyan outline)
    - Generated SAM mask (Green tint with boundary)
    - Ground contact point (Yellow marker ●)
    - Top point (Magenta marker ▲)
    - CAD reference landmark (Orange marker ✕)
    """
    img_np = np.array(image.convert("RGB")).copy()

    # 1. Overlay SAM predicted mask (translucent green)
    if gen_mask is not None and np.any(gen_mask):
        tint = np.array([40, 220, 100], dtype=np.uint8)
        img_np[gen_mask] = (img_np[gen_mask] * 0.45 + tint * 0.55).astype(np.uint8)

        # Draw contour of SAM mask
        contours, _ = cv2.findContours(
            gen_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img_np, contours, -1, (20, 255, 120), 2)

    # 2. Draw CAD GT contour (Bright Cyan)
    if gt_mask is not None and np.any(gt_mask):
        gt_contours, _ = cv2.findContours(
            gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img_np, gt_contours, -1, (0, 220, 255), 2)

    overlay_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(overlay_pil)

    # 3. Draw Ground Contact Point (Yellow circle)
    if ground_contact_pixel is not None:
        gx, gy = ground_contact_pixel
        r = 6
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=(255, 230, 0), outline=(0, 0, 0), width=2)

    # 4. Draw Top Point (Magenta triangle/marker)
    if top_pixel is not None:
        tx, ty = top_pixel
        r = 6
        draw.polygon([(tx, ty - r), (tx - r, ty + r), (tx + r, ty + r)], fill=(255, 0, 220), outline=(0, 0, 0))

    # 5. Draw CAD GT Contact Point (Cyan / Orange circle)
    if gt_contact_pixel is not None:
        cx, cy = gt_contact_pixel
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 240, 255), outline=(0, 0, 0), width=2)

    return overlay_pil


def save_entity_overlay(
    context: dict, entity_id: str, overlay_image: Image.Image
) -> Path:
    """Save diagnostic overlay image under evaluation/overlays/."""
    folder = context["evaluation"] / "overlays"
    folder.mkdir(parents=True, exist_ok=True)
    target_path = folder / f"{entity_id}.png"
    overlay_image.save(target_path, format="PNG")
    return target_path


def render_topdown_planar_solution(
    context: dict,
    entity_id: str,
    camera: dict,
    selected_ent_data: dict,
    width: int = 800,
    height: int = 600,
) -> Dict[str, Image.Image]:
    """Render 2D top-down floor plan solution validating Reverse Homography.

    Returns:
    - 'diagram': Orthographic 2D CAD blueprint with camera cone, CAD position,
                 AI prediction, and displacement error vector.
    - 'warped_texture': Unprojected perspective render floor texture warped back
                        to the 2D CAD floor plane, overlaid with CAD geometry.
    """
    cam_pos = camera.get("position", [0.0, 0.0, 1.5])
    cam_x, cam_y = float(cam_pos[0]), float(cam_pos[1])
    heading_rad = math.radians(float(camera.get("heading_deg", 0.0)))
    fov_rad = math.radians(float(camera.get("fov_deg", 60.0)))

    # Collect points of interest to establish floor plan bounding box
    xs = [cam_x]
    ys = [cam_y]

    pl_data = selected_ent_data.get("planar") if isinstance(selected_ent_data.get("planar"), dict) else {}
    gt_xy = pl_data.get("ground_truth_xy")
    gen_xy = pl_data.get("generated_xy")

    if gt_xy:
        xs.append(float(gt_xy[0]))
        ys.append(float(gt_xy[1]))
    if gen_xy:
        xs.append(float(gen_xy[0]))
        ys.append(float(gen_xy[1]))

    for rec in context.get("entities", {}).values():
        ins = rec.get("insertion_point")
        if ins and len(ins) >= 2 and np.isfinite(ins[0]) and np.isfinite(ins[1]):
            xs.append(float(ins[0]))
            ys.append(float(ins[1]))

    # Add camera forward line
    view_dist = 4.0
    tgt_x = cam_x + view_dist * math.cos(heading_rad)
    tgt_y = cam_y + view_dist * math.sin(heading_rad)
    xs.append(tgt_x)
    ys.append(tgt_y)

    min_x, max_x = min(xs) - 1.2, max(xs) + 1.2
    min_y, max_y = min(ys) - 1.2, max(ys) + 1.2

    # Enforce minimum span of 4 meters
    span_x = max(4.0, max_x - min_x)
    span_y = max(4.0, max_y - min_y)
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0

    pad = 50
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)

    def world_to_canvas(wx: float, wy: float) -> Tuple[int, int]:
        cx = int(width / 2.0 + (wx - mid_x) * scale)
        cy = int(height / 2.0 - (wy - mid_y) * scale)
        return cx, cy

    # -------------------------------------------------------------
    # 1. Base Blueprint Canvas
    # -------------------------------------------------------------
    canvas_bgr = np.full((height, width, 3), (24, 20, 16), dtype=np.uint8)  # dark slate #101418

    # Draw 1m Grid
    grid_min_x = math.floor(mid_x - (width / (2 * scale)))
    grid_max_x = math.ceil(mid_x + (width / (2 * scale)))
    grid_min_y = math.floor(mid_y - (height / (2 * scale)))
    grid_max_y = math.ceil(mid_y + (height / (2 * scale)))

    for gx in range(grid_min_x, grid_max_x + 1):
        cx, _ = world_to_canvas(gx, 0)
        if 0 <= cx < width:
            cv2.line(canvas_bgr, (cx, 0), (cx, height), (44, 38, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas_bgr, f"{gx}m", (cx + 4, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 90, 80), 1, cv2.LINE_AA)

    for gy in range(grid_min_y, grid_max_y + 1):
        _, cy = world_to_canvas(0, gy)
        if 0 <= cy < height:
            cv2.line(canvas_bgr, (0, cy), (width, cy), (44, 38, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas_bgr, f"{gy}m", (6, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 90, 80), 1, cv2.LINE_AA)

    # Draw other CAD entities (dimmed)
    for eid, rec in context.get("entities", {}).items():
        ins = rec.get("insertion_point")
        if not ins or len(ins) < 2 or eid == entity_id:
            continue
        cx, cy = world_to_canvas(float(ins[0]), float(ins[1]))
        if 0 <= cx < width and 0 <= cy < height:
            color = (80, 110, 130)
            cv2.circle(canvas_bgr, (cx, cy), 4, color, -1)
            name = rec.get("description") or rec.get("category") or eid
            cv2.putText(canvas_bgr, name[:12], (cx + 6, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 140, 160), 1, cv2.LINE_AA)

    # Draw Camera Position & FOV Cone
    cam_cx, cam_cy = world_to_canvas(cam_x, cam_y)
    cone_dist = 3.5
    left_angle = heading_rad - fov_rad / 2.0
    right_angle = heading_rad + fov_rad / 2.0

    lc_x, lc_y = world_to_canvas(cam_x + cone_dist * math.cos(left_angle), cam_y + cone_dist * math.sin(left_angle))
    rc_x, rc_y = world_to_canvas(cam_x + cone_dist * math.cos(right_angle), cam_y + cone_dist * math.sin(right_angle))
    tc_x, tc_y = world_to_canvas(tgt_x, tgt_y)

    # Camera FOV transparent cone
    cone_overlay = canvas_bgr.copy()
    cone_pts = np.array([(cam_cx, cam_cy), (lc_x, lc_y), (rc_x, rc_y)], dtype=np.int32)
    cv2.fillPoly(cone_overlay, [cone_pts], (60, 160, 100), cv2.LINE_AA)
    cv2.addWeighted(cone_overlay, 0.15, canvas_bgr, 0.85, 0, canvas_bgr)

    cv2.line(canvas_bgr, (cam_cx, cam_cy), (lc_x, lc_y), (80, 180, 120), 1, cv2.LINE_AA)
    cv2.line(canvas_bgr, (cam_cx, cam_cy), (rc_x, rc_y), (80, 180, 120), 1, cv2.LINE_AA)
    cv2.line(canvas_bgr, (cam_cx, cam_cy), (tc_x, tc_y), (100, 220, 140), 1, cv2.LINE_AA)

    # Camera body dot
    cv2.circle(canvas_bgr, (cam_cx, cam_cy), 7, (240, 200, 60), -1, cv2.LINE_AA)
    cv2.circle(canvas_bgr, (cam_cx, cam_cy), 8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas_bgr, "Camera", (cam_cx - 18, cam_cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 200, 60), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # Highlight Selected Entity CAD GT vs AI Prediction
    # -------------------------------------------------------------
    diagram_bgr = canvas_bgr.copy()

    gt_c = world_to_canvas(float(gt_xy[0]), float(gt_xy[1])) if gt_xy else None
    gen_c = world_to_canvas(float(gen_xy[0]), float(gen_xy[1])) if gen_xy else None

    # Draw CAD Ground Truth (Cyan target)
    if gt_c:
        cv2.circle(diagram_bgr, gt_c, 8, (255, 230, 0), 2, cv2.LINE_AA)  # BGR Cyan: 255, 230, 0
        cv2.drawMarker(diagram_bgr, gt_c, (255, 230, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(
            diagram_bgr,
            f"CAD GT [{gt_xy[0]:.2f}, {gt_xy[1]:.2f}]",
            (gt_c[0] + 12, gt_c[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 240, 100),
            1,
            cv2.LINE_AA,
        )

    # Draw AI Generated Prediction (Bright Yellow dot)
    if gen_c:
        cv2.circle(diagram_bgr, gen_c, 8, (0, 230, 255), -1, cv2.LINE_AA)  # BGR Yellow: 0, 230, 255
        cv2.circle(diagram_bgr, gen_c, 10, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            diagram_bgr,
            f"AI Prediction [{gen_xy[0]:.2f}, {gen_xy[1]:.2f}]",
            (gen_c[0] + 12, gen_c[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw Displacement Error Vector (Red/Orange dashed arrow)
    if gt_c and gen_c:
        cv2.arrowedLine(diagram_bgr, gt_c, gen_c, (60, 60, 255), 2, cv2.LINE_AA, tipLength=0.25)
        # Midpoint label
        mx = int((gt_c[0] + gen_c[0]) / 2.0)
        my = int((gt_c[1] + gen_c[1]) / 2.0)
        pos_err_m = pl_data.get("position_error_m", 0.0)
        dist_str = f"Error: {pos_err_m:.3f} m ({pos_err_m*100:.1f} cm)"
        # Pill box behind label
        (w_txt, h_txt), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(diagram_bgr, (mx - 6, my - h_txt - 8), (mx + w_txt + 6, my + 6), (20, 20, 40), -1)
        cv2.rectangle(diagram_bgr, (mx - 6, my - h_txt - 8), (mx + w_txt + 6, my + 6), (60, 60, 255), 1)
        cv2.putText(diagram_bgr, dist_str, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend in top-left corner
    cv2.rectangle(diagram_bgr, (10, 10), (280, 75), (20, 24, 30), -1)
    cv2.rectangle(diagram_bgr, (10, 10), (280, 75), (50, 60, 70), 1)
    cv2.drawMarker(diagram_bgr, (25, 28), (255, 230, 0), cv2.MARKER_CROSS, 10, 2)
    cv2.putText(diagram_bgr, "CAD Ground Truth Location", (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    cv2.circle(diagram_bgr, (25, 48), 5, (0, 230, 255), -1)
    cv2.putText(diagram_bgr, "AI Predicted Location (Unprojected)", (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    cv2.line(diagram_bgr, (18, 65), (32, 65), (60, 60, 255), 2)
    cv2.putText(diagram_bgr, "Position Error Vector", (40, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    diagram_pil = Image.fromarray(cv2.cvtColor(diagram_bgr, cv2.COLOR_BGR2RGB))

    # -------------------------------------------------------------
    # 2. Unprojected Perspective Floor Texture (Bird's Eye Render)
    # -------------------------------------------------------------
    # Transform: T_cad_to_canvas: (X, Y) -> (cx, cy)
    # cx = width / 2 + (X - mid_x) * scale
    # cy = height / 2 - (Y - mid_y) * scale
    T_cad_to_canvas = np.array([
        [scale, 0.0, width / 2.0 - mid_x * scale],
        [0.0, -scale, height / 2.0 + mid_y * scale],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    out = camera.get("output") or {}
    cam_w = float(out.get("width", 1024))
    cam_h = float(out.get("height", 1024))
    homo_derived = derive_homography(camera, plane_elevation=0.0, width=cam_w, height=cam_h)
    H_img_to_cad = homo_derived["H_image_to_cad"]

    M_img_to_canvas = T_cad_to_canvas @ H_img_to_cad

    render_bgr = cv2.cvtColor(np.array(context["image"].convert("RGB")), cv2.COLOR_RGB2BGR)

    # Mask only the floor/lower portion to prevent wrapping sky/ceiling artifacts
    img_h, img_w = render_bgr.shape[:2]
    floor_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    floor_mask[int(img_h * 0.40):, :] = 255
    masked_render = cv2.bitwise_and(render_bgr, render_bgr, mask=floor_mask)

    warped_floor_bgr = cv2.warpPerspective(
        masked_render,
        M_img_to_canvas,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(20, 24, 30),
    )

    # Blend diagram annotations (grid, CAD marker, AI marker) on top of warped texture
    warped_composite = cv2.addWeighted(warped_floor_bgr, 0.85, diagram_bgr, 0.35, 0)
    # Redraw crisp markers on top
    if gt_c:
        cv2.drawMarker(warped_composite, gt_c, (255, 230, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(warped_composite, "CAD GT", (gt_c[0] + 8, gt_c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 240, 100), 1, cv2.LINE_AA)
    if gen_c:
        cv2.circle(warped_composite, gen_c, 8, (0, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(warped_composite, gen_c, 10, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(warped_composite, "AI Prediction", (gen_c[0] + 8, gen_c[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
    if gt_c and gen_c:
        cv2.arrowedLine(warped_composite, gt_c, gen_c, (60, 60, 255), 2, cv2.LINE_AA, tipLength=0.25)

    texture_pil = Image.fromarray(cv2.cvtColor(warped_composite, cv2.COLOR_BGR2RGB))

    return {
        "diagram": diagram_pil,
        "warped_texture": texture_pil,
    }

