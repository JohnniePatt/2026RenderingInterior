"""Service for extracting and persisting furniture references from rendered images using instance segmentation masks."""
from datetime import datetime
import io
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image

from app.annotation.tagging import store_reference
from app.layout.persistence import resolve, save


def extract_instance_crops(
    render_image: Image.Image | Path | str | bytes,
    instance_image: Image.Image | Path | str | np.ndarray,
    instance_mapping: dict[str, Any],
    entities: Optional[dict[str, Any]] = None,
    state: dict[str, Any] | None = None,
    padding_ratio: float = 0.08,
    min_pixels: int = 25,
) -> list[dict[str, Any]]:
    """Extract individual object crops from a render image using instance segmentation.

    Args:
        render_image: File path (str/Path), PIL Image, or bytes of the render output.
        instance_image: File path to instance.png or instance.npy, or PIL Image / ndarray.
        instance_mapping: Dictionary mapping instance ID strings to entity metadata
                          (as stored in layout_state["instance_mapping"][camera_id]).
        entities: Dictionary of entities from layout_state["entities"].
        state: Optional full layout_state dictionary (for ceiling metadata, etc.).
        padding_ratio: Margin padding around the bounding box as a fraction of box dimensions (default 8%).
        min_pixels: Minimum pixel count required for an instance to be extracted (filters noise).

    Returns:
        List of dicts containing extracted crop images and rich metadata.
    """
    # 1. Load render image as PIL RGB
    if isinstance(render_image, (str, Path)):
        render_img = Image.open(render_image).convert("RGB")
    elif isinstance(render_image, bytes):
        render_img = Image.open(io.BytesIO(render_image)).convert("RGB")
    elif isinstance(render_image, Image.Image):
        render_img = render_image.convert("RGB")
    else:
        raise ValueError(f"Unsupported render_image type: {type(render_image)}")

    img_w, img_h = render_img.size

    # 2. Load instance image / array
    inst_npy = None
    inst_w, inst_h = img_w, img_h
    if isinstance(instance_image, np.ndarray):
        inst_npy = instance_image
        inst_h, inst_w = inst_npy.shape[:2]
        inst_rgb = None
    elif isinstance(instance_image, (str, Path)):
        path = Path(instance_image)
        if path.suffix.lower() == ".npy":
            inst_npy = np.load(path)
            inst_h, inst_w = inst_npy.shape[:2]
            inst_rgb = None
        else:
            inst_img = Image.open(path).convert("RGB")
            inst_w, inst_h = inst_img.size
            inst_rgb = np.array(inst_img)
    elif isinstance(instance_image, Image.Image):
        inst_img = instance_image.convert("RGB")
        inst_w, inst_h = inst_img.size
        inst_rgb = np.array(inst_img)
    else:
        raise ValueError(f"Unsupported instance_image type: {type(instance_image)}")

    scale_x = img_w / float(inst_w) if inst_w > 0 else 1.0
    scale_y = img_h / float(inst_h) if inst_h > 0 else 1.0

    entities_dict = entities or {}
    crops = []

    for inst_id_str, inst_info in instance_mapping.items():
        if not isinstance(inst_info, dict):
            continue

        ent_id = inst_info.get("entity_id") or inst_id_str
        rgb = inst_info.get("rgb")

        # Find pixel coordinates for this instance
        if inst_npy is not None:
            try:
                int_id = int(inst_id_str)
                mask = (inst_npy == int_id)
            except ValueError:
                mask = np.zeros((inst_h, inst_w), dtype=bool)
        elif inst_rgb is not None and rgb is not None:
            mask = np.all(np.abs(inst_rgb[:, :, :3].astype(int) - np.array(rgb, dtype=int)) <= 3, axis=2)
        else:
            continue

        pixel_count = int(np.sum(mask))
        if pixel_count < min_pixels:
            continue

        # Compute bounding box in instance coordinates
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue

        xmin_inst, xmax_inst = int(np.min(xs)), int(np.max(xs))
        ymin_inst, ymax_inst = int(np.min(ys)), int(np.max(ys))

        # Scale bounding box to render coordinates
        xmin = int(round(xmin_inst * scale_x))
        xmax = int(round(xmax_inst * scale_x))
        ymin = int(round(ymin_inst * scale_y))
        ymax = int(round(ymax_inst * scale_y))

        xmin = max(0, min(img_w - 1, xmin))
        xmax = max(0, min(img_w - 1, xmax))
        ymin = max(0, min(img_h - 1, ymin))
        ymax = max(0, min(img_h - 1, ymax))

        box_w = max(1, xmax - xmin + 1)
        box_h = max(1, ymax - ymin + 1)

        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)

        ent = entities_dict.get(ent_id, {})
        semantic = ent.get("semantic") or inst_info.get("semantic") or "other"
        category = ent.get("category") or inst_info.get("category") or ""
        block_name = ent.get("block_name") or inst_info.get("block_name") or ""
        notes = ent.get("notes") or inst_info.get("notes") or ""
        description = ent.get("description") or inst_info.get("description") or ""

        if semantic in ("wall", "floor"):
            # Surfaces represent materials/finishes, not 3D furniture objects.
            # Extract a centered texture patch (max 256x256) to avoid copying camera perspective.
            is_surface = True
            patch_size = 256
            cy_inst = int(np.median(ys))
            cx_inst = int(np.median(xs))
            cx = int(round(cx_inst * scale_x))
            cy = int(round(cy_inst * scale_y))

            patch_w = min(patch_size, img_w, box_w)
            patch_h = min(patch_size, img_h, box_h)
            half_w = patch_w // 2
            half_h = patch_h // 2

            x0 = max(0, min(img_w - patch_w, cx - half_w))
            y0 = max(0, min(img_h - patch_h, cy - half_h))
            x1 = min(img_w, x0 + patch_w)
            y1 = min(img_h, y0 + patch_h)

            xmin_pad, ymin_pad, xmax_pad, ymax_pad = x0, y0, x1, y1
            crop_img = render_img.crop((xmin_pad, ymin_pad, xmax_pad, ymax_pad))
        else:
            is_surface = False
            xmin_pad = max(0, xmin - pad_x)
            ymin_pad = max(0, ymin - pad_y)
            xmax_pad = min(img_w, xmax + pad_x + 1)
            ymax_pad = min(img_h, ymax + pad_y + 1)
            crop_img = render_img.crop((xmin_pad, ymin_pad, xmax_pad, ymax_pad))

        if ent_id == "auto_ceiling":
            existing_ref = ent.get("reference_image") or (state.get("ceiling", {}).get("reference_image") if state else None)
            label = "Ceiling & Cove Lighting"
            semantic = "ceiling"
        elif semantic == "wall":
            existing_ref = ent.get("reference_image")
            finish = notes or description or "Plaster"
            label = f"Wall Material Swatch ({finish})"
        elif semantic == "floor":
            existing_ref = ent.get("reference_image")
            finish = notes or description or "Flooring"
            label = f"Floor Material Swatch ({finish})"
        else:
            existing_ref = ent.get("reference_image")
            # Friendly display label
            combined = f"{notes} {description} {category} {block_name}".lower()
            if "refrigerator" in combined or "fridge" in combined:
                label = "Refrigerator & Cabinetry"
            elif "dining" in combined or "dinning" in combined or "table" in combined:
                label = "Dining Table & Chairs"
            elif "bed" in combined:
                label = "Bed"
            elif category and category.lower() != "other":
                label = category.replace("_", " ").title()
            elif block_name:
                label = block_name.replace("_", " ").title()
            else:
                label = semantic.title()

        crops.append({
            "instance_id": str(inst_id_str),
            "entity_id": ent_id,
            "rgb": rgb,
            "hex": f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}" if rgb else "#000000",
            "semantic": semantic,
            "is_surface": is_surface,
            "default_role": "Material Reference" if is_surface else "Same Room / New View Reference",
            "category": category,
            "block_name": block_name,
            "display_label": label,
            "notes": notes,
            "description": description,
            "bbox": (xmin, ymin, xmax, ymax),
            "padded_bbox": (xmin_pad, ymin_pad, xmax_pad, ymax_pad),
            "crop_size": (crop_img.width, crop_img.height),
            "pixel_count": pixel_count,
            "crop_image": crop_img,
            "has_existing_reference": bool(existing_ref),
            "existing_reference": existing_ref,
        })

    # Sort crops: furniture first, then by pixel_count descending
    crops.sort(key=lambda c: (0 if c["semantic"] == "furniture" else (1 if c["semantic"] == "ceiling" else 2), -c["pixel_count"]))
    return crops


def save_extracted_reference(
    root: Path | str,
    state: dict[str, Any],
    entity_id: str,
    crop_image: Image.Image,
    camera_id: str = "cam",
    role: Optional[str] = None,
) -> dict[str, Any]:
    """Save an extracted crop image as an authoritative reference in layout.json.

    Args:
        root: Workspace layout directory root.
        state: The current layout_state dictionary.
        entity_id: Entity ID to attach the reference image to.
        crop_image: Cropped PIL Image.
        camera_id: Camera identifier from which the crop was extracted.
        role: Optional reference role (e.g. 'Material Reference' for surfaces).

    Returns:
        Dict with 'relative_path', 'absolute_path', and updated 'state'.
    """
    root_path = Path(root)

    # Encode crop to PNG bytes
    buf = io.BytesIO()
    crop_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ent = entity_id.replace(" ", "_")
    filename = f"crop_{safe_ent}_{camera_id}_{timestamp_str}.png"

    # Determine default role if not provided
    if role is None:
        if entity_id == "auto_ceiling":
            role = "Same Room / New View Reference"
        elif entity_id in state.get("entities", {}):
            sem = state["entities"][entity_id].get("semantic")
            role = "Material Reference" if sem in ("wall", "floor") else "Same Room / New View Reference"
        else:
            role = "Same Room / New View Reference"

    # Use store_reference to save into references/furniture/
    relative_path = store_reference(root_path, filename, png_bytes)
    abs_path = resolve(root_path, relative_path)

    # Update entity or ceiling reference_image in state
    if entity_id == "auto_ceiling":
        if "ceiling" not in state or not isinstance(state["ceiling"], dict):
            state["ceiling"] = {"enabled": True}
        state["ceiling"]["reference_image"] = relative_path
        state["ceiling"]["reference_role"] = role
    elif entity_id in state.get("entities", {}):
        state["entities"][entity_id]["reference_image"] = relative_path
        state["entities"][entity_id]["reference_role"] = role
    else:
        # Fallback for any other entity not in state["entities"]
        if "entities" not in state:
            state["entities"] = {}
        state["entities"][entity_id] = {
            "semantic": "furniture",
            "dxf_handle": f"synthetic_{entity_id}",
            "reference_image": relative_path,
            "reference_role": role,
        }

    # Persist atomically to layout.json
    save(root_path, state)

    return {
        "relative_path": relative_path,
        "absolute_path": str(abs_path),
        "state": state,
    }

