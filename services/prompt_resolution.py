"""Compile selected multimodal inputs into spatial and appearance instructions.

This module reads existing exports; it never regenerates maps or assigns RGBs.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import io
import math
import re
import numpy as np
from PIL import Image
from services.conditioning_preflight import check_camera_export

VERSION = "2.2"
SAME_ROOM_ROLE = "Same Room / New View Reference"
CONDITION_INSTRUCTIONS = {
    "instance": "a CAD-derived instance segmentation map. RGB values are symbolic instance identifiers only, not desired colors, materials or appearance. Use the regions for projected position, silhouette, scale and spatial relationships. Replace them with realistic elements using the mapping below. Discard mask colors everywhere, including region boundaries, thin outlines, window frames, trim, reflections and color spill. Region edges are object boundaries, not colored physical borders. Choose surface and frame finishes from appearance references or explicit material instructions, never from mask RGB values. CRITICAL: Under no circumstances should any flat, solid segmentation mask colors (such as solid blue, cyan, magenta, green, or purple) appear in the generated image; every pixel inside every instance region must be entirely rendered with photorealistic materials, textures, shading, and complete furniture components (e.g. realistic dining chairs, table surfaces, and upholstery).",
    "depth": "a CAD-derived depth representation of the same camera view. Use it for relative surface distance, foreground/background relationships, room depth and object placement. Grayscale values represent geometry only; do not reproduce their grayscale appearance.",
    "semantic": "a CAD-derived semantic segmentation map. Its symbolic labels distinguish architectural surface categories, not appearance or material colors. Use its regions for spatial organization; do not reproduce its palette.",
    "proxy": "a CAD-derived metric proxy of the same camera view. Use it for room volume, framing, object proportions and occlusion. Simplified shapes and preview shading are geometry guides, not finished furniture or materials.",
    "planar": "a CAD-derived planar floor projection. Use it for floor-plane alignment and the projected footprint, not surface appearance.",
}
REFERENCE_ROLES = {
    "Appearance / Style Reference": "materials, lighting, atmosphere, furniture language and visual style",
    "Appearance Reference": "materials, lighting, atmosphere, furniture language and visual style",
    "Style Reference": "visual style and atmosphere",
    "Material Reference": "materials, finishes and surface textures",
    "Lighting Reference": "lighting, shadows and atmosphere",
    "Furniture Reference": "furniture design and detailing",
    "Other": "the appearance details specified by the user",
}
METADATA_FIELDS = {"semantic", "category", "description", "height", "opening_height", "sill_height",
                   "base_elevation", "elevation", "thickness", "material", "notes"}
METERS = {"mm": .001, "cm": .01, "m": 1, "in": .0254, "ft": .3048}
OBJECT_WORDS = {
    "sofa": r"\b(?:sofa|couch|settee)\b", "bed": r"\bbed\b",
    "table": r"\b(?:table|desk)\b", "chair": r"\b(?:chair|stool)\b",
    "refrigerator": r"\b(?:refrigerator|fridge)\b", "cabinet": r"\b(?:cabinet|wardrobe)\b",
    "window": r"\bwindow\b", "door": r"\bdoor\b",
}


@dataclass
class PromptBuild:
    spatial_prompt: str
    appearance_prompt: str
    final_prompt: str
    debug: dict
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def clean(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def conflict(metadata):
    category = clean(metadata.get("category")).lower().replace("_", " ")
    description = clean(metadata.get("description")).lower()
    expected = {kind for kind, pattern in OBJECT_WORDS.items() if re.search(pattern, category)}
    found = {kind for kind, pattern in OBJECT_WORDS.items() if re.search(pattern, description)}
    # Only explicit, disjoint object types are flagged. Unknown terms remain unknown.
    return bool(expected and found and expected.isdisjoint(found))


def describe(metadata, units, conflicting=False):
    sem = clean(metadata.get("semantic")) or "architectural element"
    category = clean(metadata.get("category")).replace("_", " ")
    description = clean(metadata.get("description"))
    if conflicting:
        label = "furniture element" if sem == "furniture" else "architectural element"
    elif sem == "void":
        label = f"{category} opening" if category in {"window", "door"} else "architectural opening"
        if description:
            label += f" ({description})"
    else:
        label = description or (category if category and category != "other" else sem)
    parts = [label]
    def dimension(value):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return None
        return f"{value * METERS[units]:g} m" if units in METERS else f"{value:g} {units}"
    if sem == "ceiling":
        if (dim := dimension(metadata.get("elevation"))) is not None:
            parts.append(f"at {dim} elevation")
    else:
        key = "opening_height" if sem == "void" and metadata.get("opening_height") is not None else "height"
        if (dim := dimension(metadata.get(key))) is not None:
            parts.append(f"approximately {dim} high")
        if sem == "void":
            bottom = metadata.get("sill_height", metadata.get("base_elevation"))
            if (dim := dimension(bottom)) is not None:
                parts.append(f"with a {dim} sill" if category == "window" else f"bottom elevation {dim}")
        elif (dim := dimension(metadata.get("elevation", metadata.get("base_elevation")))) is not None:
            parts.append(f"at {dim} elevation")
        if sem == "floor" and (dim := dimension(metadata.get("thickness"))) is not None:
            parts.append(f"{dim} thick")
    if material := clean(metadata.get("material")):
        parts.append(f"material: {material}")
    if not conflicting and (notes := clean(metadata.get("notes"))) and notes != description:
        parts.append(f"notes: {notes}")
    return ", ".join(parts)


def credential_errors(text, secrets=()):
    # Errors never echo matching content or credential values.
    patterns = (r"AIza[0-9A-Za-z_-]{20,}", r"\bsk-[0-9A-Za-z_-]{20,}",
                r"(?i)\b(?:api[_ -]?key|authorization)\s*[:=]\s*\S+")
    if any(secret and secret in text for secret in secrets) or any(re.search(p, text) for p in patterns):
        return ["Possible API credential in prompt content. Remove it before generation."]
    return []


def validate_prompt_request(build, secrets=()):
    errors = [*build.errors, *credential_errors(build.final_prompt, secrets)]
    if re.search(r"#[0-9a-fA-F]{6}\b", build.spatial_prompt):
        errors.append("Automatic spatial metadata contains a HEX value; remove it from the annotation.")
    if errors:
        raise ValueError("\n".join(dict.fromkeys(errors)))


def inspect_image(source):
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        size = list(image.size)
    return size, hashlib.sha256(data).hexdigest()


def compile_prompt(layout_state, camera_id, condition_images=None, reference_images=None,
                   user_design_prompt="", root_path=None, strictness="Very Strict (nanobanana)"):
    conditions = [c for c in (condition_images or []) if c.get("enabled", True)]
    references = list(reference_images or [])
    errors, warnings, rows, conflicts, ordering = [], [], [], [], []
    units = layout_state.get("source", {}).get("units", "unknown")
    if camera_id not in layout_state.get("cameras", {}):
        errors.append("Selected camera does not exist.")
    if not conditions:
        errors.append("Select at least one CAD-derived condition image.")
    camera_folder = None
    if root_path is not None:
        generated = (Path(root_path) / "generated").resolve()
        camera_folder = (generated / camera_id).resolve()
        if not camera_folder.is_relative_to(generated) or camera_folder == generated:
            errors.append("Invalid camera export directory.")
            camera_folder = None
    types = [c.get("type") for c in conditions]
    camera_check = {"status": "unverified"}
    if camera_folder is not None and camera_id in layout_state.get("cameras", {}):
        camera_check, camera_errors, camera_warnings = check_camera_export(
            camera_folder, camera_id, layout_state['cameras'][camera_id])
        errors.extend(camera_errors)
        warnings.extend(camera_warnings)
    if any(r.get("role") == SAME_ROOM_ROLE for r in references) and "instance" not in types:
        errors.append("Same Room / New View Reference requires the target camera's instance condition. Enable Instance in Spatial Conditioning Inputs.")
    if len(set(types)) != len(types):
        errors.append("Duplicate condition types are not supported.")
    for index, item in enumerate([*conditions, *references], 1):
        is_condition = index <= len(conditions)
        record = {"image": index, "kind": "condition" if is_condition else "reference",
                  "type": item.get("type") if is_condition else item.get("role", "Appearance Reference"),
                  "path": item.get("path"), "name": item.get("name", item.get("filename"))}
        if is_condition and item.get("type") not in CONDITION_INSTRUCTIONS:
            errors.append(f"IMAGE {index}: unsupported condition type.")
        try:
            source = item.get("path") if is_condition else item.get("bytes") or item.get("path")
            record["resolution"], record["sha256"] = inspect_image(source)
            if is_condition and camera_folder and Path(item["path"]).resolve() != camera_folder / f"{item['type']}.png":
                errors.append(f"IMAGE {index}: file does not belong to the selected camera export.")
        except (OSError, ValueError, TypeError, AttributeError):
            errors.append(f"IMAGE {index}: selected image is missing or unreadable.")
        ordering.append(record)
    sizes = {tuple(r["resolution"]) for r in ordering if r["kind"] == "condition" and "resolution" in r}
    if len(sizes) > 1:
        errors.append("Selected condition images have inconsistent resolutions. Re-export the same camera view.")

    visibility_source, visible_ids = "not_requested", []
    if "instance" in types:
        mapping = layout_state.get("instance_mapping", {}).get(camera_id)
        if not isinstance(mapping, dict) or not mapping:
            errors.append("Instance mapping is missing for the selected camera. Re-export its conditioning maps.")
            mapping = {}
        npy = camera_folder / "instance.npy" if camera_folder else None
        if npy is not None and npy.is_file():
            visibility_source = "instance.npy"
            try:
                array = np.load(npy, allow_pickle=False)
                if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
                    raise ValueError("Expected a nonnegative 2D integer instance array")
                visible_ids = [int(i) for i in np.unique(array) if i != 0]
                instance_record = next(r for r in ordering if r.get("type") == "instance")
                if instance_record.get("resolution") != [array.shape[1], array.shape[0]]:
                    errors.append("instance.npy and instance.png resolutions differ. Re-export conditioning maps.")
            except (OSError, ValueError, TypeError):
                errors.append("instance.npy is corrupt or not a 2D integer array. Re-export; visibility was not guessed.")
        else:
            visibility_source = "camera_mapping_fallback"
            warnings.append("instance.npy is unavailable; using the selected camera mapping as a visibility fallback (visibility unverified).")
            try:
                visible_ids = sorted(int(k) for k in mapping if int(k) > 0)
            except (ValueError, TypeError):
                errors.append("Malformed instance ID in camera mapping.")
        rgb_seen = {}
        for key, entry in mapping.items():
            try:
                instance_id = int(key)
            except (ValueError, TypeError):
                errors.append("Malformed instance ID in camera mapping.")
                continue
            if instance_id == 0:
                continue
            rows.append({"instance_id": instance_id, "rgb": entry.get("rgb") if isinstance(entry, dict) else None,
                         "entity_id": entry.get("entity_id") if isinstance(entry, dict) else None,
                         "visible": (instance_id in visible_ids) if visibility_source == "instance.npy" else None,
                         "included": instance_id in visible_ids})
        for instance_id in visible_ids:
            entry = mapping.get(str(instance_id))
            if not isinstance(entry, dict):
                errors.append(f"Visible instance {instance_id} does not resolve in the camera mapping.")
                continue
            rgb, entity_id = entry.get("rgb"), entry.get("entity_id")
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in rgb):
                errors.append(f"Instance {instance_id}: malformed RGB identifier.")
                continue
            if tuple(rgb) in rgb_seen or tuple(rgb) == (0, 0, 0):
                errors.append(f"Instance {instance_id}: RGB identifier is ambiguous or reserved for background.")
            rgb_seen[tuple(rgb)] = instance_id
            if not isinstance(entity_id, str) or not entity_id:
                errors.append(f"Instance {instance_id}: invalid entity identifier.")
                continue
            if entity_id == "auto_ceiling":
                entity = layout_state.get("ceiling")
                if not isinstance(entity, dict) or entity.get("elevation") is None:
                    errors.append("auto_ceiling does not resolve to top-level ceiling metadata with elevation.")
                    continue
                entity = {**entity, "semantic": "ceiling"}
            else:
                entity = layout_state.get("entities", {}).get(entity_id)
            if not isinstance(entity, dict) or entity.get("orphaned") or not entity.get("semantic"):
                errors.append(f"Instance {instance_id}: missing, orphaned or unannotated entity.")
                continue
            metadata = {k: entity[k] for k in METADATA_FIELDS if k in entity}
            for key in ('height','opening_height','sill_height','base_elevation','elevation','thickness'):
                value = metadata.get(key)
                if value is not None and (not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value)):
                    errors.append(f"Instance {instance_id}: invalid {key} in annotation.")
            has_conflict = conflict(metadata)
            if has_conflict:
                issue = {"entity_id": entity_id, "category": metadata.get("category"), "description": metadata.get("description")}
                conflicts.append(issue)
                warnings.append(f"Semantic conflict: {entity_id} — category: {issue['category']}; description: {issue['description']}. Correct the annotation; a neutral description is used meanwhile.")
            row = next(r for r in rows if r["instance_id"] == instance_id)
            row.update(resolved=describe(metadata, units, has_conflict), conflict=has_conflict)
        rows.sort(key=lambda r:r["instance_id"])

    if strictness == "Very Strict (nanobanana)":
        cond_names = ", ".join(f"IMAGE {idx+1} ({c.get('type')}.png)" for idx, c in enumerate(conditions))
        ref_names = ", ".join(f"IMAGE {len(conditions)+idx+1} ({r.get('name') or 'ref'})" for idx, r in enumerate(references)) or "references"
        directive_header = (
            "CRITICAL CAMERA & PERSPECTIVE DIRECTIVE (NANOBANANA STRICT EFFORT):\n"
            f"I want the interior room in a realistic image generated STRICTLY from the camera viewpoint, perspective, and boundaries of {cond_names} ONLY!\n"
            f"Please follow conditions and be VERY STRICT in {cond_names}! Under no circumstances should you copy the camera viewpoint or perspective of any reference images.\n"
            f"Decorate: you can reference {ref_names} in attachment for styling, materials, and lighting only."
        )
    elif strictness == "Strict":
        directive_header = (
            "CRITICAL CAMERA & PERSPECTIVE DIRECTIVE (STRICT):\n"
            "1. The camera position, viewing angle, field of view, and 3D spatial layout are strictly and fully dictated by the CAD conditioning maps (IMAGE 1, IMAGE 2, etc.).\n"
            "2. DO NOT adopt, copy, or blend the camera angle, perspective, or viewpoint from ANY reference images. All reference images are from a completely DIFFERENT viewpoint and are provided EXCLUSIVELY for decoration, furniture styling, materials, and textures.\n"
            "3. Every object from the reference images MUST be re-projected and rendered strictly within the 3D perspective and boundaries established by the CAD conditioning maps."
        )
    else:
        directive_header = (
            "CRITICAL CAMERA & PERSPECTIVE DIRECTIVE (BALANCED):\n"
            "The camera viewpoint and room layout are defined by the CAD conditioning maps (IMAGE 1, IMAGE 2, etc.). "
            "Use reference images for decoration, finishes, and furniture styling within the target perspective."
        )

    lines = [
        directive_header,
        "Generate a photorealistic architectural interior from the supplied CAD-derived condition images.",
        "All CAD-derived condition images represent the same room and calibrated camera view. Use them as the primary spatial reference."
    ]
    for index, condition in enumerate(conditions, 1):
        ctype = condition.get("type")
        if ctype in CONDITION_INSTRUCTIONS:
            lines.append(f"IMAGE {index} is {CONDITION_INSTRUCTIONS[ctype]}")
        if ctype == "instance":
            entries = [f"- RGB({','.join(str(v) for v in row['rgb'])}) identifies {row['resolved']}."
                       for row in rows if row.get("included") and row.get("resolved")]
            lines.append("INSTANCE MAPPING\n" + ("\n".join(entries) if entries else "No foreground instances are visible in the exported instance map."))
    same_room_preamble_added = False
    for index, reference in enumerate(references, len(conditions)+1):
        role = reference.get("role", "Appearance Reference")
        use = REFERENCE_ROLES.get(role, REFERENCE_ROLES["Other"])
        ref_name = reference.get("name") or reference.get("filename") or f"ref_{index}"
        if role == SAME_ROOM_ROLE:
            if not same_room_preamble_added:
                instruction = (
                    f"IMAGE {index} is a reference photograph of the SAME ROOM FROM ANOTHER VIEWPOINT ({ref_name}). "
                    "Decorate the interior based on this reference: match the furniture design, materials, finishes, upholstery, "
                    "architectural detailing and lighting character for corresponding elements. "
                    "The target CAD conditions take priority for camera viewpoint, 3D perspective, and framing: "
                    "NEVER copy the reference camera angle, crop, or composition. "
                    "Preserve recognizable furniture identity for corresponding elements. "
                    "Keep objects outside the target view out of frame and respect partial occlusion. "
                    "CAD proxies are incomplete representations of finished interiors: a missing separate mask for a detail does not mean that detail must be deleted. "
                    "Do not relocate explicitly out-of-view modeled objects into the frame. "
                    "Preserve the reference window-frame finish on the full frame and its edges; the window mask describes the opening extent only and supplies no frame color. "
                    "Discard mask colors everywhere, including thin outlines, window frames, and reflections."
                )
                same_room_preamble_added = True
            else:
                instruction = (
                    f"IMAGE {index} is an additional reference photograph of the SAME ROOM FROM ANOTHER VIEWPOINT ({ref_name}) "
                    "(follow the same decoration and material consistency rules above)."
                )
        else:
            instruction = f"IMAGE {index} is an appearance reference ({role}). Reference name: {ref_name}. Use it only for {use}. Do not copy its room geometry, wall positions, camera viewpoint, openings or furniture locations."
        if description := clean(reference.get("description")):
            instruction += f" User reference direction: {description}"
        lines.append(instruction)
    lines += ["ARCHITECTURAL RULE: Do not invent additional architectural openings. Doors and windows should occur only where represented by the supplied CAD-derived conditions.",
              "Conditions define spatial organization. Semantic metadata defines what the elements are. Appearance instructions and references define how the interior should look.",
              "Generate natural realistic materials, lighting, shadows, reflections and furniture details."]
    spatial = "\n\n".join(lines)
    appearance = user_design_prompt.strip()
    final = spatial + ("\n\nAPPEARANCE PROMPT\n" + appearance if appearance else "")
    debug = {"builder_version": VERSION, "camera_id": camera_id, "visibility_source": visibility_source,
             "visible_instance_ids": visible_ids if visibility_source == "instance.npy" else None,
             "included_instance_ids": visible_ids, "instances": rows, "semantic_conflicts": conflicts,
             "selected_conditions": types, "image_order": ordering, "camera_export_check": camera_check}
    result = PromptBuild(spatial, appearance, final, debug, list(dict.fromkeys(warnings)), list(dict.fromkeys(errors)))
    result.errors.extend(credential_errors(final))
    if re.search(r"#[0-9a-fA-F]{6}\b", spatial):
        result.errors.append("Automatic spatial metadata contains a HEX value; remove it from the annotation.")
    return result


def build_prompt(*args, **kwargs):
    """String interface retained for existing integrations; invalid inputs fail closed."""
    result = compile_prompt(*args, **kwargs)
    validate_prompt_request(result)
    return result.final_prompt
