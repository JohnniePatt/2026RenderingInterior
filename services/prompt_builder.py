from pathlib import Path
from typing import Any
from PIL import Image

DEFAULT_ROLES = {
    "proxy": "Spatial Geometry & Framing Condition",
    "depth": "Depth / Geometric Distance Condition",
    "instance": "Symbolic Instance Segmentation Condition",
    "semantic": "Semantic Category Classification Condition",
    "planar": "CAD Planar Floor Layout Condition",
}

from services.prompt_resolution import (CONDITION_INSTRUCTIONS, compile_prompt, build_prompt, validate_prompt_request)

DEFAULT_ROLE_DESCRIPTIONS = CONDITION_INSTRUCTIONS


def get_available_conditions(root: Path | str, camera_id: str) -> list[dict[str, Any]]:
    """Scan generated/{camera_id}/ for condition image files and return metadata."""
    root_path = Path(root)
    cam_dir = root_path / "generated" / camera_id
    known_types = ["proxy", "depth", "instance", "semantic", "planar"]
    results = []

    for ctype in known_types:
        img_path = cam_dir / f"{ctype}.png"
        exists = img_path.is_file()
        res = None
        if exists:
            try:
                with Image.open(img_path) as img:
                    res = img.size  # (width, height)
            except Exception:
                res = None

        results.append({
            "type": ctype,
            "filename": f"{ctype}.png",
            "path": str(img_path),
            "exists": exists,
            "resolution": res,
            "default_role": DEFAULT_ROLES.get(ctype, f"{ctype.capitalize()} Condition"),
            "role": DEFAULT_ROLES.get(ctype, f"{ctype.capitalize()} Condition"),
            "role_description": DEFAULT_ROLE_DESCRIPTIONS.get(ctype, ""),
            "enabled": exists and (ctype in ["depth", "instance"]),
        })
    return results



