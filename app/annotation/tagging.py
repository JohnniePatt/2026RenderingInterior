from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from app.layout.manager import safe_filename
from app.layout.persistence import resolve, save
from app.layout.schema import validate
from services.validation_concepts import resolve_concept

CATEGORIES = ["sofa", "chair", "dining_table", "coffee_table", "bed", "wardrobe", "cabinet", "desk", "refrigerator", "other"]
FIELDS = {
    "wall": {"height", "base_elevation", "material", "description", "notes"},
    "floor": {"elevation", "thickness", "material", "description", "notes"},
    "furniture": {"category", "height", "material", "description", "reference_image", "notes"},
    "void": {"category", "sill_height", "base_elevation", "opening_height", "material", "description", "notes"},
}


def apply(root, state, ids, semantic, properties):
    if semantic not in FIELDS:
        raise ValueError("Select Wall, Floor, Furniture or Void")
    if not ids:
        raise ValueError("Select at least one CAD entity")
    if state["source"]["units"].startswith("unknown"):
        raise ValueError("Confirm source units before annotating")
    candidate = deepcopy(state)
    for entity_id in ids:
        record = candidate["entities"].get(entity_id)
        if record is None or record.get("orphaned"):
            raise ValueError("Cannot annotate missing entities")
        for key in set().union(*FIELDS.values()):
            record.pop(key, None)
        record["semantic"] = semantic
        record.update({k: v for k, v in properties.items() if k in FIELDS[semantic]})
        if 'validation_concept' in properties:
            if not isinstance(properties['validation_concept'], str):
                raise ValueError('Validation Concept must be text.')
            record['validation_concept'] = properties['validation_concept'].strip()
        elif not record.get('validation_concept'):
            inferred = resolve_concept(record)
            if not inferred['needs_review']:
                record['validation_concept'] = inferred['concept']
        if semantic == "wall":
            if "height" not in record:
                raise ValueError("Missing height")
        elif semantic == "floor":
            if "thickness" not in record:
                raise ValueError("Missing thickness")
        elif semantic == "furniture":
            for required in ("height", "category"):
                if required not in record:
                    raise ValueError(f"Missing {required}")
        elif semantic == "void":
            if "category" not in record:
                raise ValueError("Missing category")
            cat = record["category"]
            if cat == "window":
                if "opening_height" not in record and "height" in properties:
                    record["opening_height"] = properties["height"]
                if "sill_height" not in record and "base_elevation" in properties:
                    record["sill_height"] = properties["base_elevation"]
                record.pop("height", None)
                record.pop("base_elevation", None)
                if "sill_height" not in record or "opening_height" not in record:
                    raise ValueError("Window requires sill_height and opening_height")
            else:
                if "opening_height" not in record and "height" in properties:
                    record["opening_height"] = properties["height"]
                if "base_elevation" not in record and "sill_height" in properties:
                    record["base_elevation"] = properties["sill_height"]
                record.pop("height", None)
                record.pop("sill_height", None)
                if "base_elevation" not in record or "opening_height" not in record:
                    raise ValueError("Door/opening requires base_elevation and opening_height")
    validate(candidate)
    save(root, candidate)
    return candidate


def clear(root, state, ids):
    candidate = deepcopy(state)
    for entity_id in ids:
        record = candidate["entities"][entity_id]
        for key in set().union(*FIELDS.values()):
            record.pop(key, None)
        record["semantic"] = None
    save(root, candidate)
    return candidate


def store_reference(root, filename, data):
    extension = Path(filename).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("Reference must be PNG, JPEG or WebP")
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("Reference image exceeds 20 MB")
    if cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
        raise ValueError("Invalid reference image")
    relative = f"references/furniture/{uuid4().hex[:12]}_{safe_filename(filename)}"
    resolve(root, relative).write_bytes(data)
    return relative
