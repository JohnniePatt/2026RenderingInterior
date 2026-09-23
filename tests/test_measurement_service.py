"""Unit and integration tests for measurement_service in Part 7B."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app.geometry.camera import new_camera
from services.validation.measurement_service import (
    generate_debug_overlay,
    get_entity_applicability,
    load_measurements,
    measure_all_entities,
    measure_entity,
    save_entity_overlay,
    save_measurements,
)


@pytest.fixture
def mock_validation_context(tmp_path):
    folder = tmp_path / "generated/camera_001"
    eval_folder = folder / "renders/evaluation/render_test/sam3"
    masks_folder = eval_folder / "masks"
    masks_folder.mkdir(parents=True)

    # 1. Create CAD GT instance map
    ids = np.zeros((100, 100), dtype=np.uint32)
    ids[30:80, 30:70] = 1  # entity_00001 (Bed)
    np.save(folder / "instance.npy", ids)

    # 2. Create generated SAM mask
    gen_mask = np.zeros((100, 100), dtype=np.uint8)
    gen_mask[32:82, 32:72] = 255
    mask_rel = "masks/mask_bed.png"
    Image.fromarray(gen_mask).save(eval_folder / mask_rel)

    # 3. Create correspondence doc
    doc = {
        "correspondences": [
            {
                "entity_id": "entity_00001",
                "instance_id": 1,
                "status": "human_verified",
                "generated_mask": mask_rel,
            }
        ]
    }
    (eval_folder / "correspondence.json").write_text(json.dumps(doc))

    camera = new_camera(5.0, 2.0, "m", heading=180.0, name="Cam1", output={"width": 100, "height": 100})
    render_img = Image.new("RGB", (100, 100), color=(120, 120, 120))

    context = {
        "root": tmp_path,
        "camera_id": "camera_001",
        "render": folder / "renders/render_test.png",
        "image": render_img,
        "evaluation": eval_folder,
        "ids": ids,
        "doc": doc,
        "aligned": True,
        "entities": {
            "entity_00001": {
                "semantic": "furniture",
                "category": "bed",
                "description": "king bed",
                "height": 0.60,
                "insertion_point": [2.5, 2.0, 0.0],
            }
        },
        "visible": [
            {"entity_id": "entity_00001", "instance_id": 1, "name": "king bed"}
        ],
    }

    return context, camera


def test_get_entity_applicability():
    furn_app = get_entity_applicability("furniture")
    assert furn_app["planar_position"]
    assert furn_app["vertical_height"]
    assert not furn_app["opening_metrics"]

    door_app = get_entity_applicability("door")
    assert door_app["opening_metrics"]

    floor_app = get_entity_applicability("floor")
    assert not floor_app["vertical_height"]
    assert not floor_app["planar_position"]


def test_measure_entity_full(mock_validation_context):
    context, camera = mock_validation_context

    result = measure_entity(context, "entity_00001", context["doc"], camera)
    assert result["measurement_status"] == "evaluated"
    assert "reproducibility" in result
    assert result["image_space"]["mask_iou"] > 0.8
    assert result["planar"]["status"] == "evaluated"
    assert "position_error_m" in result["planar"]
    assert result["vertical"]["status"] == "evaluated"
    assert result["vertical"]["cad_height_m"] == 0.60
    assert result["vertical"]["reconstructed_height_m"] > 0
    assert result["openings"] == "not_applicable"


def test_measure_all_and_persistence(mock_validation_context):
    context, camera = mock_validation_context

    summary = measure_all_entities(context, camera)
    assert summary["total_visible_entities"] == 1
    assert summary["evaluated_count"] == 1
    assert summary["cannot_evaluate_count"] == 0

    # Save
    saved_path = save_measurements(context, summary)
    assert saved_path.is_file()

    # Load
    loaded = load_measurements(context)
    assert loaded is not None
    assert loaded["stage"] == summary["stage"]
    assert len(loaded["entities"]) == 1


def test_debug_overlay(mock_validation_context):
    context, _ = mock_validation_context
    gt_mask = context["ids"] == 1
    gen_mask = np.zeros((100, 100), dtype=bool)
    gen_mask[30:80, 30:70] = True

    overlay = generate_debug_overlay(
        context["image"],
        gt_mask,
        gen_mask,
        ground_contact_pixel=(50.0, 80.0),
        top_pixel=(50.0, 30.0),
    )
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (100, 100)

    saved_p = save_entity_overlay(context, "entity_00001", overlay)
    assert saved_p.is_file()
