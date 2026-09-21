import numpy as np
from PIL import Image
import pytest
from test_dxf_parser import sample
from app.layout.manager import create
from app.layout.persistence import read
from services.segment_extractor import extract_instance_crops, save_extracted_reference


def test_extract_instance_crops_rgb():
    # 100x100 render image
    render_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    # Draw a colored patch in render
    r_arr = np.array(render_img)
    r_arr[20:60, 30:70] = [100, 150, 200]
    render_img = Image.fromarray(r_arr)

    # 100x100 instance image with matching RGB
    inst_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    inst_arr[20:60, 30:70] = [67, 99, 216]
    inst_img = Image.fromarray(inst_arr)

    mapping = {
        "1": {
            "entity_id": "ent_table",
            "rgb": [67, 99, 216],
            "semantic": "furniture",
            "category": "dining_table",
        }
    }
    entities = {
        "ent_table": {
            "semantic": "furniture",
            "category": "dining_table",
            "notes": "round dining table",
        }
    }

    crops = extract_instance_crops(
        render_image=render_img,
        instance_image=inst_img,
        instance_mapping=mapping,
        entities=entities,
        padding_ratio=0.1,
    )

    assert len(crops) == 1
    c = crops[0]
    assert c["entity_id"] == "ent_table"
    assert c["bbox"] == (30, 20, 69, 59)
    assert c["pixel_count"] == 40 * 40
    assert c["display_label"] == "Dining Table & Chairs"
    assert isinstance(c["crop_image"], Image.Image)
    # Check padded dimensions
    pad_x = int(40 * 0.1)
    pad_y = int(40 * 0.1)
    expected_w = 40 + 2 * pad_x
    expected_h = 40 + 2 * pad_y
    assert c["crop_size"] == (expected_w, expected_h)


def test_extract_instance_crops_npy_and_edge_padding():
    render_img = Image.new("RGB", (100, 100), color=(200, 200, 200))

    # Instance at the top-left edge (0, 0)
    inst_npy = np.zeros((100, 100), dtype=np.int32)
    inst_npy[0:30, 0:40] = 2

    mapping = {
        "2": {
            "entity_id": "ent_bed",
            "rgb": [230, 25, 75],
            "semantic": "furniture",
        }
    }
    entities = {
        "ent_bed": {
            "semantic": "furniture",
            "category": "bed",
            "notes": "king bed",
        }
    }

    crops = extract_instance_crops(
        render_image=render_img,
        instance_image=inst_npy,
        instance_mapping=mapping,
        entities=entities,
        padding_ratio=0.1,
    )

    assert len(crops) == 1
    c = crops[0]
    assert c["bbox"] == (0, 0, 39, 29)
    # xmin and ymin should clamp to 0 (no negative coordinates)
    assert c["padded_bbox"][0] == 0
    assert c["padded_bbox"][1] == 0
    assert c["crop_image"].width > 0
    assert c["crop_image"].height > 0


def test_save_extracted_reference(tmp_path):
    root, _ = create(tmp_path, "TestLayout", "test.dxf", sample())
    state = read(root)

    # Initialize a test entity
    ent_id = "entity_test_furniture"
    state["entities"][ent_id] = {
        "dxf_handle": "HAND_01",
        "semantic": "furniture",
        "category": "bed",
        "height": 500,
        "material": "fabric",
    }
    from app.layout.persistence import save as persist_save
    persist_save(root, state)

    crop_img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    res = save_extracted_reference(
        root=root,
        state=state,
        entity_id=ent_id,
        crop_image=crop_img,
        camera_id="cam_test",
    )

    assert "relative_path" in res
    assert res["relative_path"].startswith("references/furniture/")
    assert (root / res["relative_path"]).is_file()

    # Verify updated layout.json
    saved_state = read(root)
    assert saved_state["entities"][ent_id]["reference_image"] == res["relative_path"]


def test_save_extracted_reference_ceiling(tmp_path):
    root, _ = create(tmp_path, "TestLayout", "test.dxf", sample())
    state = read(root)

    crop_img = Image.new("RGB", (64, 32), color=(240, 240, 240))
    res = save_extracted_reference(
        root=root,
        state=state,
        entity_id="auto_ceiling",
        crop_image=crop_img,
        camera_id="cam_001",
    )

    assert "relative_path" in res
    assert res["relative_path"].startswith("references/furniture/")
    assert (root / res["relative_path"]).is_file()

    # Verify updated layout.json ceiling reference
    saved_state = read(root)
    assert saved_state["ceiling"]["reference_image"] == res["relative_path"]


def test_extract_instance_crops_surface_swatch():
    # 500x500 render image
    render_img = Image.new("RGB", (500, 500), color=(220, 220, 220))
    # Wall covering entire width and height
    inst_arr = np.zeros((500, 500, 3), dtype=np.uint8)
    inst_arr[50:450, 50:450] = [145, 30, 180]
    inst_img = Image.fromarray(inst_arr)

    mapping = {
        "1": {
            "entity_id": "ent_wall",
            "rgb": [145, 30, 180],
            "semantic": "wall",
            "notes": "painted plaster",
        }
    }
    entities = {
        "ent_wall": {
            "semantic": "wall",
            "notes": "painted plaster",
        }
    }

    crops = extract_instance_crops(
        render_image=render_img,
        instance_image=inst_img,
        instance_mapping=mapping,
        entities=entities,
    )

    assert len(crops) == 1
    c = crops[0]
    assert c["entity_id"] == "ent_wall"
    assert c["is_surface"] is True
    assert c["default_role"] == "Material Reference"
    assert "Wall Material Swatch" in c["display_label"]
    # Patch size should be capped at 256x256, not 400x400
    assert c["crop_size"] == (256, 256)


def test_save_extracted_reference_with_role(tmp_path):
    root, _ = create(tmp_path, "TestLayout", "test.dxf", sample())
    state = read(root)

    ent_id = "entity_wall_test"
    state["entities"][ent_id] = {
        "dxf_handle": "HAND_WALL",
        "semantic": "wall",
        "notes": "concrete finish",
    }
    from app.layout.persistence import save as persist_save
    persist_save(root, state)

    crop_img = Image.new("RGB", (128, 128), color=(200, 200, 200))
    res = save_extracted_reference(
        root=root,
        state=state,
        entity_id=ent_id,
        crop_image=crop_img,
        camera_id="cam_001",
        role="Material Reference",
    )

    saved_state = read(root)
    assert saved_state["entities"][ent_id]["reference_image"] == res["relative_path"]
    assert saved_state["entities"][ent_id]["reference_role"] == "Material Reference"


