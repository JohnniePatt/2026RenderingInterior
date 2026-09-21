import json
from hashlib import sha256
import numpy as np
import pytest
from PIL import Image
from services.spatial_correspondence import load_context, save_review, polygon_mask
from services.segmentation.base import validate_prompts


@pytest.fixture
def evaluation(tmp_path):
    folder = tmp_path / 'generated/camera_001'
    (folder / 'renders').mkdir(parents=True)
    ids = np.zeros((40, 80), dtype=np.uint32)
    ids[10:30, 20:60] = 1
    np.save(folder / 'instance.npy', ids)
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[ids == 1] = [17, 81, 143]
    Image.fromarray(rgb).save(folder / 'instance.png')
    mapping = {'1': {'entity_id': 'bed', 'rgb': [17, 81, 143]}, '2': {'entity_id': 'hidden', 'rgb': [90, 91, 92]}}
    state = {'entities': {'bed': {'semantic': 'furniture', 'description': 'Bed'}, 'hidden': {'semantic': 'furniture'}},
             'instance_mapping': {'camera_001': mapping}}
    for name in ('render_a', 'render_b'):
        Image.new('RGB', (80, 40)).save(folder / 'renders' / f'{name}.png')
        (folder / 'renders' / f'{name}.json').write_text(json.dumps({'camera_id': 'camera_001', 'instance_mapping': mapping,
            'prompt_builder': {'image_order': [{'kind': 'condition', 'type': 'instance', 'sha256': sha256((folder / 'instance.png').read_bytes()).hexdigest()}]}}))
    return tmp_path, state


def ctx(evaluation, name='render_a.png'):
    return load_context(*evaluation, 'camera_001', name)


def accept(context):
    prompts = {'positive_points': [], 'negative_points': [], 'box': None, 'polygon': [[10, 10], [50, 10], [50, 30]]}
    mask = polygon_mask(context['image'].size, prompts['polygon'])
    return save_review(context, entity=context['visible'][0], mask=mask, prompts=prompts, method='manual_polygon')


def test_visible_gt_and_per_render_save_restore(evaluation):
    context = ctx(evaluation)
    assert [e['entity_id'] for e in context['visible']] == ['bed']
    assert not context['evaluation'].exists()  # opening/previewing creates no accepted data
    document = accept(context)
    restored = ctx(evaluation)
    record = restored['doc']['correspondences'][0]
    assert record['human_verified'] and record['segmentation_method'] == 'manual_polygon'
    mask = np.asarray(Image.open(restored['evaluation'] / record['generated_mask']))
    assert mask.shape == (40, 80) and set(np.unique(mask)) == {0, 255}
    assert record['sam_prompt']['polygon'][0] == [10, 10]
    assert ctx(evaluation, 'render_b.png')['doc'] is None
    assert not (evaluation[0] / 'layout.json').exists()
    with pytest.raises(ValueError, match='another session'): accept(context)


@pytest.mark.parametrize('status', ['missing', 'occluded', 'unreviewed'])
def test_non_mask_statuses(evaluation, status):
    context = ctx(evaluation)
    document = save_review(context, entity=context['visible'][0], status=status)
    assert document['correspondences'][0]['status'] == status
    assert document['correspondences'][0]['generated_mask'] is None


def test_hallucination_and_sam_provenance(evaluation):
    context = ctx(evaluation)
    mask = np.zeros((40, 80), bool); mask[2:6, 3:12] = True
    prompts = {'positive_points': [[5, 4]], 'negative_points': [[20, 30]], 'box': [2, 1, 15, 10]}
    document = save_review(context, mask=mask, prompts=prompts, method='sam',
        backend={'provider': 'sam', 'model': 'vit_b', 'checkpoint_sha256': 'test'}, quality=.9, hallucination_type='door')
    entry = document['hallucinations'][0]
    assert entry['entity_id'] is None and entry['human_verified']
    assert entry['sam_prompt'] == prompts and entry['segmentation_backend']['model'] == 'vit_b'
    assert document['correspondences'][0]['status'] == 'unreviewed'


def test_mismatch_never_resizes(evaluation):
    Image.new('RGB', (40, 20)).save(evaluation[0] / 'generated/camera_001/renders/render_a.png')
    context = ctx(evaluation)
    assert not context['aligned'] and context['ids'].shape == (40, 80)
    with pytest.raises(ValueError, match='Resolution mismatch'):
        save_review(context, entity=context['visible'][0], status='missing')


def test_historical_gt_mismatch_blocks_and_saved_snapshot_survives_reexport(evaluation):
    context = ctx(evaluation)
    accept(context)
    Image.new('RGB', (80, 40), 'white').save(evaluation[0] / 'generated/camera_001/instance.png')
    np.save(evaluation[0] / 'generated/camera_001/instance.npy', np.zeros((40, 80), dtype=np.uint32))
    assert ctx(evaluation)['visible'][0]['entity_id'] == 'bed'
    with pytest.raises(ValueError, match='historical render'): ctx(evaluation, 'render_b.png')


def test_changed_inputs_and_invalid_prompts(evaluation):
    context = ctx(evaluation)
    Image.new('RGB', (80, 40), 'white').save(context['render'])
    with pytest.raises(ValueError, match='changed during review'): accept(context)
    with pytest.raises(ValueError): validate_prompts((80, 40), [[80, 20]])
    with pytest.raises(ValueError): polygon_mask((80, 40), [[1, 2], [3, 4]])


def test_swapped_gt_ids_are_rejected_even_if_png_is_unchanged(evaluation):
    np.save(evaluation[0] / 'generated/camera_001/instance.npy', np.zeros((40, 80), dtype=np.uint32))
    with pytest.raises(ValueError, match='does not match'): ctx(evaluation)
