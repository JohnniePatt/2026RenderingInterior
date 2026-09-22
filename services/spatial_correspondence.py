"""Part 7A provenance and human-confirmed masks; deliberately no spatial scores."""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image, ImageDraw
from filelock import FileLock
from services.segmentation.base import validate_prompts

STATUSES = ('unreviewed', 'matched', 'missing', 'occluded')
HALLUCINATION_TYPES = ('door', 'window', 'furniture', 'opening', 'architectural element', 'other')


def digest(data):
    return sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def png_bytes(mask):
    output = BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output, format='PNG')
    return output.getvalue()


def polygon_mask(size, points):
    validate_prompts(size, points)
    if len(set(tuple(p) for p in points)) < 3:
        raise ValueError('Manual Polygon requires at least three distinct vertices.')
    canvas = Image.new('L', size)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in points], fill=255)
    return np.asarray(canvas) > 0


def load_context(root, state, camera_id, render_name, namespace=None):
    root = Path(root).resolve()
    folder = (root / 'generated' / camera_id / 'renders').resolve()
    render = (folder / render_name).resolve()
    if not folder.is_relative_to(root) or render.parent != folder or render.suffix.lower() != '.png':
        raise ValueError('Invalid render path.')
    data = render.read_bytes()
    image = Image.open(BytesIO(data)).convert('RGB')
    evaluation = folder / 'evaluation' / render.stem
    if namespace is not None:
        if namespace != 'sam3': raise ValueError('Unknown evaluation namespace.')
        evaluation = evaluation / namespace
    doc_path = evaluation / 'correspondence.json'
    doc = json.loads(doc_path.read_text()) if doc_path.exists() else None
    metadata = json.loads(render.with_suffix('.json').read_text()) if render.with_suffix('.json').exists() else {}
    if metadata.get('camera_id', camera_id) != camera_id:
        raise ValueError('Render metadata belongs to a different camera.')
    warnings = []
    if doc:
        if doc['render_sha256'] != digest(data) or doc['camera_id'] != camera_id:
            raise ValueError('Render changed since evaluation; existing masks cannot be reused.')
        gt_path = evaluation / 'ground_truth' / 'instance.npy'
        gt_bytes = gt_path.read_bytes()
        if digest(gt_bytes) != doc['ground_truth']['sha256']:
            raise ValueError('Saved ground-truth snapshot has changed.')
        mapping = doc['ground_truth']['instance_mapping']
        entities = doc['ground_truth']['entities']
    else:
        gt_path = folder.parent / 'instance.npy'
        gt_bytes = gt_path.read_bytes()
        mapping = metadata.get('instance_mapping') or state.get('instance_mapping', {}).get(camera_id, {})
        entities = dict(state.get('entities', {}))
        entities['auto_ceiling'] = {'semantic': 'ceiling', **state.get('ceiling', {})}
        # Historic renders must never silently inherit newly exported ground truth.
        order = metadata.get('prompt_builder', {}).get('image_order', [])
        record = next((r for r in order if r.get('kind') == 'condition' and r.get('type') == 'instance'), None)
        if record and record.get('sha256'):
            if digest((folder.parent / 'instance.png').read_bytes()) != record['sha256']:
                raise ValueError('This historical render used a different instance export. Restore its matching ground truth before evaluation; current masks will not be substituted.')
        else:
            warnings.append('Legacy render has no instance hash. Ground-truth provenance is unverified; verify the selected export before confirmation.')
    ids = np.load(BytesIO(gt_bytes), allow_pickle=False)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError('instance.npy must contain nonnegative integer IDs in a 2D array.')
    if doc is None:
        visual = np.asarray(Image.open(folder.parent / 'instance.png').convert('RGB'))
        if visual.shape[:2] != ids.shape:
            raise ValueError('Ground-truth PNG and NPY resolutions differ.')
        for value in np.unique(ids):
            color = [0, 0, 0] if value == 0 else mapping.get(str(int(value)), {}).get('rgb')
            if color is None or not np.all(visual[ids == value] == color):
                raise ValueError('Ground-truth instance.npy does not match the render instance PNG/mapping. Restore matching exports.')
    visible = []
    for instance in np.unique(ids):
        if instance == 0:
            continue
        entry = mapping.get(str(int(instance)))
        if not isinstance(entry, dict) or not entry.get('entity_id'):
            raise ValueError(f'Visible instance {instance} has no entity mapping.')
        entity_id = entry['entity_id']
        entity = entities.get(entity_id)
        if not isinstance(entity, dict):
            raise ValueError(f'Missing entity metadata for {entity_id}.')
        name = entity.get('description') or entity.get('category') or entity.get('semantic') or entity_id
        visible.append({'instance_id': int(instance), 'entity_id': entity_id, 'name': name})
    aligned = image.size == (ids.shape[1], ids.shape[0])
    if not aligned:
        warnings.append(f'Resolution mismatch: render {image.width}×{image.height}; CAD {ids.shape[1]}×{ids.shape[0]}. No masks are resized. Confirmation is blocked until an explicit coordinate transformation is available.')
    return {'root': root, 'camera_id': camera_id, 'render': render, 'image': image, 'image_bytes': data,
            'render_hash': digest(data), 'evaluation': evaluation, 'doc': doc,
            'ids': ids, 'gt_bytes': gt_bytes, 'gt_path': gt_path, 'gt_hash': digest(gt_bytes),
            'mapping': mapping, 'entities': entities, 'visible': visible, 'warnings': warnings,
            'aligned': aligned, 'revision': doc.get('revision', 0) if doc else 0}


def save_review(context, *, entity=None, status='matched', mask=None, prompts=None,
                backend=None, method=None, quality=None, hallucination_type=None,
                hallucination_id=None):
    if not context['aligned']:
        raise ValueError('Resolution mismatch: confirmation is blocked; masks were not resized.')
    if status not in STATUSES:
        raise ValueError('Unknown correspondence status.')
    hallucinated = hallucination_type is not None
    if hallucinated and (hallucination_type not in HALLUCINATION_TYPES or status != 'matched'):
        raise ValueError('Invalid hallucination review.')
    if not hallucinated and entity not in context['visible']:
        raise ValueError('Select a visible CAD entity.')
    prompts = prompts or {'positive_points': [], 'negative_points': [], 'box': None, 'polygon': []}
    validate_prompts(context['image'].size, prompts.get('positive_points', []), prompts.get('negative_points', []), prompts.get('box'))
    if status == 'matched':
        if method not in ('sam', 'manual_polygon') or mask is None:
            raise ValueError('A reviewed SAM or manual polygon mask is required.')
        mask = np.asarray(mask)
        if mask.shape != (context['image'].height, context['image'].width) or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError('Mask must be nonempty, binary and at original image resolution.')
        if method == 'sam' and not backend:
            raise ValueError('SAM backend provenance is required.')
        if method == 'manual_polygon':
            expected = polygon_mask(context['image'].size, prompts.get('polygon', []))
            if not np.array_equal(expected, mask):
                raise ValueError('Polygon and mask differ; review the polygon again.')
    directory = context['evaluation']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'correspondence.json'
    with FileLock(str(directory / '.review.lock')):
        current = json.loads(path.read_text()) if path.exists() else None
        if (current or {}).get('revision', 0) != context['revision']:
            raise ValueError('Evaluation changed in another session. Reload before confirming.')
        if digest(context['render'].read_bytes()) != context['render_hash'] or digest(context['gt_path'].read_bytes()) != context['gt_hash']:
            raise ValueError('Render or ground truth changed during review. Reload before confirming.')
        now = datetime.now(timezone.utc).isoformat()
        if current is None:
            current = {'schema_version': '7A.1', 'render_id': context['render'].stem,
                       'camera_id': context['camera_id'], 'render_sha256': context['render_hash'],
                       'image_size': {'width': context['image'].width, 'height': context['image'].height},
                       'condition_size': {'width': context['ids'].shape[1], 'height': context['ids'].shape[0]},
                       'coordinate_system': 'original_image_pixels_xy', 'transforms': [],
                       'ground_truth': {'source': 'ground_truth/instance.npy', 'sha256': context['gt_hash'],
                                        'instance_mapping': context['mapping'], 'entities': context['entities'],
                                        'warnings': context['warnings']},
                       'correspondences': [{**e, 'status': 'unreviewed', 'human_verified': False}
                                           for e in context['visible']],
                       'hallucinations': [], 'created_at': now}
            atomic_write(directory / 'ground_truth/instance.npy', context['gt_bytes'])
        record = {'entity_id': None if hallucinated else entity['entity_id'],
                  'instance_id': None if hallucinated else entity['instance_id'], 'status': status,
                  'human_verified': status != 'unreviewed', 'updated_at': now,
                  'generated_mask': None, 'gt_mask_source': None if hallucinated else 'ground_truth/instance.npy',
                  'segmentation_method': method if status == 'matched' else None,
                  'sam_prompt': prompts if status == 'matched' else None,
                  'segmentation_backend': backend if method == 'sam' and status == 'matched' else None,
                  'predicted_quality': quality if method == 'sam' and status == 'matched' else None}
        if status == 'matched':
            relative = ('hallucinations/' if hallucinated else 'masks/') + uuid4().hex + '.png'
            atomic_write(directory / relative, png_bytes(mask))
            record['generated_mask'] = relative
        if hallucinated:
            record.update(hallucination_id=hallucination_id or uuid4().hex, hallucination_type=hallucination_type)
            current['hallucinations'] = [r for r in current['hallucinations'] if r['hallucination_id'] != record['hallucination_id']] + [record]
        else:
            current['correspondences'] = [record if r['entity_id'] == entity['entity_id'] else r for r in current['correspondences']]
        current.update(updated_at=now, revision=context['revision']+1)
        atomic_write(path, json.dumps(current, indent=2, ensure_ascii=False).encode())
    return current
