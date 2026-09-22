from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from uuid import uuid4
import numpy as np
from filelock import FileLock
from services.spatial_correspondence import atomic_write, png_bytes, digest, polygon_mask
from services.segmentation.base import validate_prompts

REVIEW_STATES = ('human_verified', 'rejected', 'cannot_evaluate', 'confirmed_missing', 'segmentation_failure', 'unreviewed')


def initial_document(context):
    return {'schema_version': '7A.2-sam3', 'render_id': context['render'].stem, 'camera_id': context['camera_id'],
            'render_sha256': context['render_hash'], 'image_size': {'width':context['image'].width,'height':context['image'].height},
            'generated_resolution': list(context['image'].size), 'condition_resolution': list(context['ids'].shape[::-1]),
            'coordinate_system': 'original_image_pixels_xy', 'transforms': [],
            'ground_truth': {'source':'ground_truth/instance.npy','sha256':context['gt_hash'],
                             'instance_mapping':context['mapping'],'entities':context['entities'],'warnings':context['warnings']},
            'correspondences':[], 'hallucinations':[], 'candidates':{}, 'queries':{}, 'review_history':[],
            'created_at':datetime.now(timezone.utc).isoformat()}


def commit(context, mutate):
    if not context['aligned']: raise ValueError('Resolution mismatch: no masks were resized.')
    directory = context['evaluation']; directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory/'.review.lock')):
        path = directory/'correspondence.json'
        doc = json.loads(path.read_text()) if path.exists() else None
        if (doc or {}).get('revision',0) != context['revision']: raise ValueError('Review changed in another session. Reload first.')
        if digest(context['render'].read_bytes()) != context['render_hash'] or digest(context['gt_path'].read_bytes()) != context['gt_hash']:
            raise ValueError('Inputs changed during review. Reload first.')
        if doc is None:
            doc = initial_document(context)
            atomic_write(directory/'ground_truth/instance.npy', context['gt_bytes'])
        mutate(doc, directory)
        doc.update(revision=context['revision']+1, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_write(path, json.dumps(doc,indent=2,ensure_ascii=False,allow_nan=False).encode())
    return doc


def save_automatic(context, result):
    def mutate(doc, directory):
        for key, candidate in result['candidates'].items():
            relative = f'masks/{key}.png'
            atomic_write(directory/relative, png_bytes(candidate['mask']))
            doc['candidates'][key] = {k:v for k,v in candidate.items() if k!='mask'} | {'mask_path':relative, 'backend':result['backend']}
        existing = {r['entity_id']:r for r in doc['correspondences']}
        rows = []
        for entry in result['rows']:
            row = deepcopy(entry); previous = existing.get(row['entity_id'])
            if previous and previous['status'] in REVIEW_STATES[:-1] and previous['validation_concept'] == row['validation_concept']:
                rows.append(previous); continue
            if previous: doc['review_history'].append(previous)
            selected = row['selected_candidate']
            row['generated_mask'] = doc['candidates'][selected]['mask_path'] if selected else None
            row['segmentation_backend'] = result['backend']
            row['gt_mask_source'] = 'ground_truth/instance.npy'
            rows.append(row)
        # Do not replace human decisions with a new auto-proposal owning the same object.
        claimed = {r.get('selected_candidate') for r in [*rows, *doc['hallucinations']] if r.get('human_verified')}
        for row in rows:
            if not row.get('human_verified') and row.get('selected_candidate') in claimed and row.get('selected_candidate'):
                row.update(status='ambiguous', selected_candidate=None, generated_mask=None, reason='Candidate already human-assigned.')
        doc.update(correspondences=rows, queries=result['queries'], segmentation_backend=result['backend'], thresholds=result['thresholds'])
    return commit(context, mutate)


def review(context, entity_id, status, candidate_id=None, manual=None, hallucination_type=None):
    if status not in REVIEW_STATES: raise ValueError('Unknown review state.')
    def mutate(doc, directory):
        row = next((r for r in doc['correspondences'] if r['entity_id']==entity_id), None) if entity_id else None
        if row is None and entity_id:
            e = next(e for e in context['visible'] if e['entity_id']==entity_id)
            row = {**e,'validation_concept':context.get('concepts',{}).get(entity_id,{}).get('concept',''),
                   'candidate_ids':[],'candidate_count':0}
            doc['correspondences'].append(row)
        if not entity_id:
            if hallucination_type not in ('door','window','furniture','opening','architectural element','other'):
                raise ValueError('Choose a hallucination type.')
            row = {'entity_id':None,'instance_id':None,'hallucination_id':uuid4().hex,'hallucination_type':hallucination_type,
                   'validation_concept':hallucination_type,'candidate_ids':[],'candidate_count':0}
        old = deepcopy(row)
        current_concept = context.get('concepts', {}).get(entity_id, {}).get('concept')
        if current_concept is not None and current_concept != row['validation_concept']:
            if status == 'human_verified' and manual is None:
                raise ValueError('Validation Concept changed. Re-run correspondence or use a manual mask.')
            row['validation_concept'] = current_concept
        selected = candidate_id
        method, backend, prompts, confidence = None, None, None, None
        if status == 'human_verified':
            if manual is not None:
                mask = np.asarray(manual['mask'])
                if mask.shape != context['ids'].shape or not np.isin(mask,[0,1]).all() or not mask.any(): raise ValueError('Invalid native-resolution manual mask.')
                method, backend, prompts = manual['method'], manual.get('backend'), manual['prompts']
                if method not in ('sam3_point','sam3_box','manual_polygon'): raise ValueError('Invalid segmentation method.')
                validate_prompts(context['image'].size,prompts['positive_points'],prompts['negative_points'],prompts['box'])
                if method == 'manual_polygon' and not np.array_equal(polygon_mask(context['image'].size,prompts['polygon']),mask): raise ValueError('Polygon changed since preview.')
                if method.startswith('sam3') and (backend or {}).get('provider')!='sam3': raise ValueError('SAM 3 provenance required.')
                selected = sha256(mask.astype(bool).tobytes()).hexdigest()
                relative = f'masks/{selected}.png'; atomic_write(directory/relative,png_bytes(mask))
                confidence = manual.get('quality')
                doc['candidates'][selected] = {'id':selected,'mask_path':relative,'backend':backend,'scores':{}}
                row['candidate_ids'] = list(dict.fromkeys(row['candidate_ids']+[selected]))
            elif selected not in row['candidate_ids'] or selected not in doc['candidates']:
                raise ValueError('Choose a candidate belonging to this concept.')
            for owner in [*doc['correspondences'],*doc['hallucinations']]:
                if owner is not row and owner.get('selected_candidate')==selected and owner['status'] in ('auto_matched','human_verified'):
                    raise ValueError('Candidate is already assigned. Reject its existing correspondence before reassigning.')
            candidate = doc['candidates'][selected]
            method = method or row.get('segmentation_method') or 'sam3_text'
            backend = backend or candidate.get('backend')
            confidence = confidence if confidence is not None else candidate.get('scores',{}).get(' '.join(row['validation_concept'].lower().split()))
            row.update(selected_candidate=selected, generated_mask=candidate['mask_path'], segmentation_method=method,
                       segmentation_backend=backend, confidence=confidence, sam_prompt=prompts,
                       candidate_count=len(row['candidate_ids']))
        else:
            row.update(selected_candidate=None,generated_mask=None)
        row.update(status=status,human_verified=status=='human_verified',reviewed_by_human=True,
                   updated_at=datetime.now(timezone.utc).isoformat())
        doc['review_history'].append(old)
        if not entity_id: doc['hallucinations'].append(row)
    return commit(context,mutate)
