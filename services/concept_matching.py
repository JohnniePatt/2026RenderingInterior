"""Semantic correspondence proposals, not spatial-accuracy evaluation metrics."""
from collections import defaultdict
from hashlib import sha256
import numpy as np
from services.validation_concepts import resolve_concept


def descriptor(mask):
    y, x = np.where(mask)
    if not len(x): raise ValueError('Empty candidate mask.')
    h, w = mask.shape
    return {'centroid': [float(x.mean()), float(y.mean())], 'bbox': [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
            'area': int(len(x)), 'normalized_centroid': [float(x.mean()/w), float(y.mean()/h)]}


def assignment(cost):
    """Rectangular minimum-cost one-to-one assignment (Hungarian potentials)."""
    rows, cols = cost.shape
    n = max(rows, cols)
    matrix = np.full((n, n), 2.0); matrix[:rows, :cols] = cost
    u, v = np.zeros(n+1), np.zeros(n+1)
    p, way = np.zeros(n+1, dtype=int), np.zeros(n+1, dtype=int)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minimum, used = np.full(n+1, np.inf), np.zeros(n+1, bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta, j1 = np.inf, 0
            for j in range(1, n+1):
                if not used[j]:
                    cur = matrix[i0-1, j-1] - u[i0] - v[j]
                    if cur < minimum[j]: minimum[j], way[j] = cur, j0
                    if minimum[j] < delta: delta, j1 = minimum[j], j
            for j in range(n+1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minimum[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    return {int(p[j]-1): j-1 for j in range(1,n+1) if 0 < p[j] <= rows and j <= cols}


def run_correspondence(context, backend, concepts, strong=.8, margin=.05):
    if not context['aligned']: raise ValueError('Resolution mismatch: automatic correspondence is blocked.')
    groups = defaultdict(list)
    rows, candidates, queries = [], {}, {}
    for entity in context['visible']:
        concept = concepts[entity['entity_id']]
        row = {**entity, 'validation_concept': concept['concept'], 'concept_source': concept['source'],
               'status': 'unreviewed', 'human_verified': False, 'candidate_ids': [], 'selected_candidate': None,
               'confidence': None, 'candidate_count': 0, 'segmentation_method': None,
               'gt_descriptor': descriptor(context['ids'] == entity['instance_id'])}
        rows.append(row)
        if concept['needs_review']:
            row['reason'] = concept['warning']; continue
        groups[' '.join(concept['concept'].lower().split())].append(row)
    for concept, entities in groups.items():
        observed = backend.detect_and_segment(context['image'], concept)
        ids = []
        for observed_candidate in observed:
            raw = np.asarray(observed_candidate['mask'])
            if raw.shape != context['ids'].shape or not np.isin(raw, [0, 1]).all():
                raise ValueError('Detector returned invalid mask coordinates or values.')
            mask = raw.astype(bool)
            score = float(observed_candidate['score'])
            if not np.isfinite(score) or not 0 <= score <= 1: raise ValueError('Invalid detector confidence.')
            if not mask.any(): continue
            key = sha256(mask.tobytes()).hexdigest()
            if key not in ids: ids.append(key)
            if key not in candidates:
                candidates[key] = {'id': key, 'mask': mask, 'descriptor': descriptor(mask), 'scores': {}, 'concepts': []}
            candidates[key]['scores'][concept] = score
            candidates[key]['concepts'].append(concept)
        queries[concept] = {'candidate_ids': ids, 'candidate_count': len(ids)}
        for row in entities:
            row.update(candidate_ids=ids.copy(), candidate_count=len(ids))
            if not ids: row.update(status='missing_candidate', reason='No detector candidate; object absence is not established.')
        if not ids: continue
        if len(entities) == 1:
            row = entities[0]
            if len(ids) == 1:
                score = candidates[ids[0]]['scores'][concept]
                row.update(selected_candidate=ids[0], confidence=score, segmentation_method='sam3_text',
                           status='auto_matched' if score >= strong else 'ambiguous',
                           reason='Single strong semantic candidate.' if score >= strong else 'Low confidence: review required.')
            else: row.update(status='ambiguous', reason='Multiple semantic candidates; choose one explicitly.')
            continue
        cost = np.array([[np.linalg.norm(np.subtract(row['gt_descriptor']['normalized_centroid'], candidates[key]['descriptor']['normalized_centroid'])) for key in ids] for row in entities])
        pairs = assignment(cost)
        for i, row in enumerate(entities):
            row['status'] = 'ambiguous'
            row['reason'] = 'Repeated concept: assignment requires review.'
            if i not in pairs: continue
            j = pairs[i]; key = ids[j]; score = candidates[key]['scores'][concept]
            alternatives = list(np.delete(cost[i], j)) + list(np.delete(cost[:,j], i))
            separation = min(alternatives, default=2.0) - float(cost[i,j])
            row.update(selected_candidate=key, confidence=score, segmentation_method='sam3_text_spatial_match',
                       assignment={'method': 'one_to_one_centroid', 'separation': separation})
            if score >= strong and separation >= margin and len(ids) == len(entities):
                row.update(status='auto_matched', reason='Distinct one-to-one centroid proposal; not a spatial accuracy score.')
    # Different text concepts can describe the very same detected object.
    owners = defaultdict(list)
    for row in rows:
        if row['selected_candidate']: owners[row['selected_candidate']].append(row)
    for group in owners.values():
        if len(group) > 1:
            for row in group: row.update(status='ambiguous', selected_candidate=None, reason='Candidate shared across concepts; human assignment required.')
    return {'rows': rows, 'candidates': candidates, 'queries': queries, 'backend': backend.info,
            'thresholds': {'strong': strong, 'assignment_margin': margin}}


def resolve_visible_concepts(context, current_entities):
    return {e['entity_id']: resolve_concept(current_entities.get(e['entity_id'], context['entities'].get(e['entity_id'], {}))) for e in context['visible']}
