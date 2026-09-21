"""Part 7A only: human-selected correspondence, local segmentation, explicit review."""
import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.components.v1 import declare_component
from services.segmentation.config import load_config, save_config
from services.segmentation.sam_provider import get_backend
from services.spatial_correspondence import load_context, save_review, polygon_mask, HALLUCINATION_TYPES

mask_editor = declare_component('mask_editor', path=str(Path(__file__).resolve().parents[2] / 'components/mask_editor'))


def empty_prompts():
    return {'positive_points': [], 'negative_points': [], 'box': None, 'polygon': []}


def image_url(image, mask=None):
    rgb = np.array(image.convert('RGB')).copy()
    if mask is not None:
        rgb[mask] = (.55*rgb[mask] + .45*np.array([40, 230, 180])).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(rgb).save(output, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


def render(root, state):
    st.header('Spatial Validation · Part 7A')
    st.caption('Select the CAD entity yourself. SAM extracts a boundary; only Confirm Mask saves an accepted correspondence. No spatial scores are calculated.')
    cameras = list(state.get('cameras', {}))
    if not cameras:
        st.info('Create a camera and generate a render first.'); return
    camera = st.selectbox('Validation Camera', cameras, key=f'val_camera_{root.name}')
    renders = sorted((root / 'generated' / camera / 'renders').glob('*.png'), reverse=True)
    if not renders:
        st.info('No generated renders for this camera.'); return
    render_name = st.selectbox('Generated Render', [p.name for p in renders], key=f'val_render_{root.name}_{camera}')
    try:
        context = load_context(root, state, camera, render_name)
    except (ValueError, OSError, KeyError) as exc:
        st.error(str(exc)); return
    for warning in context['warnings']:
        st.warning(warning)
    config = load_config()
    with st.expander('Local SAM settings'):
        st.caption('Pretrained Segment Anything; no training and no image-generation API. Checkpoints are local files, never downloaded on clicks.')
        with st.form('sam_settings'):
            model = st.selectbox('SAM model', ['vit_b', 'vit_l', 'vit_h'], index=['vit_b', 'vit_l', 'vit_h'].index(config['model']))
            checkpoint = st.text_input('Checkpoint path', config['checkpoint'])
            device = st.selectbox('Device', ['cpu', 'cuda'], index=0 if config.get('device') == 'cpu' else 1)
            if st.form_submit_button('Save SAM settings'):
                try:
                    save_config({'provider': 'sam', 'model': model, 'checkpoint': checkpoint, 'device': device})
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    doc = context['doc'] or {'correspondences': [], 'hallucinations': []}
    statuses = {r['entity_id']: r['status'] for r in doc['correspondences']}
    mode_key = f'val_hallucination_{root.name}_{camera}_{render_name}'
    if st.button('+ Mark Hallucinated Object'):
        st.session_state[mode_key] = True
        st.rerun()
    hallucination = st.session_state.get(mode_key, False)
    if hallucination:
        st.info('Reviewing a hallucinated object — no CAD entity will be linked.')
        if st.button('Return to CAD entities'):
            st.session_state[mode_key] = False
            st.rerun()
    entity = None
    hallucination_type = None
    if hallucination:
        hallucination_type = st.selectbox('Hallucination type', HALLUCINATION_TYPES)
    elif context['visible']:
        entity_id = st.selectbox('CAD entity', [e['entity_id'] for e in context['visible']],
            format_func=lambda eid: next(f"Instance {e['instance_id']} · {e['name']} · {eid} · {statuses.get(eid, 'unreviewed')}" for e in context['visible'] if e['entity_id'] == eid))
        entity = next(e for e in context['visible'] if e['entity_id'] == entity_id)
    else:
        st.info('No foreground CAD instances are visible. You can mark hallucinated objects.'); return
    token = sha256(json.dumps([str(root), camera, render_name, context['render_hash'], context['gt_hash'],
                              entity, hallucination_type, config, context['revision']], sort_keys=True).encode()).hexdigest()
    draft_key = f'validation_draft_{root.name}'
    if st.session_state.get(draft_key, {}).get('token') != token:
        st.session_state[draft_key] = {'token': token, 'prompts': empty_prompts(), 'result': None, 'last_event': None, 'epoch': 0}
    draft = st.session_state[draft_key]
    left, right = st.columns([2, 1])
    with right:
        if entity:
            st.image((context['ids'] == entity['instance_id']).astype(np.uint8)*255, caption='CAD Ground Truth Mask', use_container_width=True)
            st.write('Status: ' + statuses.get(entity['entity_id'], 'unreviewed'))
        else:
            st.info('Hallucination: entity_id = null. No CAD mask is assigned.')
        method = st.radio('Boundary extraction', ['SAM', 'Manual Polygon'], key=f'boundary_{token}')
        if method == 'SAM':
            interaction = st.radio('SAM controls', ['+ Object Point', '- Exclude Point', 'Use Box'])
            interaction = {'+ Object Point': 'positive', '- Exclude Point': 'negative', 'Use Box': 'box'}[interaction]
        else:
            interaction = 'polygon'
        # Switching extraction method invalidates a candidate, never a saved review.
        if draft.get('method') != method:
            draft['result'] = None
            draft['method'] = method
        retry = st.button('Retry SAM' if method == 'SAM' else 'Build Polygon Mask')
        if st.button('Clear prompts'):
            draft.update(prompts=empty_prompts(), result=None, epoch=draft['epoch']+1)
            st.rerun()
        if st.button('Undo last point'):
            field = {'positive': 'positive_points', 'negative': 'negative_points', 'polygon': 'polygon'}.get(interaction)
            if field and draft['prompts'][field]: draft['prompts'][field].pop()
            elif interaction == 'box': draft['prompts']['box'] = None
            draft['result'] = None
            draft['epoch'] += 1
            st.rerun()
        st.caption(f"{len(draft['prompts']['positive_points'])} positive · {len(draft['prompts']['negative_points'])} negative · {len(draft['prompts']['polygon'])} polygon vertices")
    result = draft['result']
    candidate = 0
    with left:
        if result:
            candidate = st.selectbox('Candidate mask', list(range(len(result['masks']))),
                                     format_func=lambda n: f"Candidate {n+1}", key=f'candidate_{token}_{draft["epoch"]}')
            if result['quality'][candidate] is not None:
                st.caption(f"SAM predicted quality: {result['quality'][candidate]:.3f} — model confidence only, not a spatial evaluation score.")
        event = mask_editor(image=image_url(context['image'], result['masks'][candidate] if result else None),
                            width=context['image'].width, height=context['image'].height, prompts=draft['prompts'],
                            mode=interaction, context=token, key=f'mask_canvas_{token}_{draft["epoch"]}', default=None)
        if event and event.get('context') == token and event.get('event_id') != draft['last_event']:
            draft['last_event'] = event['event_id']
            kind = event.get('kind')
            if kind != interaction:
                st.info('The drawing mode just changed. Click again in the updated mode.')
                st.rerun()
            if kind == 'box': draft['prompts']['box'] = event['box']
            elif kind in ('positive', 'negative', 'polygon'):
                draft['prompts'][{'positive': 'positive_points', 'negative': 'negative_points', 'polygon': 'polygon'}[kind]].append(event['point'])
            draft['result'] = None
            result = None
            retry = method == 'SAM'
            if not retry: st.rerun()
        if retry:
            try:
                prompts = draft['prompts']
                if method == 'SAM':
                    with st.spinner('Local SAM segmentation…'):
                        predicted = get_backend(config).segment(context['image'], prompts['positive_points'], prompts['negative_points'], prompts['box'])
                    draft['result'] = {'masks': predicted.masks, 'quality': predicted.quality, 'backend': predicted.backend, 'method': 'sam'}
                else:
                    mask = polygon_mask(context['image'].size, prompts['polygon'])
                    draft['result'] = {'masks': [mask], 'quality': [None], 'backend': None, 'method': 'manual_polygon'}
                draft['epoch'] += 1
                st.rerun()
            except (ValueError, OSError, ImportError, RuntimeError) as exc:
                st.error(f'Segmentation unavailable: {exc}. You can use Manual Polygon.')
        if result:
            with st.expander('Mask Only'):
                st.image(result['masks'][candidate].astype(np.uint8)*255, use_container_width=True)
        with st.expander('Original image pixel prompts'):
            st.json(draft['prompts'])
    try:
        if st.button('Confirm Mask', type='primary', disabled=result is None or not context['aligned']):
            save_review(context, entity=entity, mask=result['masks'][candidate], prompts=deepcopy(draft['prompts']),
                        method=result['method'], backend=result['backend'], quality=result['quality'][candidate],
                        hallucination_type=hallucination_type)
            st.rerun()
        if entity:
            a, b, c = st.columns(3)
            for column, label, status in [(a, 'Mark Missing', 'missing'), (b, 'Occluded / Cannot Evaluate', 'occluded'), (c, 'Reset to Unreviewed', 'unreviewed')]:
                if column.button(label, disabled=not context['aligned']):
                    save_review(context, entity=entity, status=status)
                    st.rerun()
    except (ValueError, OSError) as exc:
        st.error(str(exc))
    with st.expander('Saved correspondences', expanded=True):
        st.dataframe([{'entity': e['entity_id'], 'instance': e['instance_id'], 'status': statuses.get(e['entity_id'], 'unreviewed')} for e in context['visible']], hide_index=True)
        if context['doc']:
            st.download_button('Download correspondence.json', json.dumps(context['doc'], indent=2), file_name='correspondence.json')
            for record in [*doc['correspondences'], *doc['hallucinations']]:
                if record.get('generated_mask'):
                    st.image(str(context['evaluation'] / record['generated_mask']), width=180,
                             caption=record.get('entity_id') or f"Hallucination: {record['hallucination_type']}")
        st.caption('Accepted masks are stored per render, separately from layout.json. Draft SAM masks are never saved automatically.')
