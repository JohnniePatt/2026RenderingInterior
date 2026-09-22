"""CAD-guided SAM 3 correspondence; no Part 7B scores."""
from copy import deepcopy
from hashlib import sha256
import json
import numpy as np
from PIL import Image
import streamlit as st
from app.ui.spatial_validation_manual import mask_editor, image_url, empty_prompts
from services.spatial_correspondence import load_context, polygon_mask
from services.segmentation.sam3_provider import load_config, save_config, get_backend
from services.concept_matching import run_correspondence, resolve_visible_concepts
from services.automatic_correspondence_store import save_automatic, review


def render(root, state):
    st.header('Spatial Validation · SAM 3')
    st.caption('CAD concepts define what to search. Automatic matches are proposals. Review ambiguous or failed detections; no spatial scores are calculated.')
    cameras=list(state.get('cameras',{}))
    if not cameras: st.info('Create a camera first.'); return
    camera=st.selectbox('Validation Camera',cameras,key=f'val_camera_{root.name}')
    renders=sorted((root/'generated'/camera/'renders').glob('*.png'),reverse=True)
    if not renders: st.info('No generated renders.'); return
    name=st.selectbox('Generated Render',[p.name for p in renders],key=f'val_render_{root.name}_{camera}')
    try: context=load_context(root,state,camera,name,namespace='sam3')
    except (ValueError,OSError,KeyError) as exc: st.error(str(exc)); return
    for message in context['warnings']: st.warning(message)
    config=load_config()
    st.write('Segmentation Model: **SAM 3 (local)**')
    with st.expander('SAM 3 settings'):
        st.caption('Run and Retry use only local files. They never download weights.')
        with st.form('sam3_settings'):
            path=st.text_input('SAM 3 checkpoint folder',config['checkpoint'])
            device=st.selectbox('SAM 3 device',['cpu','cuda'],index=0 if config['device']=='cpu' else 1)
            detection=st.slider('Detection threshold',0.0,1.0,float(config['detection_threshold']))
            strong=st.slider('Strong candidate threshold',0.0,1.0,float(config['strong_threshold']))
            margin=st.slider('Assignment separation threshold',0.0,1.0,float(config['assignment_margin']))
            if st.form_submit_button('Save SAM 3 settings'):
                try:
                    save_config({**config,'checkpoint':path,'device':device,'detection_threshold':detection,'strong_threshold':strong,'assignment_margin':margin}); st.rerun()
                except ValueError as exc: st.error(str(exc))
        st.markdown('Weights require access to [facebook/sam3](https://huggingface.co/facebook/sam3). Authenticate locally, never in a research prompt.')
    concepts=resolve_visible_concepts(context,state['entities']); context['concepts']=concepts
    for eid,concept in concepts.items():
        if concept['warning']: st.warning(f"{eid}: {concept['warning']}")
    if st.button('Run Automatic Correspondence',type='primary',disabled=not context['aligned']):
        try:
            with st.spinner('Local SAM 3: querying unique concepts…'):
                result=run_correspondence(context,get_backend(config),concepts,config['strong_threshold'],config['assignment_margin'])
                save_automatic(context,result)
            st.rerun()
        except (ImportError,OSError,RuntimeError,ValueError) as exc: st.error(f'No automatic results were saved: {exc}')
    doc=context['doc'] or {'correspondences':[],'hallucinations':[],'candidates':{},'queries':{}}
    rows={r['entity_id']:r for r in doc['correspondences']}
    table=[]
    for entity in context['visible']:
        row=rows.get(entity['entity_id'],{})
        table.append({'Entity':entity['name'],'Entity ID':entity['entity_id'],'Concept':concepts[entity['entity_id']]['concept'],
                      'Detection':row.get('candidate_count',0),'Confidence':row.get('confidence'),'Status':row.get('status','unreviewed')})
        if row and row.get('validation_concept')!=concepts[entity['entity_id']]['concept']:
            st.warning(f"{entity['entity_id']}: concept changed. Re-run correspondence before accepting old candidates.")
    st.dataframe(table,hide_index=True,use_container_width=True)
    st.caption('missing_candidate is not proven object absence. auto_matched remains human_verified=false.')
    with st.expander('Part 7A Debug'):
        st.json({'camera':camera,'render':name,'visible_instance_ids':[e['instance_id'] for e in context['visible']],
                 'validation_concepts':concepts,'queries':doc.get('queries',{}),'candidates':doc.get('candidates',{}),
                 'proposed_assignment':doc.get('correspondences',[]),'resolution_checks':{'generated':list(context['image'].size),
                 'condition':list(context['ids'].shape[::-1]),'aligned':context['aligned']},'backend':doc.get('segmentation_backend')})
    if context['doc']: st.download_button('Download correspondence.json',json.dumps(doc,indent=2),file_name='correspondence.json')
    hallucination=st.checkbox('Hallucination Review (separate from CAD correspondence)',key=f'hallucination3_{root.name}_{camera}_{name}')
    if hallucination:
        kind=st.selectbox('Hallucination type',['door','window','furniture','opening','architectural element','other'])
        selected=None; row={}
        st.info('Optional review of an extra object. CAD-guided queries do not discover every hallucination.')
    else:
        kind=None
        attention=st.checkbox('Show cases needing attention only',value=True)
        options=[e for e in context['visible'] if not attention or rows.get(e['entity_id'],{}).get('status','unreviewed') not in ('auto_matched','human_verified')]
        if not options: st.info('No cases need attention. Uncheck the filter to inspect automatic matches.'); return
        eid=st.selectbox('Review CAD entity',[e['entity_id'] for e in options],format_func=lambda key: next(f"{e['name']} · {key}" for e in options if e['entity_id']==key))
        selected=next(e for e in options if e['entity_id']==eid); row=rows.get(eid,{})
    token=sha256(json.dumps([str(root),camera,name,context['render_hash'],context['gt_hash'],selected,kind,context['revision'],config,concepts],sort_keys=True).encode()).hexdigest()
    draft_key=f'sam3_draft_{root.name}'
    if st.session_state.get(draft_key,{}).get('token')!=token:
        st.session_state[draft_key]={'token':token,'prompts':empty_prompts(),'result':None,'epoch':0,'event':None}
    draft=st.session_state[draft_key]
    left,right=st.columns([2,1]); candidate_id=None; manual=None
    with right:
        if selected:
            st.image((context['ids']==selected['instance_id']).astype(np.uint8)*255,caption='CAD Ground Truth Mask',use_container_width=True)
            st.write('Status: '+row.get('status','unreviewed')); st.caption(row.get('reason',''))
        mode=st.radio('Review method',['Choose Another Candidate','Manual SAM Prompt','Manual Polygon'],index=1 if hallucination else 0,key=f'mode3_{token}')
        if draft.get('mode')!=mode: draft['result']=None; draft['mode']=mode
        if mode=='Manual SAM Prompt':
            label=st.radio('SAM 3 prompt',['+ Object Point','- Exclude Point','Use Box'])
            interaction={'+ Object Point':'positive','- Exclude Point':'negative','Use Box':'box'}[label]
        else: interaction='polygon'
    with left:
        if mode=='Choose Another Candidate':
            ids=row.get('candidate_ids',[])
            if ids:
                preferred=row.get('selected_candidate')
                candidate_id=st.selectbox('Candidate',ids,index=ids.index(preferred) if preferred in ids else 0,
                    format_func=lambda i:f"{i[:10]} · {doc['candidates'][i].get('scores',{})}")
                mask=np.array(Image.open(context['evaluation']/doc['candidates'][candidate_id]['mask_path']))>0
                st.image(image_url(context['image'],mask),caption='Generated SAM Mask Overlay',use_container_width=True)
                with st.expander('Mask Only'): st.image(mask.astype(np.uint8)*255,use_container_width=True)
            else: st.image(context['image'],use_container_width=True); st.info('No candidate. Use Manual SAM Prompt or Manual Polygon.')
        else:
            manual=manual_editor(context,config,draft,token,mode,interaction)
    stale=selected and row and row.get('validation_concept')!=concepts[selected['entity_id']]['concept'] and manual is None
    try:
        if st.button('Confirm',type='primary',disabled=not context['aligned'] or bool(stale) or (candidate_id is None and manual is None)):
            review(context,selected['entity_id'] if selected else None,'human_verified',candidate_id,manual,kind); st.rerun()
        if selected:
            for label,status in [('Reject','rejected'),('Cannot Evaluate','cannot_evaluate'),('Confirm object is missing','confirmed_missing'),('Mark segmentation failure','segmentation_failure')]:
                if st.button(label,disabled=not context['aligned']): review(context,selected['entity_id'],status); st.rerun()
    except (ValueError,OSError) as exc: st.error(str(exc))
    with st.expander('Saved Hallucination Review'):
        for item in doc['hallucinations']:
            st.write(item['hallucination_type'])
            if item.get('generated_mask'): st.image(str(context['evaluation']/item['generated_mask']),width=180)


def manual_editor(context,config,draft,token,mode,interaction):
    retry=st.button('Retry SAM 3' if mode=='Manual SAM Prompt' else 'Build Polygon Mask')
    if st.button('Clear prompts'):
        draft.update(prompts=empty_prompts(),result=None,epoch=draft['epoch']+1); st.rerun()
    predicted=draft['result']; index=0
    if predicted: index=st.selectbox('Manual candidate',list(range(len(predicted['masks']))),key=f'manual3_{token}_{draft["epoch"]}')
    event=mask_editor(image=image_url(context['image'],predicted['masks'][index] if predicted else None),
        width=context['image'].width,height=context['image'].height,prompts=draft['prompts'],mode=interaction,
        context=token,key=f'canvas3_{token}_{draft["epoch"]}',default=None)
    if event and event.get('context')==token and event.get('event_id')!=draft['event']:
        draft['event']=event['event_id']
        if event['kind']!=interaction: st.rerun()
        if interaction=='box': draft['prompts']['box']=event['box']
        else: draft['prompts'][{'positive':'positive_points','negative':'negative_points','polygon':'polygon'}[interaction]].append(event['point'])
        draft['result']=None; predicted=None; retry=mode=='Manual SAM Prompt'
        if not retry: st.rerun()
    if retry:
        try:
            prompts=draft['prompts']
            if mode=='Manual Polygon':
                draft['result']={'masks':[polygon_mask(context['image'].size,prompts['polygon'])],'quality':[None],'backend':None,'method':'manual_polygon'}
            else:
                with st.spinner('Local SAM 3 visual prompt…'):
                    result=get_backend(config).segment(context['image'],prompts['positive_points'],prompts['negative_points'],prompts['box'])
                draft['result']={'masks':result.masks,'quality':result.quality,'backend':result.backend,'method':'sam3_box' if prompts['box'] else 'sam3_point'}
            draft['epoch']+=1; st.rerun()
        except (ImportError,OSError,RuntimeError,ValueError) as exc: st.error(str(exc))
    with st.expander('Original pixel prompts'): st.json(draft['prompts'])
    if predicted:
        mask=predicted['masks'][index]
        with st.expander('Mask Only'): st.image(mask.astype(np.uint8)*255,use_container_width=True)
        return {'mask':mask,'quality':predicted['quality'][index],'backend':predicted['backend'],
                'method':predicted['method'],'prompts':deepcopy(draft['prompts'])}
    return None
