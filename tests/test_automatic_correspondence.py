from itertools import permutations
import numpy as np
import pytest
from services.concept_matching import assignment, run_correspondence, resolve_visible_concepts
from services.validation_concepts import resolve_concept
from services.spatial_correspondence import load_context
from services.automatic_correspondence_store import save_automatic, review
from test_spatial_correspondence import evaluation


class Detector:
    info={'provider':'sam3','model':'test-double','checkpoint_sha256':{'test':'test'}}
    def __init__(self,candidates): self.candidates=candidates; self.calls=[]
    def detect_and_segment(self,image,concept):
        self.calls.append(concept); return self.candidates


def context(evaluation): return load_context(*evaluation,'camera_001','render_a.png',namespace='sam3')


def candidate(mask,score=.95): return {'mask':mask,'score':score}


def test_resolution_mismatch_never_runs_detector(evaluation):
    c=context(evaluation); c['aligned']=False
    detector=Detector([])
    with pytest.raises(ValueError,match='Resolution mismatch'):
        run_correspondence(c,detector,resolve_visible_concepts(c,evaluation[1]['entities']))
    assert not detector.calls and not c['evaluation'].exists()


def test_changed_concept_cannot_confirm_stale_candidate(evaluation):
    c=context(evaluation)
    result=run_correspondence(c,Detector([candidate(c['ids']==1)]),resolve_visible_concepts(c,evaluation[1]['entities']))
    saved=save_automatic(c,result)
    c=context(evaluation); c['concepts']={'bed':{'concept':'bed'}}
    with pytest.raises(ValueError,match='Concept changed'):
        review(c,'bed','human_verified',saved['correspondences'][0]['selected_candidate'])
    assert context(evaluation)['doc']['correspondences'][0]['status']=='auto_matched'


def test_concept_precedence_and_conflicts():
    assert resolve_concept({'category':'dining_table','semantic':'furniture','description':'table'})['concept']=='dining table'
    assert resolve_concept({'semantic':'window','description':'glass'})['concept']=='window'
    conflict={'category':'sofa','description':'dining table'}
    assert resolve_concept(conflict)['needs_review']
    assert resolve_concept({**conflict,'validation_concept':'dining table'})['concept']=='dining table'
    assert not resolve_concept({**conflict,'validation_concept':'dining table'})['needs_review']


@pytest.mark.parametrize('count,score,status',[(0,.95,'missing_candidate'),(1,.95,'auto_matched'),(1,.4,'ambiguous'),(2,.95,'ambiguous')])
def test_candidate_cases(evaluation,count,score,status):
    c=context(evaluation)
    # Displaced candidate is still a semantic correspondence; no overlap requirement.
    mask=np.zeros(c['ids'].shape,bool); mask[:5,:5]=True
    second=np.zeros_like(mask); second[-5:,-5:]=True
    detector=Detector([candidate(m,score) for m in [mask,second][:count]])
    result=run_correspondence(c,detector,resolve_visible_concepts(c,evaluation[1]['entities']))
    assert result['rows'][0]['status']==status and not result['rows'][0]['human_verified']
    assert len(detector.calls)==1
    doc=save_automatic(c,result)
    assert doc['correspondences'][0]['candidate_count']==count
    assert context(evaluation)['doc']['schema_version']=='7A.2-sam3'
    if status=='auto_matched':
        row=doc['correspondences'][0]
        doc=review(context(evaluation),'bed','human_verified',row['selected_candidate'])
        assert doc['correspondences'][0]['human_verified']
    if status=='missing_candidate':
        assert doc['correspondences'][0]['generated_mask'] is None
        doc=review(context(evaluation),'bed','segmentation_failure')
        assert doc['correspondences'][0]['status']=='segmentation_failure'


def test_repeated_concept_one_query_and_unique_assignment(evaluation):
    c=context(evaluation)
    c['ids'][:]=0; c['ids'][:10,:10]=1; c['ids'][-10:,-10:]=2
    c['visible'].append({'entity_id':'chair2','instance_id':2,'name':'Chair 2'})
    concepts={e['entity_id']:{'concept':'chair','source':'manual','needs_review':False} for e in c['visible']}
    detector=Detector([candidate(c['ids']==2),candidate(c['ids']==1)])
    result=run_correspondence(c,detector,concepts)
    assert detector.calls==['chair']
    assert all(r['status']=='auto_matched' for r in result['rows'])
    assert len({r['selected_candidate'] for r in result['rows']})==2
    assert all(r['segmentation_method']=='sam3_text_spatial_match' for r in result['rows'])


def test_hungarian_finds_global_optimum():
    cost=np.array([[.4,.1,.5],[.2,.3,.8],[.6,.7,.1]])
    pairs=assignment(cost)
    assert sum(cost[i,j] for i,j in pairs.items())==min(sum(cost[i,j] for i,j in enumerate(p)) for p in permutations(range(3)))
    assert len(assignment(np.ones((3,1))))==1


def test_failed_detection_does_not_persist_missing(evaluation):
    c=context(evaluation)
    class Failed(Detector):
        def detect_and_segment(self,*args): raise RuntimeError('backend unavailable')
    with pytest.raises(RuntimeError): run_correspondence(c,Failed([]),resolve_visible_concepts(c,evaluation[1]['entities']))
    assert c['doc'] is None and not c['evaluation'].exists()


def test_human_rejection_survives_rerun(evaluation):
    c=context(evaluation); concepts=resolve_visible_concepts(c,evaluation[1]['entities'])
    result=run_correspondence(c,Detector([candidate(c['ids']==1)]),concepts)
    save_automatic(c,result)
    review(context(evaluation),'bed','rejected')
    doc=save_automatic(context(evaluation),result)
    assert doc['correspondences'][0]['status']=='rejected'
