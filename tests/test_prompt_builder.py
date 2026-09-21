import copy
import io
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from services.prompt_builder import compile_prompt, build_prompt, get_available_conditions, validate_prompt_request
from services.image_generation.gemini_provider import GeminiImageGenerator
from services.prompt_resolution import SAME_ROOM_ROLE


@pytest.fixture
def scene(tmp_path):
    folder = tmp_path / 'generated/camera_001'
    folder.mkdir(parents=True)
    for kind in ['proxy','depth','instance','semantic','planar']:
        Image.new('RGB',(4,3)).save(folder/f'{kind}.png')
    np.save(folder/'instance.npy', np.array([[0,1,2,3],[1,2,3,0],[1,2,3,0]],dtype=np.uint32))
    state = {'source':{'units':'mm'}, 'cameras':{'camera_001':{'name':'Test','position':[100,200,1500],'fov_deg':75}},
             'ceiling':{'elevation':2800,'material':'gypsum','description':'gypsum ceiling'},
             'entities':{'bed':{'semantic':'furniture','category':'bed','height':400,'description':'king-size bed',
                                'dxf_handle':'SECRET_HANDLE','block_name':'CAD_BLOCK','insertion_point':[1,2,3]},
                         'window':{'semantic':'void','category':'window','opening_height':2100,'sill_height':400},
                         'hidden':{'semantic':'furniture','category':'cabinet','description':'invisible wardrobe'}},
             'instance_mapping':{'camera_001':{'0':{'rgb':[0,0,0],'entity_id':None},
                  '1':{'rgb':[17,81,143],'entity_id':'bed'},'2':{'rgb':[190,4,89],'entity_id':'window'},
                  '3':{'rgb':[211,157,61],'entity_id':'auto_ceiling'},'4':{'rgb':[38,52,97],'entity_id':'hidden'}}}}
    conditions=get_available_conditions(tmp_path,'camera_001')
    return state,conditions,tmp_path


def compile_scene(scene, types=('instance','depth'), **kwargs):
    state,conditions,root=scene
    selected=[{**c,'enabled':True} for c in conditions if c['type'] in types]
    return compile_prompt(state,'camera_001',selected,root_path=root,**kwargs)


def test_discovery_and_selected_order(scene):
    state,conditions,root=scene
    assert len(conditions)==5 and all(c['resolution']==(4,3) for c in conditions)
    assert [c['type'] for c in conditions if c['enabled']]==['depth','instance']
    result=compile_scene(scene)
    assert not result.errors
    assert 'IMAGE 1 is a CAD-derived depth' in result.spatial_prompt
    assert 'IMAGE 2 is a CAD-derived instance' in result.spatial_prompt
    assert [r['type'] for r in result.debug['image_order']]==['depth','instance']


def test_dynamic_rgb_visibility_whitelist_and_dimensions(scene):
    result=compile_scene(scene)
    text=result.spatial_prompt
    assert 'RGB(17,81,143) identifies king-size bed, approximately 0.4 m high' in text
    assert 'RGB(190,4,89) identifies window opening, approximately 2.1 m high, with a 0.4 m sill' in text
    assert '2.8 m elevation' in text
    assert result.debug['visible_instance_ids']==[1,2,3]
    hidden=next(r for r in result.debug['instances'] if r['instance_id']==4)
    assert hidden['visible'] is False and not hidden['included']
    for forbidden in ['invisible wardrobe','RGB(38,52,97)','CAD_BLOCK','SECRET_HANDLE','insertion_point','camera_001','#000000','100%','red region','blue region']:
        assert forbidden not in text


def test_reordered_input_and_instance_only_never_mentions_depth(scene):
    result=compile_scene(scene,types=('instance',))
    assert 'depth' not in result.final_prompt.lower()
    assert 'IMAGE 1 is a CAD-derived instance' in result.final_prompt
    state,conditions,root=scene
    selected=[{**c,'enabled':True} for c in reversed(conditions) if c['type'] in {'instance','depth'}]
    result=compile_prompt(state,'camera_001',selected,root_path=root)
    assert 'IMAGE 1 is a CAD-derived instance' in result.final_prompt
    assert 'IMAGE 2 is a CAD-derived depth' in result.final_prompt


def test_no_instance_means_no_mapping_or_invisible_semantics(scene):
    result=compile_scene(scene,types=('depth','semantic','proxy'))
    assert not result.errors
    assert 'RGB(' not in result.final_prompt and 'king-size bed' not in result.final_prompt
    assert 'semantic segmentation' in result.final_prompt and 'metric proxy' in result.final_prompt


def test_conflict_is_neutral_not_silent_guess(scene):
    scene[0]['entities']['bed'].update(category='sofa',description='dinning table',notes='table set')
    result=compile_scene(scene)
    assert len(result.debug['semantic_conflicts'])==1
    assert any('Semantic conflict' in warning for warning in result.warnings)
    assert 'furniture element' in result.final_prompt
    assert 'sofa' not in result.final_prompt and 'dinning table' not in result.final_prompt


def test_reference_role_continuous_numbering_and_appearance_separation(scene):
    buffer=io.BytesIO();Image.new('RGB',(8,8)).save(buffer,format='PNG')
    appearance='Warm oak flooring with a blue sofa.'
    refs=[{'bytes':buffer.getvalue(),'role':'Lighting Reference','name':'light.png'}]
    result=compile_scene(scene,reference_images=refs,user_design_prompt=appearance)
    assert not result.errors
    assert 'IMAGE 3 is an appearance reference (Lighting Reference)' in result.spatial_prompt
    assert 'Use it only for lighting, shadows and atmosphere' in result.spatial_prompt
    assert 'Do not copy its room geometry' in result.spatial_prompt
    assert appearance not in result.spatial_prompt
    assert result.final_prompt==result.spatial_prompt+'\n\nAPPEARANCE PROMPT\n'+appearance


def test_valid_empty_instance_map_does_not_fallback(scene):
    np.save(scene[2]/'generated/camera_001/instance.npy',np.zeros((3,4),dtype=np.uint32))
    result=compile_scene(scene)
    assert not result.errors and result.debug['visible_instance_ids']==[]
    assert 'RGB(' not in result.final_prompt and 'king-size bed' not in result.final_prompt


def test_same_room_reference_prioritizes_target_view_and_retains_identity(scene):
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buffer, format='PNG')
    refs = [{'bytes': buffer.getvalue(), 'role': SAME_ROOM_ROLE, 'name': 'room.png',
             'description': 'Retain the oak dining set.'},
            {'bytes': buffer.getvalue(), 'role': 'Lighting Reference', 'name': 'light.png'}]
    result = compile_scene(scene, reference_images=refs)
    assert not result.errors
    assert 'IMAGE 3 is a reference photograph of the SAME ROOM FROM ANOTHER VIEWPOINT' in result.final_prompt
    assert 'target CAD conditions take priority' in result.final_prompt
    assert 'Preserve recognizable furniture identity' in result.final_prompt
    assert 'Keep objects outside the target view out of frame' in result.final_prompt
    assert 'Discard mask colors everywhere' in result.final_prompt
    assert 'thin outlines, window frames' in result.final_prompt
    assert 'missing separate mask for a detail does not mean that detail must be deleted' in result.final_prompt
    assert 'Do not relocate explicitly out-of-view modeled objects' in result.final_prompt
    assert 'Preserve the reference window-frame finish' in result.final_prompt
    assert 'Retain the oak dining set.' in result.final_prompt
    assert 'IMAGE 4 is an appearance reference (Lighting Reference)' in result.final_prompt
    assert result.debug['image_order'][2]['type'] == SAME_ROOM_ROLE
    # Reference numbering follows the selected inputs, including instance-only requests.
    result = compile_scene(scene, types=('instance',), reference_images=refs[:1])
    assert not result.errors
    assert 'IMAGE 2 is a reference photograph' in result.final_prompt
    result = compile_scene(scene, types=('depth',), reference_images=refs[:1])
    assert any('requires the target camera' in error for error in result.errors)
    with pytest.raises(ValueError):
        validate_prompt_request(result)


def test_missing_npy_fallback_is_explicit(scene):
    (scene[2]/'generated/camera_001/instance.npy').unlink()
    result=compile_scene(scene)
    assert not result.errors and result.debug['visibility_source']=='camera_mapping_fallback'
    assert result.debug['visible_instance_ids'] is None
    assert 'invisible wardrobe' in result.final_prompt and any('fallback' in warning for warning in result.warnings)


@pytest.mark.parametrize('failure',['camera','mapping','rgb','entity','ceiling','npy','file','shape','unmapped','duplicate_rgb','float_npy'])
def test_invalid_inputs_block_requests(scene,failure):
    state,conditions,root=scene
    folder=root/'generated/camera_001'
    if failure=='camera':state['cameras']={}
    elif failure=='mapping':state['instance_mapping']={}
    elif failure=='rgb':state['instance_mapping']['camera_001']['1']['rgb']=[400,1,1]
    elif failure=='entity':del state['entities']['bed']
    elif failure=='ceiling':del state['ceiling']
    elif failure=='npy':(folder/'instance.npy').write_bytes(b'broken')
    elif failure=='file':(folder/'instance.png').unlink()
    elif failure=='shape':np.save(folder/'instance.npy',np.ones((8,8),dtype=np.uint32))
    elif failure=='unmapped':np.save(folder/'instance.npy',np.full((3,4),99,dtype=np.uint32))
    elif failure=='duplicate_rgb':state['instance_mapping']['camera_001']['2']['rgb']=[17,81,143]
    elif failure=='float_npy':np.save(folder/'instance.npy',np.ones((3,4),dtype=float))
    result=compile_scene(scene)
    assert result.errors
    with pytest.raises(ValueError):validate_prompt_request(result)


def test_palette_is_camera_specific_and_does_not_use_global_generator(scene):
    state,conditions,root=scene
    state['instance_mapping']['camera_001']['1']['rgb']=[81,122,9]
    state['instance_mapping']['camera_other']={'1':{'rgb':[1,2,3],'entity_id':'window'}}
    result=compile_scene(scene)
    assert 'RGB(81,122,9)' in result.final_prompt and 'RGB(1,2,3)' not in result.final_prompt


def test_credentials_and_spatial_hex_are_rejected_without_echo(scene):
    result=compile_scene(scene,user_design_prompt='api_key=secret-test')
    with pytest.raises(ValueError) as exc:validate_prompt_request(result)
    assert 'secret-test' not in str(exc.value)
    result=compile_scene(scene,user_design_prompt='An arbitrary-secret-value')
    with pytest.raises(ValueError):validate_prompt_request(result,secrets=['arbitrary-secret-value'])
    scene[0]['entities']['bed']['material']='#000000'
    with pytest.raises(ValueError):validate_prompt_request(compile_scene(scene))


@patch('google.genai.Client')
def test_provider_receives_exact_documented_image_order(mock_client,scene):
    state,conditions,root=scene
    selected=[c for c in conditions if c['enabled']]
    for index,c in enumerate(selected):Image.new('RGB',(4,3),(30+index,40,50)).save(c['path'])
    buffer=io.BytesIO();Image.new('RGB',(4,3),(90,80,70)).save(buffer,format='PNG')
    refs=[{'bytes':buffer.getvalue(),'role':'Material Reference'}]
    result=compile_prompt(state,'camera_001',selected,refs,'Warm oak',root)
    validate_prompt_request(result)
    response = MagicMock()
    response.candidates[0].content.parts = [MagicMock(inline_data=MagicMock(data=b'result', mime_type='image/png'))]
    mock_client.return_value.models.generate_content.return_value = response
    GeminiImageGenerator().generate(result.final_prompt, selected, refs, {'api_key': 'test-only', 'model': 'gemini-3.1-flash-image'})
    request=mock_client.return_value.models.generate_content.call_args.kwargs
    assert [image.getpixel((0,0)) for image in request['contents'][:-1]]==[(30,40,50),(31,40,50),(90,80,70)]
    assert request['contents'][-1]==result.final_prompt
    assert 'test-only' not in request['contents'][-1]
    assert 'strictly adhering to the spatial geometry, camera perspective' in request['config'].system_instruction
    assert 'CRITICAL CAMERA & PERSPECTIVE DIRECTIVE' in result.spatial_prompt
