import io
import json
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from services.image_generation.base import GenerationResult
from test_prompt_builder import scene


def render_page(root, state):
    from pathlib import Path
    from app.ui.generate_image import render
    render(Path(root), state)


def test_prompt_ui_recompiles_selection_and_saves_metadata(scene, monkeypatch):
    from app.ui import generate_image as ui
    state, conditions, root = scene
    cfg={'id':'test','name':'Mock configuration','api_key':'test-private-value','model':'gemini-3.1-flash-image','temperature':.7}
    monkeypatch.setattr(ui,'list_configs',lambda:[cfg])
    monkeypatch.setattr(ui,'get_config',lambda _:cfg)
    output=io.BytesIO();Image.new('RGB',(8,6),(128,128,128)).save(output,format='PNG')
    requests=[]
    def generate(self, **kwargs):
        requests.append(kwargs)
        return GenerationResult(output.getvalue(),'image/png','STOP',{'model':cfg['model']})
    monkeypatch.setattr(ui.GeminiImageGenerator,'generate',generate)
    app=AppTest.from_function(render_page,kwargs={'root':str(root),'state':state}).run(timeout=15)
    assert not app.exception
    text=lambda label:next(w for w in app.text_area if w.label==label)
    assert text('Spatial / Condition Prompt').disabled
    assert text('Final Prompt').disabled
    text('Appearance Prompt').set_value('Warm oak and natural daylight').run()
    assert 'Warm oak and natural daylight' in text('Final Prompt').value
    assert 'Warm oak and natural daylight' not in text('Spatial / Condition Prompt').value
    # Disable depth; stale editable final prompts must not survive a condition change.
    next(w for w in app.checkbox if w.key==f'inc_camera_001_depth_{root.name}').uncheck().run()
    assert 'depth' not in text('Final Prompt').value.lower()
    assert next(w for w in app.button if w.label=='Confirm & Generate').disabled
    next(w for w in app.button if w.label=='Submit — Prepare Summary').click().run(timeout=15)
    assert requests == []
    next(w for w in app.button if w.label=='Confirm & Generate').click().run(timeout=15)
    assert not app.exception and not app.error
    assert len(requests)==1 and requests[0]['prompt']==text('Final Prompt').value
    assert [c['type'] for c in requests[0]['condition_images']]==['instance']
    record=json.loads(next((root/'generated/camera_001/renders').glob('*.json')).read_text())
    assert record['camera']==state['cameras']['camera_001']
    assert record['instance_mapping']==state['instance_mapping']['camera_001']
    assert record['output_resolution']=={'width':8,'height':6}
    assert record['appearance_prompt']=='Warm oak and natural daylight'
    assert record['spatial_prompt']+'\n\nAPPEARANCE PROMPT\n'+record['appearance_prompt']==record['final_prompt']
    assert cfg['api_key'] not in json.dumps(record)
    assert record['confirmed_request_id']
    assert all(isinstance(c['bytes'], bytes) for c in requests[0]['condition_images'])
    app.run()
    assert next(w for w in app.button if w.label == 'Confirm & Generate').disabled
    assert len(requests) == 1


def test_conflicts_warn_and_missing_files_disable_generation(scene,monkeypatch):
    from app.ui import generate_image as ui
    state, _, root=scene
    monkeypatch.setattr(ui,'list_configs',lambda:[])
    state['entities']['bed'].update(category='sofa',description='dinning table')
    app=AppTest.from_function(render_page,kwargs={'root':str(root),'state':state}).run(timeout=15)
    assert any('Semantic conflict' in warning.value for warning in app.warning)
    (root/'generated/camera_001/instance.npy').write_bytes(b'invalid')
    app.run()
    assert any('corrupt' in error.value for error in app.error)
    assert next(w for w in app.button if w.label=='Confirm & Generate').disabled


def test_same_room_role_updates_prompt_and_requires_instance(scene, monkeypatch):
    from app.ui import generate_image as ui
    from services.prompt_resolution import SAME_ROOM_ROLE
    state, _, root = scene
    upload = io.BytesIO()
    Image.new('RGB', (8, 6)).save(upload, format='PNG')
    upload.name = 'previous-room.png'
    monkeypatch.setattr(ui.st, 'file_uploader', lambda *args, **kwargs: [upload])
    monkeypatch.setattr(ui, 'list_configs', lambda: [])
    app = AppTest.from_function(render_page, kwargs={'root': str(root), 'state': state}).run(timeout=15)
    role = next(w for w in app.selectbox if w.label == 'Reference #1 Role')
    assert SAME_ROOM_ROLE in role.options
    role.select(SAME_ROOM_ROLE).run()
    assert not app.exception and not app.error
    final = lambda: next(w.value for w in app.text_area if w.label == 'Final Prompt')
    assert 'SAME ROOM FROM ANOTHER VIEWPOINT' in final()
    next(w for w in app.checkbox if w.key == f'inc_camera_001_instance_{root.name}').uncheck().run()
    assert any('requires the target camera' in error.value for error in app.error)
    assert next(w for w in app.button if w.label == 'Confirm & Generate').disabled
    next(w for w in app.selectbox if w.label == 'Reference #1 Role').select('Appearance / Style Reference').run()
    assert not app.exception and not app.error
    assert 'SAME ROOM FROM ANOTHER VIEWPOINT' not in final()


def test_stale_camera_export_disables_generation(scene, monkeypatch):
    from app.ui import generate_image as ui
    state, _, root = scene
    camera = state['cameras']['camera_001']
    camera.update(heading_deg=0, pitch_deg=0)
    exported = {**camera, 'position': [999, 200, 1500]}
    (root / 'generated/camera_001/metadata.json').write_text(json.dumps({
        'camera_id': 'camera_001', 'camera': exported,
        'resolution': {'width': 1024, 'height': 1024}}))
    monkeypatch.setattr(ui, 'list_configs', lambda: [])
    app = AppTest.from_function(render_page, kwargs={'root': str(root), 'state': state}).run(timeout=15)
    assert not app.exception
    assert any('do not match the selected camera' in error.value for error in app.error)
    assert next(w for w in app.button if w.label == 'Confirm & Generate').disabled


@pytest.mark.parametrize('change', ['prompt', 'condition', 'image_file', 'model'])
def test_changes_invalidate_submitted_request(scene, monkeypatch, change):
    from app.ui import generate_image as ui
    state, _, root = scene
    cfg = {'id': 'mock', 'model': 'test', 'api_key': 'private', 'temperature': .7}
    monkeypatch.setattr(ui, 'list_configs', lambda: [cfg])
    monkeypatch.setattr(ui, 'get_config', lambda _: cfg)
    def forbidden(*args, **kwargs):
        raise AssertionError('Submit and edits must never call the provider')
    monkeypatch.setattr(ui.GeminiImageGenerator, 'generate', forbidden)
    app = AppTest.from_function(render_page, kwargs={'root': str(root), 'state': state}).run(timeout=15)
    next(w for w in app.button if w.label == 'Submit — Prepare Summary').click().run()
    assert not app.exception
    assert not next(w for w in app.button if w.label == 'Confirm & Generate').disabled
    if change == 'prompt':
        next(w for w in app.text_area if w.label == 'Appearance Prompt').set_value('Oak').run()
    elif change == 'condition':
        next(w for w in app.checkbox if w.key == f'inc_camera_001_depth_{root.name}').uncheck().run()
    elif change == 'image_file':
        Image.new('RGB', (4, 3), 'white').save(root / 'generated/camera_001/depth.png')
        app.run()
    else:
        cfg['model'] = 'different-model'
        app.run()
    assert not app.exception
    assert next(w for w in app.button if w.label == 'Confirm & Generate').disabled
    assert any('Inputs changed' in item.value for item in app.info)
