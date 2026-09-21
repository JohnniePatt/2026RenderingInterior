from PIL import Image
import pytest
from services.prepared_request import prepare_request
from test_prompt_builder import scene, compile_scene


def test_snapshot_freezes_bytes_and_excludes_credentials(scene):
    state, conditions, root = scene
    build = compile_scene(scene)
    request = prepare_request(build, state, conditions, [], {'api_key': 'private', 'model': 'test'})
    data = request['conditions'][0]['bytes']
    Image.new('RGB', (4, 3), 'white').save(root / 'generated/camera_001/depth.png')
    state['cameras']['camera_001']['name'] = 'Changed'
    assert request['conditions'][0]['bytes'] == data
    assert request['state']['cameras']['camera_001']['name'] == 'Test'
    assert 'api_key' not in request['config']


def test_submit_rejects_image_changed_since_preflight(scene):
    state, conditions, root = scene
    build = compile_scene(scene)
    Image.new('RGB', (4, 3), 'white').save(root / 'generated/camera_001/depth.png')
    with pytest.raises(ValueError, match='changed during Submit'):
        prepare_request(build, state, conditions, [], {})
