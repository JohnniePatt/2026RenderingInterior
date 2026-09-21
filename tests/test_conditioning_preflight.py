import json
import pytest
from services.conditioning_preflight import check_camera_export


@pytest.fixture
def export(tmp_path):
    camera = {'position': [6.6269884, 3.0799491, 1], 'heading_deg': 208.995955,
              'pitch_deg': 0, 'fov_deg': 95.004, 'output': {'width': 1024, 'height': 1024}}
    metadata = {'camera_id': 'camera_002', 'resolution': camera['output'],
                'camera': {**camera, 'position': [6.626988, 3.079949, 1],
                           'heading_deg': 208.996, 'fov_deg': 95.00}}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    return tmp_path, camera


def test_export_rounding_is_not_a_camera_change(export):
    folder, camera = export
    report, errors, _ = check_camera_export(folder, 'camera_002', camera)
    assert not errors and report['status'] == 'matching'


@pytest.mark.parametrize('change', ['position', 'heading_deg', 'pitch_deg', 'fov_deg', 'output'])
def test_changed_camera_blocks_paid_generation(export, change):
    folder, camera = export
    if change == 'position': camera[change] = [6.000279, 2.974027, 1]
    elif change == 'output': camera[change] = {'width': 1280, 'height': 720}
    else: camera[change] += 5
    report, errors, _ = check_camera_export(folder, 'camera_002', camera)
    assert errors and report['status'] == 'stale'


def test_invalid_metadata_fails_closed(export):
    folder, camera = export
    (folder / 'metadata.json').write_text('{}')
    assert check_camera_export(folder, 'camera_002', camera)[1]
