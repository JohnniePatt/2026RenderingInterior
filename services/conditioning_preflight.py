"""Local camera/export consistency checks. No model or network calls."""
import json
import math
from pathlib import Path


def check_camera_export(folder, camera_id, camera):
    path = Path(folder) / 'metadata.json'
    report = {'status': 'unverified', 'differences': []}
    errors, warnings = [], []
    if not path.is_file():
        warnings.append('Export metadata.json is missing; camera freshness cannot be verified. Re-export conditioning maps before a paid generation.')
        return report, errors, warnings
    try:
        metadata = json.loads(path.read_text(encoding='utf-8'))
        exported = metadata['camera']
        if metadata['camera_id'] != camera_id:
            report['differences'].append('camera_id')
        # The exporter rounds position to 6 decimals, angles to 4, FOV to 2.
        for key, tolerance in [('position', 1e-6), ('heading_deg', 1e-4),
                               ('pitch_deg', 1e-4), ('fov_deg', .0051)]:
            old, new = exported[key], camera[key]
            old = old if isinstance(old, list) else [old]
            new = new if isinstance(new, list) else [new]
            if len(old) != len(new) or not all(
                math.isfinite(float(a)) and math.isfinite(float(b)) and
                (abs((float(a)-float(b)+180) % 360-180) if key == 'heading_deg'
                 else abs(float(a)-float(b))) <= tolerance for a, b in zip(old, new)
            ):
                report['differences'].append(key)
        output = camera.get('output', {'width': 1024, 'height': 1024})
        if any(metadata['resolution'][key] != output[key] for key in ('width', 'height')):
            report['differences'].append('output_resolution')
        report['exported_camera'] = exported
        report['current_camera'] = {key: camera.get(key) for key in
                                    ('position', 'heading_deg', 'pitch_deg', 'fov_deg', 'output')}
        report['status'] = 'stale' if report['differences'] else 'matching'
        if report['differences']:
            errors.append('Conditioning maps do not match the selected camera (' +
                          ', '.join(report['differences']) + '). Re-export Conditioning (PNG & NPY) in Camera Mode before generation. This is a local operation; no image-generation API is needed.')
    except (OSError, ValueError, KeyError, TypeError):
        report['status'] = 'invalid'
        errors.append('Export metadata.json is invalid; re-export conditioning maps before generation.')
    return report, errors, warnings
