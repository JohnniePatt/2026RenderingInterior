"""Freeze a reviewed request locally; this module never calls an API."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from services.prompt_builder import validate_prompt_request


def safe_config(config):
    return {key: config.get(key) for key in
            ('id', 'name', 'provider', 'model', 'temperature', 'vertexai', 'aspect_ratio')}


def request_fingerprint(build, state, config):
    data = {'prompt': build.final_prompt, 'debug': build.debug, 'state': state,
            'config': safe_config(config), 'errors': build.errors}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def prepare_request(build, state, conditions, references, config):
    validate_prompt_request(build, secrets=[config.get('api_key', '')])
    selected = [c for c in conditions if c.get('enabled', True)]
    images = []
    for item, manifest in zip([*selected, *references], build.debug['image_order'], strict=True):
        data = item.get('bytes') or Path(item['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['sha256']:
            raise ValueError('An image changed during Submit. Submit again to review the current inputs.')
        images.append({**deepcopy(item), 'bytes': bytes(data)})
    return {'id': uuid4().hex, 'fingerprint': request_fingerprint(build, state, config),
            'build': deepcopy(build), 'state': deepcopy(state), 'config': safe_config(config),
            'conditions': images[:len(selected)], 'references': images[len(selected):]}
