import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / 'local_segmentation' / 'config.json'


def load_config():
    if CONFIG_PATH.is_file():
        return json.loads(CONFIG_PATH.read_text())
    return {'provider': 'sam', 'model': 'vit_b', 'device': 'cpu',
            'checkpoint': str(Path.home() / '.cache/cad-rendering/sam/sam_vit_b_01ec64.pth')}


def save_config(config):
    if config.get('provider') != 'sam' or config.get('model') not in ('vit_b', 'vit_l', 'vit_h'):
        raise ValueError('Unsupported SAM configuration.')
    if config.get('device') not in ('cpu', 'cuda'):
        raise ValueError('Unsupported device.')
    if not Path(config['checkpoint']).expanduser().is_file():
        raise ValueError('Checkpoint file does not exist.')
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    tmp = CONFIG_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(config, indent=2))
    os.replace(tmp, CONFIG_PATH)
