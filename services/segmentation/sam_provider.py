from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version, distribution
import json
from pathlib import Path
from threading import RLock
import numpy as np
from .base import SegmentationBackend, SegmentationResult, validate_prompts


class SAMProvider(SegmentationBackend):
    def __init__(self, model, checkpoint, device='cpu'):
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this environment. Select CPU in Local SAM settings.')
        if device == 'cpu':
            torch.set_num_threads(min(4, torch.get_num_threads()))
        path = Path(checkpoint)
        self.predictor = SamPredictor(sam_model_registry[model](checkpoint=str(path)).to(device).eval())
        self.lock = RLock()
        self.image_digest = None
        self.info = {'provider': 'sam', 'model': model, 'package_version': version('segment-anything'),
                     'checkpoint_sha256': sha256(path.read_bytes()).hexdigest(), 'device': device}
        source = distribution('segment-anything').read_text('direct_url.json')
        if source:
            self.info['source_commit'] = json.loads(source).get('vcs_info', {}).get('commit_id')

    def segment(self, image, positive_points, negative_points=None, box=None):
        import torch
        rgb = np.asarray(image.convert('RGB'))
        validate_prompts(image.size, positive_points, negative_points, box)
        negatives = negative_points or []
        if not positive_points and box is None:
            raise ValueError('Add a positive object point or a box first.')
        points = positive_points + negatives
        digest = sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
        # Predictor image embeddings are mutable; serialize image set + prediction across sessions.
        with self.lock, torch.inference_mode():
            if digest != self.image_digest:
                self.predictor.set_image(rgb)
                self.image_digest = digest
            masks, scores, _ = self.predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32) if points else None,
                point_labels=np.asarray([1]*len(positive_points) + [0]*len(negatives)) if points else None,
                box=np.asarray(box, dtype=np.float32) if box else None, multimask_output=True)
        if masks.shape[1:] != (image.height, image.width):
            raise ValueError('SAM returned a mask at the wrong resolution.')
        return SegmentationResult(masks.astype(bool), [float(s) for s in scores], dict(self.info))


@lru_cache(maxsize=2)
def _load(model, checkpoint, device, mtime, size):
    return SAMProvider(model, checkpoint, device)


def get_backend(config):
    if config.get('provider') != 'sam':
        raise ValueError('Unsupported segmentation backend.')
    path = Path(config['checkpoint']).expanduser().resolve()
    if not path.is_file():
        raise ValueError('SAM checkpoint is missing. Configure a local checkpoint in SAM settings.')
    stat = path.stat()
    return _load(config['model'], str(path), config.get('device', 'cpu'), stat.st_mtime_ns, stat.st_size)
