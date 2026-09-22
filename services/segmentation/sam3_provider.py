"""Local Transformers SAM 3 concept detector and SAM 3 visual tracker adapter."""
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from threading import RLock
import json
import numpy as np
from .base import SegmentationBackend, SegmentationResult, validate_prompts

CONFIG = Path(__file__).resolve().parents[2] / 'local_segmentation/sam3.json'


def load_config():
    if CONFIG.is_file():
        return json.loads(CONFIG.read_text())
    return {'provider': 'sam3', 'model': 'facebook/sam3', 'device': 'cpu',
            'checkpoint': str(Path.home()/'.cache/cad-rendering/sam3'), 'detection_threshold': .3,
            'strong_threshold': .8, 'assignment_margin': .05}


def save_config(config):
    from services.spatial_correspondence import atomic_write
    if config.get('provider') != 'sam3' or config.get('device') not in ('cpu', 'cuda'):
        raise ValueError('Invalid SAM 3 settings.')
    for name in ('detection_threshold', 'strong_threshold', 'assignment_margin'):
        if not 0 <= float(config[name]) <= 1:
            raise ValueError('Thresholds must be between 0 and 1.')
    if config['detection_threshold'] > config['strong_threshold']:
        raise ValueError('Detection threshold cannot exceed strong-match threshold.')
    atomic_write(CONFIG, json.dumps(config, indent=2).encode())


class SAM3Provider(SegmentationBackend):
    def __init__(self, checkpoint, device='cpu', threshold=.3):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.path = Path(checkpoint)
        self.device, self.threshold = device, threshold
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use CPU or install compatible CUDA PyTorch.')
        if device == 'cpu': torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = Sam3Model.from_pretrained(str(self.path), local_files_only=True).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(str(self.path), local_files_only=True)
        self.lock, self.tracker, self.tracker_processor = RLock(), None, None
        self.cache, self.image_hash = {}, None
        files = {}
        for path in sorted(self.path.glob('*.safetensors')):
            h = sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
            files[path.name] = h.hexdigest()
        self.info = {'provider': 'sam3', 'model': 'facebook/sam3', 'implementation': 'transformers',
                     'package_version': version('transformers'), 'checkpoint_sha256': files,
                     'device': device, 'detection_threshold': threshold, 'mask_threshold': .5}
        revision = self.path / 'source_revision.json'
        if revision.is_file():
            self.info['source_revision'] = json.loads(revision.read_text())

    def detect_and_segment(self, image, concept):
        import torch
        key = sha256(image.convert('RGB').tobytes()+str(image.size).encode()).hexdigest()
        with self.lock, torch.inference_mode():
            if key != self.image_hash:
                self.cache, self.image_hash = {}, key
            if concept not in self.cache:
                inputs = self.processor(images=image.convert('RGB'), text=concept, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                data = self.processor.post_process_instance_segmentation(outputs, threshold=self.threshold,
                    mask_threshold=.5, target_sizes=inputs['original_sizes'].tolist())[0]
                candidates = []
                for mask, box, score in zip(data['masks'], data['boxes'], data['scores']):
                    mask = mask.detach().cpu().numpy().astype(bool)
                    if mask.shape != (image.height, image.width): raise ValueError('SAM 3 mask resolution mismatch.')
                    if mask.any():
                        candidates.append({'mask': mask, 'bbox': box.detach().cpu().tolist(),
                                           'score': float(score), 'concept': concept})
                self.cache[concept] = candidates
            return [{**c, 'mask': c['mask'].copy()} for c in self.cache[concept]]

    def segment(self, image, positive_points, negative_points=None, box=None):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        validate_prompts(image.size, positive_points, negative_points, box)
        if not positive_points and box is None: raise ValueError('Add a positive point or box.')
        with self.lock, torch.inference_mode():
            if self.tracker is None:
                self.tracker = Sam3TrackerModel.from_pretrained(str(self.path), local_files_only=True).to(self.device).eval()
                self.tracker_processor = Sam3TrackerProcessor.from_pretrained(str(self.path), local_files_only=True)
            kwargs = {}
            points = positive_points + (negative_points or [])
            if points:
                kwargs.update(input_points=[[points]], input_labels=[[[1]*len(positive_points)+[0]*len(negative_points or [])]])
            if box: kwargs['input_boxes'] = [[box]]
            inputs = self.tracker_processor(images=image.convert('RGB'), return_tensors='pt', **kwargs).to(self.device)
            outputs = self.tracker(**inputs, multimask_output=True)
            masks = self.tracker_processor.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes'])[0][0]
            masks = masks.numpy().astype(bool)
            if masks.shape[1:] != (image.height, image.width): raise ValueError('SAM 3 tracker resolution mismatch.')
            return SegmentationResult(masks, outputs.iou_scores[0, 0].cpu().tolist(),
                {**self.info, 'component': 'Sam3TrackerModel'})


@lru_cache(maxsize=1)
def _load(path, device, threshold, stamp):
    return SAM3Provider(path, device, threshold)


def get_backend(config):
    path = Path(config['checkpoint']).expanduser().resolve()
    if not (path/'config.json').is_file() or not list(path.glob('*.safetensors')):
        raise ValueError('SAM 3 weights are not installed. Obtain access to facebook/sam3 and run scripts/setup_sam3.py, or select an existing local checkpoint folder. No inference was run.')
    stamp = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(path.iterdir()) if p.is_file())
    return _load(str(path), config['device'], config['detection_threshold'], stamp)
