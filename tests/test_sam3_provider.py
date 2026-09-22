"""Adapter contract tests; these do not claim pretrained SAM 3 accuracy."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image


def test_sam3_adapter_native_masks_and_cached_queries(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from services.segmentation.sam3_provider import SAM3Provider
    calls = []

    class Batch(dict):
        def to(self, device): return self

    class Model:
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, **kwargs):
            calls.append('inference')
            return SimpleNamespace(pred_masks=torch.ones(1, 1, 3, 4, 8), iou_scores=torch.tensor([[[.9,.8,.7]]]))

    class Processor:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return Batch(original_sizes=torch.tensor([[4,8]]))
        def post_process_instance_segmentation(self, outputs, **kwargs):
            assert kwargs['target_sizes'] == [[4,8]]
            return [{'masks':torch.ones(1,4,8,dtype=torch.bool), 'boxes':torch.tensor([[0,0,8,4]]), 'scores':torch.tensor([.95])}]
        def post_process_masks(self, masks, sizes): return [masks[0].bool()]

    classes = {name:getattr(transformers,name) for name in ('Sam3Model','Sam3Processor','Sam3TrackerModel','Sam3TrackerProcessor')}
    for name, value in [('Sam3Model', Model()), ('Sam3Processor', Processor()),
                        ('Sam3TrackerModel', Model()), ('Sam3TrackerProcessor', Processor())]:
        def factory(*args, _value=value, **kwargs):
            assert kwargs['local_files_only'] is True
            return _value
        monkeypatch.setattr(classes[name], 'from_pretrained', factory)
    backend = SAM3Provider(tmp_path)
    image = Image.new('RGB',(8,4))
    result = backend.detect_and_segment(image,'bed')
    result[0]['mask'][:] = False
    assert backend.detect_and_segment(image,'bed')[0]['mask'].all()
    assert calls.count('inference') == 1
    masks = backend.segment(image, [[2,2]], [[1,1]], [0,0,7,3])
    assert masks.masks.shape == (3,4,8)
    inputs = next(c for c in calls if isinstance(c,dict) and 'input_points' in c)
    assert inputs['input_points'] == [[[[2,2],[1,1]]]]
    assert inputs['input_labels'] == [[[1,0]]]
    assert inputs['input_boxes'] == [[[0,0,7,3]]]
    assert masks.backend['provider'] == 'sam3'
