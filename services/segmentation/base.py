from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SegmentationResult:
    masks: np.ndarray
    quality: list
    backend: dict


def validate_prompts(size, positive_points, negative_points=None, box=None):
    width, height = size
    for points in (positive_points, negative_points or []):
        for point in points:
            if len(point) != 2 or not np.isfinite(point).all() or not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError('Point is outside original image coordinates.')
    if box is not None:
        if len(box) != 4 or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError('Box must be a nonempty rectangle in original image coordinates.')


class SegmentationBackend(ABC):
    @abstractmethod
    def segment(self, image, positive_points, negative_points=None, box=None):
        """Return candidate masks at original resolution; never assign a semantic label."""
