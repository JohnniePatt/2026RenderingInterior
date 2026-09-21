from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    image_bytes: bytes
    mime_type: str = "image/png"
    finish_reason: str = "STOP"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseImageGenerator(ABC):
    """Abstract base class for architectural AI image generators."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        condition_images: list[dict[str, Any]],
        reference_images: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> GenerationResult:
        """Execute multimodal image generation and return the result.

        Args:
            prompt: Assembled natural-language prompt.
            condition_images: List of condition dicts with 'path', 'role', 'type', etc.
            reference_images: List of reference dicts with 'bytes' or 'path', 'role', 'description'.
            config: Local API config dict (provider, model, api_key, parameters).

        Returns:
            GenerationResult containing generated image bytes and execution metadata.
        """
        pass
