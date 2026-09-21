import io
from pathlib import Path
import time
from typing import Any
from PIL import Image

from services.image_generation.base import BaseImageGenerator, GenerationResult


class GeminiImageGenerator(BaseImageGenerator):
    """Image generator implementation using Google GenAI SDK (google-genai)."""

    def generate(
        self,
        prompt: str,
        condition_images: list[dict[str, Any]],
        reference_images: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> GenerationResult:
        api_key = config.get("api_key", "").strip()
        if not api_key:
            raise ValueError("API Key is required to generate images.")

        model = config.get("model", "gemini-2.5-flash").strip()
        start_time = time.time()

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # 1. Check if Vertex AI mode is explicitly configured for legacy Imagen
        if model.lower().startswith("imagen") and config.get("vertexai", False):
            gen_config = types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio=config.get("aspect_ratio", "16:9"),
            )
            response = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=gen_config,
            )
            elapsed = time.time() - start_time

            if not response.generated_images:
                raise RuntimeError("No images were returned by the Imagen model.")

            img_bytes = response.generated_images[0].image.image_bytes
            return GenerationResult(
                image_bytes=img_bytes,
                mime_type="image/png",
                finish_reason="SUCCESS",
                metadata={
                    "model": model,
                    "provider": "google-genai",
                    "latency_seconds": round(elapsed, 3),
                    "condition_count": len(condition_images),
                    "reference_count": len(reference_images),
                },
            )

        # 2. Multimodal & Native Image models via generate_content
        # (e.g. gemini-2.5-flash-image, gemini-3.1-flash-image, gemini-2.5-flash, gemini-2.5-pro)
        contents = []

        # Preserve the image numbering established by the prompt compiler.
        active_conds = [c for c in condition_images if c.get("enabled", True)]
        for index, item in enumerate([*active_conds, *reference_images], 1):
            source = io.BytesIO(item['bytes']) if item.get('bytes') else item.get('path')
            try:
                with Image.open(source) as pil_img:
                    pil_img.load()
                    contents.append(pil_img.copy())
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f'IMAGE {index} is missing or unreadable. No generation request was sent.') from exc

        # 3. Append the prompt text after all visual inputs
        contents.append(prompt)

        # Determine target aspect ratio from config or primary condition image
        target_aspect_ratio = config.get("aspect_ratio")
        if not target_aspect_ratio and contents and isinstance(contents[0], Image.Image):
            ratio = contents[0].width / max(contents[0].height, 1)
            targets = [
                ("1:1", 1.0),
                ("4:3", 4 / 3),
                ("3:4", 3 / 4),
                ("16:9", 16 / 9),
                ("9:16", 9 / 16),
                ("3:2", 3 / 2),
                ("2:3", 2 / 3),
                ("21:9", 21 / 9),
            ]
            target_aspect_ratio = min(targets, key=lambda x: abs(ratio - x[1]))[0]

        # Setup generation config with image response modality and aspect ratio
        gen_config_kwargs = {
            "temperature": float(config.get("temperature", 0.7)),
            "system_instruction": (
                "You are an expert architectural interior rendering AI. "
                "Your highest priority is strictly adhering to the spatial geometry, camera perspective, "
                "and object positions defined by the primary CAD condition maps (Proxy, Depth and Instance maps). "
                "Never allow reference images to alter the target camera angle or spatial composition."
            ),
        }
        if target_aspect_ratio:
            try:
                gen_config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=target_aspect_ratio)
            except Exception:
                pass

        gen_config = types.GenerateContentConfig(**gen_config_kwargs)
        # Attempt requesting IMAGE modality if available
        try:
            gen_config.response_modalities = ["IMAGE", "TEXT"]
        except Exception:
            pass

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            err_str = str(exc)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "limit: 0" in err_str:
                raise RuntimeError(
                    f"Google AI Studio Quota Error (429 RESOURCE_EXHAUSTED):\n"
                    f"Free-tier API keys have a quota limit of 0 for image generation models.\n\n"
                    f"To enable image generation, please link a Pay-as-you-go billing account to your project in Google AI Studio / Google Cloud Console:\n"
                    f"https://ai.google.dev/pricing"
                ) from exc
            raise

        elapsed = time.time() - start_time

        # Inspect response for image parts
        image_bytes = None
        mime_type = "image/png"
        text_parts = []

        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = str(candidate.finish_reason)
            for part in candidate.content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    mime_type = part.inline_data.mime_type or "image/png"
                    break
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        else:
            finish_reason = "NO_CANDIDATES"

        if not image_bytes:
            # If the model returned text instead of an image, provide clear diagnostics
            returned_text = " ".join(text_parts).strip()
            msg = (
                f"Model '{model}' responded with text instead of an image.\n"
                f"Response snippet: {returned_text[:300]}...\n\n"
                f"Tip: Select an image generation capable model such as 'gemini-2.5-flash-image' or 'gemini-3.1-flash-image'."
            )
            raise RuntimeError(msg)

        return GenerationResult(
            image_bytes=image_bytes,
            mime_type=mime_type,
            finish_reason=finish_reason,
            metadata={
                "model": model,
                "provider": "google-genai",
                "latency_seconds": round(elapsed, 3),
                "condition_count": len(condition_images),
                "reference_count": len(reference_images),
            },
        )
