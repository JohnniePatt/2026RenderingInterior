import io
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
from services.image_generation.base import BaseImageGenerator, GenerationResult
from services.image_generation.gemini_provider import GeminiImageGenerator


def test_generation_result_dataclass():
    dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    res = GenerationResult(
        image_bytes=dummy_bytes,
        mime_type="image/png",
        finish_reason="STOP",
        metadata={"latency_seconds": 1.25},
    )
    assert res.image_bytes == dummy_bytes
    assert res.mime_type == "image/png"
    assert res.finish_reason == "STOP"
    assert res.metadata["latency_seconds"] == 1.25


def test_gemini_missing_api_key():
    gen = GeminiImageGenerator()
    with pytest.raises(ValueError, match="API Key is required"):
        gen.generate(
            prompt="A living room",
            condition_images=[],
            reference_images=[],
            config={"api_key": ""},
        )


@patch('google.genai.Client')
@pytest.mark.parametrize('reference', [False, True])
def test_unreadable_input_never_sends_paid_request(client, tmp_path, reference):
    item = {'path': str(tmp_path / 'missing.png')}
    with pytest.raises(ValueError, match='No generation request was sent'):
        GeminiImageGenerator().generate('test', [] if reference else [item],
                                        [item] if reference else [], {'api_key': 'fake'})
    client.return_value.models.generate_content.assert_not_called()


@patch("google.genai.Client")
def test_gemini_imagen_dispatch(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock response from generate_images
    dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRdummy"
    mock_image = MagicMock()
    mock_image.image_bytes = dummy_png
    mock_gen_img = MagicMock()
    mock_gen_img.image = mock_image
    mock_response = MagicMock()
    mock_response.generated_images = [mock_gen_img]
    mock_client.models.generate_images.return_value = mock_response

    gen = GeminiImageGenerator()
    config = {
        "provider": "google-genai",
        "model": "imagen-3.0-generate-002",
        "api_key": "AIzaSy_fake_test_key",
        "aspect_ratio": "16:9",
        "vertexai": True,
    }
    result = gen.generate(
        prompt="Photorealistic living room",
        condition_images=[],
        reference_images=[],
        config=config,
    )

    assert result.image_bytes == dummy_png
    assert result.finish_reason == "SUCCESS"
    assert result.metadata["model"] == "imagen-3.0-generate-002"
    mock_client.models.generate_images.assert_called_once()


@patch("google.genai.Client")
def test_gemini_multimodal_dispatch(mock_client_class, tmp_path):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Create dummy condition image
    cond_path = tmp_path / "proxy.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(cond_path)

    # Create dummy reference bytes
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), color="green").save(buf, format="PNG")
    ref_bytes = buf.getvalue()

    # Mock response from generate_content with inline_data
    dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRmultimodal"
    mock_part = MagicMock()
    mock_part.inline_data.data = dummy_png
    mock_part.inline_data.mime_type = "image/png"
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_candidate.content.parts = [mock_part]
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    gen = GeminiImageGenerator()
    config = {
        "provider": "google-genai",
        "model": "gemini-2.5-flash",
        "api_key": "AIzaSy_fake_test_key",
        "temperature": 0.7,
    }
    result = gen.generate(
        prompt="Modern interior rendering",
        condition_images=[{"type": "proxy", "role": "Spatial Condition", "path": str(cond_path), "enabled": True}],
        reference_images=[{"name": "ref.png", "bytes": ref_bytes, "role": "Style Reference", "description": "Nordic"}],
        config=config,
    )

    assert result.image_bytes == dummy_png
    assert result.finish_reason == "STOP"
    mock_client.models.generate_content.assert_called_once()


@patch("google.genai.Client")
def test_gemini_multimodal_text_only_error(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock response containing text part only
    mock_part = MagicMock()
    mock_part.inline_data = None
    mock_part.text = "I am a text model and cannot directly generate images without tools."
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_candidate.content.parts = [mock_part]
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    gen = GeminiImageGenerator()
    config = {
        "provider": "google-genai",
        "model": "gemini-2.5-flash",
        "api_key": "AIzaSy_fake_test_key",
    }
    with pytest.raises(RuntimeError, match="responded with text instead of an image"):
        gen.generate(
            prompt="Modern interior",
            condition_images=[],
            reference_images=[],
            config=config,
        )
