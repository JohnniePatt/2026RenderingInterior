import pytest
from pathlib import Path
from services.api_manager import (
    mask_api_key,
    list_configs,
    get_config,
    save_config,
    delete_config,
    verify_connection,
)


def test_mask_api_key():
    assert mask_api_key("") == ""
    assert mask_api_key(None) == ""
    assert mask_api_key("123") == "••••"
    assert mask_api_key("abcd") == "••••"
    assert mask_api_key("AIzaSyD-1234567890ABCDEF") == "••••••••CDEF"
    assert mask_api_key("secret_api_key_test") == "••••••••test"


def test_config_crud_lifecycle(tmp_path):
    # Initially empty
    configs = list_configs(tmp_path)
    assert len(configs) == 0

    # Save new config
    cfg_data = {
        "name": "Test Gemini Config",
        "provider": "google-genai",
        "model": "gemini-2.5-flash",
        "api_key": "AIzaSyD_Secret_Key_9999",
        "temperature": 0.5,
    }
    cfg_id = save_config(cfg_data, tmp_path)
    assert cfg_id is not None
    assert len(cfg_id) > 0

    # Verify list
    configs = list_configs(tmp_path)
    assert len(configs) == 1
    assert configs[0]["id"] == cfg_id
    assert configs[0]["name"] == "Test Gemini Config"
    assert configs[0]["masked_key"] == "••••••••9999"

    # Verify get_config
    loaded = get_config(cfg_id, tmp_path)
    assert loaded is not None
    assert loaded["id"] == cfg_id
    assert loaded["api_key"] == "AIzaSyD_Secret_Key_9999"
    assert loaded["model"] == "gemini-2.5-flash"

    # Update config
    loaded["name"] = "Updated Gemini Config"
    loaded["temperature"] = 0.8
    save_config(loaded, tmp_path)

    reloaded = get_config(cfg_id, tmp_path)
    assert reloaded["name"] == "Updated Gemini Config"
    assert reloaded["temperature"] == 0.8
    assert reloaded["api_key"] == "AIzaSyD_Secret_Key_9999"

    # Delete config
    assert delete_config(cfg_id, tmp_path) is True
    assert get_config(cfg_id, tmp_path) is None
    assert len(list_configs(tmp_path)) == 0


def test_connection_validation():
    # Missing key
    ok, msg = verify_connection({"provider": "google-genai", "api_key": ""})
    assert ok is False
    assert "missing" in msg.lower()

    # Unsupported provider
    ok, msg = verify_connection({"provider": "unknown_provider", "api_key": "dummy"})
    assert ok is False
    assert "unsupported" in msg.lower()
