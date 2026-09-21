import json
import os
from pathlib import Path
import re
import uuid

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "local_api_configs"


def get_config_dir(config_dir: Path | str | None = None) -> Path:
    target = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def mask_api_key(key: str | None) -> str:
    if not key:
        return ""
    clean = key.strip()
    if len(clean) <= 4:
        return "••••"
    return "••••••••" + clean[-4:]


def list_configs(config_dir: Path | str | None = None) -> list[dict]:
    target = get_config_dir(config_dir)
    configs = []
    for file_path in sorted(target.glob("api*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("id", file_path.stem.replace("api_", "").replace("api", ""))
                    data["masked_key"] = mask_api_key(data.get("api_key", ""))
                    data["file_path"] = str(file_path)
                    configs.append(data)
        except Exception:
            continue
    return configs


def get_config(config_id: str, config_dir: Path | str | None = None) -> dict | None:
    target = get_config_dir(config_dir)
    # Match api_{config_id}.json or api{config_id}.json
    candidates = [
        target / f"api_{config_id}.json",
        target / f"api{config_id}.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data.setdefault("id", config_id)
                    data["masked_key"] = mask_api_key(data.get("api_key", ""))
                    data["file_path"] = str(c)
                    return data
            except Exception:
                return None
    # If not found by direct name, search within contents
    for file_path in target.glob("api*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if str(data.get("id")) == str(config_id):
                    data["masked_key"] = mask_api_key(data.get("api_key", ""))
                    data["file_path"] = str(file_path)
                    return data
        except Exception:
            continue
    return None


def save_config(config_data: dict, config_dir: Path | str | None = None) -> str:
    target = get_config_dir(config_dir)
    cfg = dict(config_data)
    config_id = cfg.get("id")
    if not config_id:
        config_id = uuid.uuid4().hex[:8]
        cfg["id"] = config_id

    # Clean out non-persistent helper fields
    cfg.pop("masked_key", None)
    cfg.pop("file_path", None)

    filename = f"api_{config_id}.json"
    file_path = target / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return config_id


def delete_config(config_id: str, config_dir: Path | str | None = None) -> bool:
    target = get_config_dir(config_dir)
    deleted = False
    for c in [target / f"api_{config_id}.json", target / f"api{config_id}.json"]:
        if c.is_file():
            c.unlink()
            deleted = True
    if not deleted:
        for file_path in target.glob("api*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if str(data.get("id")) == str(config_id):
                        file_path.unlink()
                        deleted = True
                        break
            except Exception:
                continue
    return deleted


def verify_connection(config: dict) -> tuple[bool, str]:
    provider = config.get("provider", "google-genai")
    api_key = config.get("api_key", "").strip()
    model = config.get("model", "gemini-2.5-flash").strip()

    if not api_key:
        return False, "API key is missing."

    if provider == "google-genai":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            # Verify connectivity by getting model metadata or listing models
            try:
                client.models.get(model=model)
                return True, f"Connection successful! Model '{model}' is accessible."
            except Exception:
                # If specific model metadata lookup fails, try listing models to verify key validity
                pager = client.models.list(config={"page_size": 1})
                _ = next(iter(pager), None)
                return True, f"API key is valid. Model '{model}' configured."
        except Exception as exc:
            return False, f"Connection failed: {str(exc)}"

    return False, f"Unsupported provider: {provider}"


# Alias for backward compatibility, with __test__ = False so pytest doesn't collect it as a test
test_connection = verify_connection
verify_connection.__test__ = False
test_connection.__test__ = False
