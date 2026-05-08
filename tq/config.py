from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from .security import ensure_secure_dir, redact_token

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

_DEFAULTS = {
    "model_dir": "~/.tq/models",
    "binary_path": "llama-server",
    "port": 8080,
    "host": "127.0.0.1",
    "auto_api_key": True,
    "idle_timeout": 300,
}

CONFIG_DIR = os.path.expanduser("~/.tq")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.toml")


def _config_dir() -> str:
    return CONFIG_DIR


def _config_file() -> str:
    return CONFIG_FILE


def load_config() -> dict[str, Any]:
    path = _config_file()
    if not os.path.isfile(path):
        return dict(_DEFAULTS)

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return dict(_DEFAULTS)

    result = dict(_DEFAULTS)
    result.update(data)
    return result


def save_config(config: dict[str, Any]) -> None:
    ensure_secure_dir(_config_dir())
    path = _config_file()
    with open(path, "wb") as f:
        tomli_w.dump(config, f)
    os.chmod(path, 0o600)


def get_model_dir(config: Optional[dict] = None) -> str:
    cfg = config or load_config()
    return os.path.expanduser(cfg.get("model_dir", _DEFAULTS["model_dir"]))


def get_binary_path(config: Optional[dict] = None) -> str:
    cfg = config or load_config()
    return cfg.get("binary_path", _DEFAULTS["binary_path"])


def get_port(config: Optional[dict] = None) -> int:
    cfg = config or load_config()
    return int(cfg.get("port", _DEFAULTS["port"]))


def get_host(config: Optional[dict] = None) -> str:
    cfg = config or load_config()
    return cfg.get("host", _DEFAULTS["host"])


def set_value(key: str, value: str) -> None:
    config = load_config()

    int_keys = {"port"}
    bool_keys = {"auto_api_key"}

    if key in int_keys:
        config[key] = int(value)
    elif key in bool_keys:
        config[key] = value.lower() in ("true", "1", "yes")
    else:
        config[key] = value

    save_config(config)


def show_config(config: Optional[dict] = None) -> dict[str, str]:
    cfg = config or load_config()
    result = {}
    for k, v in cfg.items():
        if "token" in k.lower() or "key" in k.lower():
            result[k] = redact_token(str(v))
        else:
            result[k] = str(v)
    return result


def init_config() -> None:
    ensure_secure_dir(_config_dir())
    if not os.path.isfile(_config_file()):
        save_config(dict(_DEFAULTS))
