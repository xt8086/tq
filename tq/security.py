from __future__ import annotations

import os
import re
import hashlib

_MODEL_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)?$')
_GGUF_EXT = ".gguf"


def validate_model_name(name: str) -> str:
    if not name or not _MODEL_NAME_RE.match(name):
        raise ValueError(f"Invalid model name: {name!r}")
    if "\x00" in name:
        raise ValueError("Model name contains null byte")
    return name


def safe_model_path(model_id: str, base_dir: str) -> str:
    validate_model_name(model_id)
    safe_name = model_id.replace("/", "__")
    full_path = os.path.join(base_dir, safe_name)
    resolved = os.path.realpath(full_path)
    base_resolved = os.path.realpath(base_dir)
    if not resolved.startswith(base_resolved + os.sep) and resolved != base_resolved:
        raise ValueError(f"Path traversal detected in model ID: {model_id}")
    return resolved


def validate_gguf_path(path: str, model_dir: str) -> str:
    resolved = os.path.realpath(path)
    model_dir_resolved = os.path.realpath(model_dir)
    if not resolved.startswith(model_dir_resolved + os.sep):
        raise ValueError(f"Path escapes model directory: {path}")
    if not resolved.endswith(_GGUF_EXT):
        raise ValueError(f"Not a GGUF file: {path}")
    if not os.path.isfile(resolved):
        raise ValueError(f"File not found: {path}")
    return resolved


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def secure_write_json(path: str, data: str) -> None:
    dir_path = os.path.dirname(path)
    if os.path.islink(path):
        raise RuntimeError(f"Refusing to write: {path} is a symlink")
    tmp_path = path + ".tmp"
    try:
        fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except FileExistsError:
        os.unlink(tmp_path)
        raise


def ensure_secure_dir(path: str) -> None:
    if os.path.exists(path):
        st = os.stat(path)
        if st.st_mode & 0o077:
            os.chmod(path, 0o700)
        return
    os.makedirs(path, mode=0o700, exist_ok=True)


def redact_token(token: str) -> str:
    if len(token) <= 8:
        return "****"
    return token[:3] + "****" + token[-4:]
