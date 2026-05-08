from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

from .security import validate_model_name, safe_model_path, sha256_file

HASHES_FILE = os.path.join(os.path.expanduser("~/.tq"), "hashes.json")


def search_models(query: str, limit: int = 20) -> list[dict]:
    api = HfApi()
    results = []

    try:
        models = api.list_models(
            search=f"{query} GGUF",
            sort="downloads",
            limit=limit * 3,
        )

        seen = set()
        for model in models:
            model_id = model.id
            if model_id in seen:
                continue
            seen.add(model_id)

            try:
                files = list_repo_files(model_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if not gguf_files:
                    continue
            except Exception:
                continue

            results.append({
                "id": model_id,
                "downloads": getattr(model, "downloads", 0),
                "gguf_files": gguf_files[:5],
                "total_gguf_files": len(gguf_files),
            })

            if len(results) >= limit:
                break
    except Exception as e:
        raise RuntimeError(f"Search failed: {e}")

    return results


def download_model(
    model_id: str,
    filename: Optional[str] = None,
    model_dir: str = "~/models",
    verify_hash: bool = True,
) -> str:
    validate_model_name(model_id)

    model_dir = os.path.expanduser(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    if filename is None:
        try:
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {model_id}")
            q4_files = [f for f in gguf_files if "Q4_K_M" in f]
            filename = (q4_files + gguf_files)[0]
        except Exception as e:
            raise RuntimeError(f"Failed to list files: {e}")

    local_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=model_dir,
    )

    if verify_hash:
        _verify_or_store_hash(local_path, model_id, filename)

    return local_path


def _verify_or_store_hash(filepath: str, model_id: str, filename: str) -> None:
    hashes = _load_hashes()
    key = f"{model_id}/{filename}"

    current_hash = sha256_file(filepath)

    if key in hashes:
        expected = hashes[key]
        if current_hash != expected:
            os.unlink(filepath)
            raise RuntimeError(
                f"Hash mismatch for {key}!\n"
                f"Expected: {expected}\n"
                f"Got:      {current_hash}\n"
                f"File removed for safety."
            )
    else:
        hashes[key] = current_hash
        _save_hashes(hashes)


def _load_hashes() -> dict:
    if not os.path.isfile(HASHES_FILE):
        return {}
    try:
        with open(HASHES_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_hashes(hashes: dict) -> None:
    os.makedirs(os.path.dirname(HASHES_FILE), exist_ok=True)
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)
    os.chmod(HASHES_FILE, 0o600)