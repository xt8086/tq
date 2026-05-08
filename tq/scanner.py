from __future__ import annotations

import os
from typing import Optional

from .types import ModelMetadata
from .parser import build_model_metadata, read_gguf_metadata_from_file, GGUFParserError
from .security import validate_gguf_path


def scan_models(model_dir: str, system_wide: bool = False) -> list[ModelMetadata]:
    seen_paths = set()
    results = []

    dirs = [model_dir]
    if system_wide:
        dirs = _system_search_dirs(model_dir)

    for d in dirs:
        if not os.path.isdir(d):
            continue
        for entry in _walk_gguf_files(d):
            resolved = os.path.realpath(entry)
            if resolved in seen_paths:
                continue
            basename = os.path.basename(entry).lower()
            if basename.startswith("mmproj-") or basename.startswith("mmproj_"):
                continue
            seen_paths.add(resolved)
            try:
                meta = build_model_metadata(entry)
            except (GGUFParserError, OSError):
                meta = ModelMetadata(
                    name=os.path.basename(entry),
                    path=entry,
                    size_bytes=os.path.getsize(entry),
                )
            mmproj = _find_mmproj(entry)
            if mmproj:
                meta.mmproj_path = mmproj
                meta.is_multimodal = True
            results.append(meta)

    results.sort(key=lambda m: m.name.lower())
    return results


def _system_search_dirs(model_dir: str) -> list[str]:
    home = os.path.expanduser("~")
    dirs = [model_dir]

    common = [
        os.path.join(home, ".lmstudio", "models"),
        os.path.join(home, ".cache", "huggingface", "hub"),
        os.path.join(home, ".cache", "lm-studio", "models"),
        os.path.join(home, "models"),
        os.path.join(home, "llama.cpp", "models"),
        os.path.join(home, ".local", "share", "models"),
        "/opt/homebrew/share/models",
        "/usr/local/share/models",
    ]

    for d in common:
        if d not in dirs and os.path.isdir(d):
            dirs.append(d)

    lmstudio_hub = os.path.join(home, ".lmstudio", "hub", "models")
    if os.path.isdir(lmstudio_hub):
        dirs.append(lmstudio_hub)

    ollama_dir = os.path.join(home, ".ollama", "models")
    if os.path.isdir(ollama_dir):
        dirs.append(ollama_dir)

    return dirs


def find_model(model_dir: str, query: str) -> Optional[ModelMetadata]:
    query_lower = query.lower()
    gguf_files = list(_walk_gguf_files(model_dir))

    exact = [f for f in gguf_files if os.path.basename(f) == query]
    if exact:
        return _load_or_stub(exact[0])

    stem_match = [f for f in gguf_files if os.path.splitext(os.path.basename(f))[0] == query]
    if stem_match:
        return _load_or_stub(stem_match[0])

    partial = [f for f in gguf_files if query_lower in os.path.basename(f).lower()]
    if partial:
        return _load_or_stub(partial[0])

    return None


def resolve_model_path(model_dir: str, query: str) -> Optional[str]:
    if os.path.isfile(query) and query.endswith(".gguf"):
        return query

    if query.isdigit():
        idx = int(query) - 1
        models = scan_models(model_dir, system_wide=True)
        if 0 <= idx < len(models):
            return models[idx].path
        return None

    query_lower = query.lower()
    for d in _system_search_dirs(model_dir):
        if not os.path.isdir(d):
            continue
        for f in _walk_gguf_files(d):
            basename = os.path.basename(f)
            if basename == query or os.path.splitext(basename)[0] == query:
                return f
        for f in _walk_gguf_files(d):
            basename = os.path.basename(f)
            if query_lower in basename.lower():
                return f
    return None


def _walk_gguf_files(directory: str):
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".gguf"):
                yield os.path.join(root, fname)


def _load_or_stub(path: str) -> ModelMetadata:
    try:
        meta = build_model_metadata(path)
    except (GGUFParserError, OSError):
        meta = ModelMetadata(
            name=os.path.basename(path),
            path=path,
            size_bytes=os.path.getsize(path),
        )
    mmproj = _find_mmproj(path)
    if mmproj:
        meta.mmproj_path = mmproj
        meta.is_multimodal = True
    return meta


def _find_mmproj(model_path: str) -> Optional[str]:
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).lower()
    model_name = model_name.replace(".gguf", "").replace("-q4_k_m", "").replace("-q5_k_m", "").replace("-q4_0", "").replace("-q5_0", "").replace("-q8_0", "").replace("-f16", "").replace("-f32", "")

    for f in _walk_gguf_files(model_dir):
        basename = os.path.basename(f).lower()
        if not (basename.startswith("mmproj-") or basename.startswith("mmproj_")):
            continue
        return f

    parent_dir = os.path.dirname(model_dir)
    for f in _walk_gguf_files(parent_dir):
        basename = os.path.basename(f).lower()
        if not (basename.startswith("mmproj-") or basename.startswith("mmproj_")):
            continue
        mmproj_dir = os.path.dirname(f)
        if model_dir.startswith(mmproj_dir) or mmproj_dir.startswith(model_dir):
            return f

    return None
