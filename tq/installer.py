from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Optional

RELEASES_API = "https://api.github.com/repos/TheTom/llama-cpp-turboquant/releases/latest"
BIN_DIR = os.path.expanduser("~/.tq/bin")


def get_platform_tag() -> Optional[str]:
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        return "macos-arm64-metal"

    if system == "Windows" and machine.endswith("64"):
        if _has_nvidia():
            return "windows-x64-cuda"
        return None

    if system == "Linux":
        if _has_nvidia():
            return "linux-x64-cuda"
        if _has_amd():
            return "linux-x64-rocm"
        return "linux-x64-cpu"

    return None
        if _has_nvidia():
            return "linux-x64-cuda"
        if _has_amd():
            return "linux-x64-rocm"
        return "linux-x64-cpu"

    return None


def _has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def _has_amd() -> bool:
    return shutil.which("rocm-smi") is not None or os.path.exists("/dev/kfd")


def get_available_releases() -> list[dict]:
    import json

    try:
        req = urllib.request.Request(RELEASES_API)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        assets = []
        for asset in data.get("assets", []):
            assets.append({
                "name": asset["name"],
                "url": asset["browser_download_url"],
                "size": asset["size"],
            })
        return assets
    except Exception as e:
        raise RuntimeError(f"Failed to fetch releases: {e}")


def find_asset_for_platform(assets: list[dict], platform_tag: str) -> Optional[dict]:
    for asset in assets:
        name = asset["name"].lower()
        if platform_tag in name:
            return asset
    return None


def install_binary(force: bool = False) -> str:
    platform_tag = get_platform_tag()
    if not platform_tag:
        raise RuntimeError(
            f"Unsupported platform: {platform.system()} {platform.machine()}. "
            f"Only macOS arm64, Linux x86_64, and Windows x86_64 are currently supported."
        )

    os.makedirs(BIN_DIR, exist_ok=True)

    existing = _find_installed_binary()
    if existing and not force:
        return existing

    print(f"Platform: {platform_tag}")
    print("Fetching available releases...")

    assets = get_available_releases()
    asset = find_asset_for_platform(assets, platform_tag)
    if not asset:
        available = [a["name"] for a in assets]
        raise RuntimeError(
            f"No binary found for {platform_tag}. "
            f"Available: {', '.join(available)}"
        )

    print(f"Downloading {asset['name']} ({asset['size'] / 1024 / 1024:.0f} MB)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, asset["name"])
        urllib.request.urlretrieve(asset["url"], archive_path)
        print("Download complete. Extracting...")

        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)

        if asset["name"].endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_dir)
        elif asset["name"].endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)
        else:
            raise RuntimeError(f"Unknown archive format: {asset['name']}")

        server_path = _find_llama_server(extract_dir)
        if not server_path:
            raise RuntimeError("llama-server binary not found in archive")

        dest_dir = os.path.join(BIN_DIR, asset["name"].rsplit(".", 2)[0])
        os.makedirs(dest_dir, exist_ok=True)

        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, extract_dir)
                dst = os.path.join(dest_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        new_path = os.path.join(dest_dir, "llama-server")
        if not os.path.isfile(new_path):
            new_path = _find_llama_server(dest_dir)

        if not new_path:
            raise RuntimeError("llama-server not found after extraction")

        os.chmod(new_path, 0o755)

        print(f"Installed to: {new_path}")
        print(f"Set binary path with: tq config set binary_path {new_path}")

        return new_path


def _find_llama_server(directory: str) -> Optional[str]:
    names = ["llama-server.exe", "llama-server"] if platform.system() == "Windows" else ["llama-server"]
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f in names:
                return os.path.join(root, f)
    return None


def _find_installed_binary() -> Optional[str]:
    from .server import _find_binary
    try:
        return _find_binary("")
    except FileNotFoundError:
        return None
