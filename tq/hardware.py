from __future__ import annotations

import platform
import subprocess
import os
from typing import Optional

import psutil

from .types import HardwareProfile, GpuVendor


def detect_hardware() -> HardwareProfile:
    ram_bytes = psutil.virtual_memory().total
    gpu_vendor, gpu_name, vram_bytes, is_silicon, chip_gen = _detect_gpu()
    return HardwareProfile(
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        vram_bytes=vram_bytes,
        ram_bytes=ram_bytes,
        is_apple_silicon=is_silicon,
        apple_chip_generation=chip_gen,
    )


def _detect_gpu():
    system = platform.system()

    if system == "Darwin":
        return _detect_macos()
    elif system == "Linux":
        return _detect_linux()
    elif system == "Windows":
        return _detect_windows()
    else:
        return GpuVendor.UNKNOWN, "Unknown", 0, False, None


def _detect_macos():
    gpu_name = "Unknown Apple Silicon"
    is_silicon = True
    chip_gen = None
    vram = 0

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        cpu_brand = result.stdout.strip()

        result2 = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result2.returncode == 0:
            import json
            data = json.loads(result2.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                chip = displays[0].get("sppci_model", displays[0].get("_name", ""))
                gpu_name = chip or cpu_brand
                vr = displays[0].get("spdis_vram", "")
                if isinstance(vr, str) and vr:
                    vram = _parse_memory_string(vr)
    except Exception:
        pass

    if "Apple" in gpu_name:
        is_silicon = True
        chip_gen = _apple_chip_generation(gpu_name)

    vram = vram or psutil.virtual_memory().total

    return GpuVendor.APPLE, gpu_name, vram, is_silicon, chip_gen


def _detect_linux():
    nvidia = _detect_nvidia()
    if nvidia:
        return nvidia

    amd = _detect_amd()
    if amd:
        return amd

    return GpuVendor.UNKNOWN, "Unknown", 0, False, None


def _detect_nvidia():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if parts else "NVIDIA GPU"
            vram_mb = int(parts[1]) if len(parts) > 1 else 0
            return GpuVendor.NVIDIA, name, vram_mb * 1024 * 1024, False, None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _detect_amd():
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                parts = lines[1].split(",")
                name = parts[-1].strip() if parts else "AMD GPU"
                return GpuVendor.AMD, name, 0, False, None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if os.path.exists("/dev/kfd"):
        return GpuVendor.AMD, "AMD GPU (ROCm)", 0, False, None

    return None


def _detect_windows():
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if len(lines) > 1:
                parts = lines[1].split()
                name = parts[0] if parts else "Unknown"
                vram = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
                vendor = GpuVendor.NVIDIA if "NVIDIA" in name.upper() else (
                    GpuVendor.AMD if "AMD" in name.upper() or "RADEON" in name.upper() else GpuVendor.UNKNOWN
                )
                return vendor, name, vram, False, None
    except Exception:
        pass
    return GpuVendor.UNKNOWN, "Unknown", 0, False, None


def _parse_memory_string(s: str) -> int:
    s = s.strip().upper()
    if "GB" in s:
        return int(float(s.replace("GB", "").strip()) * 1024 ** 3)
    elif "MB" in s:
        return int(float(s.replace("MB", "").strip()) * 1024 ** 2)
    return 0


def _apple_chip_generation(name: str) -> Optional[str]:
    name_upper = name.upper()
    if "M4" in name_upper:
        return "m4"
    elif "M3" in name_upper:
        return "m3"
    elif "M2" in name_upper:
        return "m2"
    elif "M1" in name_upper:
        return "m1"
    return None
