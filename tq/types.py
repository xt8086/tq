from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class GpuVendor(enum.Enum):
    APPLE = "apple"
    NVIDIA = "nvidia"
    AMD = "amd"
    UNKNOWN = "unknown"


class QuantType(enum.Enum):
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"
    F32 = "F32"
    IQ3_M = "IQ3_M"
    IQ3_S = "IQ3_S"
    IQ4_XS = "IQ4_XS"
    UNKNOWN = "UNKNOWN"


class CacheType(enum.Enum):
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    TURBO3 = "turbo3"
    TURBO4 = "turbo4"
    F16 = "f16"


class ToolSupport(enum.Enum):
    OPENAI = "openai"
    NONE = "none"


@dataclass
class ModelMetadata:
    name: str
    path: str
    size_bytes: int
    architecture: Optional[str] = None
    context_length: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    embedding_length: Optional[int] = None
    vocab_size: Optional[int] = None
    quant_type: QuantType = QuantType.UNKNOWN
    is_moe: bool = False
    expert_count: Optional[int] = None
    chat_template: Optional[str] = None
    gguf_metadata: dict = field(default_factory=dict)
    mmproj_path: Optional[str] = None
    is_multimodal: bool = False
    tool_support: ToolSupport = ToolSupport.NONE

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def display_name(self) -> str:
        return self.path.rsplit("/", 1)[-1] if "/" in self.path else self.path


@dataclass
class HardwareProfile:
    gpu_vendor: GpuVendor
    gpu_name: str
    vram_bytes: int
    ram_bytes: int
    is_apple_silicon: bool = False
    apple_chip_generation: Optional[str] = None

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / (1024 ** 3)

    @property
    def unified_memory_gb(self) -> float:
        if self.is_apple_silicon:
            return self.ram_gb
        return self.vram_gb


@dataclass
class TQRecommendation:
    cache_type_k: CacheType
    cache_type_v: CacheType
    boundary_v: bool = False
    boundary_v_layers: int = 2
    sparse_v: bool = False
    context_length: Optional[int] = None
    reasoning: list[str] = field(default_factory=list)

    def to_flags(self) -> list[str]:
        flags = [
            "-ctk", self.cache_type_k.value,
            "-ctv", self.cache_type_v.value,
        ]
        if self.context_length is not None:
            flags.extend(["-c", str(self.context_length)])
        return flags


@dataclass
class ServerConfig:
    model_path: str
    port: int = 8080
    host: str = "127.0.0.1"
    tq: Optional[TQRecommendation] = None
    extra_flags: list[str] = field(default_factory=list)
    api_key: Optional[str] = None
    idle_timeout: int = 300
    mmproj_path: Optional[str] = None
    tool_support: str = "none"

    def to_command(self, binary_path: str) -> list[str]:
        cmd = [binary_path, "-m", self.model_path, "--port", str(self.port), "--host", self.host]
        if self.tq:
            cmd.extend(self.tq.to_flags())
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        if self.mmproj_path:
            cmd.extend(["--mmproj", self.mmproj_path])
        cmd.extend(self.extra_flags)
        return cmd


@dataclass
class ServerState:
    pid: int
    port: int
    host: str
    model_path: str
    binary_path: str
    session_token: str
    started_at: float
    idle_timeout: int = 300
    tool_support: str = "none"
    is_multimodal: bool = False
