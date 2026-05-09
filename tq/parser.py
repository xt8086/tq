from __future__ import annotations

import struct
from typing import Optional

from .types import ModelMetadata, QuantType, ToolSupport

GGUF_MAGIC = 0x46554747
GGUF_VERSION_3 = 3

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

MAX_METADATA_KEYS = 1000
MAX_STRING_LEN = 1024 * 1024
MAX_ARRAY_LEN = 200000

_QUANT_MAP = {
    "Q2_K": QuantType.Q2_K,
    "Q2_K_S": QuantType.Q2_K_S,
    "Q3_K_S": QuantType.Q3_K_S,
    "Q3_K_M": QuantType.Q3_K_M,
    "Q3_K_L": QuantType.Q3_K_L,
    "Q4_K_S": QuantType.Q4_K_S,
    "Q4_K_M": QuantType.Q4_K_M,
    "Q5_K_S": QuantType.Q5_K_S,
    "Q5_K_M": QuantType.Q5_K_M,
    "Q6_K": QuantType.Q6_K,
    "Q8_0": QuantType.Q8_0,
    "F16": QuantType.F16,
    "F32": QuantType.F32,
    "IQ3_M": QuantType.IQ3_M,
    "IQ3_S": QuantType.IQ3_S,
    "IQ4_XS": QuantType.IQ4_XS,
}


class GGUFParserError(Exception):
    pass


class _Reader:
    def __init__(self, data: bytes, offset: int = 0):
        self.data = data
        self.offset = offset

    def _remaining(self) -> int:
        return len(self.data) - self.offset

    def read_bytes(self, n: int) -> bytes:
        if self._remaining() < n:
            raise GGUFParserError(f"Unexpected end of data at offset {self.offset}, need {n}")
        result = self.data[self.offset:self.offset + n]
        self.offset += n
        return result

    def read_uint8(self) -> int:
        return struct.unpack("<B", self.read_bytes(1))[0]

    def read_int8(self) -> int:
        return struct.unpack("<b", self.read_bytes(1))[0]

    def read_uint16(self) -> int:
        return struct.unpack("<H", self.read_bytes(2))[0]

    def read_int16(self) -> int:
        return struct.unpack("<h", self.read_bytes(2))[0]

    def read_uint32(self) -> int:
        return struct.unpack("<I", self.read_bytes(4))[0]

    def read_int32(self) -> int:
        return struct.unpack("<i", self.read_bytes(4))[0]

    def read_uint64(self) -> int:
        return struct.unpack("<Q", self.read_bytes(8))[0]

    def read_int64(self) -> int:
        return struct.unpack("<q", self.read_bytes(8))[0]

    def read_float32(self) -> float:
        return struct.unpack("<f", self.read_bytes(4))[0]

    def read_float64(self) -> float:
        return struct.unpack("<d", self.read_bytes(8))[0]

    def read_string(self) -> str:
        length = self.read_uint64()
        if length > MAX_STRING_LEN:
            raise GGUFParserError(f"String too long: {length}")
        raw = self.read_bytes(length)
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return raw.decode("latin-1")

    def read_value(self, vtype: int):
        readers = {
            GGUF_TYPE_UINT8: self.read_uint8,
            GGUF_TYPE_INT8: self.read_int8,
            GGUF_TYPE_UINT16: self.read_uint16,
            GGUF_TYPE_INT16: self.read_int16,
            GGUF_TYPE_UINT32: self.read_uint32,
            GGUF_TYPE_INT32: self.read_int32,
            GGUF_TYPE_FLOAT32: self.read_float32,
            GGUF_TYPE_BOOL: self.read_uint8,
            GGUF_TYPE_STRING: self.read_string,
            GGUF_TYPE_UINT64: self.read_uint64,
            GGUF_TYPE_INT64: self.read_int64,
            GGUF_TYPE_FLOAT64: self.read_float64,
        }
        reader = readers.get(vtype)
        if reader is None:
            raise GGUFParserError(f"Unknown value type: {vtype}")
        return reader()

    def read_array(self):
        elem_type = self.read_uint32()
        length = self.read_uint64()
        if length > MAX_ARRAY_LEN:
            self._skip_array(elem_type, length)
            return None
        return [self.read_value(elem_type) for _ in range(length)]

    def _skip_array(self, elem_type: int, length: int) -> None:
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(length):
                slen = self.read_uint64()
                self.read_bytes(slen)
        elif elem_type in (GGUF_TYPE_ARRAY,):
            for _ in range(length):
                inner_type = self.read_uint32()
                inner_len = self.read_uint64()
                self._skip_array(inner_type, inner_len)
        else:
            elem_sizes = {
                GGUF_TYPE_UINT8: 1, GGUF_TYPE_INT8: 1,
                GGUF_TYPE_UINT16: 2, GGUF_TYPE_INT16: 2,
                GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4,
                GGUF_TYPE_FLOAT32: 4, GGUF_TYPE_BOOL: 1,
                GGUF_TYPE_UINT64: 8, GGUF_TYPE_INT64: 8,
                GGUF_TYPE_FLOAT64: 8,
            }
            size = elem_sizes.get(elem_type, 0)
            if size > 0:
                self.read_bytes(size * length)
            else:
                raise GGUFParserError(f"Cannot skip unknown array element type: {elem_type}")


def parse_gguf_metadata(data: bytes) -> dict:
    reader = _Reader(data)

    magic = reader.read_uint32()
    if magic != GGUF_MAGIC:
        raise GGUFParserError(f"Invalid GGUF magic: {magic:#010x}")

    version = reader.read_uint32()
    if version < 2 or version > GGUF_VERSION_3:
        raise GGUFParserError(f"Unsupported GGUF version: {version}")

    tensor_count = reader.read_uint64()
    _ = tensor_count
    metadata_kv_count = reader.read_uint64()

    if metadata_kv_count > MAX_METADATA_KEYS:
        raise GGUFParserError(f"Too many metadata keys: {metadata_kv_count}")

    metadata = {}
    for _ in range(metadata_kv_count):
        key = reader.read_string()
        value_type = reader.read_uint32()
        if value_type == GGUF_TYPE_ARRAY:
            value = reader.read_array()
        else:
            value = reader.read_value(value_type)
        metadata[key] = value

    return metadata


def read_gguf_metadata_from_file(path: str) -> dict:
    with open(path, "rb") as f:
        header_data = f.read(8 * 1024 * 1024)

    metadata = parse_gguf_metadata(header_data)

    if _needs_more_data(metadata, header_data):
        with open(path, "rb") as f:
            header_data = f.read(32 * 1024 * 1024)
        metadata = parse_gguf_metadata(header_data)

    return metadata


def _needs_more_data(metadata: dict, data: bytes) -> bool:
    if not metadata:
        return True
    reader = _Reader(data)
    try:
        magic = reader.read_uint32()
        version = reader.read_uint32()
        reader.read_uint64()
        kv_count = reader.read_uint64()
        for _ in range(kv_count):
            reader.read_string()
            vtype = reader.read_uint32()
            if vtype == GGUF_TYPE_ARRAY:
                reader.read_array()
            else:
                reader.read_value(vtype)
        return False
    except GGUFParserError:
        return True


def build_model_metadata(path: str, raw_metadata: Optional[dict] = None) -> ModelMetadata:
    import os

    if raw_metadata is None:
        raw_metadata = read_gguf_metadata_from_file(path)

    size_bytes = os.path.getsize(path)

    arch = raw_metadata.get("general.architecture")
    context_length = _get_int(raw_metadata, f"{arch}.context_length") if arch else None
    num_layers = _get_int(raw_metadata, f"{arch}.block_count") if arch else None
    num_heads = _get_int(raw_metadata, f"{arch}.attention.head_count") if arch else None
    num_kv_heads = _get_int(raw_metadata, f"{arch}.attention.head_count_kv") if arch else None
    embedding_length = _get_int(raw_metadata, f"{arch}.embedding_length") if arch else None
    vocab_size = _get_int(raw_metadata, "tokenizer.ggml.tokens", as_len=True)
    chat_template = raw_metadata.get("tokenizer.chat_template")

    expert_count = _get_int(raw_metadata, f"{arch}.expert_count") if arch else None
    is_moe = expert_count is not None and expert_count > 1

    name = raw_metadata.get("general.name", os.path.basename(path))
    quant_type = _detect_quant_type(path, raw_metadata)

    return ModelMetadata(
        name=name,
        path=path,
        size_bytes=size_bytes,
        architecture=arch,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embedding_length=embedding_length,
        vocab_size=vocab_size,
        quant_type=quant_type,
        is_moe=is_moe,
        expert_count=expert_count,
        chat_template=chat_template,
        gguf_metadata=raw_metadata,
        tool_support=_detect_tool_support(arch, chat_template, name),
    )


def _get_int(metadata: dict, key: str, as_len: bool = False) -> Optional[int]:
    val = metadata.get(key)
    if val is None:
        return None
    if as_len and isinstance(val, list):
        return len(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    return None


def _detect_quant_type(path: str, metadata: dict) -> QuantType:
    import os

    filename = os.path.basename(path).upper()

    for qname, qtype in _QUANT_MAP.items():
        if qname in filename:
            return qtype

    file_type = metadata.get("general.file_type")
    if isinstance(file_type, int):
        ft_map = {
            1: QuantType.F32, 2: QuantType.F16,
            10: QuantType.Q8_0,
            11: QuantType.Q2_K, 12: QuantType.Q3_K_S,
            13: QuantType.Q3_K_M, 14: QuantType.Q3_K_L,
            15: QuantType.Q4_K_S, 16: QuantType.Q4_K_M,
            17: QuantType.Q5_K_S, 18: QuantType.Q5_K_M,
            19: QuantType.Q6_K,
        }
        return ft_map.get(file_type, QuantType.UNKNOWN)

    return QuantType.UNKNOWN


_TOOL_CAPABLE_ARCHES = {
    "gemma2", "gemma3", "gemma",
    "qwen2", "qwen3",
    "llama", "llama3", "llama4",
    "mistral", "mixtral",
    "phi3", "phi4",
    "command-r",
    "chatglm", "glm4",
    "deepseek2", "deepseek3",
    "claude",
}

_SMALL_MODEL_PATTERNS = (
    "ministral",
    "gemma-2-2b", "gemma-2-1b",
    "phi-1",
    "smollm",
    "tinyllama",
    "qwen2-0.5b", "qwen2-1.5b",
    "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b",
    "qwen3-0.6b", "qwen3-1.7b",
    "qwen3.5-0.8b", "qwen3.5-1.5b", "qwen3.5-2b",
)


def _detect_tool_support(arch: Optional[str], chat_template: Optional[str], model_name: str) -> ToolSupport:
    template = chat_template or ""
    has_template_tools = "tool_calls" in template and ("function" in template or "tool_call" in template.lower())

    name_lower = model_name.lower().replace("_", " ").replace("-", " ")
    is_small = any(p.replace("-", " ") in name_lower for p in _SMALL_MODEL_PATTERNS)

    if is_small:
        return ToolSupport.NONE

    if has_template_tools:
        return ToolSupport.OPENAI

    if arch and arch.lower() in _TOOL_CAPABLE_ARCHES:
        return ToolSupport.OPENAI

    return ToolSupport.NONE
