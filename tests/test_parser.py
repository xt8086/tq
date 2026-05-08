import struct
import pytest
from tq.parser import (
    parse_gguf_metadata, build_model_metadata, GGUFParserError,
    _Reader, GGUF_MAGIC, GGUF_VERSION_3,
    GGUF_TYPE_UINT32, GGUF_TYPE_STRING, GGUF_TYPE_ARRAY, GGUF_TYPE_INT32,
)
from tq.types import QuantType


def _make_gguf_header(kv_pairs=None):
    if kv_pairs is None:
        kv_pairs = {}

    buf = bytearray()
    buf.extend(struct.pack("<I", GGUF_MAGIC))
    buf.extend(struct.pack("<I", GGUF_VERSION_3))
    buf.extend(struct.pack("<Q", 0))  # tensor_count
    buf.extend(struct.pack("<Q", len(kv_pairs)))

    for key, (vtype, value) in kv_pairs.items():
        key_bytes = key.encode("utf-8")
        buf.extend(struct.pack("<Q", len(key_bytes)))
        buf.extend(key_bytes)
        buf.extend(struct.pack("<I", vtype))

        if vtype == GGUF_TYPE_UINT32:
            buf.extend(struct.pack("<I", value))
        elif vtype == GGUF_TYPE_INT32:
            buf.extend(struct.pack("<i", value))
        elif vtype == GGUF_TYPE_STRING:
            val_bytes = value.encode("utf-8")
            buf.extend(struct.pack("<Q", len(val_bytes)))
            buf.extend(val_bytes)
        elif vtype == GGUF_TYPE_ARRAY:
            elem_type = value[0]
            elements = value[1]
            buf.extend(struct.pack("<I", elem_type))
            buf.extend(struct.pack("<Q", len(elements)))
            for elem in elements:
                if elem_type == GGUF_TYPE_STRING:
                    eb = elem.encode("utf-8")
                    buf.extend(struct.pack("<Q", len(eb)))
                    buf.extend(eb)
                elif elem_type == GGUF_TYPE_UINT32:
                    buf.extend(struct.pack("<I", elem))

    return bytes(buf)


class TestGGUFParser:
    def test_valid_empty_metadata(self):
        data = _make_gguf_header()
        meta = parse_gguf_metadata(data)
        assert isinstance(meta, dict)
        assert len(meta) == 0

    def test_single_string_field(self):
        data = _make_gguf_header({
            "general.name": (GGUF_TYPE_STRING, "test-model"),
        })
        meta = parse_gguf_metadata(data)
        assert meta["general.name"] == "test-model"

    def test_single_uint32_field(self):
        data = _make_gguf_header({
            "general.file_type": (GGUF_TYPE_UINT32, 16),
        })
        meta = parse_gguf_metadata(data)
        assert meta["general.file_type"] == 16

    def test_multiple_fields(self):
        data = _make_gguf_header({
            "general.architecture": (GGUF_TYPE_STRING, "llama"),
            "general.name": (GGUF_TYPE_STRING, "test"),
            "llama.context_length": (GGUF_TYPE_UINT32, 4096),
        })
        meta = parse_gguf_metadata(data)
        assert meta["general.architecture"] == "llama"
        assert meta["general.name"] == "test"
        assert meta["llama.context_length"] == 4096

    def test_array_field(self):
        data = _make_gguf_header({
            "tokenizer.ggml.tokens": (GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, ["<s>", "</s>", "<unk>"])),
        })
        meta = parse_gguf_metadata(data)
        assert meta["tokenizer.ggml.tokens"] == ["<s>", "</s>", "<unk>"]

    def test_invalid_magic(self):
        buf = struct.pack("<I", 0xDEADBEEF) + b"\x00" * 100
        with pytest.raises(GGUFParserError, match="Invalid GGUF magic"):
            parse_gguf_metadata(buf)

    def test_truncated_data(self):
        data = _make_gguf_header({
            "general.name": (GGUF_TYPE_STRING, "test"),
        })
        with pytest.raises(GGUFParserError):
            parse_gguf_metadata(data[:10])


class TestBuildModelMetadata:
    def test_build_from_metadata(self):
        data = _make_gguf_header({
            "general.architecture": (GGUF_TYPE_STRING, "llama"),
            "general.name": (GGUF_TYPE_STRING, "test-model"),
            "llama.context_length": (GGUF_TYPE_UINT32, 8192),
            "llama.block_count": (GGUF_TYPE_UINT32, 32),
        })
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(data)
            f.flush()
            path = f.name

        try:
            meta = build_model_metadata(path)
            assert meta.architecture == "llama"
            assert meta.context_length == 8192
            assert meta.num_layers == 32
        finally:
            os.unlink(path)

    def test_quant_detection_from_filename(self):
        import tempfile
        import os
        data = _make_gguf_header()
        with tempfile.NamedTemporaryFile(suffix="-Q4_K_M.gguf", delete=False) as f:
            f.write(data)
            f.flush()
            path = f.name

        try:
            meta = build_model_metadata(path)
            assert meta.quant_type == QuantType.Q4_K_M
        finally:
            os.unlink(path)