import pytest
from tq.security import validate_model_name, safe_model_path, redact_token, validate_gguf_path
from tq.types import QuantType, CacheType, TQRecommendation


class TestValidateModelName:
    def test_valid_simple(self):
        assert validate_model_name("my-model") == "my-model"

    def test_valid_with_org(self):
        assert validate_model_name("org/model-name") == "org/model-name"

    def test_valid_with_dots(self):
        assert validate_model_name("model.v2") == "model.v2"

    def test_reject_path_traversal(self):
        with pytest.raises(ValueError):
            validate_model_name("../../etc/passwd")

    def test_reject_empty(self):
        with pytest.raises(ValueError):
            validate_model_name("")

    def test_reject_null_byte(self):
        with pytest.raises(ValueError):
            validate_model_name("model\x00evil")

    def test_reject_spaces(self):
        with pytest.raises(ValueError):
            validate_model_name("my model")

    def test_reject_shell_injection(self):
        with pytest.raises(ValueError):
            validate_model_name("model; rm -rf /")


class TestSafeModelPath:
    def test_simple_name(self, tmp_path):
        path = safe_model_path("my-model", str(tmp_path))
        assert path.endswith("my-model")
        assert path.startswith(str(tmp_path))

    def test_org_model_name(self, tmp_path):
        path = safe_model_path("org/model", str(tmp_path))
        assert path.endswith("org__model")

    def test_reject_traversal(self, tmp_path):
        with pytest.raises(ValueError):
            safe_model_path("../../etc/passwd", str(tmp_path))


class TestRedactToken:
    def test_short_token(self):
        assert redact_token("abc") == "****"

    def test_long_token(self):
        result = redact_token("hf_abcdefghijklmnop1234567890")
        assert result.startswith("hf_")
        assert result.endswith("7890")
        assert "****" in result


class TestQuantType:
    def test_from_filename_q4km(self):
        from tq.parser import _detect_quant_type
        result = _detect_quant_type("llama-2-7b-Q4_K_M.gguf", {})
        assert result == QuantType.Q4_K_M

    def test_from_filename_q8(self):
        from tq.parser import _detect_quant_type
        result = _detect_quant_type("model-Q8_0.gguf", {})
        assert result == QuantType.Q8_0

    def test_from_filename_f16(self):
        from tq.parser import _detect_quant_type
        result = _detect_quant_type("model-F16.gguf", {})
        assert result == QuantType.F16

    def test_from_filename_unknown(self):
        from tq.parser import _detect_quant_type
        result = _detect_quant_type("model.gguf", {})
        assert result == QuantType.UNKNOWN

    def test_from_metadata_file_type(self):
        from tq.parser import _detect_quant_type
        result = _detect_quant_type("model.gguf", {"general.file_type": 16})
        assert result == QuantType.Q4_K_M


class TestTQRecommendation:
    def test_to_flags_basic(self):
        rec = TQRecommendation(
            cache_type_k=CacheType.Q8_0,
            cache_type_v=CacheType.TURBO4,
        )
        flags = rec.to_flags()
        assert "-ctk" in flags
        assert "q8_0" in flags
        assert "-ctv" in flags
        assert "turbo4" in flags

    def test_to_flags_with_boundary_v(self):
        rec = TQRecommendation(
            cache_type_k=CacheType.TURBO4,
            cache_type_v=CacheType.TURBO4,
            boundary_v=True,
            boundary_v_layers=2,
        )
        flags = rec.to_flags()
        assert "-ctk" in flags
        assert "-ctv" in flags

    def test_to_flags_with_context(self):
        rec = TQRecommendation(
            cache_type_k=CacheType.Q8_0,
            cache_type_v=CacheType.TURBO4,
            context_length=8192,
        )
        flags = rec.to_flags()
        assert "-c" in flags
        assert "8192" in flags