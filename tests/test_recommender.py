import pytest
from tq.recommender import recommend, _infer_quant_from_name, _compute_safe_context
from tq.types import ModelMetadata, HardwareProfile, GpuVendor, QuantType, CacheType


def _make_model(**overrides):
    defaults = dict(
        name="test-model-Q4_K_M.gguf",
        path="/models/test-model-Q4_K_M.gguf",
        size_bytes=4_500_000_000,
        quant_type=QuantType.Q4_K_M,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embedding_length=4096,
        context_length=4096,
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def _make_hw(**overrides):
    defaults = dict(
        gpu_vendor=GpuVendor.APPLE,
        gpu_name="Apple M2 Pro",
        vram_bytes=16 * 1024 ** 3,
        ram_bytes=16 * 1024 ** 3,
        is_apple_silicon=True,
        apple_chip_generation="m2",
    )
    defaults.update(overrides)
    return HardwareProfile(**defaults)


class TestRecommendQ4KM:
    def test_q4km_default(self):
        model = _make_model(quant_type=QuantType.Q4_K_M)
        hw = _make_hw()
        rec = recommend(model, hw)
        assert rec.cache_type_k == CacheType.Q8_0
        assert rec.cache_type_v == CacheType.TURBO4
        assert rec.boundary_v is True

    def test_q4km_large_model_not_prem5(self):
        model = _make_model(quant_type=QuantType.Q4_K_M, size_bytes=20_000_000_000)
        hw = _make_hw(apple_chip_generation="m3")
        rec = recommend(model, hw)
        assert rec.cache_type_k == CacheType.TURBO3

    def test_q4km_large_model_prem5(self):
        model = _make_model(quant_type=QuantType.Q4_K_M, size_bytes=20_000_000_000)
        hw = _make_hw(apple_chip_generation="m2")
        rec = recommend(model, hw)
        assert rec.cache_type_k == CacheType.Q8_0


class TestRecommendQ8:
    def test_q8_symmetric_turbo4(self):
        model = _make_model(quant_type=QuantType.Q8_0)
        hw = _make_hw()
        rec = recommend(model, hw)
        assert rec.cache_type_k == CacheType.TURBO4
        assert rec.cache_type_v == CacheType.TURBO4


class TestRecommendMoE:
    def test_moe_enables_sparse_v(self):
        model = _make_model(is_moe=True, expert_count=8)
        hw = _make_hw()
        rec = recommend(model, hw)
        assert rec.sparse_v is True


class TestRecommendBoundaryV:
    def test_boundary_v_disabled_for_small_model(self):
        model = _make_model(quant_type=QuantType.Q4_K_M, num_layers=2)
        hw = _make_hw()
        rec = recommend(model, hw)
        assert rec.boundary_v is False

    def test_boundary_v_enabled_for_low_quant(self):
        model = _make_model(quant_type=QuantType.Q3_K_M)
        hw = _make_hw()
        rec = recommend(model, hw)
        assert rec.boundary_v is True


class TestContextCapping:
    def test_caps_to_safe_context(self):
        model = _make_model(context_length=131072)
        hw = _make_hw(ram_bytes=8 * 1024 ** 3)
        rec = recommend(model, hw)
        assert rec.context_length <= 131072
        assert rec.context_length >= 512


class TestInferQuantFromName:
    def test_q4km(self):
        assert _infer_quant_from_name("llama-Q4_K_M.gguf") == QuantType.Q4_K_M

    def test_unknown(self):
        assert _infer_quant_from_name("model.gguf") == QuantType.UNKNOWN