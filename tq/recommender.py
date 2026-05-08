from __future__ import annotations

from typing import Optional

from .types import (
    ModelMetadata, HardwareProfile, TQRecommendation, QuantType, CacheType, GpuVendor,
)


def recommend(
    model: ModelMetadata,
    hw: HardwareProfile,
    context_length: Optional[int] = None,
) -> TQRecommendation:
    reasons = []
    qt = model.quant_type

    if qt == QuantType.UNKNOWN:
        qt = _infer_quant_from_name(model.name or model.display_name)

    ctx = context_length or model.context_length or 4096

    is_small_model = model.size_gb < 4.0
    is_medium_model = 4.0 <= model.size_gb < 15.0
    is_large_model = model.size_gb >= 15.0

    is_pre_m5 = hw.is_apple_silicon and hw.apple_chip_generation in ("m1", "m2")

    k_type, v_type = _recommend_cache_types(qt, is_large_model, is_pre_m5, reasons)

    boundary_v = _should_boundary_v(qt, model, reasons)

    sparse_v = model.is_moe and model.expert_count and model.expert_count > 1
    if sparse_v:
        reasons.append(f"MoE model ({model.expert_count} experts) → sparse V enabled")

    ram_budget = hw.ram_gb if hw.is_apple_silicon else hw.vram_gb
    safe_ctx = _compute_safe_context(model, hw, k_type, v_type)
    if ctx > safe_ctx:
        reasons.append(
            f"Requested ctx {ctx} exceeds safe budget {safe_ctx} → capping at {safe_ctx}"
        )
        ctx = safe_ctx

    return TQRecommendation(
        cache_type_k=k_type,
        cache_type_v=v_type,
        boundary_v=boundary_v,
        boundary_v_layers=2,
        sparse_v=sparse_v,
        context_length=ctx,
        reasoning=reasons,
    )


def _infer_quant_from_name(name: str) -> QuantType:
    upper = name.upper()
    for qname, qtype in {
        "Q2_K": QuantType.Q2_K, "Q2_K_S": QuantType.Q2_K_S,
        "Q3_K_S": QuantType.Q3_K_S, "Q3_K_M": QuantType.Q3_K_M,
        "Q3_K_L": QuantType.Q3_K_L,
        "Q4_K_S": QuantType.Q4_K_S, "Q4_K_M": QuantType.Q4_K_M,
        "Q5_K_S": QuantType.Q5_K_S, "Q5_K_M": QuantType.Q5_K_M,
        "Q6_K": QuantType.Q6_K,
        "Q8_0": QuantType.Q8_0,
        "F16": QuantType.F16, "F32": QuantType.F32,
        "IQ3_M": QuantType.IQ3_M, "IQ3_S": QuantType.IQ3_S,
        "IQ4_XS": QuantType.IQ4_XS,
    }.items():
        if qname in upper:
            return qtype
    return QuantType.UNKNOWN


def _recommend_cache_types(
    qt: QuantType, is_large: bool, is_pre_m5: bool, reasons: list[str],
) -> tuple[CacheType, CacheType]:
    if qt in (QuantType.UNKNOWN, QuantType.F32, QuantType.F16):
        k = CacheType.TURBO4
        v = CacheType.TURBO4
        reasons.append(f"Quant {qt.value} → symmetric turbo4 (safe default)")
        return k, v

    if qt in (QuantType.Q8_0,):
        k = CacheType.TURBO4
        v = CacheType.TURBO4
        reasons.append(f"Q8_0 weights → symmetric turbo4 (quality headroom)")
        return k, v

    if qt in (QuantType.Q6_K, QuantType.Q5_K_M, QuantType.Q5_K_S):
        k = CacheType.TURBO4
        v = CacheType.TURBO4
        reasons.append(f"{qt.value} weights → symmetric turbo4 (good quality headroom)")
        return k, v

    if qt in (QuantType.Q4_K_M, QuantType.Q4_K_S, QuantType.IQ4_XS):
        v = CacheType.TURBO4
        if is_large and not is_pre_m5:
            k = CacheType.TURBO3
            reasons.append(f"{qt.value} weights, large model → k=turbo3 v=turbo4 (aggressive KV compression)")
        else:
            k = CacheType.Q8_0
            reasons.append(f"{qt.value} weights → k=q8_0 v=turbo4 (asymmetric: protect K, compress V)")
        return k, v

    if qt in (QuantType.Q3_K_M, QuantType.Q3_K_S, QuantType.Q3_K_L, QuantType.IQ3_M, QuantType.IQ3_S):
        v = CacheType.TURBO4
        k = CacheType.Q8_0
        reasons.append(f"{qt.value} weights → k=q8_0 v=turbo4 (low-bit weights: protect K)")
        return k, v

    if qt in (QuantType.Q2_K, QuantType.Q2_K_S):
        v = CacheType.Q8_0
        k = CacheType.Q8_0
        reasons.append(f"{qt.value} weights → k=q8_0 v=q8_0 (ultra-low-bit: no KV compression)")
        return k, v

    k = CacheType.Q8_0
    v = CacheType.TURBO4
    reasons.append(f"Unknown quant → k=q8_0 v=turbo4 (conservative default)")
    return k, v


def _should_boundary_v(qt: QuantType, model: ModelMetadata, reasons: list[str]) -> bool:
    num_layers = model.num_layers
    if num_layers is not None and num_layers < 4:
        reasons.append(f"Only {num_layers} layers → boundary V disabled (too few layers)")
        return False

    if qt in (QuantType.Q4_K_M, QuantType.Q4_K_S, QuantType.IQ4_XS,
              QuantType.Q3_K_M, QuantType.Q3_K_S, QuantType.Q3_K_L,
              QuantType.IQ3_M, QuantType.IQ3_S,
              QuantType.Q2_K, QuantType.Q2_K_S):
        reasons.append(f"{qt.value} weights → boundary V enabled (protect first/last layers at q8_0)")
        return True

    if qt in (QuantType.UNKNOWN, QuantType.F16, QuantType.F32, QuantType.Q8_0,
              QuantType.Q5_K_M, QuantType.Q5_K_S, QuantType.Q6_K):
        reasons.append(f"{qt.value} weights → boundary V not needed (sufficient quality headroom)")
        return False

    return True


def _compute_safe_context(
    model: ModelMetadata,
    hw: HardwareProfile,
    k_type: CacheType,
    v_type: CacheType,
) -> int:
    if hw.is_apple_silicon:
        budget_gb = hw.ram_gb * 0.50
    else:
        budget_gb = hw.vram_gb * 0.70

    compute_overhead_gb = 1.0 if model.size_gb > 5 else 0.7
    available_gb = budget_gb - model.size_gb - compute_overhead_gb
    if available_gb <= 0.5:
        return 2048

    bytes_per_element_k = _cache_bytes_per_element(k_type)
    bytes_per_element_v = _cache_bytes_per_element(v_type)

    num_layers = model.num_layers or 32
    num_kv_heads = model.num_kv_heads or model.num_heads or 32
    head_dim = (model.embedding_length or 4096) // (model.num_heads or 32)

    k_elements_per_token_per_layer = num_kv_heads * head_dim
    v_elements_per_token_per_layer = num_kv_heads * head_dim
    kv_per_token_bytes = num_layers * (
        k_elements_per_token_per_layer * bytes_per_element_k +
        v_elements_per_token_per_layer * bytes_per_element_v
    )

    available_bytes = available_gb * 1024 ** 3
    max_tokens = int(available_bytes / kv_per_token_bytes) if kv_per_token_bytes > 0 else 4096

    max_tokens = max(2048, min(max_tokens, 32768))

    step = 256
    return (max_tokens // step) * step


def _cache_bytes_per_element(ct: CacheType) -> float:
    sizes = {
        CacheType.F16: 2.0,
        CacheType.Q8_0: 1.0,
        CacheType.Q4_0: 0.5,
        CacheType.TURBO3: 0.5625,
        CacheType.TURBO4: 0.625,
    }
    return sizes.get(ct, 1.0)