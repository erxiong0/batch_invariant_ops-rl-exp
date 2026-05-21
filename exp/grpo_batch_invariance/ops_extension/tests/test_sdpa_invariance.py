"""SDPA batch invariance tests."""
import math

import pytest
import torch

from batch_invariant_ops import enable_batch_invariant_mode, disable_batch_invariant_mode
from ops_extension.sdpa import sdpa_batch_invariant


@pytest.fixture(autouse=True)
def _enable_invariant_ops():
    enable_batch_invariant_mode()
    yield
    disable_batch_invariant_mode()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_causal", [True, False])
def test_sdpa_bit_equal_across_batch(dtype, is_causal):
    """同 (q_row, kv) → batch=1 与 batch=8 应 bit-equal."""
    torch.manual_seed(42)
    B_other, H, S, D = 7, 4, 128, 64
    q_row = torch.randn(1, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(1, H, S, D, device="cuda", dtype=dtype)
    v = torch.randn(1, H, S, D, device="cuda", dtype=dtype)

    out_b1 = sdpa_batch_invariant(q_row, k, v, is_causal=is_causal)

    # Build batch of 8 with q_row at slot 0
    q_batch = torch.randn(B_other + 1, H, S, D, device="cuda", dtype=dtype)
    k_batch = torch.randn(B_other + 1, H, S, D, device="cuda", dtype=dtype)
    v_batch = torch.randn(B_other + 1, H, S, D, device="cuda", dtype=dtype)
    q_batch[0] = q_row[0]
    k_batch[0] = k[0]
    v_batch[0] = v[0]

    out_b8 = sdpa_batch_invariant(q_batch, k_batch, v_batch, is_causal=is_causal)

    assert torch.equal(out_b1[0], out_b8[0]), (
        f"SDPA not batch-invariant (is_causal={is_causal}, dtype={dtype}): "
        f"max diff = {(out_b1[0] - out_b8[0]).abs().max()}"
    )


def test_sdpa_matches_reference():
    """正确性：与 torch.nn.functional.scaled_dot_product_attention 数值接近."""
    torch.manual_seed(7)
    q = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.float32)

    out = sdpa_batch_invariant(q, k, v, is_causal=True)

    # 临时关 batch-invariant 跑参考
    disable_batch_invariant_mode()
    try:
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    finally:
        enable_batch_invariant_mode()
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
