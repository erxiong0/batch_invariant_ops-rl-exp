"""RMSNorm batch invariance tests.

要求：同一行向量，单独算 (batch=1) vs 批中算 (batch=8) 必须逐 bit 相等。
"""
import pytest
import torch

from ops_extension.rms_norm import rms_norm_batch_invariant


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rms_norm_bit_equal_across_batch(dtype):
    """单行 vs 批中相同行 → torch.equal."""
    torch.manual_seed(0)
    D = 4096
    x_row = torch.randn(D, device="cuda", dtype=dtype)
    weight = torch.randn(D, device="cuda", dtype=dtype)
    eps = 1e-6

    x_b1 = x_row.unsqueeze(0)                       # (1, D)
    x_b8 = torch.randn(8, D, device="cuda", dtype=dtype)
    x_b8[0] = x_row                                 # 把目标行放到 batch 第 0 位

    out_b1 = rms_norm_batch_invariant(x_b1, weight, eps)
    out_b8 = rms_norm_batch_invariant(x_b8, weight, eps)

    assert torch.equal(out_b1[0], out_b8[0]), (
        f"RMSNorm not batch-invariant: max diff = {(out_b1[0] - out_b8[0]).abs().max()}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rms_norm_matches_reference(dtype):
    """正确性：与朴素实现数值接近（atol 容差，因为 reduction 顺序不同）."""
    torch.manual_seed(1)
    D = 2048
    x = torch.randn(4, D, device="cuda", dtype=dtype)
    weight = torch.randn(D, device="cuda", dtype=dtype)
    eps = 1e-6

    out = rms_norm_batch_invariant(x, weight, eps)
    ref = x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps).to(dtype) * weight
    atol = {torch.bfloat16: 1e-2, torch.float16: 1e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)
