"""Batch-invariant RMSNorm.

实现策略（参考 vLLM model_executor/layers/batch_invariant.py 与博客原则）：
- 每行一个 Triton 程序（数据并行，无 split-reduction）
- 固定 reduction tree：BLOCK_SIZE 固定，串行累加 tile 内 sum
- 不随 batch 大小切换策略

随后 monkey-patch 进 transformers 的 Qwen2RMSNorm / Qwen3RMSNorm，
让 transformers forward 真正命中这套实现。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols,
    eps,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row_idx * row_stride
    out_row = out_ptr + row_idx * row_stride

    # Pass 1: fixed-order sum of squares in fp32
    sum_sq = tl.zeros((), dtype=tl.float32)
    for col_off in range(0, n_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    mean_sq = sum_sq / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Pass 2: scale + multiply weight
    for col_off in range(0, n_cols, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd * w)
        tl.store(out_row + cols, y, mask=mask)


def rms_norm_batch_invariant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Batch-invariant RMSNorm. x: (..., D), weight: (D,)."""
    assert x.is_cuda and weight.is_cuda
    assert x.shape[-1] == weight.shape[0]
    out_dtype = x.dtype
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    out = torch.empty_like(x_2d)

    n_rows, n_cols = x_2d.shape
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        x_2d, weight, out,
        n_cols, float(eps),
        x_2d.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # cast back to input dtype (kernel writes fp32 inside, output preserves dtype slot)
    return out.to(out_dtype).reshape(orig_shape)


def _patched_qwen_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for transformers Qwen{2,3}RMSNorm.forward."""
    return rms_norm_batch_invariant(hidden_states, self.weight, self.variance_epsilon)


_PATCHED = False
_ORIGINAL_FORWARDS: dict = {}


def patch_transformers_rms_norm() -> None:
    """Monkey-patch transformers' Qwen2RMSNorm/Qwen3RMSNorm to use our kernel."""
    global _PATCHED
    if _PATCHED:
        return
    targets = []
    try:
        from transformers.models.qwen2 import modeling_qwen2
        targets.append(modeling_qwen2.Qwen2RMSNorm)
    except ImportError:
        pass
    try:
        from transformers.models.qwen3 import modeling_qwen3
        targets.append(modeling_qwen3.Qwen3RMSNorm)
    except ImportError:
        pass
    assert targets, "no Qwen RMSNorm class found in transformers"
    for cls in targets:
        _ORIGINAL_FORWARDS[cls] = cls.forward
        cls.forward = _patched_qwen_rms_norm_forward
    _PATCHED = True


def unpatch_transformers_rms_norm() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    for cls, fn in _ORIGINAL_FORWARDS.items():
        cls.forward = fn
    _ORIGINAL_FORWARDS.clear()
    _PATCHED = False
