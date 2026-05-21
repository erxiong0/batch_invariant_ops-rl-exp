"""Batch-invariant SDPA via mm + softmax composition.

不引入 FlexAttention 以避开 SWA 兼容性问题。SDPA 写成显式的
Q @ K^T -> softmax -> @ V 三步，让 @ 走 batch_invariant_ops 已经
patched 的 aten::mm，softmax 走 aten::_log_softmax 的反向（exp 后归一化）。

trainer 侧足够快（forward only，~2× FA2 慢但可接受）。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def sdpa_batch_invariant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Reference-style SDPA wired through batch-invariant matmul + softmax.

    Shapes: query (..., Lq, D), key (..., Lk, D), value (..., Lk, Dv).
    """
    assert dropout_p == 0.0, "training-time dropout in attn not supported in this experiment"

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    if enable_gqa and key.shape[-3] != query.shape[-3]:
        # Group-query attention: broadcast K/V heads to match Q heads.
        n_rep = query.shape[-3] // key.shape[-3]
        key = key.repeat_interleave(n_rep, dim=-3)
        value = value.repeat_interleave(n_rep, dim=-3)

    # (B, H, Lq, Lk)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        Lq, Lk = query.shape[-2], key.shape[-2]
        # mask[i, j] = True 表示需要屏蔽（j > i + (Lk - Lq)）
        causal = torch.ones(Lq, Lk, device=query.device, dtype=torch.bool).triu(diagonal=Lk - Lq + 1)
        scores = scores.masked_fill(causal, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    # log_softmax → exp → renormalized? 直接走 softmax，它内部用 _log_softmax 的反路径，
    # 等价上 = exp(log_softmax(x))，仍在我们的 patch 覆盖下
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probs, value)


_PATCHED = False
_ORIGINAL_SDPA = None


def patch_sdpa() -> None:
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention."""
    global _PATCHED, _ORIGINAL_SDPA
    if _PATCHED:
        return
    _ORIGINAL_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sdpa_batch_invariant
    _PATCHED = True


def unpatch_sdpa() -> None:
    global _PATCHED, _ORIGINAL_SDPA
    if not _PATCHED:
        return
    F.scaled_dot_product_attention = _ORIGINAL_SDPA
    _ORIGINAL_SDPA = None
    _PATCHED = False
