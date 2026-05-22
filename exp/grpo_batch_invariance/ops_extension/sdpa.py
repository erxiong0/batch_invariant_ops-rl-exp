"""Batch-invariant SDPA via mm + softmax composition.

不引入 FlexAttention 以避开 SWA 兼容性问题。SDPA 写成显式的
Q @ K^T -> softmax -> @ V 三步，让 @ 走 batch_invariant_ops 已经
patched 的 aten::mm，softmax 走 aten::_log_softmax 的反向（exp 后归一化）。

trainer 侧足够快（forward only，~2× FA2 慢但可接受）。

注意 patch 路径：只 monkey-patch `F.scaled_dot_product_attention` 是不够的。
transformers/integrations/sdpa_attention.py 在 module load 时做了
`from torch.nn.functional import scaled_dot_product_attention`，把原函数固定
绑到了自己命名空间。之后改 F.xxx 摸不到它。所以 patch_sdpa() 在替换 F. 之外
还要遍历 sys.modules，把所有 import 过原函数的模块的绑定一并替换。
"""
from __future__ import annotations

import math
import sys
from typing import List, Optional, Tuple

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

    # 必须走 F.log_softmax 才能命中 batch_invariant_ops 的 aten::_log_softmax patch；
    # aten::_softmax 没被 patch，所以 F.softmax 会跳过 batch-invariant 路径。
    # 用 log-softmax → exp 等价数学上 = softmax(x)。
    log_probs = F.log_softmax(scores.float(), dim=-1)
    probs = log_probs.exp().to(query.dtype)
    return torch.matmul(probs, value)


_PATCHED = False
_ORIGINAL_SDPA = None
# Each item: (module_name, attr_name, original_value). 用来 unpatch 时还原。
_PATCHED_BINDINGS: List[Tuple[str, str, object]] = []


def _rebind_in_sys_modules(original, replacement) -> List[Tuple[str, str, object]]:
    """遍历 sys.modules，把所有 module 命名空间里指向 `original` 的属性替换为
    `replacement`。返回被替换的 (module_name, attr_name, original) 三元组。
    """
    rebound: List[Tuple[str, str, object]] = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        # 常见名字：直接 import 的写法 `from torch.nn.functional import scaled_dot_product_attention`
        # 别名写法：`from torch.nn.functional import scaled_dot_product_attention as sdpa`
        # 我们只匹配 identity（is 比较），所以别名也会被命中。
        try:
            mod_vars = vars(mod)
        except TypeError:
            continue
        for attr_name, attr in list(mod_vars.items()):
            if attr is original:
                try:
                    setattr(mod, attr_name, replacement)
                    rebound.append((mod_name, attr_name, original))
                except (AttributeError, TypeError):
                    pass
    return rebound


def patch_sdpa(verbose: bool = True) -> None:
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention 并覆盖所有
    已经 import 它的模块命名空间。"""
    global _PATCHED, _ORIGINAL_SDPA, _PATCHED_BINDINGS
    if _PATCHED:
        return
    _ORIGINAL_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sdpa_batch_invariant
    _PATCHED_BINDINGS = _rebind_in_sys_modules(_ORIGINAL_SDPA, sdpa_batch_invariant)
    _PATCHED = True
    if verbose:
        binding_summary = ", ".join(f"{m}.{a}" for m, a, _ in _PATCHED_BINDINGS) or "<none>"
        print(f"[ops_extension.sdpa] patched F.scaled_dot_product_attention + "
              f"{len(_PATCHED_BINDINGS)} sys.modules bindings: {binding_summary}",
              file=sys.stderr, flush=True)


def unpatch_sdpa() -> None:
    global _PATCHED, _ORIGINAL_SDPA, _PATCHED_BINDINGS
    if not _PATCHED:
        return
    F.scaled_dot_product_attention = _ORIGINAL_SDPA
    for mod_name, attr_name, original in _PATCHED_BINDINGS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        try:
            setattr(mod, attr_name, original)
        except (AttributeError, TypeError):
            pass
    _PATCHED_BINDINGS = []
    _ORIGINAL_SDPA = None
    _PATCHED = False
