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
_ORIGINAL_ATTN_REGISTRY_ENTRY = None  # ALL_ATTENTION_FUNCTIONS["sdpa"] 原始函数
# Each item: (module_name, attr_name, original_value). 用来 unpatch 时还原。
_PATCHED_BINDINGS: List[Tuple[str, str, object]] = []


def _rebind_in_sys_modules(original, replacement) -> List[Tuple[str, str, object]]:
    """遍历 sys.modules，把所有 module 命名空间里指向 `original` 的属性替换为
    `replacement`。跳过本模块以保留 _ORIGINAL_SDPA 备份。
    """
    rebound: List[Tuple[str, str, object]] = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod_name == __name__ or mod_name == "ops_extension.sdpa":
            continue
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


_INVARIANT_CALL_COUNT = [0]


def _sdpa_attention_forward_invariant(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    """Drop-in replacement for transformers.integrations.sdpa_attention.sdpa_attention_forward
    that routes through `sdpa_batch_invariant`. Mirrors the upstream impl so masking,
    GQA, contiguous handling, and is_causal inference behave identically.
    """
    _INVARIANT_CALL_COUNT[0] += 1
    if _INVARIANT_CALL_COUNT[0] == 1:
        print(f"[ops_extension.sdpa] _sdpa_attention_forward_invariant CALLED for the first time "
              f"(module={type(module).__name__}, Q={tuple(query.shape)}, K={tuple(key.shape)})",
              file=sys.stderr, flush=True)

    # 强制走 repeat_kv 路径，跟 transformers eager_attention_forward 完全一致。
    # 不用 enable_gqa 旁路 —— 我们的 sdpa_batch_invariant 内部 repeat_interleave 跟
    # repeat_kv 在 stride/layout 上不同，可能让下游 matmul 落到不同的 invariant tile，
    # 引入 16-aligned 数值漂移。eager 用 repeat_kv 给出 0 diff，所以镜像它最稳。
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        try:
            from transformers.integrations.sdpa_attention import repeat_kv
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        except ImportError:
            # transformers 缺这个 helper 时降级；正常路径不应该走到。
            n_rep = module.num_key_value_groups
            B, H_kv, S, D = key.shape
            key = key[:, :, None, :, :].expand(B, H_kv, n_rep, S, D).reshape(B, H_kv * n_rep, S, D)
            B, H_kv, S, D = value.shape
            value = value[:, :, None, :, :].expand(B, H_kv, n_rep, S, D).reshape(B, H_kv * n_rep, S, D)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    if isinstance(is_causal, torch.Tensor):
        is_causal = bool(is_causal.item())

    # K/V 已经 repeat_kv 展开成跟 Q 同 head 数；不再传 enable_gqa。
    attn_output = sdpa_batch_invariant(
        query, key, value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _patch_transformers_attn_registry() -> bool:
    """Replace ALL_ATTENTION_FUNCTIONS['sdpa'] with our batch-invariant version.

    Returns True if registry was found and patched. Also prints diagnostics about
    the container type so we can spot AttentionInterface vs plain dict surprises.
    """
    global _ORIGINAL_ATTN_REGISTRY_ENTRY
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        return False
    if "sdpa" not in ALL_ATTENTION_FUNCTIONS:
        return False
    _ORIGINAL_ATTN_REGISTRY_ENTRY = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_attention_forward_invariant
    # Verify the write took effect via re-read
    readback = ALL_ATTENTION_FUNCTIONS["sdpa"]
    container_cls = type(ALL_ATTENTION_FUNCTIONS).__name__
    print(f"[ops_extension.sdpa] registry container={container_cls}, "
          f"write-readback matches={readback is _sdpa_attention_forward_invariant}",
          file=sys.stderr, flush=True)
    return True


def patch_sdpa(verbose: bool = True) -> None:
    """Three layers of SDPA monkey-patching, in order of effectiveness:

    1. Replace ALL_ATTENTION_FUNCTIONS['sdpa'] in transformers — this is the
       actual dispatch entry that model.forward consults under attn_implementation='sdpa'.
       Highest reliability.
    2. Replace F.scaled_dot_product_attention — defensive, in case some code
       path bypasses the registry.
    3. sys.modules walk to fix any pre-imported direct bindings.
    """
    global _PATCHED, _ORIGINAL_SDPA, _PATCHED_BINDINGS
    if _PATCHED:
        return

    registry_ok = _patch_transformers_attn_registry()
    _ORIGINAL_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sdpa_batch_invariant
    _PATCHED_BINDINGS = _rebind_in_sys_modules(_ORIGINAL_SDPA, sdpa_batch_invariant)
    _PATCHED = True

    if verbose:
        binding_summary = ", ".join(f"{m}.{a}" for m, a, _ in _PATCHED_BINDINGS) or "<none>"
        print(
            f"[ops_extension.sdpa] patched: "
            f"ALL_ATTENTION_FUNCTIONS['sdpa']={'yes' if registry_ok else 'no'}, "
            f"F.scaled_dot_product_attention=yes, "
            f"sys.modules bindings={len(_PATCHED_BINDINGS)} [{binding_summary}]",
            file=sys.stderr, flush=True,
        )


def unpatch_sdpa() -> None:
    global _PATCHED, _ORIGINAL_SDPA, _PATCHED_BINDINGS, _ORIGINAL_ATTN_REGISTRY_ENTRY
    if not _PATCHED:
        return
    if _ORIGINAL_ATTN_REGISTRY_ENTRY is not None:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS["sdpa"] = _ORIGINAL_ATTN_REGISTRY_ENTRY
        except ImportError:
            pass
        _ORIGINAL_ATTN_REGISTRY_ENTRY = None
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
