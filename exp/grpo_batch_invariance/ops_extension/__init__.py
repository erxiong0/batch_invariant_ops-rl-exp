"""Phase 0 扩展：在 batch_invariant_ops 的 mm/addmm/log_softmax/mean 之外，
补齐 RMSNorm（transformers 类 monkey-patch）与 SDPA（F.sdpa monkey-patch）。
"""
from ops_extension.rms_norm import (
    patch_transformers_rms_norm,
    rms_norm_batch_invariant,
    unpatch_transformers_rms_norm,
)
from ops_extension.sdpa import (
    patch_sdpa,
    sdpa_batch_invariant,
    unpatch_sdpa,
)

__all__ = [
    "enable_extended_batch_invariant_mode",
    "disable_extended_batch_invariant_mode",
    "rms_norm_batch_invariant",
    "sdpa_batch_invariant",
]

_ENABLED = False


def enable_extended_batch_invariant_mode() -> None:
    """启用 RMSNorm + SDPA 的 batch-invariant 替换。

    必须在 import transformers 模型之后（或之前——monkey-patch 是类级别，作用于所有实例）。
    与 batch_invariant_ops.enable_batch_invariant_mode() 互不冲突，建议先调后者再调本函数。
    """
    global _ENABLED
    if _ENABLED:
        return
    patch_transformers_rms_norm()
    patch_sdpa()
    _ENABLED = True


def disable_extended_batch_invariant_mode() -> None:
    global _ENABLED
    if not _ENABLED:
        return
    unpatch_sdpa()
    unpatch_transformers_rms_norm()
    _ENABLED = False
