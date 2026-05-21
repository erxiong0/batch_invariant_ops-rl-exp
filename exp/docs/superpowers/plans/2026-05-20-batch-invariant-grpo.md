# Batch Invariance × GRPO 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 swift 官方 Qwen3-1.7B + GSM8K GRPO 配方上做 A/B 实验，验证启用 batch-invariant kernel 对 RL 训练的影响（机制 + 指标 + 可复现性）。

**Architecture:** 两阶段。Phase 0 是前置工程：在 `ops_extension/` 中补齐 RMSNorm + SDPA 的 batch-invariant 实现（`batch_invariant_ops` 当前只覆盖 mm/addmm/log_softmax/mean），让 trainer 侧 forward 真正达到 batch-invariant。Phase 1 是实验脚手架：launcher（零侵入 swift）+ 诊断脚本（4-cell logprob 对比 + 旁证）+ 训练（2 mode × 3 seed × 50 step）+ 评测（GSM8K）+ 自动汇总。

**Tech Stack:** PyTorch 2.9+ / Triton / transformers / ms-swift / vLLM ≥0.17 / batch_invariant_ops (本仓库) / pytest

**Spec:** `docs/superpowers/specs/2026-05-20-batch-invariant-grpo-design.md`

---

## 文件结构总览

所有产物位于 `exp/grpo_batch_invariance/`。Plan 中的相对路径都以仓库根目录为基准。

```
exp/grpo_batch_invariance/
├── README.md
├── env/
│   ├── setup.sh
│   └── verify_env.py
├── ops_extension/
│   ├── __init__.py                              # enable_extended_batch_invariant_mode()
│   ├── rms_norm.py                              # Triton RMSNorm kernel + monkey-patch
│   ├── sdpa.py                                  # 基于 mm+softmax 拼装的 SDPA + F.sdpa monkey-patch
│   └── tests/
│       ├── __init__.py
│       ├── test_rms_norm_invariance.py
│       ├── test_sdpa_invariance.py
│       └── test_end_to_end_forward.py
├── launcher.py
├── diagnostics/
│   ├── __init__.py
│   ├── logprob_mismatch.py
│   ├── repeatability.py
│   └── plot_diagnostics.py
├── train/
│   ├── train_grpo.sh
│   └── run_all.sh
├── eval/
│   ├── eval_gsm8k.sh
│   └── eval_all.sh
└── results/                                     # 运行时填充，初始为空
    └── .gitkeep
```

---

## Phase 0 — 补齐 batch_invariant_ops

### Task 0.1: 项目骨架与 .gitkeep

**Files:**
- Create: `exp/grpo_batch_invariance/README.md`
- Create: `exp/grpo_batch_invariance/results/.gitkeep`
- Create: `exp/grpo_batch_invariance/ops_extension/__init__.py` (空壳，后续 Task 填充)
- Create: `exp/grpo_batch_invariance/ops_extension/tests/__init__.py`
- Create: `exp/grpo_batch_invariance/diagnostics/__init__.py`

- [ ] **Step 1: 创建目录骨架**

```bash
cd /Users/erxiong/Documents/si-proj/batch_invariant_ops
mkdir -p exp/grpo_batch_invariance/{env,ops_extension/tests,diagnostics,train/plugins,eval,results}
touch exp/grpo_batch_invariance/results/.gitkeep
touch exp/grpo_batch_invariance/ops_extension/__init__.py
touch exp/grpo_batch_invariance/ops_extension/tests/__init__.py
touch exp/grpo_batch_invariance/diagnostics/__init__.py
```

- [ ] **Step 2: 写 README.md（最小占位，后续 Task 会扩展）**

```markdown
# Batch Invariance × GRPO 实验

验证 batch-invariant kernel 对 ms-swift Qwen3-1.7B + GSM8K GRPO 训练的影响。

详见：
- Spec: `exp/docs/superpowers/specs/2026-05-20-batch-invariant-grpo-design.md`
- Plan: `exp/docs/superpowers/plans/2026-05-20-batch-invariant-grpo.md`

## 快速开始

```bash
bash env/setup.sh
python env/verify_env.py
pytest ops_extension/tests/ -v
python diagnostics/logprob_mismatch.py
bash train/run_all.sh
bash eval/eval_all.sh
```

## 硬件

4 × H100/A100 (80G)。
```

- [ ] **Step 3: Commit**

```bash
git add exp/grpo_batch_invariance/
git commit -m "exp: scaffold grpo_batch_invariance experiment directory"
```

---

### Task 0.2: env/setup.sh 与 env/verify_env.py（含 liger fail-fast）

**Files:**
- Create: `exp/grpo_batch_invariance/env/setup.sh`
- Create: `exp/grpo_batch_invariance/env/verify_env.py`

- [ ] **Step 1: 写 `env/setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录调用：bash exp/grpo_batch_invariance/env/setup.sh
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# batch_invariant_ops 本仓库自身（editable）
pip install -e .

# ms-swift（GRPO trainer）
pip install -U "ms-swift>=4.3"

# vLLM（必须 ≥0.17，自带 batch invariance 支持）
pip install -U "vllm>=0.17.0"

# transformers（与 swift 文档对齐）
pip install -U "transformers==5.2.*"

# 评测依赖
pip install "math_verify==0.5.2"

# 测试依赖
pip install pytest

echo "Setup done. Run: python exp/grpo_batch_invariance/env/verify_env.py"
```

```bash
chmod +x exp/grpo_batch_invariance/env/setup.sh
```

- [ ] **Step 2: 写 `env/verify_env.py`，含 liger fail-fast 检查**

```python
"""环境自检：PyTorch/vLLM 版本、GPU 数量、liger-kernel/fused-MLP 不可加载。"""
import importlib
import sys

import torch


def check_pytorch():
    major, minor = torch.__version__.split(".")[:2]
    assert (int(major), int(minor)) >= (2, 9), f"need torch>=2.9, got {torch.__version__}"
    assert torch.cuda.is_available(), "CUDA not available"
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 4, f"need 4 GPUs, got {n_gpu}"
    print(f"OK  torch={torch.__version__}  GPUs={n_gpu}")


def check_vllm():
    import vllm
    parts = vllm.__version__.split(".")
    assert (int(parts[0]), int(parts[1])) >= (0, 17), f"need vllm>=0.17, got {vllm.__version__}"
    print(f"OK  vllm={vllm.__version__}")


def check_batch_invariant_ops():
    import batch_invariant_ops
    assert hasattr(batch_invariant_ops, "enable_batch_invariant_mode")
    print("OK  batch_invariant_ops importable")


def check_no_liger_active():
    """liger-kernel 若被 import 即视为风险（它通过 monkey-patch 替换 transformers 的 MLP/RMSNorm）。"""
    forbidden = ["liger_kernel", "apex.normalization.fused_layer_norm", "flash_attn.ops.fused_dense"]
    for name in forbidden:
        try:
            mod = importlib.import_module(name)
            print(f"FAIL  forbidden module loaded: {name} -> {mod}")
            sys.exit(1)
        except ImportError:
            pass
    print("OK  no forbidden fused-kernel modules pre-loaded")


def main():
    check_pytorch()
    check_vllm()
    check_batch_invariant_ops()
    check_no_liger_active()
    print("\nAll environment checks passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑 verify_env.py**

```bash
python exp/grpo_batch_invariance/env/verify_env.py
```

Expected: 全部 OK 行，最后 "All environment checks passed."

如果输出 FAIL，按提示修复（典型问题：vllm 版本太老、torch 太老、GPU 数不足）。

- [ ] **Step 4: Commit**

```bash
git add exp/grpo_batch_invariance/env/
git commit -m "exp: env setup + verify script with liger fail-fast"
```

---

### Task 0.3: ops_extension/rms_norm.py — Triton kernel + 单元测试（TDD）

**Files:**
- Create: `exp/grpo_batch_invariance/ops_extension/tests/test_rms_norm_invariance.py`
- Create: `exp/grpo_batch_invariance/ops_extension/rms_norm.py`

- [ ] **Step 1: 写失败的单元测试**

`exp/grpo_batch_invariance/ops_extension/tests/test_rms_norm_invariance.py`:

```python
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
```

- [ ] **Step 2: 跑测试，确认失败（rms_norm_batch_invariant 不存在）**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=. pytest ops_extension/tests/test_rms_norm_invariance.py -v
```

Expected: ImportError / collection error。

- [ ] **Step 3: 写 `ops_extension/rms_norm.py`**

```python
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
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=. pytest ops_extension/tests/test_rms_norm_invariance.py -v
```

Expected: 6 个 test 全 PASS（3 dtype × 2 test）。

如果 `test_rms_norm_bit_equal_across_batch` 失败 → kernel reduction 顺序仍随 batch 变。检查 grid 与 BLOCK_SIZE 是否随 n_rows 改变。

- [ ] **Step 5: Commit**

```bash
git add exp/grpo_batch_invariance/ops_extension/rms_norm.py \
        exp/grpo_batch_invariance/ops_extension/tests/test_rms_norm_invariance.py
git commit -m "exp: add batch-invariant RMSNorm Triton kernel + tests"
```

---

### Task 0.4: ops_extension/sdpa.py — 用已 batch-invariant 的 mm+softmax 拼装 SDPA（TDD）

> **设计决策**：不引入 FlexAttention（避开 SWA 兼容性 R6）。SDPA 用 `Q @ Kᵀ → softmax → @ V` 三步拼装，其中 `@` 走 `batch_invariant_ops` 已 patch 的 `aten::mm`，softmax 用 `aten::_log_softmax` 间接覆盖（或直接走 fp32 reduction）。这样 SDPA 自动 batch-invariant，无需新写 attention kernel。代价是慢（~2× FA2），但 trainer 侧可接受。

**Files:**
- Create: `exp/grpo_batch_invariance/ops_extension/tests/test_sdpa_invariance.py`
- Create: `exp/grpo_batch_invariance/ops_extension/sdpa.py`

- [ ] **Step 1: 写失败测试**

`exp/grpo_batch_invariance/ops_extension/tests/test_sdpa_invariance.py`:

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=. pytest ops_extension/tests/test_sdpa_invariance.py -v
```

Expected: ImportError。

- [ ] **Step 3: 写 `ops_extension/sdpa.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=. pytest ops_extension/tests/test_sdpa_invariance.py -v
```

Expected: 5 test PASS（2 dtype × 2 causal + reference）。

如果 `test_sdpa_bit_equal_across_batch` 失败 → 八成是 softmax 内部走的不是 batch-invariant 路径。检查 `enable_batch_invariant_mode()` 是否已生效（fixture 是否正确触发）。

- [ ] **Step 5: Commit**

```bash
git add exp/grpo_batch_invariance/ops_extension/sdpa.py \
        exp/grpo_batch_invariance/ops_extension/tests/test_sdpa_invariance.py
git commit -m "exp: add batch-invariant SDPA via mm+softmax composition + tests"
```

---

### Task 0.5: ops_extension/__init__.py — 统一启用入口

**Files:**
- Modify: `exp/grpo_batch_invariance/ops_extension/__init__.py`

- [ ] **Step 1: 替换 __init__.py（之前是空的）**

```python
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
```

- [ ] **Step 2: 快速 import smoke test**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=. python -c "
from ops_extension import enable_extended_batch_invariant_mode, disable_extended_batch_invariant_mode
enable_extended_batch_invariant_mode()
import torch.nn.functional as F
print('sdpa patched:', F.scaled_dot_product_attention.__name__)
disable_extended_batch_invariant_mode()
print('sdpa restored:', F.scaled_dot_product_attention.__name__)
"
```

Expected:
```
sdpa patched: sdpa_batch_invariant
sdpa restored: scaled_dot_product_attention
```

- [ ] **Step 3: Commit**

```bash
git add exp/grpo_batch_invariance/ops_extension/__init__.py
git commit -m "exp: ops_extension entry point — enable/disable extended invariant mode"
```

---

### Task 0.6: end-to-end Qwen3-1.7B forward bit-equal 测试（Phase 0 总 gate）

**Files:**
- Create: `exp/grpo_batch_invariance/ops_extension/tests/test_end_to_end_forward.py`

- [ ] **Step 1: 写 e2e 测试**

```python
"""End-to-end Qwen3-1.7B forward batch invariance test.

启用 batch_invariant_ops + ops_extension 后，对相同输入 prompt:
  - batch=1 forward 拿 last-token logits
  - batch=8（含相同 prompt 在 slot 0）forward 拿 slot 0 last-token logits
两者必须 torch.equal（bit-equal）。
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch_invariant_ops import enable_batch_invariant_mode, disable_batch_invariant_mode
from ops_extension import (
    enable_extended_batch_invariant_mode,
    disable_extended_batch_invariant_mode,
)


MODEL_ID = "Qwen/Qwen3-1.7B"


@pytest.fixture(scope="module")
def model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    return model, tok


def _last_logits(model, input_ids, attn_mask):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    # last non-pad token的 logits
    last_idx = attn_mask.sum(-1) - 1
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    return out.logits[rows, last_idx]


def test_qwen_forward_bit_equal_with_full_invariance(model_and_tok):
    """启用完整 invariance 后 batch=1 与 batch=8 的 slot 0 logits bit-equal。"""
    model, tok = model_and_tok
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()
    try:
        target_prompt = "What is 13 * 47? Step by step:"
        other_prompts = [
            "Hello world.",
            "The capital of France is",
            "Solve for x: 2x + 5 = 11.",
            "List three primes.",
            "Write a haiku about GPUs.",
            "1 + 1 = ",
            "Translate 'cat' to French:",
        ]
        # batch=1
        enc1 = tok([target_prompt], return_tensors="pt", padding=True).to("cuda")
        logits1 = _last_logits(model, enc1.input_ids, enc1.attention_mask)
        # batch=8（target 在 slot 0）
        enc8 = tok([target_prompt] + other_prompts, return_tensors="pt", padding=True).to("cuda")
        logits8 = _last_logits(model, enc8.input_ids, enc8.attention_mask)

        assert torch.equal(logits1[0], logits8[0]), (
            f"forward not batch-invariant: "
            f"max diff = {(logits1[0] - logits8[0]).abs().max()}, "
            f"frac_diff = {((logits1[0] - logits8[0]).abs() > 0).float().mean()}"
        )
    finally:
        disable_extended_batch_invariant_mode()
        disable_batch_invariant_mode()


def test_qwen_forward_NOT_bit_equal_without_invariance(model_and_tok):
    """对照组：不启用 invariance，batch=1 与 batch=8 应有差异（mechanism sanity check）。"""
    model, tok = model_and_tok
    target_prompt = "What is 13 * 47? Step by step:"
    other_prompts = ["Hello.", "Hi.", "Bonjour.", "你好.", "Ciao.", "Hola.", "Olá."]

    enc1 = tok([target_prompt], return_tensors="pt", padding=True).to("cuda")
    logits1 = _last_logits(model, enc1.input_ids, enc1.attention_mask)
    enc8 = tok([target_prompt] + other_prompts, return_tensors="pt", padding=True).to("cuda")
    logits8 = _last_logits(model, enc8.input_ids, enc8.attention_mask)

    diff = (logits1[0] - logits8[0]).abs().max().item()
    # baseline 下应有非零差异，否则机制不成立
    assert diff > 0, f"baseline batch invariance unexpectedly holds (diff={diff})"
```

- [ ] **Step 2: 跑测试**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=. pytest ops_extension/tests/test_end_to_end_forward.py -v -s
```

Expected: 2 test PASS。模型下载首次约 5-10 分钟。

如果 `test_qwen_forward_bit_equal_with_full_invariance` 失败：
1. 先看哪个 layer 出问题——把 model 改为 `output_hidden_states=True`，逐层对比 batch=1 vs batch=8 第 0 行
2. 第一个出现 diff 的 layer 即未覆盖的 op；常见漏洞：transformers 在某层用了自定义 fused kernel（参考 R6/R7/R8）

- [ ] **Step 3: Commit**

```bash
git add exp/grpo_batch_invariance/ops_extension/tests/test_end_to_end_forward.py
git commit -m "exp: end-to-end Qwen3-1.7B forward bit-equal test (Phase 0 gate)"
```

**Phase 0 done-gate**: 这个测试通过即 Phase 0 完成。否则不进入 Phase 1。

---

## Phase 1 — 实验脚手架

### Task 1.1: launcher.py — 零侵入 swift 启动器

**Files:**
- Create: `exp/grpo_batch_invariance/launcher.py`

- [ ] **Step 1: 写 launcher.py**

```python
"""零侵入接入 swift CLI 的启动器。

调用方式：
    BIM_MODE=baseline   python launcher.py rlhf --rlhf_type grpo --model Qwen/Qwen3-1.7B ...
    BIM_MODE=invariant  python launcher.py rlhf --rlhf_type grpo --model Qwen/Qwen3-1.7B ...

invariant 模式同时翻三个开关：
  1) VLLM_BATCH_INVARIANT=1                       (vLLM rollout 侧)
  2) batch_invariant_ops.enable_batch_invariant_mode()  (mm/addmm/log_softmax/mean)
  3) ops_extension.enable_extended_batch_invariant_mode() (RMSNorm + SDPA)
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    mode = os.environ.get("BIM_MODE", "baseline").lower()
    assert mode in {"baseline", "invariant"}, f"bad BIM_MODE: {mode}"

    if mode == "invariant":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        from batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        from ops_extension import enable_extended_batch_invariant_mode
        enable_extended_batch_invariant_mode()
        print(f"[launcher] invariant mode ON: VLLM_BATCH_INVARIANT=1, "
              f"mm/addmm/log_softmax/mean patched, RMSNorm + SDPA patched",
              file=sys.stderr, flush=True)
    else:
        print(f"[launcher] baseline mode (no patches)", file=sys.stderr, flush=True)

    # 透传剩余参数给 swift CLI
    from swift.cli.main import cli_main
    sys.argv = ["swift"] + sys.argv[1:]
    cli_main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 烟雾测试（不真跑训练）**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=.:$(pwd)/.. BIM_MODE=invariant python -c "
import os, sys
os.environ['BIM_MODE']='invariant'
sys.argv=['launcher.py','--help']
import launcher
try: launcher.main()
except SystemExit: pass
" 2>&1 | head -5
```

Expected: 看到 `[launcher] invariant mode ON: ...` 行，然后 swift 帮助文本。

- [ ] **Step 3: Commit**

```bash
git add exp/grpo_batch_invariance/launcher.py
git commit -m "exp: launcher.py — three-switch invariant mode wrapping swift CLI"
```

---

### Task 1.2: diagnostics/logprob_mismatch.py — 4-cell rollout-vs-forward 对比

**Files:**
- Create: `exp/grpo_batch_invariance/diagnostics/logprob_mismatch.py`

- [ ] **Step 1: 写诊断脚本**

```python
"""4-cell logprob mismatch diagnostic.

对 GSM8K 前 N 条 prompt:
  - 用 vLLM rollout 一次（与训练同 sampling 参数），记录 token-level logprob_rollout
  - 用 transformers HF forward 一次，重算同 (prompt, completion) 的 logprob_train
  - 计算 delta = logprob_rollout - logprob_train、|delta| 分布

四个 cell（控制变量）:
  A baseline:    VLLM_BATCH_INVARIANT off, trainer patch off
  B trainer-only: VLLM_BATCH_INVARIANT off, trainer patch on
  C vllm-only:    VLLM_BATCH_INVARIANT on,  trainer patch off
  D both:        VLLM_BATCH_INVARIANT on,  trainer patch on

每个 cell 是独立 subprocess（因 VLLM_BATCH_INVARIANT 必须在 vLLM import 前设置）。
本脚本通过 --cell 参数运行单个 cell；--all 触发四个 subprocess。

输出: results/diagnostics/cell_{A,B,C,D}.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results" / "diagnostics"

MODEL_ID = "Qwen/Qwen3-1.7B"
N_PROMPTS = 200
MAX_NEW = 256
SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1, seed=12345)


def load_gsm8k_prompts(n: int) -> List[str]:
    """Return first n GSM8K test prompts."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").select(range(n))
    sys_msg = "You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."
    return [f"{sys_msg}\n\nQuestion: {x['question']}\nAnswer:" for x in ds]


def rollout_vllm(prompts: List[str]) -> List[Tuple[List[int], List[float]]]:
    """Return list of (token_ids, logprobs) per prompt from vLLM."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True)
    sp = SamplingParams(
        max_tokens=MAX_NEW, temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"],
        seed=SAMPLING["seed"], logprobs=1,
    )
    outputs = llm.generate(prompts, sp)
    result = []
    for out in outputs:
        comp = out.outputs[0]
        token_ids = list(comp.token_ids)
        logprobs = [lp_d[tid].logprob for tid, lp_d in zip(comp.token_ids, comp.logprobs)]
        result.append((token_ids, logprobs))
    return result


def forward_hf(prompts: List[str], rollouts) -> List[List[float]]:
    """Re-compute per-token logprob via HF forward on (prompt + completion)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    train_logprobs: List[List[float]] = []
    with torch.no_grad():
        for prompt, (gen_ids, _) in zip(prompts, rollouts):
            p_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            full_ids = torch.cat([p_ids, torch.tensor([gen_ids], device="cuda")], dim=1)
            logits = model(full_ids).logits  # (1, L, V)
            # logits[t] predicts token at t+1; so for gen tokens at positions [p_len, p_len+len(gen))
            p_len = p_ids.shape[1]
            gen_logits = logits[0, p_len - 1 : -1, :]  # (G, V)
            gen_targets = torch.tensor(gen_ids, device="cuda")
            log_softmax = F.log_softmax(gen_logits.float(), dim=-1)
            lp = log_softmax.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1).cpu().tolist()
            train_logprobs.append(lp)
    return train_logprobs


def run_cell(cell: str) -> None:
    """单个 cell 子进程的工作。"""
    if cell in ("C", "D"):
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    if cell in ("B", "D"):
        from batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        from ops_extension import enable_extended_batch_invariant_mode
        enable_extended_batch_invariant_mode()

    prompts = load_gsm8k_prompts(N_PROMPTS)
    rollouts = rollout_vllm(prompts)
    train_lps = forward_hf(prompts, rollouts)

    deltas: List[float] = []
    for (_, roll_lps), tr_lps in zip(rollouts, train_lps):
        L = min(len(roll_lps), len(tr_lps))
        for r, t in zip(roll_lps[:L], tr_lps[:L]):
            deltas.append(float(r - t))

    abs_deltas = [abs(d) for d in deltas]
    import statistics
    summary = {
        "cell": cell,
        "n_tokens": len(deltas),
        "mean_abs_delta": statistics.mean(abs_deltas),
        "max_abs_delta": max(abs_deltas),
        "frac_gt_1e-3": sum(1 for d in abs_deltas if d > 1e-3) / len(abs_deltas),
        "frac_gt_1e-6": sum(1 for d in abs_deltas if d > 1e-6) / len(abs_deltas),
        "histogram_deltas": deltas[:2000],  # 截断保存
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"cell_{cell}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[cell {cell}] wrote {out}: mean|Δ|={summary['mean_abs_delta']:.2e} "
          f"frac>1e-3={summary['frac_gt_1e-3']:.3f}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list("ABCD"))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for c in "ABCD":
            print(f"\n=== running cell {c} (subprocess, fresh env) ===", file=sys.stderr)
            env = os.environ.copy()
            env.pop("VLLM_BATCH_INVARIANT", None)
            subprocess.run(
                [sys.executable, __file__, "--cell", c],
                check=True, env=env, cwd=str(EXP_DIR),
            )
    elif args.cell:
        run_cell(args.cell)
    else:
        ap.error("need --cell {A,B,C,D} or --all")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑单个 cell 验证（先 A）**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=.:$(pwd)/.. python diagnostics/logprob_mismatch.py --cell A
```

Expected: 写出 `results/diagnostics/cell_A.json`，控制台显示 `mean|Δ|=` 一个非零值（baseline 应非零）。

- [ ] **Step 3: 跑全部 4 个 cell**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=.:$(pwd)/.. python diagnostics/logprob_mismatch.py --all
```

Expected:
- cell A: mean|Δ| ~ 1e-3 ~ 1e-2 量级，frac>1e-3 显著 > 0
- cell B/C: mean|Δ| 略降但仍非零
- cell D: **mean|Δ| < 1e-6** 且 **frac>1e-3 == 0**（Phase 0 gate 已过的旁证）

如果 D cell 不满足 → 回到 Phase 0 的 `test_end_to_end_forward.py` 修复。

- [ ] **Step 4: Commit**

```bash
git add exp/grpo_batch_invariance/diagnostics/logprob_mismatch.py
git commit -m "exp: 4-cell logprob mismatch diagnostic (rollout vs trainer)"
```

---

### Task 1.3: diagnostics/plot_diagnostics.py — 输出对比图

**Files:**
- Create: `exp/grpo_batch_invariance/diagnostics/plot_diagnostics.py`

- [ ] **Step 1: 写绘图脚本**

```python
"""Plot logprob mismatch histograms across the 4 cells.

读取 results/diagnostics/cell_{A,B,C,D}.json，输出 figures/logprob_diff_hist.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results" / "diagnostics"
FIG_DIR = EXP_DIR / "results" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    titles = {
        "A": "A: baseline (no patches)",
        "B": "B: trainer-only invariant",
        "C": "C: vLLM-only invariant",
        "D": "D: both invariant",
    }
    for ax, cell in zip(axes.flat, "ABCD"):
        path = RESULTS / f"cell_{cell}.json"
        data = json.loads(path.read_text())
        deltas = np.array(data["histogram_deltas"])
        ax.hist(deltas, bins=80, log=True)
        ax.set_title(f"{titles[cell]}\nmean|Δ|={data['mean_abs_delta']:.2e}  "
                     f"frac>1e-3={data['frac_gt_1e-3']:.2%}")
        ax.set_xlabel("logprob_rollout - logprob_train")
        ax.set_ylabel("count (log)")
    fig.suptitle("Rollout vs Trainer logprob diff — 4-cell control")
    fig.tight_layout()
    out = FIG_DIR / "logprob_diff_hist.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑生成图**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=.:$(pwd)/.. python diagnostics/plot_diagnostics.py
```

Expected: `results/figures/logprob_diff_hist.png` 生成；目视检查 D cell 直方图应集中在 0 附近、几乎不可见尾部。

- [ ] **Step 3: Commit**

```bash
git add exp/grpo_batch_invariance/diagnostics/plot_diagnostics.py
git commit -m "exp: plot 4-cell logprob diff histograms"
```

---

### Task 1.4: diagnostics/repeatability.py — 同 prompt 多次 rollout 唯一性

**Files:**
- Create: `exp/grpo_batch_invariance/diagnostics/repeatability.py`

- [ ] **Step 1: 写脚本**

```python
"""Repeatability旁证：同一 prompt 用 vLLM 跑 N 次（独立请求），统计唯一输出数。

baseline 应有几十个唯一样本（与博客原例一致），invariant 应为 1。
通过子进程实现 mode 切换（VLLM_BATCH_INVARIANT 必须在 vllm import 前设置）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results" / "diagnostics"
MODEL_ID = "Qwen/Qwen3-1.7B"
N_TRIALS = 200
PROMPT = ("Generate 30 random numbers between 0 and 1000, comma-separated. "
          "Just numbers, no prose.")


def run_one(mode: str) -> None:
    if mode == "invariant":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True)
    sp = SamplingParams(max_tokens=200, temperature=0.0, seed=0)
    outs = llm.generate([PROMPT] * N_TRIALS, sp)
    texts = [o.outputs[0].text for o in outs]
    uniq = len(set(texts))
    summary = {"mode": mode, "n_trials": N_TRIALS, "n_unique": uniq}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"repeatability_{mode}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{mode}] {N_TRIALS} trials -> {uniq} unique outputs", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "invariant"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for m in ("baseline", "invariant"):
            env = os.environ.copy()
            env.pop("VLLM_BATCH_INVARIANT", None)
            subprocess.run(
                [sys.executable, __file__, "--mode", m],
                check=True, env=env, cwd=str(EXP_DIR),
            )
    elif args.mode:
        run_one(args.mode)
    else:
        ap.error("need --mode or --all")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑全部**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=.:$(pwd)/.. python diagnostics/repeatability.py --all
```

Expected:
- `baseline`: n_unique 是 10+ 几十的量级
- `invariant`: n_unique = 1

- [ ] **Step 3: Commit**

```bash
git add exp/grpo_batch_invariance/diagnostics/repeatability.py
git commit -m "exp: repeatability diagnostic — same prompt × N trials uniqueness"
```

---

### Task 1.5: train/train_grpo.sh — 单次训练命令

**Files:**
- Create: `exp/grpo_batch_invariance/train/train_grpo.sh`

- [ ] **Step 1: 写训练 shell**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   BIM_MODE=baseline  SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
#   BIM_MODE=invariant SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
#
# 要求当前工作目录为仓库根（含 batch_invariant_ops/）。

: "${BIM_MODE:?need BIM_MODE=baseline|invariant}"
: "${SEED:?need SEED=integer}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/exp/grpo_batch_invariance"
OUTPUT_DIR="$EXP_DIR/results/runs/${BIM_MODE}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

# 寻找 swift 安装路径下的 gsm8k plugin
SWIFT_DIR=$(python -c "import swift, os; print(os.path.dirname(swift.__file__))")
GSM8K_PLUGIN="${SWIFT_DIR}/../examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py"
if [[ ! -f "$GSM8K_PLUGIN" ]]; then
  echo "gsm8k_plugin.py not found at $GSM8K_PLUGIN. Locate it under ms-swift examples and set GSM8K_PLUGIN env var." >&2
  exit 1
fi

SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

cd "$REPO_ROOT"
export PYTHONPATH="${EXP_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export BIM_MODE
# 关闭 fused MLP 路径（spec §1.3）
export DISABLE_LIGER_KERNEL=1

python "$EXP_DIR/launcher.py" rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen3-1.7B \
  --external_plugins "$GSM8K_PLUGIN" \
  --reward_funcs gsm8k_accuracy gsm8k_format \
  --columns '{"answer": "solution"}' \
  --enable_thinking false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.4 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len 10240 \
  --sleep_level 1 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --use_liger_kernel false \
  --dataset 'modelscope/gsm8k' \
  --load_from_cache_file true \
  --max_length 2048 \
  --max_completion_length 8192 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --save_steps 10 \
  --save_total_limit 100 \
  --logging_steps 1 \
  --warmup_ratio 0.0 \
  --dataloader_num_workers 4 \
  --num_generations 8 \
  --temperature 1.0 \
  --system "$SYSTEM_PROMPT" \
  --deepspeed zero2 \
  --log_completions true \
  --report_to tensorboard \
  --max_grad_norm 1.0 \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  --scale_rewards none \
  --seed "$SEED" \
  --max_steps 50 \
  --output_dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/train.log"

echo "[done] $OUTPUT_DIR"
```

```bash
chmod +x exp/grpo_batch_invariance/train/train_grpo.sh
```

- [ ] **Step 2: 5-step pilot 测速（baseline）**

先把 `--max_steps 50` 临时改为 `--max_steps 5` 跑一次，确认能启动 + 测速。

```bash
sed -i.bak 's/--max_steps 50/--max_steps 5/' exp/grpo_batch_invariance/train/train_grpo.sh
BIM_MODE=baseline SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
```

Expected: 训练正常跑 5 步，记录每步时间 t_baseline。

- [ ] **Step 3: 5-step pilot 测速（invariant）**

```bash
BIM_MODE=invariant SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
```

Expected: 训练正常跑 5 步，记录每步时间 t_invariant。

**决策点（spec R2 关联）**：若 `t_invariant / t_baseline > 2.0` 且单 run 预算超 6h，把 `--max_steps` 改为 30，并在 plan 备注中记录。否则恢复 `--max_steps 50`。

```bash
mv exp/grpo_batch_invariance/train/train_grpo.sh.bak exp/grpo_batch_invariance/train/train_grpo.sh
# 或保持 50 不变
```

- [ ] **Step 4: Commit**

```bash
git add exp/grpo_batch_invariance/train/train_grpo.sh
git commit -m "exp: train_grpo.sh single-run training command via launcher"
```

---

### Task 1.6: train/run_all.sh — 矩阵跑 6 个 run

**Files:**
- Create: `exp/grpo_batch_invariance/train/run_all.sh`

- [ ] **Step 1: 写矩阵执行 shell**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Usage: bash exp/grpo_batch_invariance/train/run_all.sh
# 顺序跑 {baseline, invariant} × {seed=42,43,44} = 6 runs。
# 跳过已存在的 output_dir（断点续跑）。

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/exp/grpo_batch_invariance"

MODES=(baseline invariant)
SEEDS=(42 43 44)

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="$EXP_DIR/results/runs/${mode}_seed${seed}"
    # 用 trainer_state.json 是否存在判定完成
    if [[ -f "$out/trainer_state.json" ]]; then
      echo "[skip] $out already finished"
      continue
    fi
    echo "[run] BIM_MODE=$mode SEED=$seed"
    BIM_MODE="$mode" SEED="$seed" bash "$EXP_DIR/train/train_grpo.sh"
  done
done

echo "all runs done"
```

```bash
chmod +x exp/grpo_batch_invariance/train/run_all.sh
```

- [ ] **Step 2: 实际执行**

```bash
bash exp/grpo_batch_invariance/train/run_all.sh
```

Expected: 6 个 run 全部完成；`results/runs/{baseline,invariant}_seed{42,43,44}/trainer_state.json` 都存在。

- [ ] **Step 3: Commit**

```bash
git add exp/grpo_batch_invariance/train/run_all.sh
git commit -m "exp: run_all.sh — 2 mode × 3 seed matrix with resume"
```

---

### Task 1.7: eval/eval_gsm8k.sh + eval_all.sh — 评测脚本

**Files:**
- Create: `exp/grpo_batch_invariance/eval/eval_gsm8k.sh`
- Create: `exp/grpo_batch_invariance/eval/eval_all.sh`

- [ ] **Step 1: 写单 checkpoint 评测脚本**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Usage: bash eval/eval_gsm8k.sh <checkpoint_dir> <output_json>
CKPT="${1:?need checkpoint dir}"
OUT="${2:?need output json path}"
mkdir -p "$(dirname "$OUT")"

CUDA_VISIBLE_DEVICES=0 swift eval \
  --model "$CKPT" \
  --enable_thinking false \
  --eval_dataset gsm8k \
  --eval_backend Native --infer_backend vllm \
  --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
  --eval_output_dir "$(dirname "$OUT")" 2>&1 | tee "${OUT}.log"

# swift eval 输出在 eval_output_dir 内，找到 gsm8k 的 result json 复制到 OUT
result=$(find "$(dirname "$OUT")" -name "gsm8k*.json" -newer "${OUT}.log" -print | head -1)
if [[ -n "$result" ]]; then
  cp "$result" "$OUT"
  echo "[eval] $CKPT -> $OUT"
fi
```

```bash
chmod +x exp/grpo_batch_invariance/eval/eval_gsm8k.sh
```

- [ ] **Step 2: 写 eval_all.sh — 扫所有 run × 5 个 step**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Usage: bash exp/grpo_batch_invariance/eval/eval_all.sh

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/exp/grpo_batch_invariance"
EVAL_DIR="$EXP_DIR/results/eval"
mkdir -p "$EVAL_DIR"

STEPS=(10 20 30 40 50)
MODES=(baseline invariant)
SEEDS=(42 43 44)

# baseline (step 0 / 初始模型) 也评一次共享，作为基准
BASELINE_OUT="$EVAL_DIR/base_qwen3-1.7b.json"
if [[ ! -f "$BASELINE_OUT" ]]; then
  bash "$EXP_DIR/eval/eval_gsm8k.sh" "Qwen/Qwen3-1.7B" "$BASELINE_OUT"
fi

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$EXP_DIR/results/runs/${mode}_seed${seed}"
    for step in "${STEPS[@]}"; do
      ckpt="$run_dir/checkpoint-${step}"
      out="$EVAL_DIR/${mode}_seed${seed}_step${step}.json"
      if [[ ! -d "$ckpt" ]]; then
        echo "[skip] no checkpoint: $ckpt"
        continue
      fi
      if [[ -f "$out" ]]; then
        echo "[skip] $out exists"
        continue
      fi
      bash "$EXP_DIR/eval/eval_gsm8k.sh" "$ckpt" "$out"
    done
  done
done

echo "all eval done"
```

```bash
chmod +x exp/grpo_batch_invariance/eval/eval_all.sh
```

- [ ] **Step 3: 执行评测**

```bash
bash exp/grpo_batch_invariance/eval/eval_all.sh
```

Expected: `results/eval/` 下生成 1 + 30 个 json（baseline + 6 run × 5 step）。

- [ ] **Step 4: Commit**

```bash
git add exp/grpo_batch_invariance/eval/eval_gsm8k.sh exp/grpo_batch_invariance/eval/eval_all.sh
git commit -m "exp: gsm8k eval scripts — single checkpoint + sweep all runs"
```

---

### Task 1.8: results 自动汇总脚本

**Files:**
- Create: `exp/grpo_batch_invariance/summarize.py`

- [ ] **Step 1: 写汇总脚本**

```python
"""Aggregate all results into results/summary.md + plots.

输入:
  results/diagnostics/cell_{A,B,C,D}.json
  results/diagnostics/repeatability_{baseline,invariant}.json
  results/eval/{mode}_seed{seed}_step{k}.json
  results/runs/{mode}_seed{seed}/trainer_state.json (训练曲线)

输出:
  results/summary.md
  results/figures/reward_curve.png
  results/figures/acc_per_step.png
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RES = EXP_DIR / "results"
FIG = RES / "figures"
RUNS = RES / "runs"
EVAL = RES / "eval"
DIAG = RES / "diagnostics"

MODES = ("baseline", "invariant")
SEEDS = (42, 43, 44)
STEPS = (10, 20, 30, 40, 50)


def load_eval_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # swift eval JSON 结构: {"results": {"gsm8k": {"accuracy": x, ...}}}
    return data.get("results", {}).get("gsm8k", {}).get("accuracy")


def load_trainer_state(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("log_history", [])


def plot_acc_per_step() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_init = load_eval_accuracy(EVAL / "base_qwen3-1.7b.json") or 0.0
    ax.axhline(baseline_init, color="gray", linestyle="--", label=f"Qwen3-1.7B init = {baseline_init:.3f}")

    for mode in MODES:
        for seed in SEEDS:
            accs = []
            for step in STEPS:
                acc = load_eval_accuracy(EVAL / f"{mode}_seed{seed}_step{step}.json")
                accs.append(acc)
            ax.plot(STEPS, accs, marker="o",
                    label=f"{mode} seed={seed}",
                    color="C0" if mode == "baseline" else "C1",
                    alpha=0.7)
    ax.set_xlabel("training step")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("GSM8K accuracy across training steps (2 mode × 3 seed)")
    ax.legend()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "acc_per_step.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_reward_curve() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        for seed in SEEDS:
            ts = load_trainer_state(RUNS / f"{mode}_seed{seed}" / "trainer_state.json")
            steps = [e["step"] for e in ts if "reward" in e]
            rewards = [e["reward"] for e in ts if "reward" in e]
            if not steps:
                continue
            ax.plot(steps, rewards,
                    label=f"{mode} seed={seed}",
                    color="C0" if mode == "baseline" else "C1",
                    alpha=0.7)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean reward")
    ax.set_title("GRPO reward curve")
    ax.legend()
    out = FIG / "reward_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def write_summary_md() -> None:
    lines: List[str] = ["# Experiment Summary\n"]

    # Diagnostics
    lines.append("## 1. Logprob mismatch (4-cell diagnostic)\n")
    lines.append("| cell | config | mean|Δ| | max|Δ| | frac>1e-3 |")
    lines.append("|---|---|---|---|---|")
    cell_desc = {"A": "baseline", "B": "trainer-only", "C": "vllm-only", "D": "both invariant"}
    for c in "ABCD":
        p = DIAG / f"cell_{c}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        lines.append(f"| {c} | {cell_desc[c]} | {d['mean_abs_delta']:.2e} | "
                     f"{d['max_abs_delta']:.2e} | {d['frac_gt_1e-3']:.3f} |")
    lines.append("")

    # Repeatability
    lines.append("## 2. Repeatability (same prompt × 200 trials)\n")
    lines.append("| mode | n_unique |")
    lines.append("|---|---|")
    for m in ("baseline", "invariant"):
        p = DIAG / f"repeatability_{m}.json"
        if p.exists():
            d = json.loads(p.read_text())
            lines.append(f"| {m} | {d['n_unique']}/{d['n_trials']} |")
    lines.append("")

    # Final accuracy
    lines.append("## 3. GSM8K accuracy by step (mean ± std across 3 seeds)\n")
    lines.append("| step | baseline | invariant | Δ |")
    lines.append("|---|---|---|---|")
    for step in STEPS:
        per_mode: Dict[str, List[float]] = {m: [] for m in MODES}
        for mode in MODES:
            for seed in SEEDS:
                acc = load_eval_accuracy(EVAL / f"{mode}_seed{seed}_step{step}.json")
                if acc is not None:
                    per_mode[mode].append(acc)
        if not (per_mode["baseline"] and per_mode["invariant"]):
            continue
        b_mean, b_std = statistics.mean(per_mode["baseline"]), statistics.pstdev(per_mode["baseline"])
        i_mean, i_std = statistics.mean(per_mode["invariant"]), statistics.pstdev(per_mode["invariant"])
        lines.append(f"| {step} | {b_mean:.4f} ± {b_std:.4f} | "
                     f"{i_mean:.4f} ± {i_std:.4f} | {i_mean - b_mean:+.4f} |")
    lines.append("")

    lines.append("## 4. Figures\n")
    lines.append("- `figures/logprob_diff_hist.png` — 4-cell logprob diff histograms")
    lines.append("- `figures/reward_curve.png` — GRPO reward across 6 runs")
    lines.append("- `figures/acc_per_step.png` — GSM8K accuracy per step")
    lines.append("")

    lines.append("## 5. Findings\n")
    lines.append("_<填写：机制是否被验证 → 训练动态差异 → 最终指标差异>_")

    out = RES / "summary.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    plot_acc_per_step()
    plot_reward_curve()
    write_summary_md()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑汇总**

```bash
cd exp/grpo_batch_invariance
PYTHONPATH=.:$(pwd)/.. python summarize.py
```

Expected: 生成 `results/summary.md` + 两张 PNG。

- [ ] **Step 3: 人工撰写 Findings 段**

打开 `results/summary.md`，把第 5 节占位文字替换为基于 diagnostics + 训练曲线 + accuracy 表的结论叙述（3-5 句）。

- [ ] **Step 4: Commit**

```bash
git add exp/grpo_batch_invariance/summarize.py exp/grpo_batch_invariance/results/summary.md
git commit -m "exp: results summarization script + findings writeup"
```

---

## 完成判定

实验完成当且仅当以下都达成：

- [ ] Phase 0: `pytest exp/grpo_batch_invariance/ops_extension/tests/ -v` 全 PASS
- [ ] 诊断 D cell `mean|Δ| < 1e-6` 且 `frac>1e-3 == 0`
- [ ] 6 个训练 run 全部完成（`trainer_state.json` 存在）
- [ ] 31 个评测 json 生成
- [ ] `results/summary.md` 含完整表格、两张图、Findings 段

---

## Deviations from spec

Plan 与 spec §3、§6 在一处不同，记录如下：

**spec 写了 `train/plugins/logprob_probe.py`（per-step rollout-vs-forward probe）**——plan 不实现这个文件。原因：
1. swift GRPO trainer 默认就向 tensorboard 记录每 step 的 `policy/clip_frac`、`ppo/approx_kl`、`reward/mean/std`、`grad_norm` 等关键 GRPO 内部指标（spec §6 第二行所述）
2. rollout-vs-forward logprob diff 不是每 step 都需要——它在训练前（初始模型）反映"机制是否被打掉"，在训练后（step 50 ckpt）反映"是否 drift 回来"。中间步骤的 diff 只是 policy 缓慢漂移的线性插值，per-step probe 信息量低、还会拖慢训练
3. 实际工作流：每个 run 完成后跑 `diagnostics/logprob_mismatch.py --cell D`，用其 step-50 ckpt 替代 base 模型采样一次，比 per-step probe 简单稳定

如果训练后发现需要 mid-training trajectory，再加一个 callback；YAGNI 优先。
