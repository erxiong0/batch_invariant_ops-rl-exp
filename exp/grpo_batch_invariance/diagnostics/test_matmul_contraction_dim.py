"""测试 matmul 在 contraction 维度 K 变化时的 bit-exact 不变性。

GEMM(M, N, K) 的 K（contraction）维由 cublas 决定 reduction 顺序。
我们之前测的"matmul middle-row invariance"里，K=D=128 永远不变（只有 M、N 变）。
但 attention 第二个 matmul `probs @ V`：
    形状 (Lq, Lk) @ (Lk, D)  → GEMM(M=Lq, N=D, K=Lk)
**Lk 变化时 K 变化**，cublas 可能选不同 split-K，使 reduction 顺序变。

本测试：
  out_short = matmul(A[:, :Lk_s], V[:Lk_s, :])           # K=Lk_s
  out_long  = matmul(A_padded[:, :Lk_l], V[:Lk_l, :])    # K=Lk_l，A 在 [Lk_s:] 位置补 0
两边数学上等价（0×V=0），但若 cublas 在 K=Lk_s vs K=Lk_l 选不同 kernel,
out_short 跟 out_long 会 bit 不等 —— 这正是 layer 5 K_cache pos=27 leak 的源头：
attention 第二个 matmul 在 Lk=47 vs Lk=48 算出来不一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


def main():
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()

    B, H, D = 1, 16, 128
    Lq = 1  # 模拟 decode：query 单行
    pairs = [(47, 48), (46, 47), (45, 47), (40, 48), (32, 48), (16, 48), (47, 49), (47, 50), (15, 16), (31, 32)]

    print(f"\n=== matmul contraction-K invariance (Lq=1, D={D}) ===")
    print(f"  '0-pad' 表示 A 在 [Lk_s..Lk_l-1] 补 0，仿真 causal mask 之外的位置")
    for Lk_s, Lk_l in pairs:
        torch.manual_seed(42)
        # A: softmax 结果一样的 shape (B, H, Lq, Lk_l)，但只前 Lk_s 列非零
        A = torch.randn(B, H, Lq, Lk_l, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(B, H, Lk_l, D, dtype=torch.bfloat16, device="cuda")

        # short: A 和 V 都截到 Lk_s
        out_short = torch.matmul(A[:, :, :, :Lk_s], V[:, :, :Lk_s, :])

        # long: A 在 [Lk_s..] 补 0；V 用全长
        A_padded = A.clone()
        A_padded[:, :, :, Lk_s:] = 0
        out_long = torch.matmul(A_padded, V)

        diff = (out_short - out_long).abs().max().item()
        note = "← 16-boundary" if (Lk_l % 16 == 0 or Lk_s % 16 == 0) else ""
        print(f"  K={Lk_s} vs K={Lk_l} (with 0-pad): diff={diff:.3e}  {note}")


if __name__ == "__main__":
    main()
