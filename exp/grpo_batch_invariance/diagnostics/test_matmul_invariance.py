"""Minimum reproducer: 是否 batch_invariant_ops 的 matmul 在变化的 M 维度下 bit-exact 不变？

复刻 attention 里的形状：(B, H, Lq, D) @ (B, H, D, Lk)。
比较：
  - 一次性算 scores_full = matmul(Q_full[:, :, :Lq, :], K^T)  → (B, H, Lq, Lk)
  - 单行算   scores_one  = matmul(Q_full[:, :, Lq-1:Lq, :], K^T) → (B, H, 1, Lk)
两者在 row Lq-1 应该 bit-exact 一致。如果不是，matmul 在 M 维不变性失败 ——
这正好对应 decode (Lq=1) vs prefill (Lq=Lk) 的对比。
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
    print(f"shapes: B={B}, H={H}, D={D}, dtype=bf16")
    print(f"{'Lk':>5} {'diff':>15}  note")

    for Lk in [8, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81, 95, 96, 97]:
        torch.manual_seed(42 + Lk)
        Q = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

        # Full Lq×Lk matmul
        scores_full = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, Lk, Lk)
        # Single row matmul (just row Lk-1)
        scores_one = torch.matmul(Q[:, :, Lk - 1 : Lk, :], K.transpose(-2, -1))  # (B, H, 1, Lk)

        diff = (scores_full[:, :, Lk - 1 : Lk, :] - scores_one).abs().max().item()
        note = "← multiple of 16" if Lk % 16 == 0 else ""
        print(f"{Lk:>5} {diff:>15.3e}  {note}")


if __name__ == "__main__":
    main()
