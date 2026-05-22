"""测试 softmax / log_softmax+exp 在变化的 M 维度下是否 bit-exact 不变。

复刻 attention 内部从 scores 到 probs 的两条路径：
  Path A (eager 风格)：  softmax(scores, dim=-1, dtype=float32).to(bf16)
  Path B (我们当前)：    log_softmax(scores.float(), dim=-1).exp().to(bf16)

每个 Lk 比较：
  full = path(scores[B,H,Lk,Lk])[:, :, Lk-1:Lk, :]
  one  = path(scores[B,H,1, Lk])
两者应 bit-exact 一致。如果某一路径在某些 Lk（特别是 16 倍数）diff > 0,
就找到 sdpa_batch_invariant 16-aligned 残差的根因。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


def main():
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()

    B, H = 1, 16
    print(f"shapes: B={B}, H={H}, dtype=bf16 in/out, float32 reduction")
    print(f"{'Lk':>5} {'softmax diff':>16} {'log_softmax+exp diff':>22}  note")

    for Lk in [8, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81, 95, 96, 97]:
        torch.manual_seed(42 + Lk)
        scores = torch.randn(B, H, Lk, Lk, dtype=torch.bfloat16, device="cuda")

        # Path A: softmax
        probs_full_A = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        probs_one_A = F.softmax(scores[:, :, Lk - 1 : Lk, :], dim=-1, dtype=torch.float32).to(torch.bfloat16)
        diff_A = (probs_full_A[:, :, Lk - 1 : Lk, :] - probs_one_A).abs().max().item()

        # Path B: log_softmax + exp
        probs_full_B = F.log_softmax(scores.float(), dim=-1).exp().to(torch.bfloat16)
        probs_one_B = F.log_softmax(scores[:, :, Lk - 1 : Lk, :].float(), dim=-1).exp().to(torch.bfloat16)
        diff_B = (probs_full_B[:, :, Lk - 1 : Lk, :] - probs_one_B).abs().max().item()

        note = "← multiple of 16" if Lk % 16 == 0 else ""
        print(f"{Lk:>5} {diff_A:>16.3e} {diff_B:>22.3e}  {note}")


if __name__ == "__main__":
    main()
