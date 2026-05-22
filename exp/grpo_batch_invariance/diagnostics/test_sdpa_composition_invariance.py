"""测试 sdpa_batch_invariant 作为一个整体在 Lq 变化时中间行是否 bit-exact 不变。

每个组件单独测过 M-invariant：matmul 单行+中间行+变 K、softmax、RMSNorm、RoPE。
但 layer divergence 显示 layer 4 K=identical 但 layer 5 K_cache@pos27 leak,
意味着 attention 整体（matmul → mask → softmax → matmul）在中间行不 invariant。

可能原因：
  1. K.transpose(-2, -1) 在不同 Lk 下 stride 不同，下游 matmul 走不同 kernel
  2. torch.ones().triu() 在不同 Lq/Lk 下 mask tensor stride/contiguity 不同
  3. masked_fill 在 mask 形状不同时走不同 fast/slow path
  4. .float() / .to(bf16) 类型转换在 batch 形状变化时不一致

直接测：random Q/K/V 长度变化时，sdpa_batch_invariant 输出在行 27 是否一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode
from ops_extension.sdpa import sdpa_batch_invariant


def main():
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()

    B, H, D = 1, 16, 128
    torch.manual_seed(42)
    L_max = 50
    Q_all = torch.randn(B, H, L_max, D, dtype=torch.bfloat16, device="cuda")
    K_all = torch.randn(B, H, L_max, D, dtype=torch.bfloat16, device="cuda")
    V_all = torch.randn(B, H, L_max, D, dtype=torch.bfloat16, device="cuda")

    print(f"\n=== sdpa_batch_invariant composition invariance (causal) ===")
    print(f"compare row 27 output across different Lq (with same prefix Q/K/V[:Lq])")
    print(f"  {'pair':>15} {'row27 diff':>15} {'rowLqM1 diff':>15} {'all rows max':>15}")
    pairs = [(47, 48), (46, 47), (45, 47), (40, 48), (32, 48), (16, 48), (15, 16), (31, 32), (63, 64)]
    for L_a, L_b in pairs:
        Q_a, K_a, V_a = Q_all[:, :, :L_a, :], K_all[:, :, :L_a, :], V_all[:, :, :L_a, :]
        Q_b, K_b, V_b = Q_all[:, :, :L_b, :], K_all[:, :, :L_b, :], V_all[:, :, :L_b, :]
        with torch.no_grad():
            out_a = sdpa_batch_invariant(Q_a, K_a, V_a, is_causal=True)
            out_b = sdpa_batch_invariant(Q_b, K_b, V_b, is_causal=True)
        # Compare overlap rows (0..L_a-1)
        diff_overlap = (out_a - out_b[:, :, :L_a, :]).abs()
        row27_diff = diff_overlap[:, :, 27, :].max().item() if L_a > 27 else float("nan")
        last_a_diff = diff_overlap[:, :, L_a - 1, :].max().item()
        all_diff = diff_overlap.max().item()
        print(f"  {f'{L_a}x{L_a} vs {L_b}x{L_b}':>15} {row27_diff:>15.3e} {last_a_diff:>15.3e} {all_diff:>15.3e}")


if __name__ == "__main__":
    main()
