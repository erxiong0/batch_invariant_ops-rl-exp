"""测试 matmul 在不同总 M 维下，中间行的 bit-exact 不变性。

之前的 test_matmul_invariance.py 只检了"最后一行"：
    matmul(Q[Lk], K[Lk]^T)[Lk-1] vs matmul(Q[1], K[Lk]^T)
都 0 diff，证明 batch_invariant matmul 对"单行 Q vs 全 Q 取最后一行"不变。

但 layer divergence 实验显示，47-prefill 跟 48-prefill 在 layer 4 attention
中间某行（具体是 row 27 区域）出 diff —— 也就是说 matmul 的中间行在 M 总长变化时
bit 不等。cublas 在 (Lq, Lk) 不同时可能选不同 tile/split-K，使得 row i 的累加顺序变。

本测试比较：
    M_short = matmul(Q[:Lq_short], K[:Lk_short]^T)           # 小尺寸
    M_long  = matmul(Q[:Lq_long], K[:Lk_long]^T)             # 大尺寸（包含 short 的所有行/列）
    diff_per_row = (M_short - M_long[:Lq_short, :Lk_short]).abs().amax over (B, H, col)
找第一行 diff > 0 的 row 就是泄漏起点。
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

    # 围绕 47/48 边界（layer divergence 报 pos=27 在 layer 5；attention 矩阵是 47x47 vs 48x48）
    pairs = [(47, 48), (46, 47), (45, 47), (40, 48), (32, 48), (16, 48), (47, 49), (47, 50)]

    print(f"\n=== matmul middle-row M-invariance (B={B}, H={H}, D={D}) ===")
    for L_short, L_long in pairs:
        torch.manual_seed(42)
        # 用 max(L_long, ...) 个 row 的统一随机种子源；slice 出 short / long
        Q_all = torch.randn(B, H, L_long, D, dtype=torch.bfloat16, device="cuda")
        K_all = torch.randn(B, H, L_long, D, dtype=torch.bfloat16, device="cuda")

        M_short = torch.matmul(Q_all[:, :, :L_short, :], K_all[:, :, :L_short, :].transpose(-2, -1))
        M_long  = torch.matmul(Q_all[:, :, :L_long, :],  K_all[:, :, :L_long, :].transpose(-2, -1))
        overlap = M_long[:, :, :L_short, :L_short]

        diff = (M_short - overlap).abs()
        # per-row max diff
        diff_per_row = diff.amax(dim=(0, 1, 3))  # (L_short,)
        max_diff = diff_per_row.max().item()
        first_div_row = int((diff_per_row > 0).nonzero()[0, 0]) if max_diff > 0 else -1

        print(f"  Q=K={L_short:>2}x{L_short:<2} vs {L_long:>2}x{L_long:<2}: "
              f"max diff={max_diff:.3e}, first divergent row={first_div_row}")


if __name__ == "__main__":
    main()
