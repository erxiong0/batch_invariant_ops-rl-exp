"""测试 Qwen3RMSNorm 和 apply_rotary_pos_emb 在 M 维（seq_len）下是否 bit-exact 不变。

Attention 路径上 matmul / softmax 都已经证明 M-invariant，但 sdpa 模式下 16-aligned
diff 还在。剩下可疑的：
  1. RMSNorm（input_layernorm / post_attention_layernorm / q_norm / k_norm 共 4 处）
  2. apply_rotary_pos_emb（rope）
两个都应该是 per-position 独立计算，理论上 M-invariant。任何一个在 16 倍数处 diff > 0
就是 sdpa 16-aligned 残差的源头。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


LK_LIST = [8, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 79, 80, 81, 95, 96, 97]


def test_qwen3_rmsnorm():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    print(f"\n=== Qwen3RMSNorm M-invariance ===")
    for hidden_size in (128, 2048):
        rms = Qwen3RMSNorm(hidden_size=hidden_size, eps=1e-6).cuda().to(torch.bfloat16)
        with torch.no_grad():
            rms.weight.copy_(torch.randn_like(rms.weight))
        print(f"\n  hidden_size={hidden_size}")
        print(f"  {'Lk':>5} {'diff':>15}  note")
        for Lk in LK_LIST:
            torch.manual_seed(42 + Lk)
            x = torch.randn(1, Lk, hidden_size, dtype=torch.bfloat16, device="cuda")
            with torch.no_grad():
                y_full = rms(x)
                y_one = rms(x[:, Lk - 1 : Lk, :])
            diff = (y_full[:, Lk - 1 : Lk, :] - y_one).abs().max().item()
            note = "← multiple of 16" if Lk % 16 == 0 else ""
            print(f"  {Lk:>5} {diff:>15.3e}  {note}")


def test_rotary():
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8

    print(f"\n=== apply_rotary_pos_emb M-invariance "
          f"(head_dim={head_dim}, num_q={num_q_heads}, num_kv={num_kv_heads}) ===")
    print(f"  {'Lk':>5} {'Q diff':>15} {'K diff':>15}  note")
    for Lk in LK_LIST:
        torch.manual_seed(42 + Lk)
        Q = torch.randn(1, num_q_heads, Lk, head_dim, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(1, num_kv_heads, Lk, head_dim, dtype=torch.bfloat16, device="cuda")
        # 在 transformers 4.54 里 cos/sin shape 是 (1, Lk, head_dim)。
        cos = torch.randn(1, Lk, head_dim, dtype=torch.bfloat16, device="cuda")
        sin = torch.randn(1, Lk, head_dim, dtype=torch.bfloat16, device="cuda")

        with torch.no_grad():
            Q_full_rot, K_full_rot = apply_rotary_pos_emb(Q, K, cos, sin)
            Q_one_rot, K_one_rot = apply_rotary_pos_emb(
                Q[:, :, Lk - 1 : Lk, :], K[:, :, Lk - 1 : Lk, :],
                cos[:, Lk - 1 : Lk, :], sin[:, Lk - 1 : Lk, :],
            )

        q_diff = (Q_full_rot[:, :, Lk - 1 : Lk, :] - Q_one_rot).abs().max().item()
        k_diff = (K_full_rot[:, :, Lk - 1 : Lk, :] - K_one_rot).abs().max().item()
        note = "← multiple of 16" if Lk % 16 == 0 else ""
        print(f"  {Lk:>5} {q_diff:>15.3e} {k_diff:>15.3e}  {note}")


def main():
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()
    test_qwen3_rmsnorm()
    test_rotary()


if __name__ == "__main__":
    main()
