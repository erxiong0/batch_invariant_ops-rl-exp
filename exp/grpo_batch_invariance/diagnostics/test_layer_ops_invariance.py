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


def test_rotary_embedding_generation():
    """Qwen3RotaryEmbedding(position_ids) → (cos, sin) 在 position_ids shape 不同时是否不变。

    这才是真正模拟 decode 和 prefill 的差别：
      decode  : position_ids = [[Lk-1]]              （单点）
      prefill : position_ids = [[0, 1, ..., Lk-1]]   （全部）
    比较 prefill output 在 Lk-1 位置 vs decode output。
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
    rope = Qwen3RotaryEmbedding(config=config).cuda()

    print(f"\n=== Qwen3RotaryEmbedding M-invariance (cache-position semantics) ===")
    print(f"  {'Lk':>5} {'cos diff':>15} {'sin diff':>15}  note")
    for Lk in LK_LIST:
        # x 只用来推 dtype/device，不影响 cos/sin 计算
        x = torch.empty(1, 1, dtype=torch.bfloat16, device="cuda")
        pos_full = torch.arange(Lk, device="cuda").unsqueeze(0)         # (1, Lk)
        pos_one = torch.tensor([[Lk - 1]], device="cuda")               # (1, 1)
        with torch.no_grad():
            cos_full, sin_full = rope(x, pos_full)
            cos_one, sin_one = rope(x, pos_one)
        # cos_full/sin_full shape: (1, Lk, head_dim)；取最后一行跟 cos_one 比
        cos_diff = (cos_full[:, Lk - 1 : Lk, :] - cos_one).abs().max().item()
        sin_diff = (sin_full[:, Lk - 1 : Lk, :] - sin_one).abs().max().item()
        note = "← multiple of 16" if Lk % 16 == 0 else ""
        print(f"  {Lk:>5} {cos_diff:>15.3e} {sin_diff:>15.3e}  {note}")


def test_apply_rotary():
    """apply_rotary_pos_emb 给定相同 cos/sin 时，对 Q[full] 取行 vs 对 Q[one] 直接旋转是否一致。"""
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
    test_rotary_embedding_generation()  # 真正模拟 cache-position 语义
    test_apply_rotary()


if __name__ == "__main__":
    main()
