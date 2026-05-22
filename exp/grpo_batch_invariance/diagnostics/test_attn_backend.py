"""测试 attention backend 对 decode-vs-prefill 16-aligned diff 的影响。

承接 test_cache_artifact.py 的结论：DynamicCache 的 torch.cat 不是元凶，
diff 精准卡在 KV length = 16 的倍数（48/64/80/96）→ 强烈指向 SDPA kernel
内部的 block tiling 在跨 16-边界时改变累加顺序。

本脚本对同一 prompt/seed 跑三种 attn_implementation：
  - sdpa（默认，复现 baseline 的 16-aligned pattern）
  - eager（纯 PyTorch matmul，无 block tiling）
  - flash_attention_2（另一套 block 划分；若未装则跳过）

判读：
  - eager diff 消失 → 确诊 SDPA kernel 的 block 切分是 artifact 源头
  - eager diff 还在 → 16-aligned 是更深层原因（rope / norm / 别的）
  - flash_attention_2 diff 换到不同位置（如 64 的倍数）→ 旁证 block-size 假设
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = ("You are a helpful math assistant. Solve the problem step by step "
          "and put your final answer within \\boxed{}.\n\n"
          "Question: What is 13 * 47?\nAnswer:")
N_TOKENS = int(os.environ.get("TAB_N_TOKENS", "64"))
SEED = 12345


def run(attn_impl: str):
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
        ).cuda().eval()
    except (ImportError, ValueError) as e:
        print(f"\n[attn={attn_impl}] SKIPPED: {e}")
        return

    enc = tok(PROMPT, return_tensors="pt").to("cuda")
    p_len = enc.input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=N_TOKENS,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.pad_token_id,
    )

    torch.manual_seed(SEED)
    with torch.no_grad():
        gen = model.generate(**enc, **gen_kwargs)
    gen_tokens = gen.sequences[0, p_len:].tolist()
    rollout_logits = [s[0].float() for s in gen.scores]

    per_step_logits = []
    with torch.no_grad():
        for t in range(N_TOKENS):
            context = enc.input_ids.tolist()[0] + gen_tokens[:t]
            ids = torch.tensor([context], device="cuda")
            out = model(ids)
            per_step_logits.append(out.logits[0, -1, :].float())

    diffs = [(rollout_logits[t] - per_step_logits[t]).abs().max().item()
             for t in range(N_TOKENS)]

    abs_positions = [p_len + t for t in range(N_TOKENS) if diffs[t] > 0]
    print(f"\n[attn={attn_impl}] prompt_len={p_len}, gen_len={N_TOKENS}")
    print(f"  positions with diff > 0:           {abs_positions}")
    print(f"  diff values at those positions:    {[f'{diffs[t]:.3e}' for t in range(N_TOKENS) if diffs[t] > 0]}")
    print(f"  fraction of positions with diff:   {sum(1 for d in diffs if d > 0)}/{N_TOKENS}")
    print(f"  max diff:                          {max(diffs):.3e}")

    # 暴露 wrapper 调用次数 —— 若 count==0 说明 dispatch 没经过我们；
    # 若 count==1 说明只第一次 prefill 经过、后续 decode + per_step 都旁路了；
    # 若 count >> 1 说明每次 attention 都走 wrapper，diff 的根因不在 dispatch 而在 wrapper 内部
    try:
        from ops_extension.sdpa import _INVARIANT_CALL_COUNT
        print(f"  [count] _sdpa_attention_forward_invariant total calls: {_INVARIANT_CALL_COUNT[0]}")
    except ImportError:
        pass

    del model
    torch.cuda.empty_cache()


def main():
    for impl in ("sdpa", "eager", "flash_attention_2"):
        run(impl)


if __name__ == "__main__":
    main()
