"""测试 KV cache 类型对 decode-vs-prefill diff 的影响。

Hypothesis: 默认 DynamicCache 每步 torch.cat 重分配，跨 16-block 时
内存对齐/重排导致数值漂移。换 StaticCache（一次性预分配 max_length）
应该消除这个 artifact。

跑两遍：DynamicCache（默认）和 StaticCache，对同一 prompt 同一 seed 看
diff 出现位置是否变化或消失。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = ("You are a helpful math assistant. Solve the problem step by step "
          "and put your final answer within \\boxed{}.\n\n"
          "Question: What is 13 * 47?\nAnswer:")
N_TOKENS = int(os.environ.get("TCA_N_TOKENS", "64"))
SEED = 12345


def run(cache_impl: str | None):
    """cache_impl: None (default DynamicCache) | 'static' | 'sliding_window' | ..."""
    enable_batch_invariant_mode()
    enable_extended_batch_invariant_mode()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

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
    if cache_impl is not None:
        gen_kwargs["cache_implementation"] = cache_impl

    torch.manual_seed(SEED)
    with torch.no_grad():
        gen = model.generate(**enc, **gen_kwargs)
    gen_tokens = gen.sequences[0, p_len:].tolist()
    rollout_logits = [s[0].float() for s in gen.scores]

    # Per-step forward (always fresh, no cache)
    per_step_logits = []
    with torch.no_grad():
        for t in range(N_TOKENS):
            context = enc.input_ids.tolist()[0] + gen_tokens[:t]
            ids = torch.tensor([context], device="cuda")
            out = model(ids)
            per_step_logits.append(out.logits[0, -1, :].float())

    # Compare
    diffs = []
    for t in range(N_TOKENS):
        diff = (rollout_logits[t] - per_step_logits[t]).abs().max().item()
        diffs.append(diff)

    cache_label = cache_impl or "dynamic (default)"
    abs_positions = [p_len + t for t in range(N_TOKENS) if diffs[t] > 0]
    print(f"\n[cache={cache_label}] prompt_len={p_len}, gen_len={N_TOKENS}")
    print(f"  positions with diff > 0:           {abs_positions}")
    print(f"  diff values at those positions:    {[f'{diffs[t]:.3e}' for t in range(N_TOKENS) if diffs[t] > 0]}")
    print(f"  fraction of positions with diff:   {sum(1 for d in diffs if d > 0)}/{N_TOKENS}")
    print(f"  max diff:                          {max(diffs):.3e}")


def main():
    # 跑默认（DynamicCache）
    run(None)
    # 跑 StaticCache
    run("static")


if __name__ == "__main__":
    main()
