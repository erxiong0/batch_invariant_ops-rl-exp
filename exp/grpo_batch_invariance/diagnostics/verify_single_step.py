"""单步 decode vs prefill 对比诊断（启用所有 invariant patches）。

目的：验证"残余 5.63e-3 gap 是 decode-vs-prefill 路径差异"还是别的原因。

流程：
  1. 取一个 prompt，model.generate 生成 N tokens，记录每步 gen.scores[t]
  2. 对每个 t，做一次 full forward on [prompt + gen_tokens[0..t]]，
     从 logits[L-1+t-1] 取相同位置的 logits
  3. 比较两组 logits 的逐元素 diff（不只是采样到的 token，是整条 vocab）

输出：
  - per-step max|Δ logits| 和 mean|Δ logits|
  - 也算采样 token 处的 log_softmax 差（与 logprob_mismatch.py 对齐）

如果 t=0 时 diff = 0 → 单步无 gap，5.63e-3 是 ≥1 步以后的累积
如果 t=0 时已经 diff > 0 → decode 路径就有问题（哪怕单步）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = ("You are a helpful math assistant. Solve the problem step by step "
          "and put your final answer within \\boxed{}.\n\n"
          "Question: What is 13 * 47?\nAnswer:")
N_TOKENS = 10
SEED = 12345


def main():
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

    # --- Rollout: generate N tokens, capture per-step logits ---
    torch.manual_seed(SEED)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=N_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.pad_token_id,
        )
    gen_tokens = gen.sequences[0, p_len:].tolist()
    rollout_logits = [s[0].float() for s in gen.scores]  # list of (V,) tensors

    print(f"\nPrompt length: {p_len}")
    print(f"Generated tokens: {gen_tokens}")
    print(f"Generated text: {tok.decode(gen_tokens)!r}\n")

    # --- Prefill: for each step t, full forward on [prompt + gen[0..t]],
    #     extract logits at position p_len-1+t (predicts gen_tokens[t]) ---
    print(f"{'step':>4} | {'max|Δlogits|':>14} | {'mean|Δlogits|':>14} | "
          f"{'Δlp(sampled)':>14} | {'sampled_tok':>11}")
    print("-" * 80)

    for t in range(N_TOKENS):
        # full forward conditioned on [prompt + gen_tokens[0..t-1]]; prediction at position p_len-1+t
        context = enc.input_ids.tolist()[0] + gen_tokens[:t]
        ids = torch.tensor([context], device="cuda")
        with torch.no_grad():
            out = model(ids)
        prefill_logits_t = out.logits[0, -1, :].float()  # (V,)

        diff = (rollout_logits[t] - prefill_logits_t).abs()
        # also compare logprob of the sampled token
        roll_lp = F.log_softmax(rollout_logits[t], dim=-1)[gen_tokens[t]].item()
        pre_lp = F.log_softmax(prefill_logits_t, dim=-1)[gen_tokens[t]].item()

        print(f"{t:>4} | {diff.max().item():>14.4e} | {diff.mean().item():>14.4e} | "
              f"{abs(roll_lp - pre_lp):>14.4e} | {gen_tokens[t]:>11}")


if __name__ == "__main__":
    main()
