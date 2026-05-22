"""Compare full forward vs per-step forward on the same gen sequence.

逻辑：
  1. 用 model.generate 拿 (gen_tokens, gen_scores)
  2. Per-step forward: 对每个 t，forward on [prompt + gen[0..t-1]]，
     从 last position 取 logits → 模拟 logprob_mismatch 实际想测的东西
     （但用的是"per-step prefill"，跟 decode 同 shape）
  3. Full forward: forward on [prompt + gen[0..N-1]] 一次性，
     从中间位置取 logits → 这是 logprob_mismatch.forward_hf 的做法
  4. 对每个 t，比较 (full_logits[t] - per_step_logits[t]).abs()

如果 full vs per-step bit-equal → 5.63e-3 gap 不来自这里，得再挖
如果不 bit-equal → 确认 gap 来源是 full-forward 的长 row log_softmax / 长 K matmul
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from batch_invariant_ops import enable_batch_invariant_mode
from ops_extension import enable_extended_batch_invariant_mode


MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = ("You are a helpful math assistant. Solve the problem step by step "
          "and put your final answer within \\boxed{}.\n\n"
          "Question: What is 13 * 47?\nAnswer:")
N_TOKENS = 30
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

    # Generate gen_tokens once (rollout path, deterministic given seed)
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
    print(f"prompt_len={p_len}, gen_len={len(gen_tokens)}, full_len={p_len+len(gen_tokens)}")

    # Path A: per-step forward (each step a fresh forward on growing prefix)
    per_step_logits = []
    with torch.no_grad():
        for t in range(N_TOKENS):
            context = enc.input_ids.tolist()[0] + gen_tokens[:t]
            ids = torch.tensor([context], device="cuda")
            out = model(ids)
            per_step_logits.append(out.logits[0, -1, :].float())

    # Path B: full forward (single call on prompt + all gen tokens)
    full_ids = torch.cat([enc.input_ids, torch.tensor([gen_tokens], device="cuda")], dim=1)
    with torch.no_grad():
        out = model(full_ids)
    full_logits = out.logits[0, p_len - 1 : -1, :].float()  # (N, V)

    # Compare per-step
    print(f"\n{'t':>3} | {'max|Δlogits|':>14} | {'mean|Δlogits|':>14} | {'Δlp(gen_tok)':>14}")
    print("-" * 70)
    for t in range(N_TOKENS):
        diff = (per_step_logits[t] - full_logits[t]).abs()
        ps_lp = F.log_softmax(per_step_logits[t], dim=-1)[gen_tokens[t]].item()
        full_lp = F.log_softmax(full_logits[t], dim=-1)[gen_tokens[t]].item()
        print(f"{t:>3} | {diff.max().item():>14.4e} | {diff.mean().item():>14.4e} | "
              f"{abs(ps_lp - full_lp):>14.4e}")


if __name__ == "__main__":
    main()
