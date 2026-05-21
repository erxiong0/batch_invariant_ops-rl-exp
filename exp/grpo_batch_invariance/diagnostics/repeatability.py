"""Repeatability旁证：同一 prompt 用 HF generate 跑 N 次（独立调用），统计唯一输出数。

vLLM is not used (driver 12.4 incompatible). HF generate at temperature 0
on identical input/seed will produce identical output by construction
even without batch invariance (since no batching across requests in HF).
This test instead exercises batching: generate the same prompt in a batch
of N concurrent prompts, and check whether the same target prompt at slot 0
produces identical output regardless of what else is in the batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results" / "diagnostics"
MODEL_ID = "Qwen/Qwen3-1.7B"
TARGET_PROMPT = ("Generate 30 random numbers between 0 and 1000, comma-separated. "
                 "Just numbers, no prose.")
N_OTHER_PROMPTS = [
    "Hello world.", "The capital of France is", "Solve for x: 2x + 5 = 11.",
    "List three primes.", "Write a haiku about GPUs.", "1 + 1 = ",
    "Translate 'cat' to French:",
]
N_TRIALS = 50


def run_one(mode: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if mode == "invariant":
        from batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        from ops_extension import enable_extended_batch_invariant_mode
        enable_extended_batch_invariant_mode()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    # Each trial: target prompt is at slot 0 of a batch with randomly-chosen others.
    import random
    rng = random.Random(0)
    target_outputs = []
    with torch.no_grad():
        for _ in range(N_TRIALS):
            other_sample = rng.sample(N_OTHER_PROMPTS, k=min(7, len(N_OTHER_PROMPTS)))
            batch_prompts = [TARGET_PROMPT] + other_sample
            enc = tok(batch_prompts, return_tensors="pt", padding=True).to("cuda")
            gen = model.generate(
                **enc,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )
            target_out = gen[0, enc.input_ids.shape[1]:].tolist()
            target_outputs.append(tuple(target_out))

    uniq = len(set(target_outputs))
    summary = {"mode": mode, "n_trials": N_TRIALS, "n_unique": uniq}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"repeatability_{mode}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{mode}] {N_TRIALS} trials, target at slot 0 with random co-batch -> {uniq} unique", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "invariant"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        import subprocess
        for m in ("baseline", "invariant"):
            print(f"\n=== {m} ===", file=sys.stderr)
            subprocess.run([sys.executable, __file__, "--mode", m], check=True, cwd=str(EXP_DIR))
    elif args.mode:
        run_one(args.mode)
    else:
        ap.error("need --mode or --all")


if __name__ == "__main__":
    main()
