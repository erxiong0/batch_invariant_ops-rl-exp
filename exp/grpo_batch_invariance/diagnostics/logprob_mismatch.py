"""2-cell logprob mismatch diagnostic (HF rollout vs HF trainer forward).

vLLM is not used (driver 12.4 doesn't support vllm batch-invariance). Both
rollout and trainer use HF transformers, so when patches are enabled, both
sides go through our batch-invariant ops. This is actually a CLEANER test of
the batch-invariance hypothesis than the original 4-cell vllm-based design.

两个 cell（控制变量）:
  A baseline:    trainer-side patches off
  B invariant:   trainer-side patches on (mm/addmm/log_softmax/mean + RMSNorm + SDPA)

每个 cell 是独立 subprocess（让 patch 注册到全局 torch.library，避免状态污染）。
单独 cell: --cell {A,B}；触发两个 subprocess: --all。

输出: results/diagnostics/cell_{A,B}.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import statistics
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results" / "diagnostics"

MODEL_ID = "Qwen/Qwen3-1.7B"
N_PROMPTS = 100         # HF generate is slow → smaller sample than the original 200
MAX_NEW = 128           # ditto
SEED = 12345


def load_gsm8k_prompts(n: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").select(range(n))
    sys_msg = ("You are a helpful math assistant. Solve the problem step by step "
               "and put your final answer within \\boxed{}.")
    return [f"{sys_msg}\n\nQuestion: {x['question']}\nAnswer:" for x in ds]


def rollout_hf(prompts: List[str]):
    """Generate completions with HF model.generate and capture per-token logprobs.

    Returns list of (gen_token_ids, gen_logprobs).
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    torch.manual_seed(SEED)
    out: List[Tuple[List[int], List[float]]] = []
    with torch.no_grad():
        for prompt in prompts:
            enc = tok(prompt, return_tensors="pt").to("cuda")
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tok.pad_token_id,
            )
            gen_ids = gen.sequences[0, enc.input_ids.shape[1]:].tolist()
            # gen.scores: tuple length N_new, each (1, vocab); pre-softmax logits at decode time
            logprobs: List[float] = []
            for tok_id, score in zip(gen_ids, gen.scores):
                lp = F.log_softmax(score[0].float(), dim=-1)[tok_id].item()
                logprobs.append(lp)
            out.append((gen_ids, logprobs))
    return out


def forward_hf(prompts: List[str], rollouts):
    """Recompute per-token logprob via single-shot HF forward on (prompt + completion)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    out: List[List[float]] = []
    with torch.no_grad():
        for prompt, (gen_ids, _) in zip(prompts, rollouts):
            if not gen_ids:
                out.append([])
                continue
            p_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            full_ids = torch.cat([p_ids, torch.tensor([gen_ids], device="cuda")], dim=1)
            logits = model(full_ids).logits  # (1, L, V)
            p_len = p_ids.shape[1]
            gen_logits = logits[0, p_len - 1 : -1, :]  # (G, V)
            gen_targets = torch.tensor(gen_ids, device="cuda")
            log_softmax = F.log_softmax(gen_logits.float(), dim=-1)
            lp = log_softmax.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1).cpu().tolist()
            out.append(lp)
    return out


def run_cell(cell: str) -> None:
    if cell == "B":
        from batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        from ops_extension import enable_extended_batch_invariant_mode
        enable_extended_batch_invariant_mode()

    prompts = load_gsm8k_prompts(N_PROMPTS)
    rollouts = rollout_hf(prompts)
    train_lps = forward_hf(prompts, rollouts)

    deltas: List[float] = []
    for (_, roll_lps), tr_lps in zip(rollouts, train_lps):
        L = min(len(roll_lps), len(tr_lps))
        for r, t in zip(roll_lps[:L], tr_lps[:L]):
            deltas.append(float(r - t))

    if not deltas:
        print(f"[cell {cell}] WARN: no tokens collected", file=sys.stderr)
        return

    abs_deltas = [abs(d) for d in deltas]
    summary = {
        "cell": cell,
        "n_tokens": len(deltas),
        "mean_abs_delta": statistics.mean(abs_deltas),
        "max_abs_delta": max(abs_deltas),
        "frac_gt_1e-3": sum(1 for d in abs_deltas if d > 1e-3) / len(abs_deltas),
        "frac_gt_1e-6": sum(1 for d in abs_deltas if d > 1e-6) / len(abs_deltas),
        "histogram_deltas": deltas[:2000],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"cell_{cell}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[cell {cell}] wrote {out}: mean|Δ|={summary['mean_abs_delta']:.2e} "
          f"frac>1e-3={summary['frac_gt_1e-3']:.3f}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["A", "B"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for c in ("A", "B"):
            print(f"\n=== running cell {c} (subprocess, fresh state) ===", file=sys.stderr)
            subprocess.run([sys.executable, __file__, "--cell", c], check=True, cwd=str(EXP_DIR))
    elif args.cell:
        run_cell(args.cell)
    else:
        ap.error("need --cell {A,B} or --all")


if __name__ == "__main__":
    main()
