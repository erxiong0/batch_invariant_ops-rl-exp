"""4-cell logprob mismatch diagnostic.

对 GSM8K 前 N 条 prompt:
  - 用 vLLM rollout 一次（与训练同 sampling 参数），记录 token-level logprob_rollout
  - 用 transformers HF forward 一次，重算同 (prompt, completion) 的 logprob_train
  - 计算 delta = logprob_rollout - logprob_train、|delta| 分布

四个 cell（控制变量）:
  A baseline:    VLLM_BATCH_INVARIANT off, trainer patch off
  B trainer-only: VLLM_BATCH_INVARIANT off, trainer patch on
  C vllm-only:    VLLM_BATCH_INVARIANT on,  trainer patch off
  D both:        VLLM_BATCH_INVARIANT on,  trainer patch on

每个 cell 是独立 subprocess（因 VLLM_BATCH_INVARIANT 必须在 vLLM import 前设置）。
本脚本通过 --cell 参数运行单个 cell；--all 触发四个 subprocess。

输出: results/diagnostics/cell_{A,B,C,D}.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results" / "diagnostics"

MODEL_ID = "Qwen/Qwen3.5-2B"
N_PROMPTS = 200
MAX_NEW = 256
SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=-1, seed=12345)


def load_gsm8k_prompts(n: int) -> List[str]:
    """Return first n GSM8K test prompts."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").select(range(n))
    sys_msg = "You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."
    return [f"{sys_msg}\n\nQuestion: {x['question']}\nAnswer:" for x in ds]


def rollout_vllm(prompts: List[str]) -> List[Tuple[List[int], List[float]]]:
    """Return list of (token_ids, logprobs) per prompt from vLLM."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True)
    sp = SamplingParams(
        max_tokens=MAX_NEW, temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"],
        seed=SAMPLING["seed"], logprobs=1,
    )
    outputs = llm.generate(prompts, sp)
    result = []
    for out in outputs:
        comp = out.outputs[0]
        token_ids = list(comp.token_ids)
        logprobs = [lp_d[tid].logprob for tid, lp_d in zip(comp.token_ids, comp.logprobs)]
        result.append((token_ids, logprobs))
    return result


def forward_hf(prompts: List[str], rollouts) -> List[List[float]]:
    """Re-compute per-token logprob via HF forward on (prompt + completion)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    train_logprobs: List[List[float]] = []
    with torch.no_grad():
        for prompt, (gen_ids, _) in zip(prompts, rollouts):
            p_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            full_ids = torch.cat([p_ids, torch.tensor([gen_ids], device="cuda")], dim=1)
            logits = model(full_ids).logits  # (1, L, V)
            # logits[t] predicts token at t+1; so for gen tokens at positions [p_len, p_len+len(gen))
            p_len = p_ids.shape[1]
            gen_logits = logits[0, p_len - 1 : -1, :]  # (G, V)
            gen_targets = torch.tensor(gen_ids, device="cuda")
            log_softmax = F.log_softmax(gen_logits.float(), dim=-1)
            lp = log_softmax.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1).cpu().tolist()
            train_logprobs.append(lp)
    return train_logprobs


def run_cell(cell: str) -> None:
    """单个 cell 子进程的工作。"""
    if cell in ("C", "D"):
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    if cell in ("B", "D"):
        from batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        from ops_extension import enable_extended_batch_invariant_mode
        enable_extended_batch_invariant_mode()

    prompts = load_gsm8k_prompts(N_PROMPTS)
    rollouts = rollout_vllm(prompts)
    train_lps = forward_hf(prompts, rollouts)

    deltas: List[float] = []
    for (_, roll_lps), tr_lps in zip(rollouts, train_lps):
        L = min(len(roll_lps), len(tr_lps))
        for r, t in zip(roll_lps[:L], tr_lps[:L]):
            deltas.append(float(r - t))

    abs_deltas = [abs(d) for d in deltas]
    import statistics
    summary = {
        "cell": cell,
        "n_tokens": len(deltas),
        "mean_abs_delta": statistics.mean(abs_deltas),
        "max_abs_delta": max(abs_deltas),
        "frac_gt_1e-3": sum(1 for d in abs_deltas if d > 1e-3) / len(abs_deltas),
        "frac_gt_1e-6": sum(1 for d in abs_deltas if d > 1e-6) / len(abs_deltas),
        "histogram_deltas": deltas[:2000],  # 截断保存
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"cell_{cell}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[cell {cell}] wrote {out}: mean|Δ|={summary['mean_abs_delta']:.2e} "
          f"frac>1e-3={summary['frac_gt_1e-3']:.3f}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list("ABCD"))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for c in "ABCD":
            print(f"\n=== running cell {c} (subprocess, fresh env) ===", file=sys.stderr)
            env = os.environ.copy()
            env.pop("VLLM_BATCH_INVARIANT", None)
            subprocess.run(
                [sys.executable, __file__, "--cell", c],
                check=True, env=env, cwd=str(EXP_DIR),
            )
    elif args.cell:
        run_cell(args.cell)
    else:
        ap.error("need --cell {A,B,C,D} or --all")


if __name__ == "__main__":
    main()
