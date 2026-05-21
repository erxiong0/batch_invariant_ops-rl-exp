"""Repeatability旁证：同一 prompt 用 vLLM 跑 N 次（独立请求），统计唯一输出数。

baseline 应有几十个唯一样本（与博客原例一致），invariant 应为 1。
通过子进程实现 mode 切换（VLLM_BATCH_INVARIANT 必须在 vllm import 前设置）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results" / "diagnostics"
MODEL_ID = "Qwen/Qwen3.5-2B"
N_TRIALS = 200
PROMPT = ("Generate 30 random numbers between 0 and 1000, comma-separated. "
          "Just numbers, no prose.")


def run_one(mode: str) -> None:
    if mode == "invariant":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True)
    sp = SamplingParams(max_tokens=200, temperature=0.0, seed=0)
    outs = llm.generate([PROMPT] * N_TRIALS, sp)
    texts = [o.outputs[0].text for o in outs]
    uniq = len(set(texts))
    summary = {"mode": mode, "n_trials": N_TRIALS, "n_unique": uniq}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"repeatability_{mode}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{mode}] {N_TRIALS} trials -> {uniq} unique outputs", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "invariant"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for m in ("baseline", "invariant"):
            env = os.environ.copy()
            env.pop("VLLM_BATCH_INVARIANT", None)
            subprocess.run(
                [sys.executable, __file__, "--mode", m],
                check=True, env=env, cwd=str(EXP_DIR),
            )
    elif args.mode:
        run_one(args.mode)
    else:
        ap.error("need --mode or --all")


if __name__ == "__main__":
    main()
