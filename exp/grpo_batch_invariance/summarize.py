"""Aggregate all results into results/summary.md + plots.

输入:
  results/diagnostics/cell_{A,B,C,D}.json
  results/diagnostics/repeatability_{baseline,invariant}.json
  results/eval/{mode}_seed{seed}_step{k}.json
  results/runs/{mode}_seed{seed}/trainer_state.json (训练曲线)

输出:
  results/summary.md
  results/figures/reward_curve.png
  results/figures/acc_per_step.png
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RES = EXP_DIR / "results"
FIG = RES / "figures"
RUNS = RES / "runs"
EVAL = RES / "eval"
DIAG = RES / "diagnostics"

MODES = ("baseline", "invariant")
SEEDS = (42, 43, 44)
STEPS = (10, 20, 30, 40, 50)


def load_eval_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # swift eval JSON 结构: {"results": {"gsm8k": {"accuracy": x, ...}}}
    return data.get("results", {}).get("gsm8k", {}).get("accuracy")


def load_trainer_state(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("log_history", [])


def plot_acc_per_step() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_init = load_eval_accuracy(EVAL / "base_qwen3-1.7b.json") or 0.0
    ax.axhline(baseline_init, color="gray", linestyle="--", label=f"Qwen3-1.7B init = {baseline_init:.3f}")

    for mode in MODES:
        for seed in SEEDS:
            accs = []
            for step in STEPS:
                acc = load_eval_accuracy(EVAL / f"{mode}_seed{seed}_step{step}.json")
                accs.append(acc)
            ax.plot(STEPS, accs, marker="o",
                    label=f"{mode} seed={seed}",
                    color="C0" if mode == "baseline" else "C1",
                    alpha=0.7)
    ax.set_xlabel("training step")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("GSM8K accuracy across training steps (2 mode × 3 seed)")
    ax.legend()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "acc_per_step.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_reward_curve() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        for seed in SEEDS:
            ts = load_trainer_state(RUNS / f"{mode}_seed{seed}" / "trainer_state.json")
            steps = [e["step"] for e in ts if "reward" in e]
            rewards = [e["reward"] for e in ts if "reward" in e]
            if not steps:
                continue
            ax.plot(steps, rewards,
                    label=f"{mode} seed={seed}",
                    color="C0" if mode == "baseline" else "C1",
                    alpha=0.7)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean reward")
    ax.set_title("GRPO reward curve")
    ax.legend()
    out = FIG / "reward_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def write_summary_md() -> None:
    lines: List[str] = ["# Experiment Summary\n"]

    # Diagnostics
    lines.append("## 1. Logprob mismatch (4-cell diagnostic)\n")
    lines.append("| cell | config | mean|Δ| | max|Δ| | frac>1e-3 |")
    lines.append("|---|---|---|---|---|")
    cell_desc = {"A": "baseline", "B": "trainer-only", "C": "vllm-only", "D": "both invariant"}
    for c in "ABCD":
        p = DIAG / f"cell_{c}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        lines.append(f"| {c} | {cell_desc[c]} | {d['mean_abs_delta']:.2e} | "
                     f"{d['max_abs_delta']:.2e} | {d['frac_gt_1e-3']:.3f} |")
    lines.append("")

    # Repeatability
    lines.append("## 2. Repeatability (same prompt × 200 trials)\n")
    lines.append("| mode | n_unique |")
    lines.append("|---|---|")
    for m in ("baseline", "invariant"):
        p = DIAG / f"repeatability_{m}.json"
        if p.exists():
            d = json.loads(p.read_text())
            lines.append(f"| {m} | {d['n_unique']}/{d['n_trials']} |")
    lines.append("")

    # Final accuracy
    lines.append("## 3. GSM8K accuracy by step (mean ± std across 3 seeds)\n")
    lines.append("| step | baseline | invariant | Δ |")
    lines.append("|---|---|---|---|")
    for step in STEPS:
        per_mode: Dict[str, List[float]] = {m: [] for m in MODES}
        for mode in MODES:
            for seed in SEEDS:
                acc = load_eval_accuracy(EVAL / f"{mode}_seed{seed}_step{step}.json")
                if acc is not None:
                    per_mode[mode].append(acc)
        if not (per_mode["baseline"] and per_mode["invariant"]):
            continue
        b_mean, b_std = statistics.mean(per_mode["baseline"]), statistics.pstdev(per_mode["baseline"])
        i_mean, i_std = statistics.mean(per_mode["invariant"]), statistics.pstdev(per_mode["invariant"])
        lines.append(f"| {step} | {b_mean:.4f} ± {b_std:.4f} | "
                     f"{i_mean:.4f} ± {i_std:.4f} | {i_mean - b_mean:+.4f} |")
    lines.append("")

    lines.append("## 4. Figures\n")
    lines.append("- `figures/logprob_diff_hist.png` — 4-cell logprob diff histograms")
    lines.append("- `figures/reward_curve.png` — GRPO reward across 6 runs")
    lines.append("- `figures/acc_per_step.png` — GSM8K accuracy per step")
    lines.append("")

    lines.append("## 5. Findings\n")
    lines.append("_<填写：机制是否被验证 → 训练动态差异 → 最终指标差异>_")

    out = RES / "summary.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    plot_acc_per_step()
    plot_reward_curve()
    write_summary_md()


if __name__ == "__main__":
    main()
