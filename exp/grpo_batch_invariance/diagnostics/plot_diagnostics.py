"""Plot logprob mismatch histograms across the 4 cells.

读取 results/diagnostics/cell_{A,B,C,D}.json，输出 figures/logprob_diff_hist.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results" / "diagnostics"
FIG_DIR = EXP_DIR / "results" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    titles = {
        "A": "A: baseline (no patches)",
        "B": "B: trainer-only invariant",
        "C": "C: vLLM-only invariant",
        "D": "D: both invariant",
    }
    for ax, cell in zip(axes.flat, "ABCD"):
        path = RESULTS / f"cell_{cell}.json"
        data = json.loads(path.read_text())
        deltas = np.array(data["histogram_deltas"])
        ax.hist(deltas, bins=80, log=True)
        ax.set_title(f"{titles[cell]}\nmean|Δ|={data['mean_abs_delta']:.2e}  "
                     f"frac>1e-3={data['frac_gt_1e-3']:.2%}")
        ax.set_xlabel("logprob_rollout - logprob_train")
        ax.set_ylabel("count (log)")
    fig.suptitle("Rollout vs Trainer logprob diff — 4-cell control")
    fig.tight_layout()
    out = FIG_DIR / "logprob_diff_hist.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
