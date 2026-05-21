#!/usr/bin/env bash
set -euo pipefail

# Usage: bash exp/grpo_batch_invariance/train/run_all.sh
# 顺序跑 {baseline, invariant} × {seed=42,43,44} = 6 runs。
# 跳过已存在的 output_dir（断点续跑）。

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/exp/grpo_batch_invariance"

MODES=(baseline invariant)
SEEDS=(42 43 44)

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="$EXP_DIR/results/runs/${mode}_seed${seed}"
    # 用 trainer_state.json 是否存在判定完成
    if [[ -f "$out/trainer_state.json" ]]; then
      echo "[skip] $out already finished"
      continue
    fi
    echo "[run] BIM_MODE=$mode SEED=$seed"
    BIM_MODE="$mode" SEED="$seed" bash "$EXP_DIR/train/train_grpo.sh"
  done
done

echo "all runs done"
