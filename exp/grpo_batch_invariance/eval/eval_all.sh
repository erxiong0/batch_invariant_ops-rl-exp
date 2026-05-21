#!/usr/bin/env bash
set -euo pipefail
# Usage: bash exp/grpo_batch_invariance/eval/eval_all.sh

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/exp/grpo_batch_invariance"
EVAL_DIR="$EXP_DIR/results/eval"
mkdir -p "$EVAL_DIR"

STEPS=(10 20 30 40 50)
MODES=(baseline invariant)
SEEDS=(42 43 44)

# baseline (step 0 / 初始模型) 也评一次共享，作为基准
BASELINE_OUT="$EVAL_DIR/base_qwen3-1.7b.json"
if [[ ! -f "$BASELINE_OUT" ]]; then
  bash "$EXP_DIR/eval/eval_gsm8k.sh" "Qwen/Qwen3-1.7B" "$BASELINE_OUT"
fi

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$EXP_DIR/results/runs/${mode}_seed${seed}"
    for step in "${STEPS[@]}"; do
      ckpt="$run_dir/checkpoint-${step}"
      out="$EVAL_DIR/${mode}_seed${seed}_step${step}.json"
      if [[ ! -d "$ckpt" ]]; then
        echo "[skip] no checkpoint: $ckpt"
        continue
      fi
      if [[ -f "$out" ]]; then
        echo "[skip] $out exists"
        continue
      fi
      bash "$EXP_DIR/eval/eval_gsm8k.sh" "$ckpt" "$out"
    done
  done
done

echo "all eval done"
