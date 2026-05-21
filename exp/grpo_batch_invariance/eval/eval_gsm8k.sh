#!/usr/bin/env bash
set -euo pipefail
# Usage: bash eval/eval_gsm8k.sh <checkpoint_dir> <output_json>
CKPT="${1:?need checkpoint dir}"
OUT="${2:?need output json path}"
mkdir -p "$(dirname "$OUT")"

CUDA_VISIBLE_DEVICES=0 swift eval \
  --model "$CKPT" \
  --enable_thinking false \
  --eval_dataset gsm8k \
  --eval_backend Native --infer_backend pt \
  --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
  --eval_output_dir "$(dirname "$OUT")" 2>&1 | tee "${OUT}.log"

# swift eval 输出在 eval_output_dir 内，找到 gsm8k 的 result json 复制到 OUT
result=$(find "$(dirname "$OUT")" -name "gsm8k*.json" -newer "${OUT}.log" -print | head -1)
if [[ -n "$result" ]]; then
  cp "$result" "$OUT"
  echo "[eval] $CKPT -> $OUT"
fi
