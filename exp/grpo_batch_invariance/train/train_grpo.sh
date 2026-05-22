#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   BIM_MODE=baseline  SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
#   BIM_MODE=invariant SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
#
# 要求当前工作目录为仓库根（含 batch_invariant_ops/）。

: "${BIM_MODE:?need BIM_MODE=baseline|invariant}"
: "${SEED:?need SEED=integer}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/exp/grpo_batch_invariance"
OUTPUT_DIR="$EXP_DIR/results/runs/${BIM_MODE}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"

# 寻找 swift 安装路径下的 gsm8k plugin
SWIFT_DIR=$(python -c "import swift, os; print(os.path.dirname(swift.__file__))")
GSM8K_PLUGIN="${SWIFT_DIR}/../examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py"
if [[ ! -f "$GSM8K_PLUGIN" ]]; then
  echo "gsm8k_plugin.py not found at $GSM8K_PLUGIN. Locate it under ms-swift examples and set GSM8K_PLUGIN env var." >&2
  exit 1
fi

SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

cd "$REPO_ROOT"
export PYTHONPATH="${EXP_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export BIM_MODE
# 关闭 fused MLP 路径（spec §1.3）
export DISABLE_LIGER_KERNEL=1

python "$EXP_DIR/launcher.py" rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen3-1.7B \
  --external_plugins "$GSM8K_PLUGIN" \
  --reward_funcs gsm8k_accuracy gsm8k_format \
  --columns '{"answer": "solution"}' \
  --enable_thinking false \
  --use_vllm false \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --use_liger_kernel false \
  --dataset 'modelscope/gsm8k' \
  --load_from_cache_file true \
  --max_length 2048 \
  --max_completion_length 8192 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --save_steps 10 \
  --save_total_limit 100 \
  --logging_steps 1 \
  --warmup_ratio 0.0 \
  --dataloader_num_workers 4 \
  --num_generations 8 \
  --temperature 1.0 \
  --system "$SYSTEM_PROMPT" \
  --deepspeed zero2 \
  --log_completions true \
  --report_to tensorboard \
  --max_grad_norm 1.0 \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  --scale_rewards none \
  --seed "$SEED" \
  --max_steps 50 \
  --output_dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/train.log"

echo "[done] $OUTPUT_DIR"
