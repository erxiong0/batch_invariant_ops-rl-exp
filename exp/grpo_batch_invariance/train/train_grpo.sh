#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   BIM_MODE=baseline  SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
#   BIM_MODE=invariant SEED=42 bash exp/grpo_batch_invariance/train/train_grpo.sh
#
# Optional env overrides (并跑/做对照时用):
#   CUDA_VISIBLE_DEVICES=4,5,6,7    指定显卡（默认 0,1,2,3）
#   NPROC_PER_NODE=4                 worker 数（默认 4；需匹配 CUDA_VISIBLE_DEVICES 数量）
#   MASTER_PORT=29501                同节点并跑时**必填**（torchrun 默认 29500 会撞车）
#   RUN_TAG=sdpa                     给 output_dir 加后缀
#                                    (results/runs/<BIM_MODE>_seed<SEED>_<TAG>/)
#
# 要求当前工作目录为仓库根（含 batch_invariant_ops/）。

: "${BIM_MODE:?need BIM_MODE=baseline|invariant}"
: "${SEED:?need SEED=integer}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/exp/grpo_batch_invariance"
RUN_NAME="${BIM_MODE}_seed${SEED}${RUN_TAG:+_${RUN_TAG}}"
OUTPUT_DIR="$EXP_DIR/results/runs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

# 默认走我们自带的 bim_gsm8k_plugin.py —— 它在 worker import 时跑 BIM_MODE invariant 初始化
# 和 trl GRPOTrainer attr 兜底。env var 优先（想用上游 plugin 时可覆盖）。
if [[ -z "${GSM8K_PLUGIN:-}" ]]; then
  GSM8K_PLUGIN="${EXP_DIR}/train/plugins/gsm8k/bim_gsm8k_plugin.py"
fi
if [[ ! -f "$GSM8K_PLUGIN" ]]; then
  echo "GSM8K plugin not found at $GSM8K_PLUGIN" >&2
  exit 1
fi
echo "[train_grpo] using GSM8K_PLUGIN=$GSM8K_PLUGIN" >&2

SYSTEM_PROMPT='You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}.'

cd "$REPO_ROOT"
export PYTHONPATH="${EXP_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export BIM_MODE
# 关闭 fused MLP 路径（spec §1.3）
export DISABLE_LIGER_KERNEL=1
# vllm 0.11.1 拉的 flashinfer-python 0.5.2 跟 flashinfer-cubin 0.6.8.post1 版本不
# 一致（pip resolver 把 cubin 拉到了更新版本），flashinfer 内部 version check 会
# 拒绝启动；ABI 在 minor 版本内稳定，bypass 这个 check 是 vllm 官方推荐的临时方案。
export FLASHINFER_DISABLE_VERSION_CHECK=1
# bim_gsm8k_plugin 在每个 worker 里 hook trl._compute_loss，记录 per-step 的
# importance ratio 统计 (mean/max |ratio-1|, frac outside clip band) 到这个目录下
# ratio_rank{N}.jsonl。是 Thinking Machines claim 的直接验证数据。
export BIM_RATIO_LOG_DIR="$OUTPUT_DIR/ratio_stats"
mkdir -p "$BIM_RATIO_LOG_DIR"
echo "[train_grpo] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE OUTPUT_DIR=$OUTPUT_DIR BIM_RATIO_LOG_DIR=$BIM_RATIO_LOG_DIR" >&2

python "$EXP_DIR/launcher.py" rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen3-1.7B \
  --external_plugins "$GSM8K_PLUGIN" \
  --reward_funcs gsm8k_accuracy gsm8k_format \
  --columns '{"answer": "solution"}' \
  --enable_thinking false \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.4 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
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
