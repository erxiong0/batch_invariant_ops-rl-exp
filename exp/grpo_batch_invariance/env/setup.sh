#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录调用：bash exp/grpo_batch_invariance/env/setup.sh
# 前置：NVIDIA driver 550+ (CUDA 12.4)。如有 driver ≥570 走 cu128 stack 性能更好，但本脚本兼容 12.4。
# 装一个能 import 的 vLLM（cu124 build），但训练 --use_vllm false，rollout 走 HF transformers。
# 不依赖 vLLM 的 batch-invariant 版本（driver 12.4 上没有）。
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# 1) PyTorch + cu124：torch 2.6.0 是 cu124 wheel 时代最高稳定版
#    同时清掉 torchvision —— 旧 ABI 残留会让 transformers import 时 torchvision::nms 注册失败。
#    也清掉 vllm —— 镜像里可能预装 cu13 build (vllm 0.21+)，下面会重装 cu124 兼容版本。
#    本实验纯文本，不需要 torchvision；如以后要跑视觉模型再 pip install torchvision==0.21.0
pip uninstall -y torch triton torchvision vllm 2>/dev/null || true
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

# 2) batch_invariant_ops 本仓库（editable）
pip install -e .

# 3) ms-swift（GRPO trainer）：4.2.x 兼容 torch 2.6
pip install -U "ms-swift>=4.2,<4.3"

# 3b) trl：swift 4.2.1 args 解析时 hard require `trl>=0.26`，所以不能下钉太老。
#     0.26–0.29 区间稳定；trl 0.26+ 的 training_step 引用 self.current_gradient_accumulation_steps,
#     该 attr 需要 transformers ≥4.50 的 _inner_training_loop 注入（见 step 6 钉 4.54+）。
pip install "trl>=0.26,<0.30"

# 4) deepspeed：train_grpo.sh 用 --deepspeed zero2，没装会在 SftArguments._init_deepspeed 报
#    PackageNotFoundError。0.14–0.15 跟 torch 2.6 + swift 4.2 兼容。
pip install "deepspeed>=0.14,<0.16"

# 5) vllm：实验里 --use_vllm false，但 swift.infer_engine.vllm_engine 在 grpo_trainer
#    导入链里无 guard 地 `import vllm`，所以必须有一个能 import 的 vllm。0.8.x 是
#    cu124 wheel 时代，跟 torch 2.6 兼容。装更高版本（0.10+）会回到 libcudart.so.13 missing。
pip install "vllm==0.8.5"

# 6) transformers：≥4.50 才会在 _inner_training_loop 里注入 current_gradient_accumulation_steps,
#    trl 0.26+ 的 training_step 依赖这个 attr。<4.55 是因为 4.55+ 要 torch≥2.9。
pip install -U "transformers>=4.54,<4.55"

# 7) 评测依赖
pip install "math_verify==0.5.2"

# 8) 测试 + 绘图
pip install pytest matplotlib

echo ""
echo "Setup done. NEXT:"
echo "  python exp/grpo_batch_invariance/env/verify_env.py"
