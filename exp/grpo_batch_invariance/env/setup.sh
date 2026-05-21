#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录调用：bash exp/grpo_batch_invariance/env/setup.sh
# 前置：NVIDIA driver 550+ (CUDA 12.4)。如有 driver ≥570 走 cu128 stack 性能更好，但本脚本兼容 12.4。
# 不安装 vLLM —— driver 12.4 上没有 batch-invariant 的 vLLM 版本，本实验用 HF transformers 做 rollout。
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# 1) PyTorch + cu124：torch 2.6.0 是 cu124 wheel 时代最高稳定版
#    同时清掉 torchvision —— 旧 ABI 残留会让 transformers import 时 torchvision::nms 注册失败。
#    本实验纯文本，不需要 torchvision；如以后要跑视觉模型再 pip install torchvision==0.21.0
pip uninstall -y torch triton torchvision 2>/dev/null || true
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

# 2) batch_invariant_ops 本仓库（editable）
pip install -e .

# 3) ms-swift（GRPO trainer）：4.2.x 兼容 torch 2.6
pip install -U "ms-swift>=4.2,<4.3"

# 4) transformers：与 swift 4.2.x 兼容的范围；不再钉 5.2（5.2 要 torch≥2.9）
pip install -U "transformers>=4.46,<4.55"

# 5) 评测依赖
pip install "math_verify==0.5.2"

# 6) 测试 + 绘图
pip install pytest matplotlib

echo ""
echo "Setup done. NEXT:"
echo "  python exp/grpo_batch_invariance/env/verify_env.py"
