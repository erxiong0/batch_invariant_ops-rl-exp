#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录调用：bash exp/grpo_batch_invariance/env/setup.sh
# 前置：NVIDIA driver ≥570（支持 CUDA 12.8 runtime）。检查 `nvidia-smi` 的 "CUDA Version" 列。
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# 1) PyTorch：显式钉到 cu128 wheel（huaweicloud 镜像默认会拉 cu130，太新跑不动 driver 570）。
#    走官方 pytorch index 拿 cu128 wheel；版本 2.11.x 与 vllm 0.17 兼容。
pip uninstall -y torch triton 2>/dev/null || true
pip install "torch==2.11.0" "triton" --index-url https://download.pytorch.org/whl/cu128

# 2) batch_invariant_ops 本仓库自身（editable）
pip install -e .

# 3) ms-swift（GRPO trainer）。最新发布 4.2.x；GRPO 自 3.x 稳定，4.2 即可。
pip install -U "ms-swift>=4.2"

# 4) vLLM（必须 ≥0.17，自带 batch invariance 支持）
pip install -U "vllm>=0.17.0"

# 5) transformers（与 swift 文档对齐）
pip install -U "transformers==5.2.*"

# 6) 评测依赖
pip install "math_verify==0.5.2"

# 7) 测试依赖
pip install pytest

echo "Setup done. Run: python exp/grpo_batch_invariance/env/verify_env.py"
