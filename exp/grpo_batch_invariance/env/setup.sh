#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录调用：bash exp/grpo_batch_invariance/env/setup.sh
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# batch_invariant_ops 本仓库自身（editable）
pip install -e .

# ms-swift（GRPO trainer）。4.3+ 是文档版本号，发布版最新为 4.2.x；GRPO 自 3.x 稳定，4.2 即可。
pip install -U "ms-swift>=4.2"

# vLLM（必须 ≥0.17，自带 batch invariance 支持）
pip install -U "vllm>=0.17.0"

# transformers（与 swift 文档对齐）
pip install -U "transformers==5.2.*"

# 评测依赖
pip install "math_verify==0.5.2"

# 测试依赖
pip install pytest

echo "Setup done. Run: python exp/grpo_batch_invariance/env/verify_env.py"
