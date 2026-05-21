"""环境自检：PyTorch + CUDA 可用、GPU 数量、liger-kernel/fused-MLP 不可加载。

注：本实验已经迁移到 HF rollout（不用 vLLM），所以不检查 vllm。
"""
import importlib
import sys

import torch


def check_pytorch():
    major, minor = torch.__version__.split(".")[:2]
    assert (int(major), int(minor)) >= (2, 6), f"need torch>=2.6, got {torch.__version__}"
    assert torch.cuda.is_available(), "CUDA not available"
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 4, f"need 4 GPUs, got {n_gpu}"
    print(f"OK  torch={torch.__version__}  cuda={torch.version.cuda}  GPUs={n_gpu}")


def check_batch_invariant_ops():
    import batch_invariant_ops
    assert hasattr(batch_invariant_ops, "enable_batch_invariant_mode")
    print("OK  batch_invariant_ops importable")


def check_no_liger_active():
    """liger-kernel 若被 import 即视为风险（它通过 monkey-patch 替换 transformers 的 MLP/RMSNorm）。"""
    forbidden = ["liger_kernel", "apex.normalization.fused_layer_norm", "flash_attn.ops.fused_dense"]
    for name in forbidden:
        try:
            mod = importlib.import_module(name)
            print(f"FAIL  forbidden module loaded: {name} -> {mod}")
            sys.exit(1)
        except ImportError:
            pass
    print("OK  no forbidden fused-kernel modules pre-loaded")


def main():
    check_pytorch()
    check_batch_invariant_ops()
    check_no_liger_active()
    print("\nAll environment checks passed.")


if __name__ == "__main__":
    main()
