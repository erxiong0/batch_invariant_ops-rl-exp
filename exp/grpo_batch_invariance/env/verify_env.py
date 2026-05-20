"""环境自检：PyTorch/vLLM 版本、GPU 数量、liger-kernel/fused-MLP 不可加载。"""
import importlib
import sys

import torch


def check_pytorch():
    major, minor = torch.__version__.split(".")[:2]
    assert (int(major), int(minor)) >= (2, 9), f"need torch>=2.9, got {torch.__version__}"
    assert torch.cuda.is_available(), "CUDA not available"
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 4, f"need 4 GPUs, got {n_gpu}"
    print(f"OK  torch={torch.__version__}  GPUs={n_gpu}")


def check_vllm():
    import vllm
    parts = vllm.__version__.split(".")
    assert (int(parts[0]), int(parts[1])) >= (0, 17), f"need vllm>=0.17, got {vllm.__version__}"
    print(f"OK  vllm={vllm.__version__}")


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
    check_vllm()
    check_batch_invariant_ops()
    check_no_liger_active()
    print("\nAll environment checks passed.")


if __name__ == "__main__":
    main()
