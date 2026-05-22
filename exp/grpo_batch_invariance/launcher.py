"""零侵入接入 swift CLI 的启动器。

调用方式：
    BIM_MODE=baseline   python launcher.py rlhf --rlhf_type grpo --model Qwen/Qwen3-1.7B ...
    BIM_MODE=invariant  python launcher.py rlhf --rlhf_type grpo --model Qwen/Qwen3-1.7B ...

invariant 模式翻两个开关（vLLM 已不使用，本实验用 HF rollout）：
  1) batch_invariant_ops.enable_batch_invariant_mode()  (mm/addmm/log_softmax/mean)
  2) ops_extension.enable_extended_batch_invariant_mode() (RMSNorm + SDPA)
"""
from __future__ import annotations

import os
import sys


def _patch_trl_grpo_grad_accum() -> None:
    """trl >= 0.26 的 GRPOTrainer.training_step 读 self.current_gradient_accumulation_steps,
    但 swift 4.2.1 的 GRPOTrainer 子类构造链上没把这个 attr 设进来，第一步就 AttributeError。
    包一层 __init__ wrapper：原 init 跑完后若 attr 缺失就用 args.gradient_accumulation_steps 补。
    """
    try:
        from trl import GRPOTrainer as TrlGRPOTrainer
    except Exception as e:
        print(f"[launcher] WARN: trl GRPOTrainer patch skipped: {e}", file=sys.stderr)
        return
    _orig_init = TrlGRPOTrainer.__init__

    def _wrapped_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if not hasattr(self, "current_gradient_accumulation_steps"):
            gas = getattr(getattr(self, "args", None), "gradient_accumulation_steps", 1)
            self.current_gradient_accumulation_steps = gas
            print(f"[launcher] patched current_gradient_accumulation_steps={gas}",
                  file=sys.stderr, flush=True)

    TrlGRPOTrainer.__init__ = _wrapped_init


def main() -> None:
    mode = os.environ.get("BIM_MODE", "baseline").lower()
    assert mode in {"baseline", "invariant"}, f"bad BIM_MODE: {mode}"

    if mode == "invariant":
        from batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        from ops_extension import enable_extended_batch_invariant_mode
        enable_extended_batch_invariant_mode()
        print(f"[launcher] invariant mode ON: "
              f"mm/addmm/log_softmax/mean patched, RMSNorm + SDPA patched",
              file=sys.stderr, flush=True)
    else:
        print(f"[launcher] baseline mode (no patches)", file=sys.stderr, flush=True)

    _patch_trl_grpo_grad_accum()

    # 透传剩余参数给 swift CLI
    from swift.cli.main import cli_main
    sys.argv = ["swift"] + sys.argv[1:]
    cli_main()


if __name__ == "__main__":
    main()
