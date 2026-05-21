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

    # 透传剩余参数给 swift CLI
    from swift.cli.main import cli_main
    sys.argv = ["swift"] + sys.argv[1:]
    cli_main()


if __name__ == "__main__":
    main()
