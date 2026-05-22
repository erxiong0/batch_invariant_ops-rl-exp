"""零侵入接入 swift CLI 的启动器（rlhf 子命令实际由 torchrun 拉起 N 个 worker
进程，每个 worker 重新 `python -m swift.cli.rlhf`，完全绕过本 launcher）。

所以 BIM_MODE invariant 模式开启 + trl GRPOTrainer attr 兜底实际发生在
`train/plugins/gsm8k/bim_gsm8k_plugin.py`（swift 通过 --external_plugins 在
每个 worker 里 import 这个文件）。本 launcher 只负责把命令行透传给 swift CLI,
并在父进程也打印一行状态，方便人肉确认。
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    mode = os.environ.get("BIM_MODE", "baseline").lower()
    assert mode in {"baseline", "invariant"}, f"bad BIM_MODE: {mode}"
    print(f"[launcher] BIM_MODE={mode} (worker init happens in --external_plugins)",
          file=sys.stderr, flush=True)

    from swift.cli.main import cli_main
    sys.argv = ["swift"] + sys.argv[1:]
    cli_main()


if __name__ == "__main__":
    main()
