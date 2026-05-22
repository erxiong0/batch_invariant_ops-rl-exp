"""GSM8K reward plugin + batch-invariant init that fires in EVERY worker process.

Why this file exists rather than reusing the upstream gsm8k_plugin.py directly:

ms-swift's `rlhf` CLI launches N worker processes via torchrun, each as
`python -m swift.cli.rlhf`. Those workers bypass our launcher.py entirely, so
the patches that launcher.py was applying (BIM_MODE invariant ops + trl
GRPOTrainer attr backfill) never reach the actual trainers. Workers only
inherit env vars (BIM_MODE) and the modules they themselves import.

`--external_plugins` is one such required import: swift loads it at args
parse time on every worker, before constructing the trainer. So we use it
as the choke point: apply patches at module-import time, then define the
same reward classes as upstream gsm8k_plugin.py.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List


# ----------------------------------------------------------------------------
# Worker-side init (runs on every torchrun child via --external_plugins import)
# ----------------------------------------------------------------------------

def _enable_invariant_mode_if_requested() -> None:
    mode = os.environ.get("BIM_MODE", "baseline").lower()
    if mode != "invariant":
        print(f"[bim_plugin] worker pid={os.getpid()} baseline mode",
              file=sys.stderr, flush=True)
        return
    # Ensure the exp dir is importable so `ops_extension` resolves regardless
    # of how this plugin was launched (swift puts only the plugin's parent dir
    # on sys.path).
    exp_dir = Path(__file__).resolve().parents[3]  # .../exp/grpo_batch_invariance
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()
    from ops_extension import enable_extended_batch_invariant_mode
    enable_extended_batch_invariant_mode()
    print(f"[bim_plugin] worker pid={os.getpid()} invariant mode ON "
          f"(mm/addmm/log_softmax/mean + RMSNorm + SDPA patched)",
          file=sys.stderr, flush=True)


def _patch_trl_grpo_grad_accum() -> None:
    """trl >= 0.26 GRPOTrainer.training_step reads self.current_gradient_accumulation_steps,
    but swift 4.2.1's GRPOTrainer subclass construction chain never sets it. Backfill
    lazily in training_step before delegating to the original implementation.
    """
    try:
        from trl import GRPOTrainer as TrlGRPOTrainer
    except Exception as e:
        print(f"[bim_plugin] WARN: trl patch skipped: {e}", file=sys.stderr)
        return
    _orig_step = TrlGRPOTrainer.training_step
    if getattr(_orig_step, "_bim_patched", False):
        return  # idempotent across re-imports
    _notice = {"emitted": False}

    def _wrapped_step(self, *args, **kwargs):
        if not hasattr(self, "current_gradient_accumulation_steps"):
            gas = getattr(getattr(self, "args", None), "gradient_accumulation_steps", 1)
            self.current_gradient_accumulation_steps = gas
            if not _notice["emitted"]:
                print(f"[bim_plugin] backfilled current_gradient_accumulation_steps={gas} "
                      f"on pid={os.getpid()}", file=sys.stderr, flush=True)
                _notice["emitted"] = True
        return _orig_step(self, *args, **kwargs)

    _wrapped_step._bim_patched = True  # type: ignore[attr-defined]
    TrlGRPOTrainer.training_step = _wrapped_step


# Fire both at import time, before swift loads any trainer class.
_patch_trl_grpo_grad_accum()
_enable_invariant_mode_if_requested()


# ----------------------------------------------------------------------------
# Reward functions — verbatim from ms-swift upstream gsm8k_plugin.py
# ----------------------------------------------------------------------------

from swift.rewards import ORM, orms  # noqa: E402  (must be after patches)


class GSM8KAccuracy(ORM):

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract the last #### number from text."""
        text = text[-500:] if len(text) > 500 else text
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].replace(',', '').replace(' ', '').strip()
        matches = re.findall(r'####\s*([\-\d,\.\s]+)', text)
        if matches:
            return matches[-1].replace(',', '').replace(' ', '').strip()
        return ''

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, gt_answer in zip(completions, solution):
            gt_num = self.extract_answer(gt_answer)
            pred_num = self.extract_answer(completion)
            correct = False
            if pred_num and gt_num:
                try:
                    correct = abs(float(pred_num) - float(gt_num)) < 1e-5
                except (ValueError, OverflowError):
                    correct = pred_num == gt_num
            rewards.append(1.0 if correct else 0.0)
        return rewards


class GSM8KFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
            )
            rewards.append(1.0 if has_answer else 0.0)
        return rewards


orms['gsm8k_accuracy'] = GSM8KAccuracy
orms['gsm8k_format'] = GSM8KFormat
