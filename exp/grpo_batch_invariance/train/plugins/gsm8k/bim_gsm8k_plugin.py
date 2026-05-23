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

    # vLLM rollout 侧：让每个 vLLM worker 在 init_batch_invariance() 时把
    # vllm_is_batch_invariant() 看到为 True。必须在 vLLM worker 进程构造之前设。
    # 我们这个 plugin 在 worker import 时就会跑，比 vLLM colocate init 早。
    # 注意 colocate 模式下 vLLM 跟 trainer 共享同一进程，所以 vLLM 在 torch.library
    # 注册的 mm/addmm/log_softmax/mean 替换对 trainer forward 也生效 — 不需要我们
    # 重复调 batch_invariant_ops.enable_batch_invariant_mode（实测重复注册会抛
    # RuntimeError "kernel registered twice for _log_softmax"）。
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

    # 仍然需要 ops_extension：它 patch 的是 transformers 类层面（Qwen3RMSNorm.forward
    # 替换 + ALL_ATTENTION_FUNCTIONS["sdpa"] 字典项替换 + F.scaled_dot_product_attention
    # 替换），不通过 torch.library，跟 vLLM 的 aten op patches 不冲突。
    exp_dir = Path(__file__).resolve().parents[3]  # .../exp/grpo_batch_invariance
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from ops_extension import enable_extended_batch_invariant_mode
    enable_extended_batch_invariant_mode()
    print(f"[bim_plugin] worker pid={os.getpid()} invariant mode ON "
          f"(VLLM_BATCH_INVARIANT=1 -> vLLM patches aten ops; ops_extension patches "
          f"transformers RMSNorm + SDPA)",
          file=sys.stderr, flush=True)


def _backfill_grad_accum(trainer) -> None:
    if not hasattr(trainer, "current_gradient_accumulation_steps"):
        gas = getattr(getattr(trainer, "args", None), "gradient_accumulation_steps", 1)
        trainer.current_gradient_accumulation_steps = gas


def _patch_grpo_training_step() -> None:
    """Patch BOTH trl.GRPOTrainer AND swift.rlhf_trainers.grpo_trainer.GRPOTrainer.

    trl >= 0.26 reads self.current_gradient_accumulation_steps in training_step,
    but swift 4.2.1's subclass construction never sets it. We wrap training_step
    on both layers — whichever one MRO actually hits will backfill the attr
    before delegating downward.
    """
    patched_any = False

    # Layer 1: swift's subclass — this is the immediate super() target from the
    # transformers Trainer loop, so this is where the call lands first.
    try:
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer as SwiftGRPOTrainer
        _orig = SwiftGRPOTrainer.training_step
        if not getattr(_orig, "_bim_patched", False):
            def _swift_wrapped(self, *args, **kwargs):
                _backfill_grad_accum(self)
                return _orig(self, *args, **kwargs)
            _swift_wrapped._bim_patched = True  # type: ignore[attr-defined]
            SwiftGRPOTrainer.training_step = _swift_wrapped
            print(f"[bim_plugin] patched swift.GRPOTrainer.training_step on pid={os.getpid()}",
                  file=sys.stderr, flush=True)
            patched_any = True
    except Exception as e:
        print(f"[bim_plugin] WARN: swift GRPOTrainer patch skipped: {e}",
              file=sys.stderr, flush=True)

    # Layer 2: trl base — defense if swift ever calls trl.training_step directly
    # (e.g. via super().training_step from another mixin layer we don't see).
    try:
        from trl import GRPOTrainer as TrlGRPOTrainer
        _orig = TrlGRPOTrainer.training_step
        if not getattr(_orig, "_bim_patched", False):
            def _trl_wrapped(self, *args, **kwargs):
                _backfill_grad_accum(self)
                return _orig(self, *args, **kwargs)
            _trl_wrapped._bim_patched = True  # type: ignore[attr-defined]
            TrlGRPOTrainer.training_step = _trl_wrapped
            print(f"[bim_plugin] patched trl.GRPOTrainer.training_step on pid={os.getpid()}",
                  file=sys.stderr, flush=True)
            patched_any = True
    except Exception as e:
        print(f"[bim_plugin] WARN: trl GRPOTrainer patch skipped: {e}",
              file=sys.stderr, flush=True)

    if not patched_any:
        print(f"[bim_plugin] WARN: no GRPOTrainer class found to patch on pid={os.getpid()}",
              file=sys.stderr, flush=True)


def _patch_grpo_ratio_stats() -> None:
    """Hook swift GRPOTrainer to log per-step rollout-vs-trainer logprob drift.

    Thinking Machines' core claim is about numerical drift between vLLM rollout
    logprob and HF trainer forward logprob on the same (prompt, completion). In
    swift's grpo_trainer, vLLM rollout logprobs land in inputs['rollout_per_token_logps']
    and trainer forward logprobs are returned by _get_per_token_logps_and_entropies
    as per_token_logps. These should be bit-equal under full invariance.

    Note: swift overrides trl._compute_loss into a thin dispatcher; actual logic
    is in _compute_loss_and_metrics (in swift.rlhf_trainers.grpo_trainer, see
    L1-250 of that method's source). We patch THAT method, not trl's.

    Also captures the GRPO importance ratio (coef_1) for the trl-style drift
    measurement: log_ratio = per_token_logps - old_per_token_logps where
    old_per_token_logps may be None (then = detach short-circuit → ratio≡1).
    """
    ratio_log_dir = os.environ.get("BIM_RATIO_LOG_DIR")
    if not ratio_log_dir:
        return  # opt-in
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    log_path = Path(ratio_log_dir) / f"ratio_rank{rank}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer as SwiftGRPOTrainer
    except Exception as e:
        print(f"[bim_plugin] WARN: ratio stats patch skipped: {e}", file=sys.stderr, flush=True)
        return

    _orig = SwiftGRPOTrainer._compute_loss_and_metrics
    if getattr(_orig, "_bim_ratio_patched", False):
        return

    def _wrapped(self, model, inputs):
        import torch
        import json

        # Run original first (it computes per_token_logps internally and returns
        # (loss, metrics_data)). We piggyback on its computation by re-running
        # the same _get_per_token_logps_and_entropies inside a no_grad block
        # to capture per_token_logps for our stats — cheap because torch caches
        # the model outputs and the inputs slice is the same.
        result = _orig(self, model, inputs)

        try:
            # Recompute per_token_logps for stats (no_grad, eval state unchanged).
            with torch.no_grad():
                per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    model, inputs, compute_entropy=False)

                completion_mask = inputs.get("completion_mask")
                if completion_mask is None:
                    return result
                m = completion_mask.float()
                n_tok = m.sum().clamp(min=1.0)

                # --- trl-style ratio (vs old_per_token_logps) ---
                old = inputs.get("old_per_token_logps")
                old_supplied = old is not None
                if old is None:
                    old = per_token_logps.detach()
                log_ratio_trl = per_token_logps - old
                coef_1 = torch.exp(log_ratio_trl)
                abs_lr_trl = log_ratio_trl.abs() * m
                abs_dev = (coef_1 - 1.0).abs() * m
                eps_low = float(getattr(self, "epsilon_low", 0.2))
                eps_high = float(getattr(self, "epsilon_high", 0.2))
                outside_band = (((coef_1 < 1 - eps_low) | (coef_1 > 1 + eps_high)).float() * m).sum() / n_tok

                # --- rollout-vs-trainer drift (Thinking Machines core claim) ---
                rollout = inputs.get("rollout_per_token_logps")
                rollout_supplied = rollout is not None
                if rollout is not None:
                    log_ratio_rollout = per_token_logps - rollout
                    abs_lr_roll = log_ratio_rollout.abs() * m
                    mean_abs_lr_roll = float(abs_lr_roll.sum().item() / max(n_tok.item(), 1))
                    max_abs_lr_roll = float(abs_lr_roll.max().item())
                    coef_1_roll = torch.exp(log_ratio_rollout)
                    abs_dev_roll = (coef_1_roll - 1.0).abs() * m
                    mean_abs_dev_roll = float(abs_dev_roll.sum().item() / max(n_tok.item(), 1))
                    max_abs_dev_roll = float(abs_dev_roll.max().item())
                else:
                    mean_abs_lr_roll = max_abs_lr_roll = mean_abs_dev_roll = max_abs_dev_roll = None

                stats = {
                    "step": int(getattr(getattr(self, "state", None), "global_step", -1)),
                    "rank": rank,
                    "n_tokens": int(n_tok.item()),
                    # trl-style importance ratio (vs old_per_token_logps)
                    "old_supplied": bool(old_supplied),
                    "mean_abs_log_ratio_trl": float(abs_lr_trl.sum().item() / max(n_tok.item(), 1)),
                    "max_abs_log_ratio_trl": float(abs_lr_trl.max().item()),
                    "mean_abs_ratio_minus_1_trl": float(abs_dev.sum().item() / max(n_tok.item(), 1)),
                    "max_abs_ratio_minus_1_trl": float(abs_dev.max().item()),
                    "frac_outside_clip_band_trl": float(outside_band.item()),
                    "epsilon_low": eps_low,
                    "epsilon_high": eps_high,
                    # rollout vs trainer drift (Thinking Machines)
                    "rollout_supplied": bool(rollout_supplied),
                    "mean_abs_log_ratio_rollout": mean_abs_lr_roll,
                    "max_abs_log_ratio_rollout": max_abs_lr_roll,
                    "mean_abs_ratio_minus_1_rollout": mean_abs_dev_roll,
                    "max_abs_ratio_minus_1_rollout": max_abs_dev_roll,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(stats) + "\n")
        except Exception as e:
            if not getattr(self, "_bim_ratio_warned", False):
                import traceback
                print(f"[bim_plugin] WARN: ratio stats logging failed: {e}",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                self._bim_ratio_warned = True
        return result

    _wrapped._bim_ratio_patched = True  # type: ignore[attr-defined]
    SwiftGRPOTrainer._compute_loss_and_metrics = _wrapped
    print(f"[bim_plugin] ratio stats logging ON pid={os.getpid()} rank={rank} -> {log_path}",
          file=sys.stderr, flush=True)


# Fire patches at import time. Print a banner first so a missing prefix in logs
# immediately tells us the plugin wasn't loaded vs loaded-but-patch-skipped.
print(f"[bim_plugin] loaded by pid={os.getpid()} (BIM_MODE={os.environ.get('BIM_MODE', 'baseline')})",
      file=sys.stderr, flush=True)
_patch_grpo_training_step()
_patch_grpo_ratio_stats()
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
