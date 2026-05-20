# Batch Invariance × GRPO 实验

验证 batch-invariant kernel 对 ms-swift Qwen3.5-2B + GSM8K GRPO 训练的影响。

详见：
- Spec: `exp/docs/superpowers/specs/2026-05-20-batch-invariant-grpo-design.md`
- Plan: `exp/docs/superpowers/plans/2026-05-20-batch-invariant-grpo.md`

## 快速开始

```bash
bash env/setup.sh
python env/verify_env.py
pytest ops_extension/tests/ -v
python diagnostics/logprob_mismatch.py --all
bash train/run_all.sh
bash eval/eval_all.sh
python summarize.py
```

## 硬件

4 × H100/A100 (80G)。
