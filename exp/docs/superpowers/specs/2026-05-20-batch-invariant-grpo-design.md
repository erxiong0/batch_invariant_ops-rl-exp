# Batch Invariance × GRPO 对比实验设计

**日期**: 2026-05-20
**目标**: 在 swift 官方 Qwen3.5-2B GRPO + GSM8K 配方上，验证启用 `batch_invariant_ops` 对 RL 训练的影响——既要验证机制（rollout vs trainer logprob 偏差），也要验证最终指标（GSM8K accuracy），还要验证可复现性。

## 1. 背景与假设

Thinking Machines 博客 *Defeating Nondeterminism in LLM Inference* 指出：RL 训练中 vLLM rollout 阶段产生的 token logprob，与 trainer forward 阶段对同一 (prompt, completion) 重新计算的 logprob 并不相等，原因是两次调用的 batch shape 不同，导致 reduction 顺序、kernel tile 形状、padding 处理都不同。这种差异使得 GRPO/PPO 中的 importance ratio `π_new/π_old ≠ 1`（即使 policy 还没更新），破坏 on-policy 假设，引入隐性偏差并放大 clip 行为。

启用 batch-invariant kernels（同时在 vLLM 和 trainer 两侧）后，rollout 与 forward 的 logprob 应**逐 token 完全相等**，importance ratio 在第 0 步应恰为 1。本实验设计用于量化这一假设带来的训练动态与最终指标差异。

## 2. 实验设计

### 2.1 配方对齐

完全复用 ms-swift `Qwen3.5 最佳实践` 文档中的 GRPO 命令（[文档链接](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html)），不做任何超参修改：
- 模型: `Qwen/Qwen3.5-2B`，full fine-tuning，bf16
- 数据集: `modelscope/gsm8k`
- Reward: `gsm8k_accuracy` + `gsm8k_format`（来自 `examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py`）
- 关键超参: `lr=1e-6`, `num_generations=8`, `temperature=1.0`, `epsilon=0.2`, `epsilon_high=0.28`, `scale_rewards none`, `per_device_train_batch_size=4`, `gradient_accumulation=4`
- 后端: vLLM colocate, `gpu_mem_util=0.4`, deepspeed zero2
- 硬件: 4 × H100/A100 (80G)

### 2.2 A/B 变量

唯一区分两组的是两个开关：

| 组 | `VLLM_BATCH_INVARIANT` | `set_batch_invariant_mode()` 包裹 trainer step |
|---|---|---|
| baseline | 未设置（关） | 关 |
| invariant | `1` | 开 |

### 2.3 矩阵

`{baseline, invariant} × {seed=42, 43, 44}` = **6 个 50-step run**。
预算估计：每 run 1.5-3h（invariant 模式可能略慢），合计 9-18h。

## 3. 代码结构

所有产物位于 `exp/grpo_batch_invariance/`，与 `exp/` 根目录解耦。

```
exp/grpo_batch_invariance/
├── README.md                      # 复现指引与硬件要求
├── env/
│   ├── setup.sh                   # pip install ms-swift, vllm>=0.17, batch_invariant_ops -e
│   └── verify_env.py              # 启动前自检：PyTorch>=2.9, vllm batch-inv 支持, GPU 数量
├── launcher.py                    # 零侵入接入 swift 的启动器（见 §4）
├── diagnostics/
│   ├── logprob_mismatch.py        # 核心诊断：4 cell 对比 logprob diff
│   ├── repeatability.py           # 同 prompt × N 次 rollout 的 bit-equal 率
│   └── plot_diagnostics.py        # 输出 logprob_diff_hist.png 等
├── train/
│   ├── train_grpo.sh              # 单 run 训练命令（参数化 MODE / SEED / OUTPUT_DIR）
│   ├── run_all.sh                 # 顺序跑 6 个 run（也支持 --only 跳过已完成的）
│   └── plugins/
│       └── logprob_probe.py       # swift external_plugin，在每 step 记录
│                                  #   rollout-vs-forward logprob diff 直方图统计
├── eval/
│   ├── eval_gsm8k.sh              # swift eval per checkpoint
│   └── eval_all.sh                # 扫所有 run × {step 10,20,30,40,50}
└── results/
    ├── runs/<mode>_<seed>/        # tensorboard + swanlab 日志、checkpoint
    ├── eval/<mode>_<seed>_step<k>.json
    ├── figures/                   # 三张主图（见 §6）
    └── summary.md                 # 自动生成的对比表
```

## 4. 接入方式（launcher.py）

零侵入 swift。两个开关必须同时翻转——这是设计中最关键的不变量：

```python
# launcher.py
import os
import sys

mode = os.environ.get("BIM_MODE", "baseline")  # baseline | invariant
assert mode in ("baseline", "invariant")

if mode == "invariant":
    # 开关 1：vLLM 侧 C++/CUDA kernel 路径
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    # 开关 2：trainer 侧 torch.library aten 替换。enable_* 是进程级全局，
    # 不用 context manager（colocate 模式下两侧共享同进程 Python 调用栈）
    from batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()

# 透传剩余参数给 swift CLI
from swift.cli.main import cli_main
sys.argv = ["swift"] + sys.argv[1:]
cli_main()
```

调用方式：

```bash
BIM_MODE=invariant python launcher.py rlhf --rlhf_type grpo --model Qwen/Qwen3.5-2B ...
BIM_MODE=baseline   python launcher.py rlhf --rlhf_type grpo --model Qwen/Qwen3.5-2B ...
```

**风险 1 处理**: 如果 colocate 模式下 `enable_batch_invariant_mode()` 与 vLLM 内部 batch-invariant kernel 冲突（双重 patch），降级方案是只设 `VLLM_BATCH_INVARIANT=1`，在 swift trainer 的 `compute_loss` hook 中用 `with set_batch_invariant_mode():` 局部包裹（通过 `external_plugins` 注入）。

## 5. 诊断脚本（diagnostics/logprob_mismatch.py）

实验最有信息量的一段，跑训练前先验证机制。**这一步如果都没区别，训练对比就没意义。**

流程：
1. 加载 Qwen3.5-2B（不训练）
2. 取 GSM8K 前 200 条 prompt
3. 用 vLLM rollout 一遍（与训练同 sampling 参数），记录每个 token 的 `logprob_rollout`
4. 同模型 HF forward 一遍 (prompt, completion) 拼接序列，记录 `logprob_train`
5. 计算: `delta = logprob_rollout - logprob_train`、`ratio = exp(delta)`、`frac(|delta| > 1e-3)`
6. **四个 cell** 对比（控制变量）：

   | cell | vLLM batch-inv | trainer batch-inv | 预期 |
   |---|---|---|---|
   | A (baseline) | off | off | delta 显著非零 |
   | B | off | on | delta 仍大（rollout 侧未变） |
   | C | on | off | delta 仍大（trainer 侧未变） |
   | D (invariant) | on | on | delta 应为 0（bit-equal）|

7. 输出 `figures/logprob_diff_hist.png`，模仿 `batch-invariant.png` 的双栏直方图风格

`repeatability.py` 做正交验证：同 prompt 用 vLLM 跑 1000 次，统计唯一输出数（baseline 应有几十个、invariant 应为 1，对应博客原例）。

## 6. 指标采集

| 类别 | 指标 | 来源 | 频率 |
|---|---|---|---|
| 训推偏差 | `mean(|Δlogprob|)`, `frac(|Δ|>1e-3)`, `IS-ratio` 分位数 | `plugins/logprob_probe.py` | 每 step |
| GRPO 内部 | clip 比例、approx KL、reward mean/std、advantage std、grad_norm | swift trainer 默认日志 | 每 step |
| 最终指标 | GSM8K accuracy | `swift eval` | 每 10 step |
| 可复现性 | 3 seed 间 final acc 的 std；同 seed bit-equal 率 | `summary.md` 聚合 | 实验结束 |

## 7. 产出

- `results/figures/logprob_diff_hist.png` — 诊断阶段，4 cell 直方图
- `results/figures/reward_curve.png` — 6 run 的 reward over step，按 mode 染色，seed 用线型区分
- `results/figures/acc_per_step.png` — 6 run 的 GSM8K acc @ step 10/20/30/40/50
- `results/summary.md` — 自动生成对比表，含 3 seed 的 mean ± std
- `README.md` 末尾的 **Findings** 段：用 3-5 句结论叙述「机制是否被验证 → 训练动态差异 → 最终指标差异」

## 8. 风险与备选

| 风险 | 触发条件 | 备选 |
|---|---|---|
| **R1**: 双重 batch-inv patch 冲突 | invariant 组训练崩溃或 logprob 不一致 | trainer 侧改用 `with set_batch_invariant_mode():` 局部包裹（plugin 注入） |
| **R2**: invariant kernel 拖慢训练 >2× | 50 step 预算超 6h/run | 5-step pilot 测速后，把矩阵缩为 30 step 或砍到 2 seed |
| **R3**: Qwen3.5-2B 不在 vLLM batch-inv 验证清单 | 诊断脚本 D cell 仍有 delta | 降级到 Qwen3-1.7B（已验证），保持其他配方不变 |
| **R4**: vLLM ≥0.17 与 batch_invariant_ops 当前版本不兼容 | env/verify_env.py 报错 | 钉一个 vllm + torch 版本组合（在 setup.sh 中记录） |
| **R5**: HF forward 重算 logprob 时显存爆 | 8192 max_completion_length × 200 prompt | 诊断脚本里改成微批 16 + grad disabled |

## 9. 执行顺序

实验按以下顺序，每一步都是后一步的 gate：

1. `env/setup.sh` + `env/verify_env.py` — 环境就位
2. `diagnostics/logprob_mismatch.py` — **gate**：D cell 必须满足 `mean(|Δlogprob|) < 1e-6` 且 `frac(|Δ|>1e-3) == 0`（即 bit-equal），同时 A cell 必须满足 `mean(|Δlogprob|) > 1e-4`（确认机制存在），否则停下来定位 R1/R3
3. `diagnostics/repeatability.py` — 旁证
4. `train/train_grpo.sh BIM_MODE=baseline SEED=42` — pilot 5 step，测速、确认 plugin 正常打点
5. `train/run_all.sh` — 6 run 全跑
6. `eval/eval_all.sh` — 30 个 eval 点
7. `results/summary.md` 自动生成 + 人工撰写 Findings

## 10. 非目标（YAGNI）

明确**不做**以下事，避免范围蔓延：
- 不实现新的 batch-invariant kernel（直接用 `batch_invariant_ops` 已有的 mm/addmm/log_softmax/mean）
- 不改 swift 源码（全部走 launcher + external_plugins）
- 不做 LoRA 对比（只对比 baseline vs invariant 在 full FT 下）
- 不做超过 GSM8K 的 benchmark
- 不做多机训练
- 不做 GKD（文档里有，本实验只看 GRPO）
