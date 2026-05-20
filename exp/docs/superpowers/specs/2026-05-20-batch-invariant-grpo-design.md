# Batch Invariance × GRPO 对比实验设计

**日期**: 2026-05-20
**目标**: 在 swift 官方 Qwen3.5-2B GRPO + GSM8K 配方上，验证启用 `batch_invariant_ops` 对 RL 训练的影响——既要验证机制（rollout vs trainer logprob 偏差），也要验证最终指标（GSM8K accuracy），还要验证可复现性。

## 1. 背景与假设

Thinking Machines 博客 *Defeating Nondeterminism in LLM Inference* 指出：RL 训练中 vLLM rollout 阶段产生的 token logprob，与 trainer forward 阶段对同一 (prompt, completion) 重新计算的 logprob 并不相等，原因是两次调用的 batch shape 不同，导致 reduction 顺序、kernel tile 形状、padding 处理都不同。这种差异使得 GRPO/PPO 中的 importance ratio `π_new/π_old ≠ 1`（即使 policy 还没更新），破坏 on-policy 假设，引入隐性偏差并放大 clip 行为。

启用 batch-invariant kernels（同时在 vLLM 和 trainer 两侧）后，rollout 与 forward 的 logprob 应**逐 token 完全相等**，importance ratio 在第 0 步应恰为 1。本实验设计用于量化这一假设带来的训练动态与最终指标差异。

### 1.1 必须覆盖的三大 kernel

博客与 vLLM 实现都明确：消除 batch-shape 依赖要求**完整覆盖**以下 reduction-heavy kernel：

| Kernel | batch-dependent 原因 | vLLM (rollout) 已覆盖 | `batch_invariant_ops` (trainer) 当前 |
|---|---|---|---|
| MatMul / `mm` / `addmm` | 小 batch 触发 Split-K | ✅ batch-invariant matmul | ✅ `matmul_persistent` |
| **RMSNorm** | 小 batch 触发 split-reduction | ✅ Triton RMSNorm | **❌ 缺**（Qwen3.5 大量使用）|
| **Attention (SDPA)** | Split-KV / FlashDecoding 的 split 数随 batch 变 | ✅ FlexAttention + fixed split-tile-size | **❌ 缺**（trainer 走 SDPA / FA2）|
| log_softmax | head reduction | ✅ | ✅ |
| mean.dim | 一般 reduction | ✅ | ✅ |

→ **trainer 侧若不补齐 RMSNorm 与 Attention 的 batch-invariant 替换，rollout vs forward 的 logprob 不可能逐 token 一致**。因此本实验必须以"补齐这两个 op"为前置工程（Phase 0），否则后续训练对比会被无关 numerics 噪声淹没。

### 1.2 forward 完整路径的覆盖盘点（含 MLP）

为避免漏算其他 reduction 路径，逐模块列出 Qwen3.5-2B Dense forward 的覆盖情况：

| 模块 | 内部 op | reduction? | 覆盖来源 |
|---|---|---|---|
| Token embedding | gather | 否 | 天然 batch-invariant |
| RMSNorm (pre-attn / pre-mlp / final) | `mean(x²) + rsqrt + scale` | 是 | **Phase 0** |
| Q/K/V/O projection | matmul | 是 | `mm`/`addmm` ✓ |
| RoPE | pointwise rotate | 否 | 天然 |
| Attention (SDPA) | `softmax(QKᵀ/√d) · V` | 是 | **Phase 0** |
| MLP gate/up/down_proj | matmul (×3) | 是 | `mm`/`addmm` ✓ |
| MLP SiLU + gate×up | pointwise | 否 | 天然 |
| LM head | matmul | 是 | `mm`/`addmm` ✓ |
| Loss (CE) | log_softmax + gather | 是 | `_log_softmax` ✓ |

→ **MLP 不需要单独 patch**：MLP 的所有 batch-shape-dependent 部分都是 3 个 matmul，由 `mm`/`addmm` 已覆盖；SwiGLU 的非线性与逐元素乘是 pointwise，天然 invariant。

### 1.3 绕过路径与显式锁定（覆盖性前提）

以下三条会让 matmul/RMSNorm patch 被绕过，必须在训练命令与 `verify_env.py` 启动检查中显式拒绝：

| 绕过路径 | 何时触发 | 拒绝方式 |
|---|---|---|
| **Fused SwiGLU MLP** (liger-kernel `LigerSwiGLUMLP`, apex fused MLP) | 启用 fused/liger 路径时，gate/up/down/SiLU/mul 融成单个自定义 Triton kernel，不走 `aten::mm` | 训练命令显式 `--use_liger_kernel false`；`verify_env.py` 检测到 liger 已 import 或被 transformers 启用即 fail-fast |
| **Grouped GEMM** (`aten::_grouped_mm`) | MoE / `--experts_impl grouped_mm` | 本实验 Qwen3.5-2B 是 Dense，天然不触发；spec 锁定模型为 Dense 2B |
| **量化 MLP** (`aten::_weight_int8_mm` 等) | INT8/INT4 训练 | 本实验 `--torch_dtype bfloat16`，不触发 |

后两项对本实验配置不会触发；第一项需要在 verify_env.py 中加 fail-fast 检查，并在训练命令中显式关闭。

## 2. 实验设计

### 2.0 Phase 0：补齐 batch_invariant_ops（前置工程）

**目标**：在 `batch_invariant_ops` 中新增 RMSNorm 与 SDPA 的 batch-invariant 实现，让 `enable_batch_invariant_mode()` 之后，trainer 侧的 forward 真正达到 batch-invariant。

**新增内容**（放在 `exp/grpo_batch_invariance/ops_extension/`，独立 Python 包，import 后再调 `enable_batch_invariant_mode()` 注册到 torch.library）：

1. **`rms_norm_batch_invariant`**
   - 替换 `aten::rms_norm`（PyTorch 2.4+ 原生 op）
   - 同时对 transformers 的 `Qwen2RMSNorm`/`Qwen3RMSNorm` 做 monkey-patch（如果它没走 `aten::rms_norm`）
   - Triton 实现参考 vLLM `model_executor/layers/batch_invariant.py` 中的 RMSNorm kernel：one-row-per-block 数据并行 + 固定 reduction tree，禁用 split-reduction
   - 数学定义：`y = x / sqrt(mean(x²) + eps) * weight`

2. **`sdpa_batch_invariant`**
   - 替换 `aten::scaled_dot_product_attention`
   - 实现策略：使用 PyTorch `FlexAttention` 并强制 `BLOCK_M / BLOCK_N` 与 KV 分块为固定值（不随 batch/seq 变），等价于 vLLM 的 fixed split-tile-size 策略
   - 强制后端：在 patch 中显式 `with sdpa_kernel(SDPBackend.MATH):` 作为 fallback 路径（math 路径本身确定但慢；FlexAttention 是优先路径）
   - swift 训练命令需配合：去掉 `--attn_impl flash_attention_2`，改为不传或显式 `--attn_impl sdpa`，让 transformers 走 `nn.functional.scaled_dot_product_attention`，从而命中我们的 patch

3. **单元测试** `tests/test_invariance.py`：
   - 对 RMSNorm 和 SDPA 分别构造 `batch=1` vs `batch=8` 的输入（同首行），断言输出逐 bit 相等（torch.equal）
   - 这是 Phase 0 的 done-gate

### 2.1 配方对齐

完全复用 ms-swift `Qwen3.5 最佳实践` 文档中的 GRPO 命令（[文档链接](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html)），不做任何超参修改：
- 模型: `Qwen/Qwen3.5-2B`，full fine-tuning，bf16
- 数据集: `modelscope/gsm8k`
- Reward: `gsm8k_accuracy` + `gsm8k_format`（来自 `examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py`）
- 关键超参: `lr=1e-6`, `num_generations=8`, `temperature=1.0`, `epsilon=0.2`, `epsilon_high=0.28`, `scale_rewards none`, `per_device_train_batch_size=4`, `gradient_accumulation=4`
- 后端: vLLM colocate, `gpu_mem_util=0.4`, deepspeed zero2
- 硬件: 4 × H100/A100 (80G)
- **Attention 后端**: 与默认配方略不同——必须显式 `--attn_impl sdpa`，让 forward 走 `nn.functional.scaled_dot_product_attention`，从而命中 Phase 0 的 patch。这是 baseline 与 invariant 组**共同**的设置，避免引入第二个变量。

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
├── ops_extension/                 # Phase 0：补齐 RMSNorm + SDPA 的 batch-invariant 实现
│   ├── __init__.py                # enable_full_batch_invariant_mode() 入口
│   ├── rms_norm.py                # Triton kernel + aten::rms_norm patch + transformers monkey-patch
│   ├── sdpa.py                    # FlexAttention with fixed split-tile-size + aten::scaled_dot_product_attention patch
│   └── tests/
│       ├── test_rms_norm_invariance.py    # batch=1 vs batch=8 逐 bit 相等断言
│       ├── test_sdpa_invariance.py        # 同上
│       └── test_end_to_end_forward.py     # Qwen3.5-2B 整体 forward 在 batch=1 / 8 / 32 上 bit-equal
├── launcher.py                    # 零侵入接入 swift 的启动器（见 §4）
├── diagnostics/
│   ├── logprob_mismatch.py        # 核心诊断：4 cell 对比 logprob diff
│   ├── repeatability.py           # 同 prompt × N 次 rollout 的 bit-equal 率
│   └── plot_diagnostics.py        # 输出 logprob_diff_hist.png 等
├── train/
│   ├── train_grpo.sh              # 单 run 训练命令（参数化 MODE / SEED / OUTPUT_DIR）
│   └── run_all.sh                 # 顺序跑 6 个 run（也支持 --only 跳过已完成的）
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
    # 开关 2：trainer 侧 torch.library aten 替换 + Phase 0 扩展（RMSNorm + SDPA）
    from batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()                       # mm/addmm/log_softmax/mean
    from ops_extension import enable_extended_batch_invariant_mode
    enable_extended_batch_invariant_mode()              # rms_norm/scaled_dot_product_attention
                                                        # + transformers Qwen RMSNorm monkey-patch

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

**风险 1 处理**: 如果 colocate 模式下 trainer 侧的 patch 与 vLLM 内部 batch-invariant kernel 冲突（双重 patch），降级方案是把 trainer 侧的两次 `enable_*` 改为 context manager 包裹（在 swift trainer 的 `compute_loss` / 自定义 logprob 计算位置用 `with set_batch_invariant_mode():` + `with extended_batch_invariant():` 嵌套），通过 `external_plugins` 注入。注意：transformers 模块的 monkey-patch（RMSNorm 类替换）无法用 context manager，必须进程级开启——若必须降级到局部包裹，需保留 RMSNorm 替换不变，只把 aten 级 op 替换收成局部。

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
| 训推偏差 | `mean(|Δlogprob|)`, `frac(|Δ|>1e-3)`, `IS-ratio` 分位数 | `diagnostics/logprob_mismatch.py`（训练前 + 训练后 ckpt 各跑一次）| 每 run 2 次 |
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
| **R6**: SDPA patch 与 transformers attention 实现不兼容 | Qwen3.5 的 attention 模块直接调 `F.scaled_dot_product_attention` 还是走自定义路径未知；FlexAttention 在 bf16 + causal mask + 滑窗 (sliding window attention, Qwen3.5 用) 下的支持范围有限 | 先验证 transformers 实际调用栈；若 FlexAttention 不支持 SWA，退化为 SDPA-MATH 后端（慢但 batch-invariant），同时把 5-step pilot 测速作为决策点 |
| **R7**: RMSNorm patch 未命中 transformers fused 实现 | transformers 可能不调 `aten::rms_norm`，而是自定义 `Qwen2RMSNorm.forward` | 双重保险：既 patch `aten::rms_norm`，也对 transformers 的 `Qwen2RMSNorm` / `Qwen3RMSNorm` 类做 monkey-patch；单元测试用真实模型 forward 而不是裸 `F.rms_norm` 验证 |
| **R8**: liger-kernel / 其他 fused MLP 路径绕过 `aten::mm` patch | swift 默认是否启用 liger-kernel 需要确认；transformers 自身的 `attn_implementation` 之外，MLP 部分若被 fused 也会绕过 | 训练命令显式 `--use_liger_kernel false`；`verify_env.py` 检测 liger / apex fused MLP / 其他自定义 SwiGLU kernel 是否已加载并 fail-fast；不依赖 transformers 默认值 |

## 9. 执行顺序

实验按以下顺序，每一步都是后一步的 gate：

1. `env/setup.sh` + `env/verify_env.py` — 环境就位
2. **Phase 0**: 实现 `ops_extension/rms_norm.py` + `ops_extension/sdpa.py`，跑 `ops_extension/tests/` 全绿 — **gate**：单元测试中 batch=1 与 batch=8 必须 `torch.equal`
3. `diagnostics/logprob_mismatch.py` — **gate**：D cell 必须满足 `mean(|Δlogprob|) < 1e-6` 且 `frac(|Δ|>1e-3) == 0`（即 bit-equal）；A cell 必须满足 `mean(|Δlogprob|) > 1e-4`（确认机制存在）；否则停下来定位 R1/R3/R6
4. `diagnostics/repeatability.py` — 旁证
5. `train/train_grpo.sh BIM_MODE=baseline SEED=42` — pilot 5 step，测速、确认 plugin 正常打点
6. `train/run_all.sh` — 6 run 全跑
7. `eval/eval_all.sh` — 30 个 eval 点
8. `results/summary.md` 自动生成 + 人工撰写 Findings

## 10. 非目标（YAGNI）

明确**不做**以下事，避免范围蔓延：
- **本实验确实需要新实现 RMSNorm + SDPA 的 batch-invariant kernel**（Phase 0），这是必要前提，不是 YAGNI 项
- 不向 `batch_invariant_ops` 上游 PR Phase 0 的实现（先放在 `ops_extension/`，实验成功后再考虑 upstream）
- 不改 swift 源码（全部走 launcher + external_plugins + transformers monkey-patch）
- 不做 LoRA 对比（只对比 baseline vs invariant 在 full FT 下）
- 不做超过 GSM8K 的 benchmark
- 不做多机训练
- 不做 GKD（文档里有，本实验只看 GRPO）
- 不优化 Phase 0 kernel 的性能至 cuBLAS/FA2 水平（接受 ~20% 性能损失，与博客一致）
