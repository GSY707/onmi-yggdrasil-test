# V2-A：推理介质实验

日期：2026-07-13  
当前判定：A0 visible text-CoT 已形成正向 probe；当前结构的 A1 mechanism smoke 已通过；A2 的 2B K8/T8 MLP full-data latent probe 将普通 test 提到 `0.6016`，但 composition-heldout 只有 `0.3672`，仍没有相对 text-CoT 的稳定 heldout Pareto。新的 token-wise source adapter 只把 composition 提到 `0.4063`（bottleneck=128），test 为 `0.5859`，仍未形成 Pareto；Attention + Dense FFN 形式的 latent-attention full512 probe 为 `0.5078/0.4141/0.4531`（test/composition/length），no-latent 与 shuffled-latent 均为 `0.0938`；同构 0.8B latent probe 的 test 为 `0.2578`。逐步 verifier self-critical RL full512 probe 为 `0.5625/0.3906/0.5547`，verifier test 寄存器准确率仅 `0.0968`、整态命中为 `0`，策略熵塌缩。随后新增的 A1.5 分层正控制证明 P0/P2 ordinary 可执行、P1 ordinary hidden→latent 可拟合，但 P0/P1/P2 的 composition-heldout 分别只有 `0.2734`、`0.3242`、`0.2773` final，state full 仍为 `0` 或 `0.3290`；A1.5 Gate 仍未通过。未启动 V2-A3/V2-A4 或 V2-B。

本文件记录当前 V2-A 的实现合同、可复现实验入口和证据边界。架构规范以 [`Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md`](Project-Yggdrasil%20多模态潜变量推理架构白皮书%20V2.md) 为准，阶段顺序和 Gate 以 [`next-stage-test-plan.md`](next-stage-test-plan.md) 为准。

## 1. 实验范围

V2-A 只比较推理介质，不引入视觉、Boundary-MoE、FFN-MoE 或动作。对照链为：

```text
同一成熟文本基座
├─ direct answer
├─ answer-only（无显式 CoT）
├─ visible text CoT
└─ continuous latent recurrence（K 个 latent vectors，T 次 transition）
```

latent 主答案头只能读取最终 latent state；默认使用 K-token mean，也提供 flattened-token 诊断读出。输入 hidden states、答案旁路、teacher trace 和隐藏文本 token 采样都不进入答案头。teacher hidden 或中间寄存器 state 若启用，只能作为训练期辅助损失，不能进入模型 forward 的答案路径。

## 2. A0 合同

### 2.1 基座与生成边界

当前默认冻结基座为 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，revision 为 `15852e8c16360a2fea060d615a32b45270f8a8fc`，许可证为 Apache-2.0。V2-A 从官方多模态 checkpoint 只提取 `language_model` 权重，不激活视觉塔。应用户要求，Qwen3.5-0.8B 也用当前 `symbolic-state-machine.v4`、同一 prompt contract 和同构 latent 配置重跑；其 revision 为 `2fc06364715b967f1860aea9cf38778875588b17`。0.8B 作为候选基座保留实证结果，但目前不替换 2B 默认基座。

text baseline 不设人为总输出长度上限：实现使用 16 token transport chunk，持续生成到语义终态、EOS 或模型物理上下文边界；chunk 只是执行传输单位，不是质量上限。遇到长时间不终止的样本，可选显式 per-example wall-time safety timeout，并将其记录为未终止。当前 Windows 缺少 `flash-linear-attention`/`causal-conv1d` fast path，Transformers 回退到 PyTorch 实现；该环境成本不能被包装成 latent 优势。

### 2.2 任务与数据

任务是三寄存器符号状态追踪：`amber`、`cobalt`、`jade` 初始各持一个 `A`–`J` 单 token 符号；`SWAP a <-> b` 交换两个寄存器，`COPY a -> b` 覆写目标但不清除源。每个 query 选择一个最终寄存器，答案是其最终符号。生成器追踪 target dependency steps，避免答案只依赖最后一个表面词。

当前 schema 为 `yggdrasil.v2-a.symbolic-state-machine.v4`：

| split | 程序长度 | 额外约束 |
| --- | --- | --- |
| train / validation / test | 2–3 | 普通组合 |
| composition-heldout | 3–4 | `swap -> copy` 组合不在 train 出现 |
| length-heldout | 5–6 | 长度超出训练范围 |

`artifacts/v2-a/data/manifest.json` 记录 1024 个 unique examples、各 split 直方图、schema 和跨 split fingerprint 不重叠。模型输入只使用 `question`，不包含 `answer` 或 `trace_text`。

## 3. 可运行命令

以下命令从仓库根目录执行；本机实际验证环境是 Python 3.11.9、PyTorch 2.11.0+cu128、Transformers 5.13.1 和 RTX 4070 Laptop GPU。

```powershell
.\.venv\Scripts\python.exe experiments\v2_a_reasoning_medium.py prepare-data `
  --output-dir artifacts\v2-a\data
```

运行 A0 三路 deterministic baseline：

```powershell
.\.venv\Scripts\python.exe experiments\v2_a_reasoning_medium.py baseline `
  --data-dir artifacts\v2-a\data `
  --output artifacts\v2-a\a0\baseline_results.json `
  --split test `
  --max-examples 32 `
  --device cuda
```

验证 0.8B 的无总长度上限 text-CoT（当前数据 32 条 smoke；继续生成到语义终态、EOS 或物理上下文边界）：

```powershell
.\.venv\Scripts\python.exe experiments\v2_a_reasoning_medium.py baseline `
  --model-id Qwen/Qwen3.5-0.8B `
  --revision 2fc06364715b967f1860aea9cf38778875588b17 `
  --data-dir artifacts\v2-a\data `
  --output artifacts\v2-a\a0\probe_qwen3p5_0p8b_current_text_cot_nocap32.json `
  --split test `
  --max-examples 32 `
  --modes text_cot `
  --device cuda
```

运行当前 A1 默认 `K=8/T=8` latent reasoner：

```powershell
.\.venv\Scripts\python.exe experiments\v2_a_reasoning_medium.py train-latent `
  --data-dir artifacts\v2-a\data `
  --output-dir artifacts\v2-a\a1\k8-t8-seed20260712 `
  --steps 400 `
  --batch-size 16 `
  --encoder-batch-size 8 `
  --device cuda
```

中断后重跑同一命令会从 `latest.pt` 恢复；使用 `--no-resume` 才显式重启。`--teacher-hidden-weight` 默认为 `0`，仅用于隔离训练监督诊断，不能作为主架构的答案旁路。

用于当前修正后 probe 的命令形态为：

```powershell
.\.venv\Scripts\python.exe experiments\v2_a_reasoning_medium.py train-latent `
  --output-dir artifacts\v2-a\a2\k4-t8-mlp-source-mean-full512-seed20260713 `
  --latent-tokens 4 `
  --recurrent-steps 8 `
  --recurrent-blocks 2 `
  --transition-backend mlp `
  --transition-bottleneck 512 `
  --source-mean-init `
  --steps 400 `
  --batch-size 16 `
  --device cuda
```

`source-adapter-backend=mlp`、`source-mean-init`、`answer-pooling=flatten`、`state-supervision-weight` 和 `recurrent-blocks=0` 都是诊断开关；默认入口仍是单层 frozen hidden、copied Qwen blocks、mean pooling、无辅助监督。source adapter 只逐 token 变换，不混合 token；已删除的 source-layer bank 和 token mixer 只保留失败 artifact。0.8B 的同构 full-data probe 使用：

```powershell
.\.venv\Scripts\python.exe experiments\v2_a_reasoning_medium.py train-latent `
  --model-id Qwen/Qwen3.5-0.8B `
  --revision 2fc06364715b967f1860aea9cf38778875588b17 `
  --output-dir artifacts\v2-a\a2\qwen3p5-0p8b-k8-t8-mlp-source-mean-full512-seed20260713 `
  --latent-tokens 8 --recurrent-steps 8 --recurrent-blocks 2 `
  --transition-backend mlp --transition-bottleneck 512 `
  --source-mean-init --steps 400 --batch-size 16 `
  --encoder-batch-size 8 --learning-rate 5e-5 `
  --no-resume --device cuda
```

逐步 verifier self-critical RL 的 full512 probe（奖励只计真实程序步骤，hidden cache schema 为 v5）：

```powershell
.\.venv\Scripts\python.exe experiments\v2_a_reasoning_medium.py train-latent `
  --output-dir artifacts\v2-a\a2\k8-t8-mlp-source-adapter128-verifier-rl-full512-seed20260713 `
  --latent-tokens 8 --recurrent-steps 8 --recurrent-blocks 2 `
  --transition-backend mlp --transition-bottleneck 512 `
  --source-mean-init --source-adapter-backend mlp --source-adapter-bottleneck 128 `
  --verifier-rl-weight 0.5 --steps 400 --batch-size 16 `
  --encoder-batch-size 8 --learning-rate 2e-4 --no-resume --device cuda
```

## 4. 当前证据

### 4.1 A0：text-CoT 是强基线，但尚非 formal Gate

`artifacts/v2-a/a0/probe8_qwen3p5_2b_canonical_all.json` 的 deterministic test probe：

| 路径 | exact | parse |
| --- | ---: | ---: |
| direct | 0.125 | 1.00 |
| answer-only | 0.50 | 1.00 |
| visible text CoT | 1.00 | 1.00 |

`probe4_qwen3p5_2b_composition_cot.json` 的 composition-heldout exact 为 1.00；`probe4_qwen3p5_2b_length_cot.json` 的 length-heldout exact 为 0.75。它们是单 seed probe，不构成 multi-seed formal Gate，但已足以作为 A1/A2 的强质量参照。0.8B 结果单独列为基座对照，不与 2B 的质量数字混合。

当前 schema 下的 0.8B 无 cap text-CoT smoke 位于 `artifacts/v2-a/a0/probe_qwen3p5_0p8b_current_text_cot_nocap32.json`：32 条 test exact `0.75`、parse `0.78125`、生成 reasoning tokens `3534`；`generation.artificial_output_token_cap` 为 `null`。它低于 2B 强基线，且部分样本在本次 smoke 中未生成可解析的 `FINAL`，因此不能替代 2B A0 Gate，也不应与旧 schema 的 0.8B artifact 混合。

### 4.2 A1：当前结构的 2B mechanism smoke 已通过

当前证据路径为 `artifacts/v2-a/a1/smoke-k8-t8-qwen3p5-2b-current/results.json`。同一目录先执行 2 steps，再不加 `--no-resume` 恢复到第 3 step：

- `K=8`、`T=8`、2 个独立 Dense recurrent blocks；Qwen3.5-2B 的 full-attention layer 选择为 `[19, 23]`；
- 冻结 text backbone `1,881,825,088` 参数，latent reasoner 可训练参数 `121,711,629`；
- latest/best checkpoint 均写出，`recovery.resumed_from` 指向 `latest.pt`；
- no-bypass 通过：`answer_head_input = final latent mean only`，无 `lm_head`、`input_ids`、`labels`、teacher/trace 命中；
- no-latent、middle-step batch shuffle 和半轨迹截短干预均可执行，且干预改变输出；
- smoke 质量为 test `0/2`、validation `0/2`，不解释为任务学习或架构优势。

此前同名旧目录的 2B smoke 使用结构修正前的 layer 选择，当前文档只引用 `-current` 结果，避免把旧机制证据当作现行实现。

### 4.3 A2：K/T 与结构/监督探针未形成泛化

以下均为同一 seed、训练集最多 128 examples、评估集各 16 examples 的 probe；exact 是最终质量，不是 teacher-forced 指标。

| 配置 | steps / lr | test | validation | composition | length |
| --- | ---: | ---: | ---: | ---: | ---: |
| K1/T8 | 40 / 2e-4 | 0.0625 | 0 | 0.125 | 0 |
| K4/T8 | 40 / 2e-4 | 0.1875 | 0.125 | 0.0625 | 0 |
| K8/T8 | 40 / 2e-4 | 0.125 | 0.1875 | 0.0625 | 0.25 |
| K4/T8，lr=5e-5 | 120 / 5e-5 | 0.1875 | 0.125 | 0.0625 | 0.0625 |
| K4/T8，full layers | 400 / 5e-5 | 0.1875 | 0.125 | 0.0625 | 0.0625 |
| K4/T8，full layers + transition gate | 40 / 5e-5 | 0.1875 | 0.125 | 0.0625 | 0.0625 |
| K4/T8，teacher hidden 权重=0.5 | 40 / 5e-5 | 0.0625 | 0 | 0.125 | 0 |

对应结果分别索引于 `artifacts/v2-a/a2/k1-t8-seed20260712/`、`k4-t8-seed20260712/`、`k8-t8-seed20260712/`、`k4-t8-lr5e-5-seed20260712/`、`k4-t8-full-layers-seed20260712/`、`k4-t8-gated-full-seed20260712/` 和 `k4-t8-teacher05-seed20260712/`。full-layers 400-step probe 已排除“只增加训练步数”这一小修解释；transition gate probe 也没有变化；teacher hidden 只用于训练期 cosine 辅助损失，结果反而下降，因此不纳入主路径。

K1/K4/K8 与学习率 probe 产生于 full-attention layer 修正前，不能当作严格同结构容量曲线；`full-layers`、`gated-full` 和当前 A1 smoke 是修正后的对照。所有数字因此只作为方向性 probe，不足以升级成公平的 A2 formal sweep。

代表性 K4 full-layers 结果的 test causal exact 为：no-latent `0.125`、middle-step shuffled latent `0.1875`、trajectory half-length `0.1875`。当前没有形成“破坏关键 latent step 就同步破坏答案”的稳定因果证据，且距离 text-CoT `1.0` 很大。

### 4.4 修正后结构与目标诊断

这些 probe 使用 prompt contract v3（encoder 与 visible CoT 使用同一状态追踪 instruction；demonstrations 作为显式可计量选项），并加入 `fit` 指标；除 full512 外，训练集最多 128 examples、评估集各 16 examples。它们用于定位瓶颈，不是 formal 结果。

| 配置 | train / fit | test | validation | composition | length |
| --- | ---: | ---: | ---: | ---: | ---: |
| K4/T2，copied Qwen | 128 / 未记录 | 0.1875 | 0.125 | 0.0625 | 0.0625 |
| K4/T8，cross-only + source mean | 128 / 0.5625 | 0.3750 | 0.3750 | 0.3750 | 0.5000 |
| K4/T8，copied block1 + gate=-5 | 128 / 0.5859 | 0.3750 | 0.3125 | 0.3750 | 0.4375 |
| K4/T8，MLP transition + source mean | 128 / 0.5625 | 0.3125 | 0.3750 | 0.1875 | 0.3750 |
| K4/T8，MLP + source mean，full512 | 512 / 0.5918 | **0.4766** | 0.4375 | 0.4219 | 0.4453 |
| K8/T2，MLP + source mean，full512 | 512 / 0.6250 | 0.5859 | **0.5000** | 0.3750 | 0.4531 |
| K8/T4，MLP + source mean，full512 | 512 / 0.5605 | 0.5234 | 0.4219 | 0.3672 | 0.4063 |
| K8/T8，MLP + source mean，2B full512 | 512 / 0.5898 | **0.6016** | **0.4922** | 0.3672 | **0.5078** |
| K8/T8，latent attention + Dense FFN，source mean，full512 | 512 / 0.6738 | 0.5078 | **0.6016** | 0.4141 | 0.4531 |
| K8/T8，MLP + source mean + masked state supervision=0.5 | 512 / 0.6504 | 0.5703 | 0.5313 | 0.3984 | 0.4219 |
| K8/T8，MLP + source query init=mean，full512 | 512 / 0.5508 | 0.5078 | 0.3750 | 0.3672 | 0.4219 |
| K8/T8，MLP + source query init=mean_last，full512 | 512 / 0.4395 | 0.3516 | 0.2656 | 0.2578 | 0.2266 |
| K8/T16，MLP + source mean，full512 | 512 / 0.6289 | 0.5469 | **0.5000** | **0.4141** | 0.4297 |
| K16/T8，MLP + source mean，full512 | 512 / 0.6270 | 0.5078 | 0.4844 | 0.4063 | 0.3906 |
| K8/T8，MLP + source layer bank，中间+末层 | 512 / 0.4785 | 0.4375 | 0.3281 | 0.3750 | 0.3438 |
| K8/T8，MLP + token-wise source adapter，bottleneck=512 | 512 / 0.6738 | 0.5469 | 0.5391 | **0.4531** | 0.5000 |
| K8/T8，MLP + token-wise source adapter，bottleneck=128 | 512 / 0.6895 | 0.5859 | 0.5156 | 0.4063 | 0.5078 |
| K8/T8，MLP + token-wise source adapter，bottleneck=128，seed=20260713 | 512 / 0.6328 | 0.5078 | 0.5000 | **0.4609** | 0.4766 |
| K8/T8，adapter128 + step-level verifier self-critical RL | 512 / **0.7227** | 0.5625 | 0.5703 | 0.3906 | **0.5547** |
| K8/T8，adapter128 + register-slot supervision=0.1 | 512 / 0.6523 | 0.5859 | 0.5078 | 0.3750 | 0.5078 |
| K8/T8，MLP + source mean，0.8B full512 | 512 / 0.2070 | 0.2578 | 0.1250 | 0.1328 | 0.2109 |
| K8/T8，MLP + source mean + source reread，gate=0 | 512 / 0.6406 | 0.5234 | 0.4531 | 0.3203 | 0.3594 |
| K8/T8，MLP + source mean，flatten | 512 / 0.3301 | 0.3516 | 0.2500 | 0.2422 | 0.2188 |
| K4/T8，MLP + state supervision=0.5 | 128 / 0.5156 | 0.3125 | 0.2500 | 0.1250 | 0.3750 |
| K4/T8，MLP + flattened tokens | 128 / 0.5469 | 0.2500 | 0.4375 | 0.0625 | 0.2500 |
| K4/T8，MLP + source mean + 2 demonstrations | 128 / 0.1172 | 0.0625 | 0.0625 | 0 | 0.0625 |
| K4/T8，MLP + source reread each step | 128 / 0.3750 | 0.1875 | 0.2500 | 0.1250 | 0.2500 |
| K4/T8，MLP + query-state supervision=0.5 | 128 / 0.5859 | 0.3125 | 0.3125 | 0.2500 | 0.3750 |

结果路径依次为 `k4-t2-encoder-cot-contract-seed20260713/`、`k4-t8-cross-only-source-mean-400-seed20260713/`、`k4-t8-source-mean-block1-gate-5-400-seed20260713/`、`k4-t8-mlp-source-mean-400-seed20260713/`、`k4-t8-mlp-source-mean-full512-seed20260713/`、`k8-t2-mlp-source-mean-full512-seed20260713/`、`k8-t4-mlp-source-mean-full512-seed20260713/`、`k8-t8-mlp-source-mean-full512-seed20260713/`、`k8-t8-latent-attention-source-mean-full512-seed20260713/`、`k8-t8-mlp-source-mean-state-supervision-mask-full512-seed20260713/`、`k8-t8-mlp-source-query-mean-full512-seed20260713/`、`k8-t8-mlp-source-query-mean-last-full512-seed20260713/`、`k8-t16-mlp-source-mean-full512-seed20260713/`、`k16-t8-mlp-source-mean-full512-seed20260713/`、`k8-t8-mlp-source-layer2-mean-full512-seed20260713/`、`k8-t8-mlp-source-adapter-full512-seed20260713/`、`k8-t8-mlp-source-adapter128-full512-seed20260713/`、`k8-t8-mlp-source-adapter128-full512-seed20260713-r2/`、`k8-t8-mlp-source-adapter128-verifier-rl-full512-seed20260713/`、`k8-t8-mlp-source-adapter128-register-slot-full512-seed20260713/`、`qwen3p5-0p8b-k8-t8-mlp-source-mean-full512-seed20260713/`、`k8-t8-mlp-source-mean-reread-open-full512-seed20260713/`、`k8-t8-mlp-source-mean-flatten-full512-seed20260713/`、`k4-t8-mlp-state-supervision-400-seed20260713/`、`k4-t8-mlp-source-mean-flatten-400-seed20260713/`、`k4-t8-mlp-source-mean-demos2-400-seed20260713/`、`k4-t8-mlp-source-mean-reread-400-seed20260713/` 和 `k4-t8-mlp-query-state-400-seed20260713/`。这些目录名沿用运行时命名；主表结果 JSON 的训练 seed 为 `20260712`，adapter128 第二个 seed 与 verifier-RL、latent-attention、masked state-supervision probe 为 `20260713`。

最后的 token-mixer smoke 位于 `artifacts/v2-a/a2/k4-t8-mlp-token-mixer-smoke-seed20260713/`：5 steps、32 train、8-example splits，test `0/8`、fit `0.125`。它只证明连续 token mixing 的 wiring 和 no-bypass，没有形成投入长 probe 的正向理由。

另有一个已删除的 latent-token self-mixer 变体：`k8-t8-mlp-source-mean-latent-mixer-full512-seed20260713/`。它的 fit/test/composition/length 为 `0.1582/0.1953/0.1016/0.1094`，说明该实现没有形成正向方向；artifact 只用于失败追溯，不再保留为当前 CLI 设计。

source-query-init 的 mean/mean_last 变体已在 full512 完成后删除 CLI 与模型入口；它们分别为 `0.5078/0.3672/0.4219` 和 `0.3516/0.2578/0.2266`（test/composition/length），只保留 artifact 作为失败追溯。source-layer bank 的中间+末层变体为 `0.4375/0.3750/0.3438`，source token mixer smoke 为 `0/0.125`（test/fit）；两者均已从当前代码与 CLI 删除，只保留 artifact 追溯。

register-slot process supervision（前三个 latent slots 对齐三寄存器）位于 `k8-t8-mlp-source-adapter128-register-slot-full512-seed20260713/`，结果为 `0.5859/0.3750/0.5078`；它没有把 adapter 的 composition 增益转化为答案泛化，当前 CLI 与训练头已删除。

step-level verifier self-critical RL 位于 `k8-t8-mlp-source-adapter128-verifier-rl-full512-seed20260713/`，结果为 test/composition/length `0.5625/0.3906/0.5547`。它确实进入反传并记录 sampled/greedy reward，但有效步骤 mask 修正后 test verifier greedy register accuracy 仅 `0.0968`、greedy state exact 与 final state exact 均为 `0`；训练末尾 policy entropy 约 `0.0017`，属于策略塌缩，不是 verifier 学会状态程序。旧 v4 cache 在此前 smoke 中把 padding 最终状态计入奖励，产生的 `0.9167` register accuracy 已判为无效，不再引用。

latent-attention + Dense FFN probe 位于 `k8-t8-latent-attention-source-mean-full512-seed20260713/`，结果为 test/composition/length `0.5078/0.4141/0.4531`；它使 composition 高于 MLP baseline，但普通 test、length 和 causal shuffled-latent 均不支持稳定递归收益。该 backend 已从当前模型、CLI 与测试删除，只保留 artifact 追溯，避免把单项 heldout 增益保留成并列设计。

masked state-supervision probe 位于 `k8-t8-mlp-source-mean-state-supervision-mask-full512-seed20260713/`，结果为 test/composition/length `0.5703/0.3984/0.4219`。它修正了过程监督对 padding 状态的错误计权，但相对 MLP baseline 仍牺牲普通 test/length，且 shuffled-latent `0.1016` 高于 no-latent `0.0625`；因此只能作为监督修正证据，不能作为状态保真或因果递归通过证据。

结论是：完整数据和 identity-initialized MLP 能把 K4 latent probe 从旧的 `0.1875` 提到 `0.4766`，K8/T8 在同一 2B full-data 预算下达到普通 test `0.6016`；但 T2/T4/T8/T16、K8/K16、source query init 和 source layer bank 之间没有同时改善普通 test、composition 和 length 的 Pareto 点。直接加入 Attention + Dense FFN 的 latent-attention probe 也未形成 Pareto：composition `0.4141`，但 test/length 只有 `0.5078/0.4531`，no-latent 与 shuffled-latent 同为 `0.0938`，因此已删除当前入口。masked state supervision 将过程监督改为只计真实步骤，但结果 `0.5703/0.3984/0.4219` 仍牺牲普通 test/length，shuffled-latent 还高于 no-latent。token-wise source adapter 是当前最有希望的接口方向：bottleneck=128 的两个 seed 分别为 `0.5859/0.4063/0.5078` 和 `0.5078/0.4609/0.4766`（test/composition/length），bottleneck=512 为 `0.5469/0.4531/0.5000`；register-slot process supervision 又回落到 `0.5859/0.3750/0.5078`。adapter 稳定地改善了部分 composition，但普通 test 和 length 没有同步超过 K8/T8 baseline，因此仍不是 A2 通过。step-level verifier RL 的 `0.5625/0.3906/0.5547` 也没有 Pareto，且 verifier 状态指标接近零、策略熵塌缩；它说明 reward 已进入训练目标，但当前 reward/policy 接口不能恢复可泛化状态。source reread、K8 flatten 和 latent-token mixer 均下降；0.8B 同构 latent 只有 `0.2578`，明显低于它自己的 0.8B text-CoT smoke `0.75`。这说明 adapter 的 heldout 部分增益尚不能解释为推理介质优势，仍需要更严格的容量/成本匹配与状态保真目标。copied Qwen block 在 gate 打开时破坏 fit，query-state supervision 和 masked state supervision 只降低/调整辅助 loss，没有改善最终答案，2 个 demonstrations 也没有收益。不能据此宣称连续 latent 已通过介质 Gate。

补充检查显示最佳 K8/T8 MLP/full-data run 的 transition gates 仍约为 `-2.002`，source-reread run 的 read gate 约为 `-0.002`（保持开放但质量下降）；verifier-RL probe 末尾 policy entropy 约 `0.0017`，说明策略塌缩而非状态程序收敛。当前收益主要来自 source-mean + MLP 的静态/弱递归表示，不能包装成已验证的多步 recurrence 优势。

## 5. A1.5 分层正控制续实验

A1.5 不沿用旧 v4 数据或旧 A2 的 K/T 结论，而是以独立 schema `yggdrasil.v2-a1.5.symbolic-state-machine.v1` 重新检查必要步骤、operation span、逐步 state 和组合 heldout。完整命令、artifact、干预和问题因果链见 [`docs/v2-a1.5-latent-foundation.md`](v2-a1.5-latent-foundation.md) 与 [`tmp/V2-A1.5 result.md`](../tmp/V2-A1.5%20result.md)。

这一轮的稳定结果是：P0 结构化正控制在 ordinary test final/state `1.0/1.0`，但 composition `0.2734/0`、length `1.0/0.2266`；P1 Qwen3.5-2B 4096-cache warm-up/joint surrogate 在 ordinary validation/test final/state `1.0/1.0`，composition final/state `0.3242/0.3290`；P2 K=8 learned workspace ordinary test `1.0/1.0`，composition `0.2773/0`，length `1.0/0.6484`。P2 的 same-answer composition shuffle accuracy `0.2773`、changed prediction rate `0.8125`，说明样本身份依赖仍在。用户要求的 Qwen3.5-0.8B no-cap visible baseline 已提供独立入口；test/composition/length zero-shot 各128条的 parse/final/state 分别为 `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`，2-shot final/state 分别为 `0.1328/0.0156`、`0.1953/0`、`0.0938/0`。未终态 completion 用显式 wall-time safety timeout 记录，不用隐藏 `max_new_tokens`，因此其批量矩阵不能与 latent 结果混为质量结论。

A1.5 的问题定位比旧 A2 更清楚：recurrent transition 本身可运行，失败集中在 operation relation binding、未见 bigram 的组合泛化、length 上的答案/state 脱钩、hidden→latent 组合信息保真和 visible baseline 的终止/显存边界。Gate 仍未通过，V2-A3/V2-A4/V2-B 继续停止。

## 6. Gate 判定、担忧与下一步

当前结论是：

1. A0 数据、基座、解析、prompt contract、无总输出长度限制和成本记录链已落地；2B visible text-CoT 是正向强基线，0.8B 当前 smoke 可跑但质量和可解析率更弱，不能替换 2B。
2. A1 当前实现的机制、checkpoint/resume、no-bypass 和干预 smoke 已通过，但 smoke 质量为零不能升级成任务学习证据。
3. A2 已覆盖 K/T、prompt、初始化、copied/MLP transition、latent-attention probe、mean/flatten readout、source reread、full-data、masked state/query-state supervision、step-level verifier RL 以及 0.8B/2B 基座对照；2B K8 full-data latent test `0.6016` 仍低于 2B text-CoT `1.0`，且 composition-heldout 没有同步改善；masked state probe 为 `0.5703/0.3984/0.4219`，verifier-RL probe 为 `0.5625/0.3906/0.5547`，状态 verifier 未学会；0.8B latent test `0.2578` 也低于其 text-CoT `0.75`。Gate A2 未通过，不能进入 V2-A3 audit 或 V2-A4 formal，更不能启动 V2-B。

主要担忧不是输出长度，而是冻结基座的 token-level 状态信息在 latent cross-attention 前后的表示几何与监督目标不匹配。旧 verifier smoke 的 padding 奖励 shortcut 已被 v5 有效步骤 mask 清除；正式 RL probe 随后出现熵塌缩和近零 verifier 泛化。source adapter 的 composition 改善说明逐 token 适配可能是正确方向，但当前 test 没有同步超过 baseline，且没有多 seed；继续堆 `K/T`、训练步数或辅助 loss 不能替代 adapter 正则化/状态保真目标的正式设计。在新的 encoder/latent 接口形成稳定正向证据前，不扩展到 audit 或多模态。

## 7. 证据边界

- 所有结果都是单 seed smoke/probe；没有 multi-seed、formal manifest、audit readout、原始文本能力 retention 或 quality-cost Pareto 结论。
- `artifacts/` 是本地运行工件；只有被本文件、阶段计划和目录索引同时引用时才构成当前 repo truth。
- A2 失败不等于“连续 latent 架构已被证明不可能”，只说明当前 Qwen3.5-2B、prompt/cache、latent 初始化、transition、读出和训练目标组合没有形成达到强 text-CoT 基线的可重复泛化。
