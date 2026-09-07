# V2-A Closure C1 单 seed 架构资格结果复盘

日期：2026-08-25

正式身份：`V2-A-CLOSURE-C1-SINGLE-SEED-20260825-1`

机器终态：`FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY`

停止点：`G007 trace credit`

## 1. 核心结论

C1 的唯一正式运行完整启动并自然结束，但没有通过架构资格：G001 identity/no-bypass 与 G002 Overfit32 通过，六个 epoch 的 primary train/select 也按冻结合同完成；随后 G007 在 ERE、CPS 两族的全 token 与 content-token accuracy 上同时失败，停止树因此没有运行 G003–G006、G008–G011。终局为 `authorizes="nothing"`、`c2_authorized=false`、`v2a_passed=false`。

这个结果不是“模型完全没学到”。正确 owner 的轨迹 NLL 明显优于同族错配 owner，说明 learned trajectory 中存在可泛化的样本相关信号；但该信号不足以让冻结 trace probe 准确恢复 CT1 的具体 token，尤其是非固定语法内容。最简洁的表述是：**C1 学到了轨迹相对身份，却没有形成合同要求的可解码精确推理状态。**

它也不是整个 V2-A 架构的最终否定。因为 behavior、OOD、causal、hidden necessity、recurrence necessity 与 slot integrity 均按停止树未运行，现有证据不能回答这些能力是否存在。它只否定当前 `Boundary -> K=8 -> T=10 shared core + CT1 dense credit` 候选满足这份 C1 必要条件；不得把失败改写成 C2/Pareto 结论，也不得据此恢复 H1/WD、routed projection 或“先写后删”。

## 2. 单次执行链与证据身份

四个执行阶段均使用固定 identity，正式 root 没有覆盖、删除或重跑：

| 阶段 | 终态 | 关键规模与耗时 | result / seal SHA-256 |
| --- | --- | --- | --- |
| cache preflight | `PASS_V2_A_CLOSURE_C1_CACHE_PREFLIGHT` | 26,624 条全 bank CPU 资格检查，加最长八条真实 Qwen CUDA 前向；5,499.10 s | `D98B962F…E2B4` / `BC774BF2…C35` |
| cache formal | `PASS_V2_A_CLOSURE_C1_CACHE_QUALIFICATION` | 26,624 条、16,596,184 source tokens、416 shards、67,998,434,400 bytes；cache build 6,118.38 s，总计 16,057.77 s | `A0586871…D4A` / `68CA34BB…DC34` |
| learner preflight | `PASS_V2_A_C1_PREFLIGHT` | 最长八条真实 cache row；trace chunk 64/512 forward+backward；零 optimizer step、零 model write | `2B2749D8…753` / `27A0DA8E…A6EA` |
| C1 formal | `FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY` | 6,544 optimizer steps、14 model writes；总计 7,174.04 s | `FE9F22B8…972C` / `BEB46576…FA19` |

cache formal 的 K001–K008 全部通过，最终 seal 独立 replay `867/867`。C1 formal 的 source identity 为 `F910F5F47EE066606BF2A7AB65F65B875B9E143EC7C3929D7D1BCD8FB992729C`，运行前后稳定；最终 seal 独立 replay `55/55`。选中 checkpoint 为 update `5,120`，SHA-256 `3FEC2FF80CC65FAEEB5E033F77D686D35E42B113730BC555F0FF1E47B8FEF46B`。

learner preflight 曾报告 memory-efficient attention backward 的非确定性警告。冻结实现使用 fixed seed 与 `torch.use_deterministic_algorithms(..., warn_only=true)`，所以该证据是合同规定的一次性 fixed-seed 运行，不是 bitwise rerun 可重复性证明；这不改变本次自然 FAIL，也不授权以复跑检查数值波动。

## 3. Gate 发生了什么

| 阶段 | 状态 | 证据含义 |
| --- | --- | --- |
| G001 identity/no-bypass | PASS | C0R、cache、Qwen revision、source identity、fresh initialization 与禁止输入检查成立。 |
| G002 Overfit32 | PASS | 通路可训练、trace probe 可提供梯度，且 probe 剥离不改变 answer。 |
| Primary train/select | PASS | 恰好完成 6,144 updates；schedule、finite、exposure 与 trace-only selection 检查全通过。 |
| G007 trace credit | **FAIL** | 两族都只有 owner-relative NLL margin 通过；精确 token 与 content-token accuracy 均失败。 |
| G003–G006、G008–G011 | NOT_RUN | 合同顺序要求 G007 先通过；不能把未测阶段记作失败或通过。 |

G002 在 update `400` 首次满足全部条件：answer `32/32`，ERE/CPS 的全 token 与 content-token accuracy 都为 `1.0`；correct-owner NLL margin 分别为 `5.22508` / `4.51038` nat/token。物理剥离 probe 前后仍为 `32/32`，prediction invariance `1.0`，FP32 logits max-abs-diff `0.0`，剥离后 trace-probe 参数为零。这排除了实现断路、probe 偷接 answer head，以及小样本根本不可优化这三类解释。

Primary training 完成 `6,144` updates，8,192 个训练样本各暴露六次，共处理 49,152 example exposures、27,503,904 source tokens 与 3,145,728 trace tokens。训练全 finite，吞吐为 `3.55677 updates/s`，训练 wall time `1,727.41 s`，峰值 CUDA allocated memory `617,660,928 bytes`。冻结 trace-only 规则选中 update `5,120`；ERE/CPS validation trace NLL 分别为 `2.13154` / `2.39088`，answer、OOD 或 causal 结果没有参与选择。

## 4. G007 的直接结果

G007 要求每族全 token accuracy `>=0.80`、content-token accuracy `>=0.60`，并要求正确 owner 相对同族错配 owner 的 NLL margin point `>=0.15` 且 record-bootstrap lower `>=0.05`。

| family | 全 token accuracy | content-token accuracy | owner NLL margin point | margin bootstrap lower | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| ERE | `0.580391` | `0.372911` | `2.126506` | `1.977064` | accuracy 两项 FAIL；margin PASS |
| CPS | `0.411512` | `0.307978` | `0.920545` | `0.834268` | accuracy 两项 FAIL；margin PASS |

这里的反差很重要。NLL margin 说明同一个 probe 面对正确 trajectory 时，平均给目标序列的概率远高于错配 trajectory；它不是纯随机或完全忽略 latent。可是 argmax token accuracy 仍远低于合同门槛，content token 更差，说明 trajectory 携带的是粗粒度、分布式或不充分的信息，而不是稳定可读的 CT1 状态。

现有证据无法在下列解释中做唯一选择：latent medium 的容量/动力学不足；CT1 对当前 latent 表示要求了过强的逐 token 可逆性；或训练目标与 checkpoint selection 优先改善 NLL、却没有形成足够尖锐的正确 token 决策。区分它们需要另立只读诊断或新合同，不能在已失败 root 上调 trace 权重、延长 updates、换 checkpoint 或换 seed。

## 5. 计算利用率与提速边界

这轮并非 CPU-only。cache 的真实 Qwen hidden 生成和 C1 训练都在 CUDA 上执行，训练也使用 BF16 autocast；但当前运行时只有一张 CUDA 卡。Windows 看到 Intel UHD 与 NVIDIA RTX 4070 Laptop GPU，PyTorch `2.13.0+cu130` 和 `nvidia-smi` 都只枚举到 RTX 4070 一个 CUDA device。因此所谓“两块 GPU”在当前机器上是核显加独显，不是两张可直接做 PyTorch CUDA 并行的卡。

CPU 侧确有明显的串行瓶颈。当前系统枚举到 i9-13900HX 的 24 个物理核、32 个逻辑处理器；即使按用户希望只使用其中 16 个物理核，现实现的 full-bank trace materialization、tokenization、hash/replay、shard 组装与 mmap 打包也主要是 Python 顺序循环，没有 worker pool。cache formal 总耗时 `4.46 h`，其中真正的 CUDA cache build 为 `1.70 h`；C1 formal 总耗时 `1.99 h`，其中 primary CUDA training 只有 `28.79 min`。因此观察到的低 CPU/GPU 利用率是真实的基础设施问题，不是这次 G007 失败的证据解释。

若未来授权新合同，提速应作为基础设施直接切换，并在正式启动前证明与串行参考 byte-identical/metric-identical：

1. 将纯 CPU 的 CT1 formatting/tokenization/hash/replay 改为固定分区、固定 merge 顺序的 16-worker process pipeline；
2. 用 source-length bucketing 与 token-budget batching 减少 Qwen batch=8 的 padding 浪费，并将 CPU tokenizer、pinned-memory H2D、CUDA forward、CPU shard writer 做有界流水；
3. 对训练/评测增加 mmap prefetch、pinned batches，并在不改变 effective batch、sample order、loss 与 optimizer step 语义的前提下提高 microbatch 或使用梯度累积；
4. 只有 `nvidia-smi` 与 `torch.cuda.device_count()` 都发现第二张 NVIDIA CUDA 卡后，才注册 DDP/双卡 cache sharding；Intel UHD 不能冒充第二张 CUDA 卡。

这些修改不能回填本次 sealed root，也不能成为重跑 C1 的理由。它们只应进入未来新身份的 preflight 性能合同。

## 6. 当前路线位置

C0R 已证明“数据、trace 尺子和公平输入合同可用”；C1 又证明“候选通路能在小样本上训练，并能在完整预算中学到 owner-relative trace signal”。但 C1 没有证明 latent state 达到可部署资格，也没有进入答案泛化和因果必要性评测。V2-A 当前因此停在 **single-seed architecture eligibility FAIL**，C2 matched Pareto、C3 自然语言审计、V2-B 与 V2-C 均未授权。

下一步不是自动再训练，也不是回到“先写后删”。应先决定是否授权一个不改变 sealed root 的失败归因阶段，专门区分“CT1 尺子过强”与“latent state 信息不足”；在这个判断形成前，不应为追过 G007 而局部调权重或堆训练步数。
