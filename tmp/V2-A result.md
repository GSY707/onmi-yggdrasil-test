# V2-A 总结

V2-A 已完成一轮完整的“基座—文本基线—latent smoke—结构探针—full512 probe—失败收口”实验闭环。

核心结论是：

> 当前连续 latent 实现没有通过 A2 推理介质 Gate。  
> 最好的 latent 普通 test 为 `0.6016`，但显式 text-CoT 达到 `1.0`，且 latent 没有在 composition-heldout、length-heldout 和因果干预上形成稳定优势。

这不是“连续 latent 已被证明不可能”，而是当前这组：

- Qwen3.5 hidden 表示；
- prompt/cache 合同；
- latent 初始化；
- transition；
- source adapter；
- state/verifier 监督；
- answer readout；
- 训练目标；

没有形成可重复泛化。

---

## 一、实验目标与合同

V2-A 只研究“推理介质”，不引入视觉、Boundary-MoE、FFN-MoE、动作或 Agent 控制面。

对照链是：

```text
同一成熟文本基座
├─ direct answer
├─ answer-only
├─ visible text-CoT
└─ continuous latent recurrence
```

latent 主答案头只能读取最终 latent state，不能读取：

- 原始 `input_ids`
- `labels`
- `trace_text`
- teacher hidden
- 隐藏文本 token
- 基座 `lm_head`

任务是三寄存器符号状态机：

- 寄存器：`amber`、`cobalt`、`jade`
- 值域：`A`–`J`
- 操作：`SWAP`、`COPY`
- 普通训练长度：2–3 步
- composition-heldout：包含训练中未出现的 `swap -> copy`
- length-heldout：5–6 步

数据 schema 为 `yggdrasil.v2-a.symbolic-state-machine.v4`，共 1024 个 unique examples，并检查了跨 split fingerprint 泄漏。

---

## 二、A0：文本基线与 0.8B 结果

### 1. Qwen3.5-2B 是强文本基线

2B deterministic probe：

| 路径 | exact |
|---|---:|
| direct | 0.125 |
| answer-only | 0.500 |
| visible text-CoT | **1.000** |

heldout：

- composition-heldout：`1.000`
- length-heldout：`0.750`

因此 latent 不是在解决一个已经被 direct answer 饱和的任务，而是在和一个真正有效的文本推理路径比较。

### 2. Qwen3.5-0.8B 已按用户要求实际测试

0.8B 使用了：

- 同一数据 schema
- 同一 prompt contract
- 同构 latent 配置
- 无人为总输出长度上限

0.8B visible text-CoT smoke：

- exact：`0.7500`
- parse：`0.78125`
- reasoning tokens：`3534`
- `artificial_output_token_cap`: `null`

这里的“无输出上限”是真实的。生成只在以下条件之一发生时停止：

- 语义终态
- EOS
- 模型物理上下文边界

代码中的 128-token chunk 只是传输块，不是输出长度上限。

0.8B latent full512：

| fit | test | composition | length |
|---:|---:|---:|---:|
| 0.2070 | 0.2578 | 0.1328 | 0.2109 |

因此 0.8B 没有解决 latent 泛化问题，反而明显弱于自己的 text-CoT，也不适合替换 2B 默认基座。

需要注意：当前 Windows 环境缺少 `flash-linear-attention` 和 `causal-conv1d` fast path，Transformers 回退到 PyTorch 实现。因此不能将 0.8B 的运行成本直接包装成 latent 成本优势。

---

## 三、A1：latent mechanism smoke

当前结构的 A1 smoke 已完成：

- `K=8`
- `T=8`
- 2 个 recurrent blocks
- 2B frozen text backbone
- full-attention layer `[19, 23]`
- checkpoint/resume
- no-bypass
- no-latent
- shuffled-latent
- trajectory truncation

当前 smoke 的质量为：

- test：`0/2`
- validation：`0/2`

所以 A1 的结论是：

> 机制可运行、可以反传、可以保存和恢复、没有明显答案旁路；但 smoke 不提供任务学习或架构优势证据。

A1 通过的是 wiring/mechanism，不是质量 Gate。

---

## 四、A2 实现成果

### 1. 冻结基座与 hidden cache

实现了从 Qwen3.5 官方多模态 checkpoint 中只提取 `language_model` 权重的路径：

- 不激活视觉塔
- frozen text backbone
- latent reasoner 独立训练
- hidden states 可缓存
- 结果记录参数量、显存、训练时间、latent transitions 和恢复状态

hidden cache 已从 v4 更新到 v5，新增：

- `last_hidden` source representation
- 有效程序步骤 mask
- state/query-state label
- cache identity 与 prompt contract 校验

### 2. latent 结构与诊断开关

实现并测试了：

- learned latent queries
- source mean initialization
- MLP transition
- copied Qwen transition
- source adapter
- source reread
- mean pooling
- flatten pooling
- state head
- query-state head
- no-bypass 检查
- latent trajectory intervention
- checkpoint/resume

### 3. verifier-RL

实现了真正的 self-critical policy gradient，而不是只记录一个 reward：

- sampled state rollout
- greedy rollout baseline
- per-register reward
- full-state exact reward
- final-step reward
- valid-step mask
- sampled/greedy reward statistics
- policy entropy
- 可反传 RL loss

但这条路最终失败，详见后文。

---

## 五、A2 主要实验结果

以下是当前最重要的 full512 结果。除 0.8B 外，主要为 2B、单 seed、512 train examples、128 examples per heldout split，因此属于 probe，不是 formal。

| 配置 | fit | test | composition | length |
|---|---:|---:|---:|---:|
| K4/T8，MLP + source mean | 0.5918 | 0.4766 | 0.4219 | 0.4453 |
| K8/T2，MLP + source mean | 0.6250 | 0.5859 | 0.3750 | 0.4531 |
| K8/T4，MLP + source mean | 0.5605 | 0.5234 | 0.3672 | 0.4063 |
| K8/T8，MLP + source mean | 0.5898 | **0.6016** | 0.3672 | **0.5078** |
| K8/T16，MLP + source mean | 0.6289 | 0.5469 | **0.4141** | 0.4297 |
| K16/T8，MLP + source mean | 0.6270 | 0.5078 | 0.4063 | 0.3906 |
| latent-attention + Dense FFN | 0.6738 | 0.5078 | 0.4141 | 0.4531 |
| masked state supervision=0.5 | 0.6504 | 0.5703 | 0.3984 | 0.4219 |
| source adapter 512 | 0.6738 | 0.5469 | **0.4531** | 0.5000 |
| source adapter 128，seed 1 | 0.6895 | 0.5859 | 0.4063 | 0.5078 |
| source adapter 128，seed 2 | 0.6328 | 0.5078 | **0.4609** | 0.4766 |
| adapter128 + verifier-RL | **0.7227** | 0.5625 | 0.3906 | **0.5547** |
| source reread | 0.6406 | 0.5234 | 0.3203 | 0.3594 |
| flatten readout | 0.3301 | 0.3516 | 0.2422 | 0.2188 |
| 0.8B latent | 0.2070 | 0.2578 | 0.1328 | 0.2109 |

### 主要判断

没有任何配置同时超过 K8/T8 MLP baseline 的：

```text
test 0.6016 / composition 0.3672 / length 0.5078
```

也没有任何配置接近 2B visible text-CoT 的：

```text
test 1.0 / composition 1.0 / length 0.75
```

---

# 六、遇到的问题与失败原因

## 问题一：latent 的普通质量低于 text-CoT

最好的 latent 普通 test 是 `0.6016`，而 visible text-CoT 是 `1.0`。

这说明当前 latent 路径没有证明：

- 可以保留文本推理的状态信息；
- 可以完成多步状态更新；
- 可以避免语言化而保持同等质量；
- 可以在同等质量下取得成本优势。

当前结果更像是“一个可训练的分类器式 latent bottleneck”，而不是已经形成了可靠的连续推理介质。

---

## 问题二：heldout 泛化和普通 test 不能同时提升

source adapter 是最有希望的方向：

- adapter128 seed1 composition：`0.4063`
- adapter128 seed2 composition：`0.4609`
- adapter512 composition：`0.4531`

但代价是：

- 普通 test 没有超过 `0.6016`
- length 没有稳定提高
- 两个 seed 的普通 test 波动明显

也就是说，adapter 可能改善了部分 source hidden 的接口适配，却没有建立真正的状态推理能力。

它更像是：

> 对输入表示做了局部校准，而不是恢复了可泛化的 latent reasoning process。

因此 adapter 只能作为下一轮信息保真设计候选，不能进入 A3。

---

## 问题三：K/T 没有形成单调容量收益

K/T 结果没有呈现清晰规律：

- K8/T2 普通 test 较高，但 composition 较低；
- K8/T8 普通 test 最高；
- K8/T16 的 composition 较高，但 test/length 下降；
- K16/T8 没有超过 K8/T8；
- 更大的 T 或 K 没有稳定收益。

这排除了一个简单解释：

> “只要加大 latent 数量或递归步数，就会自然得到更好的推理。”

当前更可能是表示接口和训练目标先出了问题，容量不是主瓶颈。

---

## 问题四：MLP transition 没有真正的 latent token 交互

当前 MLP transition 对每个 latent token 独立变换：

```text
H[k] -> MLP -> H'[k]
```

K 个 latent workspace token 之间没有真正的信息交换。

这导致：

- 多 token workspace 的协同能力有限；
- 每个 token 更像独立特征槽；
- 多步 transition 可能只是重复非线性变换；
- K8 不一定比 K1 更像真正的多向量推理。

这与白皮书中“Attention + Dense FFN”的 latent Transformer 参考结构存在偏差。

因此曾经加入 identity-initialized latent-attention transition 进行验证。

但 latent-attention full512 结果是：

- test：`0.5078`
- composition：`0.4141`
- length：`0.4531`
- no-latent：`0.0938`
- shuffled-latent：`0.0938`

它虽然提高了 composition，却没有形成：

- 普通 test Pareto
- length 泛化
- shuffled-latent 因果下降
- 稳定的递归依赖

所以该 backend 已从当前代码和 CLI 删除，只保留 artifact。

---

## 问题五：copied Qwen block 没有形成有效递归

复制 Qwen 顶层 block 是最接近白皮书参考结构的路径，但结果反而较差：

- fit 受损；
- gate probe 没有改善；
- transition gate 长期接近关闭；
- copied block 没有带来稳定 heldout 提升。

当前最佳 MLP/full-data run 的 transition gate 仍约为：

```text
-2.002
```

这意味着实际有效 gate 很小。

因此当前最佳结果不能包装成“已经验证的多步 recurrence”，更准确的表述是：

> source mean + identity-initialized MLP 形成了一个静态或弱递归表示。

---

## 问题六：过程监督最初存在 padding shortcut

state labels 按 split 的最大程序长度补齐，短程序后面的步骤会重复最终状态。

最初的 state CE 和 verifier reward 把这些重复状态当作真实步骤。

这造成两个问题：

1. verifier 可能只学最后状态；
2. padding 状态会虚假抬高 reward/accuracy。

旧 smoke 曾得到 verifier register accuracy `0.9167`，后来确认这是 padding shortcut，不是状态程序学会。

之后完成了：

- hidden cache v5
- 有效步骤 mask
- masked state CE
- masked query-state CE
- masked RL reward
- 最终有效步骤索引计算

修正后的 verifier full512：

- register accuracy：`0.0968`
- state exact：`0`
- final state exact：`0`

修正后的 masked state supervision：

- test：`0.5703`
- composition：`0.3984`
- length：`0.4219`
- no-latent：`0.0625`
- shuffled-latent：`0.1016`

所以 mask 修复是必要的计量修正，但没有解决推理能力问题。

---

## 问题七：verifier-RL 发生策略塌缩

verifier-RL 使用了：

- sampled rollout
- greedy self-critical baseline
- register accuracy reward
- full state exact reward
- final state reward

full512 结果：

- fit：`0.7227`
- test：`0.5625`
- composition：`0.3906`
- length：`0.5547`
- verifier register accuracy：`0.0968`
- state exact：`0`
- final state exact：`0`
- policy entropy：约 `0.0017`

训练后期 sampled 和 greedy 行为几乎一致，advantage 接近零，RL loss 失去有效探索。

这说明：

- reward 已经真的进入了反传；
- 但 reward 太稀疏、策略过快确定化；
- verifier policy 没有学到可泛化状态；
- reward loss 不能代替最终答案和因果指标。

因此当前 RL 结果只能说明“正式 reward 目标已接入并暴露了策略塌缩”，不能说 verifier 设计成功。

---

## 问题八：因果递归证据不足

理论上，真正的 latent recurrence 应该满足：

- 删除关键 latent step，答案明显下降；
- 打乱关键 latent step，答案明显下降；
- 截短 trajectory，答案明显下降；
- no-latent 显著差于正常路径。

实际结果多次出现：

- no-latent 与 shuffled-latent 接近；
- shuffled-latent 甚至高于 no-latent；
- half-trajectory 结果接近甚至超过正常 test；
- latent-attention 的 no/shuffled 都是 `0.0938`；
- masked state supervision 的 shuffled-latent `0.1016` 高于 no-latent `0.0625`。

这说明当前 answer head 很可能依赖：

- 初始 source summary；
- 静态 latent 表示；
- 浅层 shortcut；
- 非关键的末端表示；

而不是依赖一条真正有因果意义的多步状态轨迹。

---

## 问题九：readout 和输入增强没有解决瓶颈

### flatten readout

把最终 latent token 全部 flatten 后读出：

```text
test 0.3516
composition 0.2422
length 0.2188
```

明显不如 mean pooling。

### source reread

每个 latent step 重新读取 source：

```text
test 0.5234
composition 0.3203
length 0.3594
```

质量下降，说明不断重新读取 source 没有形成更好的状态更新，反而可能削弱 recurrence。

### query init

source mean / source last-token 初始化都没有稳定提升：

- source query init mean：`0.5078/0.3672/0.4219`
- source query init mean_last：`0.3516/0.2578/0.2266`

### demonstrations

向 encoder 注入两条 text-CoT demonstration：

```text
test 0.0625
composition 0
length 0.0625
```

说明 demonstrations 不是当前 latent 的稳健修复方案，反而可能引入分布干扰。

---

## 问题十：训练 fit 与 heldout 存在明显错位

例如 verifier-RL：

```text
fit 0.7227
test 0.5625
```

adapter128 seed1：

```text
fit 0.6895
test 0.5859
```

latent-attention：

```text
fit 0.6738
test 0.5078
```

这显示 latent reasoner 可以在训练集上学到一定映射，但不能稳定外推到：

- 新组合；
- 更长程序；
- 新的寄存器依赖；
- 真正的多步状态轨迹。

因此不能用训练 loss、fit accuracy 或 state loss 判断介质成功。

---

## 问题十一：成本优势尚未成立

当前记录了：

- frozen backbone 参数；
- trainable latent 参数；
- latent transitions；
- peak CUDA memory；
- training time；
- cache；
- checkpoint；
- processed examples。

但还没有完整的：

- FLOPs 估算；
- 端到端推理 latency；
- 同质量成本匹配；
- 多 seed 成本/质量曲线；
- 原始文本能力 retention。

此外，Qwen3.5 在当前环境缺少 fast path，不能把本地运行差异当作架构成本结论。

所以当前没有 quality-cost Pareto 证据。

---

# 七、已经删除的失败设计

按照项目治理规则，以下设计在负向 full probe 后已从当前代码/CLI 删除：

- source layer bank
- source query init 专用入口
- source token mixer
- latent-token self-mixer
- register-slot supervision head
- latent-attention transition backend

它们的 artifact 和数字保留在文档中，用于失败追溯，但不再保留并列兼容入口。

当前代码只保留：

- copied Qwen transition
- MLP transition
- token-wise source adapter
- source mean/reread
- mean/flatten readout
- masked state/query-state supervision
- verifier-RL 诊断路径

---

# 八、最终 Gate 判定

## A0

状态：已完成 probe，未达到 formal。

- 2B visible text-CoT 是强基线；
- 0.8B 已按无输出上限要求完成对照；
- 数据、prompt、解析、cache 和成本链可复现。

## A1

状态：mechanism smoke 已通过。

- 前后向可运行；
- checkpoint/resume 可运行；
- no-bypass 通过；
- 干预链可运行；
- 但 smoke 质量为零，不能解释成任务学习。

## A2

状态：未通过。

Gate A2 要求至少一个连续方案同时具备：

- heldout 泛化；
- length 外推；
- 多 seed 稳定性；
- 相对 text-CoT 的质量或成本优势；
- 可靠的 latent 因果依赖。

当前没有任何配置满足这些条件。

## A3/A4/V2-B

均未启动：

- A3 audit readout 未启动；
- A4 formal multi-seed 未启动；
- V2-B 多模态/MoE 未启动。

这是有意的路线门禁，而不是遗漏。

---

# 九、当前最重要的结论

V2-A 证明了三件事：

1. **文本能力确实能作为 latent 实验的强对照和输入来源。**
2. **当前 latent 方案可以运行、可以训练，但还没有形成可靠的连续推理介质。**
3. **问题不在“有没有再加一点 K/T 或 loss 权重”，而在信息保真、状态更新和 reward/control 接口。**

下一步不应继续堆小型超参 probe，而应重做：

- encoder hidden 到 latent workspace 的信息保真接口；
- latent state 的显式可验证结构；
- 真正因果的状态更新机制；
- 可探索且不塌缩的 verifier reward；
- text ↔ latent 的可检查转换或重建目标；
- 多 seed、成本匹配和 audit readout。

详细真源：

- [V2-A 实验记录](<C:/Users/24408/Documents/onmi yggdrasil test/docs/v2-a-reasoning-medium-experiment.md>)
- [下一阶段测试计划](<C:/Users/24408/Documents/onmi yggdrasil test/docs/next-stage-test-plan.md>)
- [目录索引](<C:/Users/24408/Documents/onmi yggdrasil test/docs/DIRECTORY_REFERENCE.md>)
- [训练实现](<C:/Users/24408/Documents/onmi yggdrasil test/src/yggdrasil_v2/reasoning_medium/train.py>)