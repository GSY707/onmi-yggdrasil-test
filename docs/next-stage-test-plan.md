# 下一阶段测试任务规划

日期：2026-07-11

架构真源：`docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md`

总路线：`docs/Project-Yggdrasil V2 从架构验证到商用路线图.md`

当前状态：V2-A 尚未实现；本文件只冻结最近两部分实验，不提供不存在的训练命令或结果。

## 1. 路线直接切换

当前路线不再继续 AV-J-D，也不继续给 AV-J/AV-J-B/AV-J-C 的 record reward、candidate mask 或 no-source shortcut 打补丁。

旧 Stage A—AV-J-C 保留为代理实验历史，最高只能提供 mechanism/surrogate 证据。新的当前主线从 V2-A 开始：

1. **V2-A：推理介质。**比较显式文本思维链、单向量 latent recurrence 和多向量 latent recurrence。
2. **V2-B：多模态与双层 MoE。**使用 V2-A 胜出介质，验证 Boundary-MoE、FFN-MoE 和文本/视觉/动作融合。

在 V2-A 正式通过前，不实现 V2-B；在 V2-B 通过前，不加入工作树运行层、长期记忆、主动采样、渐进生成或离线专家晋升。

## 2. 共同证据口径

### 2.1 证据等级

- `smoke`：前后向、checkpoint、结果 JSON 或单个接口能运行；
- `probe`：单 seed 或缩小数据的方向信号；
- `formal`：预注册配置、多 seed、heldout、消融和成本指标齐全；
- `architecture-fidelity`：成熟基座、连续 recurrence、无旁路、审计和专家边界符合 V2；
- `architecture-formal`：高保真实现相对同预算强基线形成可重复优势。

### 2.2 所有实验必须报告

- 参数量：冻结参数、可训练参数和 active parameters 分开；
- 数据量：unique tokens/examples 与 processed tokens 分开；
- 质量：最终 exact/episode success、heldout、长度外推和分任务指标；
- 因果：no/shuffled input、no/shuffled latent、截短 trajectory 和必要 modality 消融；
- 成本：训练时间、推理延迟、采样次数、latent transitions、峰值显存、KV/激活和估算 FLOPs；
- 稳定性：smoke 之外至少记录 seed，formal 默认 3 seeds；
- 恢复：latest/best checkpoint、resume、设备、schema version 和中断原因；
- 边界：失败原因、未测试项和不能外推的能力。

### 2.3 禁止的成功口径

以下指标不能单独判定成功：loss、latent cosine、MSE、单次 answer accuracy、teacher-forced 指标、模型自述或更大参数量。

## 3. V2-A：推理介质实验

### 3.1 V2-A0：成熟基座与任务基线

#### 目标

选择一个能稳定完成任务的最小成熟文本基座，并建立公平的文本推理基线。

#### 必须完成

- 确定基座、tokenizer、推理模式和许可证；
- 选择确实需要多步状态更新的纯文本任务；
- 建 train/validation/test、composition-heldout 和 length-heldout；
- 跑 direct answer、无显式 CoT 和显式文本 CoT；
- 记录生成 reasoning token 数、KV、延迟和最终质量；
- 排除模板、标签、长度和答案候选泄漏。

#### Gate A0

文本 CoT 基线在多 seed 或可重复 deterministic 配置下稳定；任务不是靠直接映射即可饱和；恢复和成本计量链可用。

### 3.2 V2-A1：连续 latent recurrence smoke/probe

#### 参考结构

```text
成熟文本基座（完全冻结）
-> 输入 hidden states
-> K 个 learned latent queries
-> 复制基座顶部 2 个 block 的独立 recurrent reasoner
-> 固定 T 次 latent transition
-> 只读最终 latent 的答案头
```

第一版固定：

| 参数 | 起始值 |
| --- | --- |
| `D_latent` | 基座 residual width |
| `K` | 8 |
| `T` | 8 |
| recurrent blocks | 2 |
| FFN | Dense |
| 基座 | 完全冻结 |
| 输出 | 单文本答案专家 |
| audit readout | 暂不联合训练 |

#### Gate A1

前后向、checkpoint/resume、无答案旁路、固定 `K/T` 的任务学习和结果 schema 均通过；该 Gate 只证明机制可训练，不提供介质优越性结论。

### 3.3 V2-A2：`K/T` 容量—步骤曲线

#### 对照

1. A0：显式文本 CoT；
2. A1：`K=1` 单向量 recurrence；
3. A2：`K=4/8/16` 多向量 recurrence；
4. 固定 `T=8` 扫 `K`；
5. 选择合适 `K` 后扫 `T=2/4/8/16`；
6. 固定步数与计算匹配两种口径。

#### 必须消融

- answer head 直读输入；
- `no-latent`；
- shuffled latent step；
- trajectory 截短；
- 关键 step intervention；
- 隐藏文本 token 采样检查。

#### Gate A2

至少一个连续方案在 heldout、长度外推和多 seed 中形成稳定 Pareto 改善。若只靠更大 `K/T` 和更多计算获胜，判为容量收益，不判介质通过。

### 3.4 V2-A3：自然语言 audit readout

#### 训练顺序

1. 冻结或大部分冻结已训练 latent reasoner；
2. audit decoder 只读取 `H_1...H_T` 与允许的来源/阶段记录；
3. 不提供标准答案；
4. readout 不足时才以低权重短程联合训练。

#### Gate A3

- 打乱/删除关键 latent step 会同步破坏思维链和结果；
- readout 中间判断能预测后续 state/动作/答案；
- 受控干预产生方向一致变化；
- `no-latent` 与 `shuffled-latent` 明显下降；
- readout 不只是复述问题或最终答案。

### 3.5 V2-A4：architecture-fidelity formal

#### 正式配置要求

- 预注册基座、数据、`K/T`、学习率、冻结矩阵和 seeds；
- 同基座文本 CoT、direct 和 latent 对照；
- 至少 3 seeds；
- heldout、长度外推、消融、audit 和成本齐全；
- 原始文本能力 retention 套件；
- latest/best checkpoint 与可复现实验 manifest。

#### V2-A 通过标准

连续 latent recurrence 同时满足：

1. 无离散隐藏 CoT 和答案旁路；
2. 最终质量不低于强文本基线，或在同成本下有实质提升；
3. 在同质量下具有可重复成本优势，或在同成本下具有可重复质量优势；
4. 多向量容量收益与介质收益被分开报告；
5. 成熟文本能力没有不可接受退化；
6. audit readout 通过因果忠实性 Gate。

若未通过，停在 V2-A，重做 transition、训练监督、任务或基座；不进入 V2-B。

## 4. V2-B：多模态与双层 MoE

V2-B 只有在 V2-A 通过后启动。

### 4.1 V2-B0：不可单解的文本 + 视觉任务

- 任务必须同时依赖文本约束和视觉事实；
- 同 prompt 风格但图像事实不同的 hard negatives；
- 冲突文本/图像样本；
- no-text、no-image、shuffled-image 和 text-only 基线；
- 视觉先语言化与直接视觉 latent 两条路径。

### 4.2 V2-B1：Boundary-MoE + Dense core

- 文本/视觉按显式 modality routing；
- 专家内部异构，边界统一 `D_latent`；
- 先 raw concat 保真，再比较 Attention Pump；
- 潜变量核心使用 V2-A 胜出配置和 Dense FFN；
- 输出先固定文本专家。

Gate：Boundary-MoE 相对普通统一接口/早期拼接形成因果正确的质量或成本收益；no/shuffled modality 明显下降。

### 4.3 V2-B2：Attention Pump 与 bypass

比较：

1. raw expert tokens；
2. pump-only；
3. pump+residual/bypass。

`K_pump` 固定并单独报告，route weight 不直接决定输出长度。Gate 同时看重建、任务、消融和成本。

### 4.4 V2-B3：FFN-MoE 2×2 对照

| 外部接口 | 核心 FFN |
| --- | --- |
| 普通接口 | Dense |
| Boundary-MoE | Dense |
| 普通接口 | FFN-MoE |
| Boundary-MoE | FFN-MoE |

FFN-MoE 从 4 experts、top-1 和约 1.25 capacity factor 起步。Boundary router 与 FFN router 分别记录使用率、entropy、collapse 和任务族分布。

Gate：组合相对各单项产生多 seed 可重复净收益；若只增加总参数而没有 active-compute 或质量收益，回退 Dense。

### 4.5 V2-B4：文本 + 视觉 + 动作

- 加入动作状态、动作候选和动作输出专家；
- 使用 READ/REASON/AUDIT/EMIT/STOP；
- 输出先显式选择单专家；
- 检查动作合法性、视觉事实、文本约束和审计一致性；
- 任一单模态不能独立解决任务。

### 4.6 V2-B5：教师调用到自主调用

1. 静态输入；
2. 教师指定 READ/EMIT；
3. 删除调用标签后自主调用；
4. 加入无效调用、空结果、冲突、超时和精读成本。

Gate：同质量下降低总读取成本，且能从错误调用恢复。该阶段通过后，路线进入总路线图 R3/A1。

## 5. 当前不执行的内容

以下内容已经进入 V2 和总路线，但不属于最近实现：

- 多输出专家候选 arbiter；
- 图像/视频/音频高保真生成与编辑；
- 工作树、记忆树和 provider KV 联动；
- 长程 Agent、多 Agent 与跨窗口恢复；
- 离线新增专家与授权晋升；
- 渐进生成和动态资源预算；
- 商用 serving、数据治理、SLO、计费和客户试点。

## 6. 旧路线收口

以下路线不再继续：

- AV-J-D；
- 继续优化 AV-J-C 粗粒度 record reward；
- 把 object/cell/count/relation slot 当作目标架构必须采用的内部本体；
- 单阶段 from-scratch Micro-Omni 作为 V2 正式验证；
- Attention Pump 唯一通道；
- 内部宪法自评后直接固化生产权重；
- 工作树 drop 等同 KV 无损回收。

旧脚本、测试、报告和 artifact 只保留历史证据地位。后续若实施 V2，应删除或重写锁定旧契约的代码和测试，不建立兼容 wrapper。

## 7. 最近执行顺序

1. 完成 V2-A0 基座与任务选择；
2. 写 V2-A 文档化实验规格和数据 schema；
3. 实现 V2-A1 latent reasoner smoke；
4. 完成 A2 `K/T` sweep；
5. 完成 A3 audit readout；
6. 预注册并运行 A4 formal；
7. 只有 A4 通过后，开始 V2-B0 文本 + 视觉任务设计。

当前没有可运行的 V2-A 命令、脚本或 formal artifact。后续实现时必须同步本文件和 `docs/DIRECTORY_REFERENCE.md`，不得把计划项写成已完成项。

## 8. 担忧与不确定性

1. 成熟文本基座可能无法在本机 8GB 显存上按理想配置训练；应优先冻结、缓存 hidden states、使用小型 latent reasoner，而不是退回 from-scratch 代理模型冒充目标架构。
2. 复制顶部 block 形成 recurrent reasoner 是参考实现，不保证最优；R1 失败时应与共享层或独立 reasoner 对照。
3. audit decoder 可能事后合理化；没有因果干预不得判可审计。
4. 计算匹配比参数匹配更重要；latent 方案不能靠更多 step 和更宽状态获得不公平优势。
5. Boundary-MoE 与 FFN-MoE 同时训练可能使归因和优化不稳定，因此必须先 Boundary/Dense，再做 2×2。
6. 商用路线需要首个明确垂直场景；当前文档只覆盖架构和研究路径，不承诺市场已经验证。
