# 下一阶段测试任务规划

日期：2026-07-17

架构真源：`docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md`

总路线：`docs/Project-Yggdrasil V2 从架构验证到商用路线图.md`

当前状态：V2-A0 数据/基座链路、2B text-CoT probe 和当前结构的 V2-A1 mechanism smoke 已实现；旧 A2 probe 未形成稳定 Pareto。A1.8 使结构化 core 通过 T24，A1.9 证明 oracle-role-segmented Qwen hidden 可驱动该 frozen core。A1.10–A1.17 把 exact-symbolic Reasoner 的失败定位到 relation-addressed transition、soft closure 与训练目标悖论。A1.18/A1.18B 解决诊断 core 的逐步全局状态信用分配，A1.19H 又把固定三寄存器推广为 equality-only opaque handles + variable-cardinality continuous payload，formal/causal `3/3`。A1.20B 的无 oracle full-text Boundary 在 heldout validation 失败后，A1.20C 加入分层 entity/program compiler 与 straight-through execution credit。A1.20C 的训练 token anchors、hard-forward equivalence 和 answer 均通过，但目标臂 overfit32 trajectory/final-state 只有 `0.875/0.90625`；state 与 local compiler objective 的全局梯度 cosine 为 `-0.7366`。完整 2×2、formal/causal、A1.21P、A1.22A 与 V2-B 均未启动，V2-A 当前不通过。

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

当前默认候选为 Qwen3.5-2B（revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，Apache-2.0）；按用户要求，Qwen3.5-0.8B（revision `2fc06364715b967f1860aea9cf38778875588b17`）已在同一 schema、prompt 和 latent 配置下重跑。0.8B 当前 32-example text-CoT smoke exact `0.75`、同构 latent test `0.2578`，质量明显弱于 2B text-CoT，暂不替换默认基座。V2-A 只从官方 checkpoint 提取 language model 权重，不激活视觉塔。任务 schema 为 `yggdrasil.v2-a.symbolic-state-machine.v4`：三寄存器 `amber/cobalt/jade`、`A-J` 单 token 符号、canonical `SWAP/COPY` 操作、`swap->copy` composition heldout 和 5–6 步 length heldout。

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

0.8B 的 `artifacts/v2-a/a1/smoke-k8-t8-qwen3p5/results.json` 只证明早期机制可跑；当前 2B 证据必须以 `artifacts/v2-a/a1/smoke-k8-t8-qwen3p5-2b-current/results.json` 为准。该结果已完成前后向、实际 checkpoint/resume、no-bypass 和干预链 smoke，使用 full-attention layer `[19,23]`，质量仍只能按 smoke 解释。

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

### 3.6 V2-A1.5：潜空间建立与递归正控制（当前执行结果）

A1.5 是在旧 A2 之后新增的分层正控制，不继续旧 K/T sweep。它使用独立 schema `yggdrasil.v2-a1.5.symbolic-state-machine.v1`，要求 train 覆盖 1–4 步，普通 test 与 composition-heldout 同为 2–4 步，length-heldout 为 5–6 步；每一步通过 no-op 反事实检查必要性，并保存 operation span/mask。

当前结果：

- P0 结构化 explicit-register recurrent core 已通过 32-example overfit；4096-example/192k sampled training 的 best ordinary test final/state full exact 为 `1.0/1.0`，说明 shared transition、state CE 和训练链可运行；
- P0 composition-heldout final/state 为 `0.2734/0`，length-heldout final/state full exact 为 `1.0/0.2266`，所以逐步状态与未见组合仍未通过；
- P1 已真实生成 Qwen3.5-2B FP16、分片、无静默截断的 hidden cache；small probe 仅为 underfit 诊断，4096-cache warm-up/joint formal ordinary validation final/state `1.0/1.0`，并支持 best checkpoint reload 与五 split `p1-evaluate`；
- A1.5 matched text-CoT 入口已支持 Qwen3.5-0.8B/2B、zero-shot/2-shot 和无人工总输出 cap；0.8B test/composition/length zero-shot 各128条 formal 分别为 parse/final/state `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`，2-shot test/composition/length final/state `0.1328/0.0156`、`0.1953/0`、`0.0938/0`，test zero-shot 有10条 safety timeout。批量 runner 增加透明 per-example wall-time safety timeout，但512条扩展矩阵仍未完成；
- P2 learned K=8 slots 已完成 formal 训练与干预：ordinary 通过，composition 与 same-answer shuffle 失败，length state full 仅 `0.6484`；
- A1.5 未通过，A3 audit、A4 formal 和 V2-B 均保持停止。

详细记录、命令和 artifacts：`docs/v2-a1.5-latent-foundation.md`、`tmp/V2-A1.5 result.md`、`artifacts/v2-a/a1_5/`。下一轮优先修正 operation-composition binding、same-answer identity dependence 与 length state fidelity；P1 ordinary interface 已建立，不返回旧 A2 的 K/T sweep。

### 3.7 V2-A1.6/A1.7：连续递归核心闭包（当前执行结果）

A1.6 建立了三寄存器 relation-addressed continuous core，但正式 C0 的 relation-heldout trajectory full 仅 `0.765625`。后续只读诊断表明 oracle-reset one-step 与 predicted hard re-embed diagnostic 在 test/length/relation/causal 四个 split 均为 `1.0`，把失败定位到连续 latent 写回的递归闭包；旧 final-vs-trajectory Gate 无效，旧 relation split 也不是严格单变量 holdout。

A1.7 直接切换到独立 schema 和实现，不兼容修补 A1.6：state slot 只含 value content，register key 只负责寻址；唯一新增训练约束是权重 `1.0`、target stop-gradient 的 canonical value prototype cosine closure。唯一 relation holdout 为 `COPY amber→jade`，train/validation/test/length 覆盖其他 11 个 family×directed-pair 组合，relation 每例恰好一次 holdout，background 全在训练支持内。

当前结果：

- data audit 十项 Gate 全部通过，relation heldout 位置 1–4 各 `128`，cross-split fingerprint overlap `0`，causal deletion necessary rate `1.0`；
- overfit32 在 step `200` 达到五项 fit-only 指标全 `1.0`；
- seed `20260715/20260716/20260717` 的 fresh formal C0 与 causal intervention 均通过；目标 seed 的 test/length/relation trajectory full 为 `1.0/0.996094/0.998047`，causal prefix/replacement/deletion/shuffled 为 `0.999349/0.997394/1.0/1.0`；
- 同 seed、同数据、同容量的 2×2 消融中，content-only/address-mixed 与 closure on/off 四种组合均通过 5–6 步 formal 和 causal Gate。因此短程通过不能证明地址/内容分离或 closure 必要，修正后的训练组合覆盖和 relation sequence 是重要贡献；
- fingerprint 零重叠的 8/12/16 步压力评测要求两个 split 的 aggregate 和每个长度 trajectory full 均不低于 `0.95`，三个目标 seed 为 `0/3` 通过。T16 supported 为 `0.894531/0.941406/0.894531`，说明任意加深递归尚未成立；
- closure-off 的 T16 supported 在 content-only/address-mixed 下分别降到 `0.449219/0.679688`，而 closure-on 为 `0.894531/0.953125`。closure 对长程漂移有实质作用；地址/内容分离尚无独立正收益；
- 当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断为 `40%–55%`、中心约 `48%`，完整白皮书 V2 为 `20%–35%`。这些是分层工程概率，不是统计置信区间；
- A1.7 只证明结构化 relation-addressed core 的短程可复现性，不证明 Qwen hidden、匿名 workspace 或完整 V2-A；本轮按合同不进入 C1。

详细记录与 artifacts：`docs/v2-a1.7-core.md`、`tmp/V2-A1.7 result.md`、`artifacts/v2-a/a1_7/core-assessment-summary.json`。A1.7 的长程失败已由 A1.8 后续实验继续归因；不能单独使用本节把结构化正控制升级为完整架构结论。

### 3.8 V2-A1.8：随机深度长程递归（已完成）

A1.8 没有更改 A1.7 的 content-only slots、address keys、shared transition、shared state/answer head 或 closure。唯一主动变量是把训练长度从 T1–4 切换为 T1–16，并在每个 batch 内对 16 个长度严格均衡采样。best checkpoint 以 validation 最差长度优先选择。

三组独立 data/model seed 为 `20260721/20260821`、`20260722/20260822`、`20260723/20260823`。每组数据各 `22,784` 条，彼此以及与 A1.7 data/stress fingerprint overlap 均为 `0`。length-balanced overfit32 通过。

正式结果：

- 三个 run 的 short T1–6 最低 trajectory 均为 `1.0`；
- T16 supported 最低 `0.996094`、relation 最低 `1.0`；
- OOD T24 supported 最低 `1.0`、relation 最低 `0.996094`；
- 诊断性 T32 supported 最低 `0.988281`、relation 最低 `1.0`；
- 三个 run 的 causal prefix/replacement/deletion/shuffled 全部为 `1.0`；
- initial-slot 扰动平均放大率在 T16/T24/T32 均低于 `1.0`，prototype margin 保持为正；
- 三 run 共处理 `204,800` examples、`1,740,800` latent transitions，实测训练 `610.5` 秒；成本 benchmark 与机器可读总表均已保存。

**A1.8 总 Gate 为 3/3 passed。**A1.7 T16 supported/relation mean 从 `0.910156/0.928385` 提升到 A1.8 的 `0.998698/1.0`。由于架构与 closure 不变，结果否定了 shared transition 在 T16 必然内在发散，并强烈支持训练 horizon mismatch；但 A1.8 regimen 同时改变了 horizon coverage 和 transition exposure，没有 compute-matched 地拆开两者贡献。

这个结果仍只属于强结构化 COPY/SWAP core；后续 A1.9 已在不修改 core 的前提下继续验证真实 Qwen hidden boundary，并以 3/3 通过。A1.8 的概率与下一步判断仅保留为阶段快照。

详细记录与 artifacts：`docs/v2-a1.8-long-horizon.md`、`tmp/V2-A1.8 result.md`、`artifacts/v2-a/a1_8/assessment-summary.json`。

### 3.9 V2-A1.9：冻结 Qwen hidden 边界（已完成）

A1.9 冻结 Qwen3.5-2B 和三个已经通过的 A1.8 core，只训练 value、family 和共享 register 边界映射。source、target、query 共用 register adapter；query 只能选择最终 slot；不允许 full-source、answer、query bypass、新答案头、core/Qwen 解冻或旧 A1.6 compatibility wrapper。训练继续使用 T1–16 batch 内均衡长度，正式 Gate 保持 short、T8/T12/T16、T20/T24、relation、trajectory、pointer 与 answer/state identity，并新增 core hash 不变和 boundary mapping Gate。

Qwen cache 只保存 oracle role spans 的 pooled contextual hidden，不保存 full-source hidden。由于 span hidden 仍可能携带全局上下文，A1.9 必须在普通 Gate 通过后执行 replacement/deletion/step shuffle、query swap、same-answer/different-trajectory、no-hidden 和 independent role shuffle；干预必须同时检查对重新计算 oracle 的跟随和对旧 oracle 的放弃，不能只报告 final answer changed rate。

A1.9 的 overfit32 在 step `100` 达到 mapping、pointer、trajectory 与 answer 全 `1.0`。三个正式 run 的 cache audit、formal 和 hidden intervention 均通过；三组 cache 各 `22,784` 条记录且 fingerprint 两两 overlap 为 `0`。short T1–6 最低 trajectory `1.0`，T16 supported/relation 最低 `1.0/0.996094`，T24 均为 `1.0/1.0`，诊断性 T32 为 `0.996094/1.0`，boundary mapping 最低 `1.0`。core hash 全部不变。

prefix、replacement、deletion、operation shuffle、query swap 和 same-answer/different-trajectory 对新 oracle 的跟随率在三个 run 中全部为 `1.0`；replacement/same-answer 对旧 trajectory 的保留率为 `0`，no-hidden 和 independent role shuffle 的 trajectory full exact 为 `0`。因此 **A1.9 总 Gate 为 3/3 passed**，可以声称 oracle-role-segmented frozen Qwen hidden 能因果忠实地驱动 frozen structured core。

三组 Qwen role encoding 共 `3,451.30` 秒、cache 占用 `9,177,138,012` bytes；计时不含模型/tokenizer 首次加载。cached boundary/core benchmark 也不包含在线 Qwen 编码，且未与 text-CoT 对照，因此不是 Pareto 证据。完整合同与结果见 `docs/v2-a1.9-qwen-boundary.md`、`tmp/V2-A1.9 result.md` 和 `artifacts/v2-a/a1_9/assessment-summary.json`。

A1.9 仍使用 oracle character spans、typed value/family/register 输入和显式三寄存器 COPY/SWAP core，不覆盖从完整文本自主发现 role、匿名 K-slot workspace、通用 recurrent Transformer reasoner 或 matched text-CoT Pareto。当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断更新为 `48%–63%`、中心约 `55%`，完整白皮书 V2 为 `24%–40%`；这不是统计置信区间，上调只来自真实 Qwen hidden 接入风险下降。

### 3.10 V2-A1.10：完整文本匿名工作区与通用 reasoner（已完成，失败）

A1.10 同时删除 A1.9 的 oracle character spans、typed role adapter、显式三寄存器 slots/keys 和 COPY/SWAP 专用 transition。Qwen3.5-2B 完全冻结，cache 只保存完整 last hidden 与 attention mask；`K=8`、`D=256` 的匿名 learned queries 先读取完整 source，再由两层共享 pre-norm self-attention/cross-attention/Dense FFN 递归更新。模型前向不接收 operation mask、program length、role tensor 或 register identity。T1–16 统一运行 16 步并在程序结束后要求状态保持。

修正版 overfit32 在 best step `800` 达到 T1–16 trajectory/final/answer 全 `1.0`。三个正式 run 各训练 4,000 steps，cache audit 全部通过且跨 run train fingerprint overlap 为 `0`，但 formal 为 `0/3`：T16 supported trajectory 为 `0.003906/0/0`，relation 为 `0/0/0`；T24 supported/relation 全为 `0`。T16/T24 final state 与 answer 仍约 `0.4–0.6`，说明模型学到部分终态或统计信号，却没有形成逐步状态推进。训练集均衡诊断 trajectory 同样接近 `0`，失败不是只出现在 relation/OOD。

三个 ordinary formal 均失败，按预注册没有运行 hidden intervention。三组 full-token cache 共 `38,358,822,720` bytes，Qwen 净编码 `4,767.62` 秒；三组训练 `3,256.60` 秒、`6,144,000` recurrent transitions。A1.10 只能否定当前联合配置，不能分别否定 full-text boundary、匿名 workspace 或通用 reasoner。完整合同与结果见 `docs/v2-a1.10-anonymous-workspace.md`、`tmp/V2-A1.10 result.md` 和 `artifacts/v2-a/a1_10/assessment-summary.json`。

当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断下调为 `30%–45%`、中心约 `37%`，完整白皮书 V2 为 `14%–28%`。这不是统计置信区间。

### 3.11 V2-A1.11：正交故障定位（已完成，定位成立）

本阶段没有继续联合配置的 K/T/层数/步数 sweep，而是完成两个对称诊断：

1. `A1.11-Boundary`：full-token frozen Qwen hidden → learned typed role reader → frozen A1.8 core，只删除 oracle spans。修正版 overfit32 的 trajectory/answer/mapping 全为 `1.0`，但 source/target pointer 最低均为 `0.9642857143`，严格 Gate failed；只读 hard re-embedding 后所有指标为 `1.0`，formal 按停止规则未运行；
2. `A1.11-Reasoner`：最终使用 exact symbolic typed roles，不加载 Qwen/cache/adapter/core → anonymous `K=8` generic recurrent reasoner。overfit32 在 best step 800 全指标 `1.0`；三组 formal 为 `0/3`，short/in-range/relation/T20–24/causal trajectory 均为 `0–0.003906`，训练集均衡诊断 trajectory 也只有 `0–0.003906`。

Reasoner 在精确输入下独立失败，已经否定“两个健康组件只在组合后失败”。Boundary 另有连续 latent 与 frozen address geometry 的严格接口缺陷，但 task-level formal 因 overfit stop rule 未知。完整合同与结果见 `docs/v2-a1.11-fault-localization.md`、`tmp/V2-A1.11 result.md` 和 `artifacts/v2-a/a1_11/assessment-summary.json`。

当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断调整为 `22%–35%`、中心约 `28%`，完整白皮书 V2 为 `10%–22%`。这不是统计置信区间；下调来自 exact-symbolic 条件仍无法形成大分布逐步算法，保留的上行空间来自 A1.8/A1.9 已证明最小结构化状态与真实 Qwen hidden 分别可工作。

### 3.12 V2-A1.12–A1.17：Reasoner 二级根因定位（已完成）

A1.12 的 BIND/CURSOR/BOTH 三臂均先过 overfit32、随后 formal `0/3`，排除 entity-addressable state 与 aligned operation cursor 的独立/联合充分性。A1.13 的 GENERIC-CLOSURE state `0/3`、STRUCTURED-CE `1/3`、STRUCTURED-CLOSURE state `3/3`/full `1/3`；唯一完整通过 run 的因果干预通过。A1.13F 固定 4000-step 后三条单因素参考仍为 state `0/3`、`0/3`、`1/3`，所以 early-stop 不是主因。

A1.15 query-coupled + answer CE 与 A1.16 query-coupled + no answer CE 均 overfit32 通过、新 seed state/full `0/3`。A1.17 回到 A1.13 的 model seed `20261321/22/23`，逐 tensor 验证共享初始化位相等；paired coupled-CE 和 coupled-noCE 仍分别为 state/full `0/3`，而 independent pooled-answer reference state `3/3`。机器分类为 `independent_answer_auxiliary_gradient_required`：当前 state 算法依赖独立 pooled-answer objective 的全局辅助梯度，但该旁路本身不能成为可靠因果答案接口。

该结论触发并约束了 A1.18：推理答案固定为 query-coupled state，训练期只允许不进入部署图的全局辅助目标，禁止恢复推理答案旁路。

### 3.13 V2-A1.18/A1.18B：训练目标机制解决（已完成）

QAUX 与 FINAL-SAUX 的 overfit32 均通过。QAUX 只作实现阳性控制；FINAL-SAUX 从最终 workspace mean 预测完整三寄存器终态，paired seed formal/causal 为 `2/3`。两个通过run在辅助头删除后通过 relation、T24 与全部因果干预；失败seed从 step `200` 到 `4000` 始终没有进入算法盆地，因此 final-only 全局状态梯度方向正确但初始化不稳定。

A1.18B 只改变全局状态监督密度：TSAUX 使用一组跨步共享训练头，在每个递归步从 global workspace mean 预测当步完整三寄存器 state；正式 answer CE 保持 `0`。它先在 FINAL-SAUX 失败的相同 seed 上于 step `600` 修复 state，随后统一重跑三个 paired seed，再用 model seed `20261821/22/23` 做三个 fresh run。paired 与 fresh 的 formal/causal 都为 `3/3`；六个部署模型的 causal trajectory/answer 都为 `1.0`，全部 counterfactual Gate 通过，disable-recurrence 和 wrong-start trajectory 都为 `0`。

formal 前全部 `training_auxiliary_head.*` 参数均物理删除，正式 loader 会拒绝仍含辅助参数的 checkpoint。机器分类为 `per_step_global_state_credit_assignment_confirmed`、`mechanism_solved=true`、`diagnostic_core_architecture_validated=true`。这证明 answer-specific auxiliary 不是必要条件；缺失机制是每一步 global workspace → complete state 的密集信用分配。

原计划下一阶段直接做 state-target source annealing，现已因路线重审后移。A1.10/A1.11 的匿名通用 reasoner 事实上已经在每个递归步用 workspace mean 预测完整 state 并施加 trajectory CE，但在 exact-symbolic 条件下仍 formal `0/3`；这说明 A1.18B 的成功不能只归因于新增监督，还依赖 relation addressing、显式实体地址和 closure。当前不得先削弱唯一成功配置的监督，也不得直接恢复旧 A2。完整机制结果见 `docs/v2-a1.18-training-scaffold.md`、`tmp/V2-A1.18 result.md` 与 `artifacts/v2-a/a1_18b/assessment-summary.json`。

### 3.14 V2-A1.19H：可泛化混合 core（已完成）

A1.19H 保留 relation-addressed shared transition、soft closure、TSAUX、query-coupled answer 和部署剥离，但分两个顺序子门把固定三寄存器 scaffold 改为任务无关的 addressable hybrid workspace。`H1 opaque-handle` 保持实体数、操作、数据和预算不变，只删除 slot index = register identity，逐样本随机分配 opaque handle 并测试 handle/slot 联合 permutation。H1 formal `3/3` 后才运行 `H2 variable-cardinality`，把实体数量扩展为训练内多个 `N` 与 heldout `N`，其余合同不变。

两个子门均先做 overfit32；通过后使用三个 fresh seed 跑 formal/causal，并测试 OOD horizon、relation holdout、handle permutation/alias、同值不同实体、query swap、旧轨迹拒绝、disable recurrence 和辅助头剥离；H2 另加 OOD entity count。soft continuous addressing 只作 H2 通过后的同预算消融，不是前置 Gate。任一子门 formal 不到 `3/3` 时停止 full-text、Pareto 与 V2-B。新增任务/关系族留到 A1.21P，不在 A1.19H 混入。

A1.19H 已完成。H1 删除 fixed register/slot identity 后 formal/causal `3/3`；H2 在训练 `N=2,3,4`、heldout `N=5` 和 heldout relation 合同下 formal/causal `3/3`。三个 H2 run 的 `N=5` 与 `N=5+relation` trajectory/final/answer 均为 `1.0`，OOD horizon 最低 aggregate trajectory 为 `0.983073`。机器分类 `generalized_hybrid_core_confirmed`、`a119h_complete=true`。部署 artifact 不含训练辅助参数，handle 没有语义 embedding，也不存在 fixed-register 或 task-family-specific executor。

run-3 前发现旧训练管线被 Python→CUDA 标量编码和每步地址校验同步阻塞。新管线使用一次性 validated tensor cache、GPU index-select、预生成采样与 forward 路由预计算；60-step 同初始化对照加速 `10.744×`，final loss 与所有参数差均为 `0`。旧 run-1 checkpoint 的优化后全 split 回归与原 formal JSON 完全相同。完整记录见 `docs/v2-a1.19h-hybrid-core.md`、`tmp/V2-A1.19H result.md` 与 `artifacts/v2-a/a1_19h/assessment-summary.json`。

### 3.15 V2-A1.20B：learned full-text boundary（已停止，失败）

冻结通过的 A1.19H core，使用完整 Qwen hidden 训练 Boundary 自主输出连续 value payload 与必要的 source/target/query handle、relation/type/control。输入不得包含 oracle span mask、oracle role tensor、程序步骤 mask、答案 metadata 或 hard re-embedding 诊断补丁。环境原生 source identity/位置 metadata 可以保留，但任务语义角色必须从完整文本学习。

先做 overfit32，并分别验证 value、source pointer、target pointer、query pointer、relation family 与 handle geometry；全部严格通过后才运行三个 fresh Boundary formal。正式 task Gate 必须覆盖 H2 的 short/in-range/relation/OOD/N5/N5+relation，并在 formal 通过后执行 no-hidden、hidden shuffle、role-preserving text counterfactual、query swap、operation replacement/deletion/shuffle 和 core hash 不变检查。Boundary formal 不到 `3/3` 时停止 A1.21P/A1.22A，不得用 short joint tuning 或 hard re-embedding 掩盖。

A1.20B 通过后进入 A1.21P matched R1 Pareto，再进入 A1.22A natural-language audit。训练目标来源不再是独立硬门：可执行 trace、程序状态和 verifier-filtered teacher record 都允许用于训练，但必须报告生成成本/错误率并在部署前删除辅助路径。零 teacher 与自监督 next-state 降级为可选效率研究。A1.21P/A1.22A 同时通过才关闭 V2-A/R1。路线决策真源见 `docs/v2-a-route-reassessment-2026-07-17.md`。

A1.20B 实际已执行并停止。overfit32 mapping/trajectory/final/answer 全为 `1.0`；run-1 的 full-token cache audit 全通过，固定 5000-step 训练完成，但 best checkpoint 的 2048 条 validation trajectory/final/answer 为 `0.297363/0.409668/0.608887`，最差 cell mapping/trajectory/answer 为 `0.315341/0/0.227273`，没有资格进入 formal split。

逐样本 Boundary 全字段联合 exact 仅 `0.087891`；N4 全部 value 正确仅 `0.007331`，T13–16 的完整 program/source/target 序列 exact 均为 `0`。全 oracle Boundary 使同一 frozen core 恢复 trajectory/final/answer `1.0`，排除 core 失效。state CE 的梯度实测只到 continuous value path，不到 threshold/argmax 后的 entity/operation presence、family、source、target、query logits；完整 loss 只能通过各字段局部 CE 训练这些控制。机器分类 `full_text_entity_binding_and_program_extraction_failure`。

因此 run-1 formal cache/eval、causal、run-2/run-3、A1.21P 与 A1.22A 均按合同未运行。继续前必须重新立项解决 Boundary entity/program binding 与离散控制执行级信用分配；不能直接增加 seed/step，也不能用 hard re-embedding 补丁掩盖。完整记录见 `docs/v2-a1.20b-full-text-boundary.md`、`tmp/V2-A1.20B result.md` 与 `artifacts/v2-a/a1_20b/assessment-summary.json`。

### 3.16 V2-A1.20C：分层编译器 × execution-credit 修复（已停止，失败）

A1.20C 复用 A1.20B 的 frozen full-token Qwen cache、frozen A1.19H-H2 core、数据与 seed，预注册 compiler `flat/hierarchical` × credit `hard-local/straight-through` 2×2。hierarchical compiler 先建立 entity table，再读取 ordered family/source/target 与 query；训练 token anchor target 不进入模型前向。straight-through 模式在 forward 保持 hard prefix mask、family 和 pointer，在 backward 使用 soft surrogate 驱动 frozen core。

目标臂 `hierarchical + straight-through` 的 supervision audit、4 项回归测试和 CUDA smoke 均通过。hard-forward state/answer logit 差为 `0`，core hash 不变。fixed-5000 overfit32 的 best checkpoint 位于 step 4800，anchor all-sequence exact、value、family、answer 均为 `1.0`，但 trajectory/final-state/state-token 只有 `0.875/0.90625/0.964474`，mapping minimum 为 `0.90625`；四条 `N=4` 样本仍有 entity/operation count 与 pointer-validity 联动错误，最差 cell mapping/trajectory 为 `0/0`。

失败根因已进一步定位。pointer local CE 在未执行 predicted entity mask 的 logits 上优化，而实际执行会用 hard entity mask 改写 pointer 候选集合，current ST bridge 未连续化这个 validity-set 决策。best checkpoint 的 state CE 与其余 local mapping/payload/anchor objective 梯度 cosine 全局为 `-0.7366`，entity path 为 `-0.9680`，presence heads 为 `-0.9944`；固定 `3e-4` 学习率又使 total loss 从 step 4800 的 `0.2429` 回升到 step 5000 的 `0.6103`。机器分类 `anchor_localization_solved_but_execution_objectives_conflict`。

因此另外三个 2×2 正式训练臂、heldout validation matrix、三 seed formal/causal、A1.21P 与 A1.22A 均未运行。继续前必须新立项把 entity-count/pointer-validity 联合连续化，分阶段或投影冲突梯度，并引入 joint-stage 学习率衰减；然后从 fresh initialization 重过 overfit32。不能把 anchor 或 answer `1.0` 当作机制通过。完整记录见 `docs/v2-a1.20c-boundary-repair.md`、`tmp/V2-A1.20C result.md` 与 `artifacts/v2-a/a1_20c/overfit32/hierarchical__straight_through/failure-diagnostic.json`。

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

1. 已完成 V2-A0 基座候选筛选、canonical 任务选择、数据 schema 和可复现数据生成；
2. Qwen3.5-2B 普通/组合 heldout text-CoT probe 已形成正向信号，length-heldout 仍有错误；
3. 已完成当前结构的 2B V2-A1 `K=8/T=8` latent mechanism smoke、checkpoint/resume、no-bypass 和干预链；
4. 已完成 A2 K/T（含 K16/T8 与 K8/T2/T4/T16）、prompt contract、source-mean 初始化、copied/MLP transition、latent-attention probe、mean/flatten readout、source reread、token-wise source adapter、full-data、masked state/query-state supervision、step-level verifier RL 以及 0.8B/2B 基座对照；source-layer bank、token mixer 和 latent-attention 的失败入口已删除；2B K8/T8 full-data MLP test `0.6016`，composition-heldout `0.3672`，source adapter bottleneck=128 为 `0.5859/0.4063/0.5078`（test/composition/length），masked state supervision 为 `0.5703/0.3984/0.4219`，latent-attention 为 `0.5078/0.4141/0.4531` 且 no/shuffled latent 均 `0.0938`，verifier-RL 为 `0.5625/0.3906/0.5547` 且状态 verifier 未学会，0.8B 同构 latent test `0.2578`，均未形成相对 text-CoT 的稳定 Pareto，Gate A2 未通过；
5. A1.6 formal relation Gate 失败后，已通过只读诊断定位 continuous closure 故障；
6. A1.7 已完成 data、overfit32、三个初始化 seed 的 formal/causal、2×2 结构/closure 消融和 8/12/16 步压力；短程 Gate 为 `3/3`，旧严格长程 Gate 为 `0/3`，closure 的长程价值成立，但地址分离的独立收益未成立；
7. A1.8 已完成三组独立 data/model seed 的 T1–16 random-depth training、T20/T24 OOD、T32 诊断、因果、稳定性和成本；总 Gate `3/3` 通过，A1.7 漂移主要归因为 horizon mismatch；
8. A1.9 Qwen hidden boundary 已完成：三组 cache audit、formal 与 hidden causal Gate 均通过，总 Gate `3/3`；
9. A1.10 已完成 full-token + anonymous K-slot + generic recurrent reasoner 联合切换；overfit32 通过但 formal `0/3`，所有 hidden intervention 按 Gate 停止；
10. A1.11 正交故障定位已完成：Boundary 严格 overfit pointer Gate failed；exact-symbolic Reasoner overfit passed、formal `0/3`；纯组合故障解释被否定，所有 formal 后干预按 Gate 停止；
11. A1.12 binding × cursor 已完成：三臂 overfit32 全通过、formal 全为 `0/3`，两项候选不足；
12. A1.13 transition × closure 与 A1.13F fixed-budget audit 已完成：联合臂 state `3/3`/full `1/3`，单因素臂不稳定且不是 early-stop 主导；
13. A1.15/A1.16 fresh-seed query-coupled 两臂均 state/full `0/3`；
14. A1.17 同共享初始化配对审计已完成：coupled-CE/noCE 均 `0/3`，独立 pooled-answer objective 的优化脚手架作用成立，但架构未通过；
15. A1.18 FINAL-SAUX paired formal/causal 为 `2/3`，证明 final-only 全局状态监督可替代答案梯度但 seed 不稳定；
16. A1.18B TSAUX 已完成三个 paired 与三个 fresh seed：两组 formal/causal 都为 `3/3`，部署辅助头全部删除，机制 Gate 通过；
17. A1.19H generalized hybrid core 已完成：H1/H2 overfit32、三 seed formal 与 formal 后 causal 均通过，heldout N5 与 N5+relation 三 seed 全通过，机器分类 `generalized_hybrid_core_confirmed`；
18. A1.20B learned full-text boundary 已完成 run-1 并在 formal 前 eligibility 失败；
19. A1.20C 分层 compiler × straight-through 修复已完成监督审计、实现、smoke、目标臂 fixed-5000 overfit32 与实际 checkpoint 梯度审计；目标臂 strict Gate 失败，完整 2×2、formal/causal、A1.21P、A1.22A 和 V2-B0 均保持停止。当前没有自动授权的下一阶段，继续前必须先修复 entity-count/pointer-validity 信用和 state-vs-local 负梯度冲突。

当前已有 A1.8 structured core 到 T24、A1.9 oracle-role-segmented frozen Qwen hidden boundary、A1.10–A1.17 分层失败归因、A1.18B TSAUX 训练机制闭环，以及 A1.19H generalized hybrid core 的多 seed formal/causal artifact。最新证据证明 equality-only opaque addressing + shared continuous transition + closure + per-step global complete-state credit assignment 可以稳定跨实体数量学习 exact-symbolic state，且正式输出不需要答案旁路。A1.20C 又证明 hierarchical token-role localization 可以在 full-text overfit32 学到全 exact，straight-through 也能保持 hard forward 并恢复执行级梯度；但未受约束的 state/local objective 会在 entity count、pointer validity 和 shared reader 上产生强负梯度冲突。learned full-text Boundary、matched Pareto 与完整 V2-A formal artifact 仍不存在。不得把 A1.19H symbolic core Gate 或 A1.20C answer/anchor `1.0` 写成完整 V2-A 通过。

## 8. 担忧与不确定性

1. 成熟文本基座可能无法在本机 8GB 显存上按理想配置训练；应优先冻结、缓存 hidden states、使用小型 latent reasoner，而不是退回 from-scratch 代理模型冒充目标架构。
2. 复制顶部 block 形成 recurrent reasoner 是参考实现，不保证最优；R1 失败时应与共享层或独立 reasoner 对照。
3. audit decoder 可能事后合理化；没有因果干预不得判可审计。
4. 计算匹配比参数匹配更重要；latent 方案不能靠更多 step 和更宽状态获得不公平优势。
5. Boundary-MoE 与 FFN-MoE 同时训练可能使归因和优化不稳定，因此必须先 Boundary/Dense，再做 2×2。
6. 商用路线需要首个明确垂直场景；当前文档只覆盖架构和研究路径，不承诺市场已经验证。
7. Qwen3.5-0.8B 在当前 Windows 环境缺少 `flash-linear-attention`/`causal-conv1d` fast path；文本塔可以运行但 baseline latency 较高，不能把该环境差异包装成 latent 成本优势。当前 32-example text-CoT smoke 的 `artificial_output_token_cap` 为 `null`，未设置人为总输出长度上限。
8. 当前 2B text-CoT 在普通 test 与 composition-heldout probe 为正向强基线；修正后的 2B K8/T8 full-data MLP latent 为 `0.6016/0.3672/0.5078`（test/composition/length），K8/T4 为 `0.5234/0.3672/0.4063`，K16/T8 为 `0.5078/0.4063/0.3906`，source adapter bottleneck=128 为 `0.5859/0.4063/0.5078`，K8/T2/T16 也未形成 Pareto；0.8B 同构 latent test 为 `0.2578`。仍没有多 seed 稳定性和 Pareto 优势，不能把训练 loss、teacher/state 辅助 loss、adapter 的单项 heldout 增益或单次准确率包装成 A2 通过。
9. 将 text-CoT 的 2 个 trace demonstrations 注入 encoder 的 probe 反而降至 test 0.0625，说明 demonstrations 不是当前 latent 的稳健修复，后续不能把它当作默认输入合同。
10. Qwen3.5 原生 thinking 的单样本探针在 206 token 后未形成可解析终态；native thinking 不能被当作 A0 visible CoT 的替代 Gate。
11. token-wise source adapter 在 bottleneck=128 的两个 seed 中把 composition-heldout 提升到 `0.4063/0.4609`，但普通 test 为 `0.5859/0.5078`、length 为 `0.5078/0.4766`，没有同时超过 K8/T8 baseline `0.6016/0.3672/0.5078`。bottleneck=512 的普通 test 为 `0.5469`。该方向只能作为下一轮信息保真/正则化设计的候选，不能直接进入 A3。
12. 将前三个 latent slot 绑定到 amber/cobalt/jade 的 register-slot supervision full512 probe 为 `0.5859/0.3750/0.5078`，没有把 adapter 的 composition 增益转化为答案泛化；过程监督仍不能替代正式 verifier/reward 设计。
13. 首次 step-level verifier self-critical RL full512 probe 使用 v5 有效步骤 mask，结果为 `0.5625/0.3906/0.5547`，test greedy register accuracy `0.0968`、state/final exact 均为 `0`，末尾 policy entropy 约 `0.0017`。RL 已进入反传但发生策略塌缩，不能把 reward loss 或 sampled/greedy reward 当成状态学习证据；后续若重做，必须先解决 reward 信号稀疏和 policy collapse。
14. 为符合白皮书的 Attention + Dense FFN 参考结构，曾加入 identity-initialized latent-attention transition；full512 的 test/composition/length 为 `0.5078/0.4141/0.4531`，no-latent 与 shuffled-latent 都为 `0.0938`。它没有形成因果递归或 Pareto 优势，当前代码与 CLI 已删除，不保留并列旧入口。
15. state/query-state supervision 已改为只计真实程序步骤，masked state full512 为 `0.5703/0.3984/0.4219`，但 test/length 仍低于无监督 MLP baseline，且 shuffled-latent 高于 no-latent；mask 修复解决了计量错误，不等于过程监督形成了有效 verifier。
16. A1.9 的每个 role hidden 都来自 oracle character span，而且 Qwen hidden 本身具有全局上下文；hidden 反事实已经排除多种身份/答案捷径，但不能替代自主 role discovery。
17. A1.18B 已把 A1.17 的辅助梯度拆开：FINAL-SAUX 为 `2/3`，TSAUX paired/fresh 均 formal/causal `3/3`，说明 answer 标签不是必要条件，稳定性来自每步 global workspace → complete state 的密集信用分配。该结论要求后续架构保留 training-only TSAUX 等价机制、state-only checkpoint selection、部署剥离和 causal Gate，不能恢复推理答案旁路。
18. A1.9/A1.10 cache 生成都依赖当前缺少 `flash-linear-attention`/`causal-conv1d` fast path 的 Qwen fallback。A1.10 三组 full-token cache 净编码 `4,767.62` 秒、占用 `38.36` GB，还不含模型/tokenizer 首次加载；cached reasoner 延迟不能与在线 text-CoT 延迟直接比较。
19. A1.11 Boundary 的 hard re-embedding 是只读诊断，不是被验证的新架构；若未来使用 prototype-anchored/discrete addressing，必须直接切换合同并重新做 fresh formal，不能把诊断偷偷变成兼容补丁。
20. 当前 TSAUX 使用每步完整三寄存器 oracle state。它解决结构化 core 的训练机制，不解决开放任务的 target 来源；但白皮书不要求零 teacher 训练。后续允许 executable/verifier-filtered teacher target，必须报告成本、覆盖和错误率，并禁止部署/推理时 teacher 绕过。
21. 纯匿名 `K-slot` 与全连续寻址都不再被当作必须坚持的核心承诺。当前候选 core 是 `S_t=(A_t,H_t)` relation-addressable hybrid workspace；离散 sidecar 只能承载身份、地址、类型和控制，不能承载答案或完整语义 state。
22. A1.19H 已通过，但只形成 hybrid core 证据。完整 R1 仍必须在 full-text、同基座、计算匹配条件下形成 Pareto，并训练自然语言 audit readout 通过因果忠实性 Gate；不得用 symbolic trajectory 指标代替 A3/A4。
23. A1.19H-H2 的旧训练路径存在严重 CPU/GPU 同步浪费。后续 A1.20B–A1.22A 的 cached training 必须在正式启动前做 tensor cache、热路径同步、固定 shape 和吞吐基准审计；GPU 功率不是单独 Gate，step/s、端到端延迟和数值等价才是主指标。
24. A1.20B 已按上述要求在正式训练前完成 Qwen cache batch benchmark、mmap cached training 与同步/prefetch 对照。低功率的剩余部分来自约 `2.77M` 参数 Boundary、batch 16 和周期性全 validation，而不是逐样本 Qwen 编码。更关键的失败是 threshold/argmax 控制切断 state CE：局部 mapping CE 在 train batch 可接近零，但 heldout 完整程序联合 exact 只有 `0.087891`。下一轮必须先解决执行级离散信用分配和分层 entity/program binding，不能用更高 GPU 利用率替代机制修复。
25. A1.20C 已把 overfit32 训练推进到约 `4.24 step/s`，GPU 在热路径可达到高利用率；性能不再是本轮失败主因。真正残留的是 pointer-validity 的 hard mask 断点和实际负梯度冲突：state-vs-local 全局 cosine `-0.7366`、presence heads `-0.9944`。下一修复必须改变训练接口和优化顺序，不能只延长固定 `3e-4` 训练。
