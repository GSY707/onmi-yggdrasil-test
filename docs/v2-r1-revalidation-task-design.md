# V2-R1R 跨任务重验证：高层冻结设计与历史 P0-D 合同

原始设计日期：2026-08-02；当前状态更新：2026-08-12

文档地位：第 1–3、7–15 节仍是 R1 高层任务、模型、公平基线和停止规则的设计真源；第 4–6、16–18 节均只保留历史 P0-D 设计与被拒绝方案；第 19–23 节记录测量链、v12–v17 production-data 与 P0-M v1–v5 边界；第 24–30 节记录 P1 v1–v6R，第 31–35 节记录 v7/v8/v8R/v8D/v8L 正式失败。v17 P0-D accepted，P0-M v5 accepted training-path smoke；v6/v6R 已资格化 causal-temporal witness，v7–v8L 均未关闭 CPS causal answer，当前匿名 K=8 + lexical-anchor 主线已经停线。所有已消耗 fixed roots 禁止覆盖或重跑，完整 P1 与 P2 仍未完成。

证据状态：P0-D v1–v16 保持各自历史 accepted/rejected 身份；v17 以独立 repair qualification 和 fresh seed 正式关闭 production data + verifier。P0-M v1–v4 保留为训练工程失败诊断；v5 的 M01–M08 全 true，只接受 cache/模型/baseline/吞吐通路可训练。P1 v3 的 T01–T06 与 R01–R09 已关闭 v2 telemetry/recovery 障碍，但 K=8 validation、OOD、causal 与 middle intervention 全部失败；它正式拒绝当前训练出的候选和端到端学习路径，仍不证明抽象架构、K 容量或 Pareto。A1.20D 只保留为历史机制正控制，不作为本轮初始化、数据源或实现基座。

## 1. 核心判断

本轮不再回答“COPY/SWAP 切面能否继续优化”，而只回答一个跨任务问题：

> 同一个无任务分支的连续 recurrent latent core，能否从完整自然语言输入中学习两种不同计算过程，并相对同基座 text-CoT 在质量—成本上形成稳定优势？

两种计算过程分别是：

- `ERE`：执行一条给定事件序列，持续更新状态后回答查询；
- `CPS`：从同一初始状态并行评估多条候选计划，检查前置条件、延迟后果和资源，再选择唯一最低成本解。

这两个任务共享底层“文本定义临时语义、状态随步骤变化”的能力，但一个要求单轨状态执行，另一个要求多候选模拟与比较。若只有 ERE 通过，只能证明状态执行器；若 ERE 与 CPS 都通过但没有成本 Pareto，只能证明 latent core 可用，不能证明它是更好的推理介质。

本轮只验证白皮书 H1/H2：成熟文本语义能否进入连续 latent workspace，以及同一 recurrent transition 能否跨任务工作。不同时验证 Boundary-MoE、FFN-MoE、Attention Pump、自然语言 audit、主动 READ、工作树、记忆树或 V2-C。

## 2. 权限边界与信息通路

代码可以直接维护下列不变量，因为它们不是需要模型学习的语义：

- padding、attention mask 和最大输入长度；
- 一个样本允许输出哪些局部标签；
- 从公开序列长度计算出的 recurrent 计算预算；
- 数据版本、seed、来源、fingerprint、训练阶段和成本账本；
- simulator、teacher trace、训练期真假命题和评测 verifier。

模型必须学习：

- episode 内 nonce 规则或 action 的含义；
- 实体、条件、关系、资源和候选计划之间的语义；
- 随步骤发生的状态更新；
- 哪个局部标签对应最终答案。

正式模型输入只有四项：`source_hidden`、`source_attention_mask`、`reasoning_budget`、`valid_choice_mask`。答案、AST、teacher state、规则索引、候选合法性、span/role mask、entity handle、正确计划索引和 simulator 输出都不能进入 forward。

## 3. 统一样本与输出合同

### 3.1 表面输入

ERE 与 CPS 都渲染为自然语言文档，顺序统一为：

1. 一句任务说明；
2. episode 内临时词汇或 action/rule 定义；
3. 初始世界；
4. 一个或多个有序序列；
5. 查询或目标与预算；
6. 局部答案标签说明。

标题和标点属于普通文本，模型不接收由代码生成的 section mask。训练模板与 language OOD 模板完全分离；定义、事实、候选和标签顺序都要随机化。

答案头固定输出 `9` 个局部类别，对应 `A` 到 `I`。每个 episode 重新随机分配这些字母的含义。代码只用 `valid_choice_mask[9]` 屏蔽该样本不存在的标签，不提供正确标签位置。

### 3.2 数据记录

每条 JSONL 记录必须至少包含：

```text
schema_version
generator_version
example_id
family                 # ere / cps
split
source_text
reasoning_budget
valid_choice_mask[9]
answer_index
semantic_fingerprint
surface_fingerprint
pair_id / pair_role
program_ast             # 仅 simulator/audit 可见
teacher_trace           # 仅训练辅助和 text-CoT 可见
causal_certificate      # 仅 audit 可见
```

数据加载必须显式实现三种 view：

- `model_view`：只暴露表面文本派生的 frozen-Qwen hidden、mask、预算和合法输出 mask；
- `training_view`：在 model view 外增加 answer 与训练期 claim；
- `audit_view`：可以读取 AST、simulator state、因果证书和 fingerprint。

测试必须证明 model view 不含任何 teacher、AST、answer、span、role、entity、candidate validity 或 simulator 字段。

### 3.3 公开计算预算

两个任务使用同一公式：

```text
L = 输入中任一有序调用序列的最大长度
T = min(24, L + 2)
```

ERE 的有序序列是事件；CPS 的有序序列是各候选计划。`T` 只由表面可见结构决定，不根据正确规则、正确候选或答案调整。训练分布通常为 `T=6–12`，长程 OOD 可以到 `T=22`。core 不使用长度为 `24` 的 learned step table，而使用由 `(t, T)` 计算的固定 Fourier 特征，因此可以外推到未见步数。

## 4. ERE 数据生成算法

### 4.1 世界状态

每个 episode 采样：

- train/validation 为 `3–5` 个实体，entity OOD 为 `6` 或 `8` 个；
- 两个 categorical attributes，每个 attribute 有四个 episode-local nonce value；
- 一个有向 relation；
- `2–5` 条 episode-local nonce rule；
- 一条事件序列和一个最终 attribute/relation query。

实体、attribute、value、relation 和 rule 名称都从 pronounceable nonce 生成器采样，并在 episode 内唯一。名称不携带稳定 operator 含义。

### 4.2 允许的规则原语

simulator 只实现以下通用 AST 原语；它们用于生成真值，不进入模型：

| 原语 | 语义 |
| --- | --- |
| `SET` | 把目标实体的一个 attribute 设为 literal value |
| `COPY` | 把 source 当前 attribute 复制给 target |
| `SWAP` | 同时交换两个实体某 attribute 的旧值 |
| `IF` | 根据当前 predicate 选择两个 effect 之一 |
| `LINK` / `UNLINK` | 添加或删除一条有向 relation |
| `FOREACH_LINKED` | 对当前 relation 邻居集合应用同一个 effect |

一条 nonce rule 由一或两个原语组成，并用自然语言定义。事件只调用 nonce rule 与它的参数，不能直接出现 `SET/COPY/...` 名称。

### 4.3 构造而非盲目拒绝采样

生成器先构造一条影响查询的 causal spine，再插入干扰项：

1. 选择 query entity、attribute/relation 和初始值；
2. 选择长度 `d` 的必要更新链，使信息至少跨两个不同实体或一个条件与一个后续 effect；
3. 将 spine 的每一步实例化为 nonce rule 调用；
4. 插入不会改变最终 query、但表面相似的事件和未被调用规则；
5. 随机化定义、事实和事件的表述顺序；
6. simulator 执行并生成最终答案、逐步 delta 和 dependency graph；
7. 删除或翻转 spine 上一个指定事件，验证答案改变，形成 causal certificate。

train/validation 至少 `90%` 样本的最短 query dependency depth 为 `3`；length OOD 至少为 `8`。任何没有可验证必要事件的样本都不得靠随机补齐进入正式数据。

### 4.4 组合与语言拆分

train 可以出现所有单原语，以及 `IF→SET`、`IF→COPY`、`LINK→COPY` 等基础组合。composition OOD 保留以下组合拓扑，只允许其中的单项原语在 train 出现：

- 条件决定 relation mutation，后续 `FOREACH_LINKED` 读取变化后的邻居集合；
- 早期 `SWAP` 改变后续条件真假，再触发 `COPY/SET`；
- 同一 query 同时依赖一次 relation 更新和一次 attribute 传播。

language OOD 使用完全独立的模板族，包括被动语态、条件后置、定义顺序倒置和同义表达；不能只替换少量词。

### 4.5 ERE teacher 与反事实

teacher trace 每步只描述：调用了哪个表面 rule、条件判断、发生的 state delta，以及最终如何读取 query。它是一个可验证解释，不是唯一正确 latent 轨迹。

每个 causal pair 共享实体/rule 名称、模板、标签映射和绝大多数文本，只翻转一个必要条件、事件参数或 relation fact；pair 的正确答案必须改变。另生成少量“轨迹不同但最终答案相同”的诊断 pair，防止把 latent 差异简单等同于标签差异，但它们不计入主 flip Gate。

## 5. CPS 数据生成算法

### 5.1 世界与 action

每个 episode 包含：

- `4–10` 个 boolean/categorical facts；
- `1–3` 个取值为 `0–6` 的 integer resources；
- train/validation 为 `4–6` 个 nonce actions，OOD 为 `6–10` 个；
- 每个 action 的 conjunction precondition、state/resource effect 和正整数 cost；
- 一个 goal conjunction、一个总 cost budget，以及候选计划。

action 名称是 episode-local nonce；precondition、effect 和 cost 都在输入中自然语言定义。simulator 顺序执行 candidate：任一步 precondition 不满足即整条计划非法；合法计划还必须达到 goal、满足最终约束和 budget。

### 5.2 候选的构造

生成器先构造一条合法目标计划 `P*`，再从它产生 hard negatives，而不是随机拼 action：

1. 从目标反向建立长度 `2–5` 的 prerequisite/effect 链；horizon OOD 为 `6–10`；
2. 给 `P*` 分配唯一最低成本；
3. 通过局部变换产生候选：删除 prerequisite、交换依赖顺序、替换为资源不足 action、加入会破坏最终约束的 action、加入冗余高成本 action、使用只差一个否定 precondition 的 action；
4. 至少保留一个“合法且达到目标但成本更高”的候选，以及一个“表面接近但非法”的候选；
5. 以 `1/6` 的目标比例生成 `NONE`：所有候选都失败，同时保留至少两种不同失败原因；
6. 随机化候选顺序，再随机把候选和 NONE 映射到局部字母标签。

train/validation 固定为 `5` 条候选加 `NONE`，distractor/horizon OOD 可以扩到 `8` 条候选加 `NONE`。`valid_choice_mask` 只反映标签数量。

### 5.3 组合拆分

train 包含单一 unlock、单资源消耗、单 relation/fact precondition 和直接成本比较。composition OOD 保留下列组合：

- unlock 后才能补充资源，再执行目标 action；
- 早期低成本 action 破坏后续必要 fact，而高成本前缀保留可行性；
- 两条都达到局部目标，但只有一条满足最终否定约束；
- 资源、action cost 与总 budget 共同决定唯一最优，而非只看 plan 长度。

language OOD 与 ERE 一样使用独立模板族。distractor OOD 增加未被任何合格计划使用的 actions、无关 facts 和近似候选，但不能改变主能力定义。

### 5.4 CPS teacher 与反事实

teacher trace 对每条 candidate 给出逐步 legality、必要 state/resource delta、累计 cost、goal/constraint 结果和最终比较。文本 CoT 可按候选顺序输出；latent 辅助监督按统一 prefix depth 并行提出真假命题。

causal pair 只改变一个 resource、precondition literal、effect 或 cost，并要求最优标签或 NONE 状态翻转。pair 共享标签映射和候选顺序，避免把表面变化或重新排列误当因果能力。

## 6. Split、fingerprint 与 P0 数据 Gate

### 6.1 规模

P0 使用独立 generator seed：每族 train `4,096`、validation `512`，每个正式 OOD split `512`。P0 通过后，P1 用新的 generator seed 重新生成每族 train `8,192`、validation `1,024`、每个 split `1,024`；P1 不是 P0 train 的简单扩容。

ERE splits：`validation`、`composition_ood`、`length_ood`、`entity_ood`、`language_ood`、`causal_pairs`。

CPS splits：`validation`、`composition_ood`、`horizon_ood`、`distractor_ood`、`language_ood`、`causal_pairs`。

所有输入 tokenized 后必须不超过 `1,024` token；生成器应使用紧凑模板并拒绝重生超长记录，任何缓存和 baseline 都不得静默截断。

### 6.2 fingerprint

`semantic_fingerprint` 对 nonce 名称做 alpha-renaming，移除 paraphrase 与答案字母映射，并将无语义顺序 canonicalize；它保留初始状态、规则/action AST、事件或候选集合、query/goal 和预算。CPS 候选集合在 fingerprint 中按 canonical AST 排序，因此只重排候选不能躲过 overlap audit。

`surface_fingerprint` 对最终 source text 逐字节计算 SHA-256。train 与所有 heldout 的 semantic overlap 和 surface overlap都必须为 `0`；不同 heldout split 之间也必须为 `0`，causal pair 内部除外且必须通过 pair id 显式声明。

### 6.3 P0 硬 Gate

P0 只判断数据、simulator、teacher 和训练接口是否有效，不把某种模型是否偏好 CoT 错写成数据属性。旧合同中的“text-CoT >=0.75 且比 direct 高 0.15”从 P0 删除，移到 P1 作为实测比较；direct 模型可能在 hidden 中完成推理，CoT 差值不能证明数据是否递归。

P0 必须同时满足：

1. 所有 split 的 simulator 重放、answer verifier 和 teacher-trace verifier 为 `100%`；
2. 未声明 pair 的跨 split semantic/surface overlap 为 `0`；
3. train 标签频率相对均匀分布最大偏差不超过 `0.03`，512 规模 split 不超过 `0.07`；
4. 至少 `95%` 样本拥有经 simulator 验证的必要事件、条件、资源或 action；
5. causal pair 的指定单点修改率和答案翻转率都为 `100%`；
6. ERE dependency depth、CPS hard-negative/valid-suboptimal coverage 满足前述构造约束；
7. majority、label position、长度 bucket、last mention、question-only unigram、full-text unigram Naive Bayes 的准确率都不高于该 split 平均随机正确率 `+0.10`；
8. 训练 claim 的 positive/negative 数量平衡，claim-only unigram/length heuristic 不高于 `0.60`；
9. tokenizer 无静默截断，所有 record/schema/hash/seed/version 完整；
10. `model_view` 禁止字段审计完全通过。

P0 后追加三个非正式 trainability smoke：单任务 K=8 overfit64、ERE+CPS 联合 overfit128、direct/text-CoT overfit64。它们只验证实现和 loss 通路，不替代 P1；K=8 单任务 answer 至少 `0.95`、joint 每任务至少 `0.90`、claim 至少 `0.90`，否则停在训练实现分析。

## 7. 冻结模型：R1R-Latent v1

### 7.1 成熟文本基座

固定使用：

```text
model_id = Qwen/Qwen3.5-2B
revision = 15852e8c16360a2fea060d615a32b45270f8a8fc
text hidden width = 2048
hidden layer = final language-model hidden
base trainable parameters = 0 for latent paths
```

输入只编码一次，训练时缓存完整 token hidden 为 float16；在线 P2 必须重新计入 tokenizer 与 Qwen encode，不能用 cache latency 代替。

### 7.2 Boundary

Boundary 在 ERE/CPS、K=1/K=8 之间结构相同，只允许 K 配置改变：

```text
X [B,S,2048]
-> non-affine LayerNorm(2048)
-> Linear(2048,512)
-> RMSNorm(512)
= E [B,S,512]
```

Boundary 是逐 token 投影，不做 learned section decode、entity extraction、candidate pooling 或 token 间 self-attention。这样它不能成为隐藏的任务执行器；跨 token 组合必须由成熟基座已有语义与 recurrent core 完成。Boundary 参数接受主任务与辅助 loss 梯度，但无任务分支。

### 7.3 recurrent core

固定配置：

```text
D_latent = 512
K = 1 or 8
recurrent block depth = 2
attention heads = 8
FFN hidden = 2048, SwiGLU
dropout = 0
T = public reasoning budget, maximum 24
```

`H_0` 由同一个 learned generic slot seed 加固定 slot Fourier position 得到。slot 不带 entity、candidate、source/target 或任务角色。

每个 recurrent layer 在每一步执行：

```text
U = H + gated_cross_attention(RMSNorm(H), E)
V = U + gated_self_attention(RMSNorm(U))
H' = V + gated_SwiGLU(RMSNorm(V))
```

两层在一个 step 内参数不同，但整组两层在所有 recurrent step、两个任务和所有序列长度上共享。每层 source K/V 在一次 forward 开始时预计算一次，并在 T 步复用；attention 使用 PyTorch SDPA，不能在 Python 中逐样本计算。

固定 `(t,T)` Fourier control feature 经一个共享 linear 写入所有 slot。不得使用 task embedding、operation embedding、candidate slot assignment、task-specific transition 或长度 `24` 的 learned lookup table。

### 7.4 latent-only readout

最终 readout 只接收 `H_T`：

```text
pooled = learned_query_attention(H_T)
logits = Linear(RMSNorm(pooled), 9)
logits = logits.masked_fill(~valid_choice_mask, -inf)
```

readout 不能持有 `E`、Qwen hidden、input ids、teacher claim 或 task id 的引用。代码测试必须用函数签名、forward hook 和梯度图共同证明没有 source-to-answer bypass。

## 8. 训练期通用监督

### 8.1 为什么不用显式三寄存器 scaffold

ERE 与 CPS 没有共同的固定寄存器布局；若把 simulator state、正确 entity slot 或 candidate validity 直接注入 core，就会重新得到任务专用执行器。另一方面，只用 final answer CE 又会重现长程信用不足。

本轮采用训练期 `claim verifier`：simulator 针对某个 prefix 生成关于当前状态的真假自然语言命题，辅助头从 `H_t` 判断命题真假。它提供逐步全局信用，但不规定 slot 布局或唯一 latent 轨迹。

### 8.2 claim 构造

ERE claim 示例类型：某实体当前 attribute value、某 relation 是否存在、某条件在当前 prefix 是否成立。

CPS claim 示例类型：某 candidate 在 prefix 后是否合法、当前 resource/cost、某 fact 是否成立、最终是否达到 goal/constraint。

每个 prefix 生成表面匹配的 positive/negative pair；negative 只替换一个 value、关系方向、candidate label、resource 或真假词。每个训练 step 每样本采样两对 claim，positive/negative 平衡。claim 本身不能进入 core 或答案头。

claim 由同一个 frozen Qwen 编码并缓存；其 token hidden 经过与 source 共享的 Boundary 投影后做 masked mean，得到 `q_claim[512]`。训练期共享 probe 用 `q_claim` 对 `H_t` 做一次 attention，再输出二分类。probe 参数在两个任务共享。

对齐关系固定为：`H_1` 对应初始状态，`H_(j+1)` 对应执行所有序列的第 `j` 个 prefix 后状态，`H_T` 用于最终比较/回答。CPS 对所有 candidates 使用同一 prefix depth并行生成 claim，不分配 oracle candidate slot。

### 8.3 loss 与退火

总训练上限为 `2,400` optimizer updates，联合 batch 中 ERE/CPS 样本权重各 `0.5`：

| updates | loss |
| ---: | --- |
| `1–400` | `0.25 * L_answer + 1.0 * L_claim` |
| `401–1800` | `1.0 * L_answer + 0.5 * L_claim` |
| `1801–2400` | `1.0 * L_answer` |

正式 checkpoint 必须在 answer-only 阶段产生。保存时物理删除 claim encoder/probe 参数；删除后的 state dict 重新加载并独立评测，不能只把 loss weight 设为零。

optimizer 固定为 AdamW，Boundary LR `1e-4`，core LR `2e-4`，readout/claim probe LR `3e-4`，weight decay `0.01`，warmup `5%`，cosine decay，global grad clip `1.0`。默认 bf16；若实测数值不稳定才允许 fp32 core 对照，并记录为合同偏离候选而非静默修改。

每 `100` update 在两个 validation 上评测；在完成 `1,200` update 前不得早停。选择 checkpoint 时用两个任务 validation accuracy 的较小值，不能用平均值掩盖单任务失败。

### 8.4 梯度冲突预注册

前 `200` update 每 `20` 步记录 ERE loss 与 CPS loss 在共享 Boundary/core 上的 gradient cosine。只有当连续十次的 median `< -0.15` 且两个任务 validation 差距 `>0.05`，才允许启用 core-only PCGrad；这是预注册修复分支。未触发不得为了提高某个 split 随意加 task weight。触发、未触发和修复前后都必须写 artifact。

## 9. 公平文本基线

四条主路径为独立 checkpoint：

1. `direct SFT`：输入后只生成 `Answer: <local label>`；
2. `text-CoT SFT`：生成 simulator teacher trace，再生成同一个 local label；
3. `K=1 latent`；
4. `K=8 latent`。

direct/text-CoT 固定使用同一个 Qwen revision，并在顶部四层 attention 与 MLP projection 上使用 LoRA `rank=8, alpha=16, dropout=0`；其余参数冻结。两个文本基线使用相同训练 episode、shuffle seed、最大 optimizer update 和 LoRA 结构。direct 只接受 answer supervision；text-CoT 与 latent claim 都来自同一 simulator/teacher source，teacher 生成成本记入账本。

文本基线使用官方 chat template并关闭 thinking；input 不超过 `1,024` token，direct target 不超过 `16` token，CoT target 不超过 `512` token，均禁止静默截断。训练目标只计算 assistant output token CE。

公平性同时报告：

- equal-example：四条路径看到相同 episode 数；
- equal-GPU-hour：在最短完成路径的 GPU 时间处截取其他路径的最近 checkpoint；
- trainable parameter、teacher token、训练 FLOPs proxy 和 peak VRAM；
- 在线 tokenizer、Qwen encode、autoregressive decode、latent transition、KV/activation、median/p95 latency。

不得再用 prompt-only direct/text-CoT 对比监督训练 latent。

## 10. 执行阶段与 Gate

### 10.1 P0-D：数据与 verifier

按第 6 节生成完整 P0，运行全部硬 Gate。任一 Gate 失败时只允许修复 generator、simulator、render、schema 或 audit，再以新 generator version 全量重生；不能跳到模型训练。

### 10.2 P0-M：训练通路 smoke

P0-D 通过后依次运行：

1. ERE K=8 overfit64；
2. CPS K=8 overfit64；
3. ERE+CPS shared K=8 overfit128；
4. direct/text-CoT 各自 overfit64；
5. 100-step throughput benchmark。

任何 smoke 失败先定位 loss、mask、缓存、数值或吞吐，不得扩大数据掩盖实现问题。P0-M 通过不代表架构通过。

### 10.3 P1：单 seed 跨任务 falsification

P1 使用新的 8192/1024 数据和一组 fresh Boundary/core seed。K=8 必须同时满足：

- ERE、CPS validation 均 `>=0.85`；
- 每个主 OOD split `>=0.75`；
- causal pair answer-flip accuracy `>=0.80`；
- middle-step zero/batch-shuffle latent intervention 使主指标下降 `>=0.40`；
- `T=1` 相对完整 T 在 length/horizon 主 split 下降 `>=0.15`；
- claim probe 物理删除后 accuracy 变化绝对值 `<=0.01`；
- source bypass、answer leakage、task-specific parameter audit 全部通过。

K=1 跑完全相同数据与训练合同，用于容量曲线，不预设它必须失败。direct/text-CoT 的 validation/OOD 和 CoT 相对 direct 差值在这里报告，不再作为 P0 数据 Gate。

### 10.4 P2：matched Pareto

只有 P1 通过才在线评测，每任务至少 `128` 条、每条路径相同输入、同一 GPU、随机交错运行顺序。K=8 必须在 ERE 和 CPS 分别满足以下之一：

- 相对 text-CoT 质量不低于 `-0.02`，同时 end-to-end latency 或 FLOPs/KV 主成本降低至少 `25%`，另一主要成本轴不恶化超过 `10%`；
- 主质量提高至少 `0.05`，同时总在线成本不超过 text-CoT 的 `1.10x`。

若 K=1 与 K=8 在 paired bootstrap 区间内等质且更便宜，路线直接切换到 K=1，删除多 slot 复杂度。

### 10.5 P3：fresh seeds

只有 P2 通过才运行三个独立 generator、Boundary/core initialization 和 data-order seed。每个 seed 都必须重新过 data audit、P1 行为/因果 Gate 和 P2 Pareto。主报告使用最差任务、最差 split、最差 seed和 paired bootstrap 区间，不追求 `1.0`。

## 11. 因果与架构完整性审计

正式评测至少包含：

- `zero_latent@middle`：中间状态清零；
- `batch_shuffle_latent@middle`：只在 choice-mask 相同的样本间置换；
- `slot_permutation`：K=8 slot 次序置换，判断 readout 是否依赖偶然固定 slot；
- `truncate_T=1/2/4/full`：得到 recurrence 深度曲线；
- `wrong_definition`：用 causal pair 的 rule/action 定义替换原定义；
- `source_token_shuffle`：只作破坏性诊断，不作为语义 Gate；
- `aux_stripped_reload`：物理删除训练头后重新加载；
- `head_signature` 与 forward-hook audit：答案头从未读取 source/teacher；
- state-dict 名称 audit：不存在 `ere_*`、`cps_*` 或 operator/candidate 专用参数。

不要求 latent 与 teacher state exact 对齐。claim accuracy、trajectory probe 和完整 simulator state decoder 都是诊断，正式成功依赖行为、因果和成本。

## 12. GPU 与缓存工程合同

所有命令必须使用项目 `.venv\Scripts\python.exe`；系统 `python` 当前是 CPU-only，不得误用。正式训练前必须：

1. 确认 GPU 可用和空闲显存；已有无关进程占用较高时等待或明确报告，不抢占用户进程；
2. 对 Qwen cache batch `4/8/12` 做实测，选择不 OOM 的最高吞吐配置；
3. hidden cache 按长度排序分 shard，float16、mmap 读取，不保存 answer/oracle 字段；
4. DataLoader 使用 pinned memory、persistent workers 和 non-blocking copy；
5. source K/V 每个 layer 每个 forward 只投影一次；
6. 不在训练热路径做 `.item()`、`.cpu()`、逐样本 simulator、逐样本 Qwen 或频繁 `cuda.synchronize()`；
7. 先记录 100 step 的 samples/s、source tokens/s、step median/p95、GPU utilization/power、VRAM、数据等待比例和数值一致性。

GPU 功率是故障信号，不是单独 Gate。若功率持续低于 `20W`，必须先区分 GPU 等待数据、模型过小、频繁 validation、同步或其他进程争用；以 step/s 和 profiler 证据决定修复。目标是在 latent 热路径达到稳定 GPU 工作区间，不能为了功率盲目扩大无意义计算。

## 13. 独立实现布局与 CLI

新实现必须位于独立 package，不向旧 `reasoning_medium` 添加兼容分支：

```text
src/yggdrasil_v2/r1_revalidation/
  schema.py
  symbols.py
  simulator.py
  ere.py
  cps.py
  render.py
  audit.py
  cache.py
  model.py
  claims.py
  train_latent.py
  train_text.py
  evaluate.py
  cost.py
  assess.py
experiments/v2_r1_revalidation.py
tests/test_v2_r1r_{data,audit,model,training,cost}.py
artifacts/v2-r1r/
```

CLI 至少暴露：

```text
generate-p0
audit-p0
cache-smoke
train-smoke
benchmark-train
generate-p1
cache-p1
train-direct
train-text-cot
train-latent --slots 1|8
evaluate-p1
intervene
benchmark-online
assess
```

每条命令接受显式 config、seed、input/output 路径并拒绝覆盖已有正式 artifact。artifact 保存完整 config、git commit、dirty status、环境、模型 revision、输入 SHA-256、开始/结束时间、GPU 时间和 Gate 状态。

## 14. 停机解释

- P0-D 失败：任务或数据合同无效；不能训练 latent。
- P0-M 失败：实现、优化或监督通路有问题；不能归因架构。
- ERE 通过、CPS 失败：当前模型是状态执行器，不是跨任务 reasoner；停止 P2。
- 两任务 P1 通过但没有 Pareto：latent 可工作但没有介质优势；停止 V2 多模态主线。
- 必须增加 task-specific parser、transition、slot 或阈值才通过：R1R 失败。
- K=1 支配 K=8：采用 K=1，并删除多 slot 路线。
- P2 通过而三 seed 失败：候选不稳定，不进入自然语言 audit。
- P3 通过：只关闭 R1；随后另立 A1.22A audit，不能直接宣称完整 V2 成立。

## 15. 执行 agent 不得自行改变的项目

未经主设计层重新冻结，执行 agent不得：

- 改任务原语、split、规模、阈值、Qwen revision、D/K/T、loss schedule 或 baseline LoRA 结构；
- 给 ERE/CPS 增加不同模型模块、task embedding 或不同 recurrent budget 公式；
- 把 AST、span、role、candidate validity、simulator state 或答案放进 forward；
- 用 prompt-only baseline、cached online latency、小于规定样本数或 best-seed 结果宣布通过；
- 在 Gate 失败后继续后续阶段；
- 启动 OPS、A1.22A、V2-B 或 V2-C。

若规格存在无法实现、显存不够或测试互相矛盾，执行 agent 必须停下并提交具体证据，由主设计层判断；不能自行“合理化”修改。

## 16. 历史：P0-D v2 主设计层修订（已否决）

本节只保存 v1→v2 的诊断与当时合同，不再具有执行权限。v2/v3 均已失败；所有当前 P0-D 实现、阈值、路径和阶段权限以第 17 节 v4 为准，不得从本节恢复旧 CLI 或兼容层。

### 16.1 为什么 v1 的机器通过无效

generator v1 满足了它自己实现的十项布尔检查，却没有满足这些检查背后的任务合同。主设计层用只读 surface heuristic 对完整 14,336 条记录复核后得到：

- ERE 只需读取第一个 `SET` 定义中的 literal，再在标签说明中查表，即可在 train、全部 OOD 与 causal pairs 上达到 `1.0`；无需执行任何事件。
- CPS 在所有非 causal split 上，只需判断 Candidate 1 的首 action 是否也是第一条 action 定义：若是则返回 Candidate 1 的局部标签，否则返回 NONE 标签，即可达到 `1.0`。更弱的“总选 Candidate 1”在常规 split 约为 `0.833`，在 horizon OOD 为 `1.0`。
- CPS train 与 composition OOD 调用了同一个基本拓扑；后者没有保留组合。ERE language OOD 同时换了组合模式，因而把语言变化与计算变化混在一起。
- language OOD 只改变开场句和少量引导词，没有实现被动语态、条件后置、定义子句倒置等独立语法族。
- CPS 的 `candidate is legal at prefix` claim 没有从 prefix state 求真；不同 split 的正 claim 真值准确率只有约 `0.38–0.57`。成对翻转 label 只能制造数量平衡，不能制造正确监督。
- causal certificate 保存的 `changed_path` 没有与 AST 的真实唯一 diff path 比较；ERE 路径甚至指向不存在的 `events/.../primitives` 结构。
- regex `contract_token_count` 不是 Qwen tokenizer 的保守上界：14,336 条记录的真实 Qwen token 数都高于该计数。当前数据实测最大值为 `745`，所以样本本身未超长，但 v1 Gate 9 的证明方法无效。

因此 `artifacts/v2-r1r/p0-v1/p0-assessment.json::passed=true` 只表示旧审计器自洽，正式路线判定仍为 P0-D 失败。旧目录必须保留为 `rejected diagnostic`，不得覆盖、删除或用于 P0-M。

### 16.2 版本、产物与执行边界

修订实现固定使用 `generator_version=r1r-p0-generator-v2`，formal 输出固定为 `artifacts/v2-r1r/p0-v2/`，smoke 输出使用独立目录。任何 v2 记录或 manifest 都不得引用 v1 JSONL。设计文档 hash 取包含本节的当前文件 SHA-256。

本轮仍然只允许修改 `schema/symbols/simulator/ere/cps/render/audit`、P0 CLI、P0 测试和 P0 文档；不得创建 model、cache、train、Boundary、core 或 baseline 实现，不得启动 GPU。任一 v2 Gate 失败，停在 P0-D。

### 16.3 表面合同与随机化

source text 必须完整、无歧义地声明 episode 语义。ERE 要显式列出两个 attribute 名称与各自 value domain，初始状态使用 `entity: attribute=value`；不得再依赖“第一个值就是第一个未命名属性”的隐式约定。空 goal/final restriction 必须渲染为 `none`，不能产生残缺句。

生成 AST 后再使用独立的 surface RNG 随机化下列无语义顺序：entity/value 声明顺序、rule/action 定义顺序、初始 fact/relation/resource 的显示顺序、candidate 顺序、标签 legend 顺序和 available-choice 顺序。事件内部的时间顺序与 candidate 内的 action 顺序是语义，不能打乱。candidate 重排后必须重建 `candidate_i`、teacher、claim、label mapping 与 causal certificate 的索引。定义位置、candidate 位置、正确 candidate 位置和 NONE 必须有显式频率审计；不能靠答案字母均衡替代 candidate 位置均衡。

### 16.4 ERE v2 构造

ERE 仍使用第 4 节的通用原语，但必须先构造真实 dependency graph，再渲染。train/validation 至少均衡覆盖 `copy_chain`、`conditional_branch`、`swap_then_condition` 和 `relation_foreach` 四种基础家族；composition OOD 只使用第 4.4 节保留组合，language OOD 的 AST 家族与 validation 匹配，不得混入额外组合偏移。

每条记录至少声明三步 causal spine；train/validation 至少 `90%`、length OOD `100%` 的记录必须满足：对 spine 中每个声明为必要的事件分别做一次 no-op/delete intervention，最终答案都改变。audit 必须自己运行这些 intervention，不能信任 `dependency.depth` 字段。若某个事件删除会使调用非法，应使用同参数的显式 no-op 规则替换，并在 certificate 中记录方法。

答案来源必须在下列 provenance 家族间平衡，任一非 causal split 的最大占比不得超过 `0.35`：初始状态经 COPY 传播、条件真分支、条件假分支、SWAP 后来源、relation 邻居选择。关键 literal、条件真假、query entity、query attribute、关键 rule 的定义位置和关键 event 的位置都由独立随机变量决定；不能再让“第一条 SET 的 literal”恒等于答案。

至少实现并正式审计以下 surface-only ERE heuristic：initial-query value、first/last SET literal、first/last rule literal、first/last event 的静态输出、last-mentioned value、忽略事件的答案、忽略 IF 当前真值而固定选 then/else。每个 split 的任何 heuristic 都不得高于该 split 平均随机正确率 `+0.10`。这些 heuristic 必须真的从 `source_text` 解析其线索，不能读取 `answer_index`、AST 或 audit 字段。

### 16.5 CPS v2 构造

CPS 必须把训练基础能力与 composition OOD 分开。train/validation 可单独或两两出现 unlock、resource replenish/consume、fact/relation precondition、final constraint、cost/budget；不得出现第 5.3 节四种完整保留组合。composition OOD 才组合 unlock→补充资源→目标、早期动作破坏后续必要 fact、局部 goal 相同但 final constraint 不同、resource/action cost/budget 联合裁决。每条记录保存由 audit 根据 AST 重新计算的 `composition_signature`；formal Gate 要求 train/validation 与 composition OOD 的保留 signature 交集为零，同时确认组成原语在 train 中各自出现过。

`P*` 和 hard negatives 构造完成后必须随机重排 candidate；正确 candidate 的表面位置在非 NONE 样本中近似均匀，512 split 最大相对偏差不超过 `0.10`、train 不超过 `0.05`。NONE 比例保持 `1/6 ± 0.02`。action 定义也独立重排。候选角色不能由 `Candidate 1`、定义位置或 plan 长度固定表达。

至少实现并正式审计以下 surface-only CPS heuristic：总选 first/last candidate、first action 与 first definition 匹配否则 NONE、最短/最长 plan、原始 action cost 求和最低但忽略 legality、只检查 precondition、只检查 goal、只检查 budget、只检查 final constraint。任何 split 的任何 heuristic 都不得超过随机正确率 `+0.10`。同时用 audit AST 计算 candidate-role 与位置的互信息或条件多数准确率；超过同一阈值即失败。

### 16.6 teacher、claim 与 certificate

teacher trace 必须由 simulator fresh replay 得到。claim 必须从 prefix state 生成，再由与生成分离的 predicate evaluator 逐条复算；`claim_truth_accuracy` 必须为 `1.0`。positive/negative pair 只能替换一个 value、方向、数值或真假词，且两条复算真值必须相反。

ERE claim 至少覆盖 attribute value、relation existence 和 condition truth；CPS claim 至少覆盖 prefix action legality、resource/cost、fact truth、goal/final-constraint 状态。claim 中的 candidate 称谓必须和 source 的表面编号或局部 label 一致，不能使用 source 中不存在的零基 `candidate_0`。每族每个正式 split 的 claim kind 最小占比不得低于 `0.10`，正负数保持相等，原有 claim-only shortcut Gate 保留。

causal pair 的 `changed_path` 必须等于 `_deep_diff_paths(base_ast, flip_ast)` 唯一返回的真实路径；audit 还要检查 `specified_change.from/to` 与该路径两端值一致。非 pair causal certificate 也要由 audit 施加其指定修改并复算。不能只检查 pair 两边写了相同的错误证书。

### 16.7 language OOD 的实质性 Gate

language OOD 必须和 validation 使用相同的 AST archetype、难度、长度、答案来源与候选角色分布，只改变 renderer family。OOD renderer 至少实现并标记三类真实句法变化：effect/precondition 或 predicate/consequence 子句倒置、条件后置、主动/被动语态互换；不能只改 opening 或 section 标题。

测试要对同一批 AST 同时渲染 train 与 OOD 文本：semantic fingerprint 必须相同，删除 opening/title 后的正文必须仍不同；每个 OOD template 的句法 feature 标记必须覆盖其声明变化。audit 必须比较 AST/distribution signature，防止把 composition shift 当 language shift。

### 16.8 tokenizer、heuristic 训练与独立性

token Gate 固定加载 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` 的实际 tokenizer，按未来 cache 使用的同一 special-token 设置记录 `qwen_token_count`。生成和 audit 都拒绝任何 `>1024` 的 source；regex 计数只能作为生成前快速预筛，不能写成 formal 证明。manifest 记录 tokenizer class、revision、transformers version 与 special-token config。

统计学习 heuristic 同时报告 split-local grouped cross-validation，以及只在 train 拟合、在 validation/OOD 评估的结果。pair 必须按 pair id 同组。人工结构 heuristic 不需要拟合，但必须覆盖第 16.4/16.5 节。audit 的 formal conjunction 必须使用全部结果，不得只把报告写入 JSON 而不接 Gate。

simulator 与 generator 可以共享通用数据结构，但测试必须包含手写小世界及独立期望值，覆盖每个原语、每类 CPS failure、prefix claim、certificate diff 和语言模板。至少加入六个故障注入测试，证明 audit 会拒绝：固定正确 candidate 位置、ERE seed-literal 泄漏、反转 claim、composition signature 污染、只换 opening 的 language OOD、伪造 token count/changed path。

### 16.9 P0-D v2 正式 conjunction

v2 assessment 只有下列项目全部为真且正式规模完整时才可 `passed=true`：

1. simulator、teacher 与独立手写测试通过；
2. 未声明 semantic/surface overlap 为零；
3. answer、choice、candidate/action/definition position 与 NONE 比例通过；
4. ERE 多事件必要性和两族 causal necessity 通过；
5. causal pair 单点 diff、答案 flip 与真实 certificate path 通过；
6. train/composition/language/length/entity/horizon/distractor 结构合同通过；
7. source text 完整无歧义；
8. language OOD 的正文句法与分布匹配通过；
9. 所有统计与结构化 shortcut heuristic 通过；
10. claim truth、kind coverage、正负平衡与 claim-only heuristic 通过；
11. 实际 Qwen tokenizer、schema、hash、seed、version 与 provenance 通过；
12. `model_view` forbidden-field audit 通过；
13. formal scale 完整。

若任何一项失败，机器 assessment 必须为 false，并在 P0-D 停止。只有主设计层独立复核 `p0-v2` 后，才可能另行授权 P0-M。

## 17. P0-D v4：reference-first 最小合同

### 17.1 判决与直接切换

v3 主设计层验收见 `docs/v2-r1r-p0-v3-main-review.md`。v3 的教训不是“再增加几个 Gate”，而是 Gate 数量过多却没有正向参考，审计器、生成器和 manifest 仍可共同形成自洽盲区。v4 因此缩减合同面：只保留八个能被独立正控、负控和完整 assessment 共同验证的 Gate；每个任务约束归入一个明确所有者，不再把同一失败拆成多个互相依赖的布尔值。

本节直接覆盖第 4–6 节、第 16 节以及 D2 source snapshot 中旧 v3 第 17 节的所有 P0-D 实现细节。第 7–12 节冻结的共享模型、公平基线、因果评估与 Pareto 合同不变。历史 v1/v2/v3 代码只存在于 Git diff 或 artifact snapshot；当前实现不得兼容读取旧 schema，也不得保留旧 CLI alias。

固定标识为：

- `contract_version=r1r-p0-contract-v4`；
- `generator_version=r1r-p0-generator-v4`，但 R0 阶段不得实现或调用 production generator；
- record、manifest、audit、assessment 与 seal schema 统一升级为 `.v4`；
- R0/R1/R2 artifact 分别使用 `artifacts/v2-r1r/p0d-v4-r0-*`、`p0d-v4-r1-*`、`p0d-v4-r2-*`；
- formal 唯一路径为 `artifacts/v2-r1r/p0-v4/`，没有主设计层授权时 CLI 必须在写入前拒绝；
- model、cache、Boundary、core、baseline、GPU 和训练仍全部禁止。

当前首次执行权限只到 R0 audit-reference。R0 即使通过也必须停止；执行 agent 无权实现 production ERE/CPS generator、创建下一阶段授权或运行 R1。

### 17.2 八个 Gate 与机器判定模型

v4 assessment 只允许下列八个精确 Gate id：

| Gate | 唯一职责 |
| --- | --- |
| `G01_audit_reference_bidirectionality` | 正确 reference 能通过、定向 fault 能精确失败、metamorphic 非故障保持通过 |
| `G02_artifact_integrity_and_model_view` | sealed artifact、snapshot、tokenizer、schema、依赖 provenance 与禁止字段 |
| `G03_semantic_replay_and_supervision` | simulator、answer、teacher trace、claim 真值与 claim 配额 |
| `G04_split_pair_and_language_integrity` | overlap、composition、causal pair、language pair 与 fold 声明 |
| `G05_ere_core_difficulty` | ERE depth、provenance、逐事件必要性与保留组合 |
| `G06_cps_core_difficulty` | CPS 反向依赖图、逐记录 hard negative、NONE、valid-suboptimal 与组合结构 |
| `G07_deterministic_surface_independence` | label/choice/candidate/action/template 精确配额与 source-only structured heuristic |
| `G08_learned_shortcut_resistance` | grouped CV、train-fit-heldout、claim-only 与全文统计捷径 |

contract ledger 对每个 Gate 和 required metric 都必须保存：

```text
clause_id
gate_id
metric_id
evaluator_id
value_type
comparator
threshold
positive_fixture_id
fault_fixture_ids
required_evidence_fields
```

每个 metric report 固定保存 `value`、`checked`、`threshold`、`comparator`、`passed`、`failures` 和 `evaluator_id`。assessment 不信任 Gate 自报的 `passed`，而是从冻结 registry 的精确 metric set 重新计算；缺项、额外项、`None`、NaN/Inf、`checked=0`、异常、空 vocabulary、零解析或非布尔结论都使所属 Gate 为 false。Gate report 只做聚合展示，不能拥有 registry 外的私有通过逻辑。

registry 的 required metric id、比较器与负控所有者精确冻结如下。所有行的 `positive_fixture_id` 都是 `R0_GOOD`；表中 fault id 就是唯一允许的 `fault_fixture_ids`，执行层不得自行合并、删减或新增 required metric。`rate` 必须同时保存 numerator/denominator/failing record ids，`min/max/deviation` 必须保存逐 split 原值和决定极值的 record/group ids，布尔值必须保存全部子检查与失败 id；只有 G01 是显式 meta-conjunction，其他 Gate 不得用一个笼统布尔值吞掉表中细项。

| Gate | required metric id | comparator / threshold | fault |
| --- | --- | --- | --- |
| G01 | `g01_bidirectional_reference_matrix_valid` | `eq true`；内部精确要求 positive target Gate `7/7`、fault `20/20`、metric positive/kill coverage `1.0`、metamorphic `4/4`、import boundary true | F401 |
| G02 | `g02_input_seal_valid` | `eq true` | F402 |
| G02 | `g02_snapshot_source_set_exact` | `eq true` | F402 |
| G02 | `g02_manifest_schema_count_hash_valid` | `eq true` | F402 |
| G02 | `g02_runtime_and_stream_provenance_valid` | `eq true` | F402 |
| G02 | `g02_model_view_exact_rate` | `eq 1.0` | F403 |
| G02 | `g02_forbidden_field_count` | `eq 0` | F403 |
| G02 | `g02_tokenizer_provenance_valid` | `eq true` | F404 |
| G02 | `g02_token_recount_match_rate` | `eq 1.0` | F404 |
| G02 | `g02_max_source_tokens` | `le 1024` | F404 |
| G03 | `g03_semantic_answer_replay_rate` | `eq 1.0` | F405 |
| G03 | `g03_teacher_delta_budget_exact_rate` | `eq 1.0` | F405 |
| G03 | `g03_ere_claim_contract_rate` | `eq 1.0` | F406 |
| G03 | `g03_cps_claim_contract_rate` | `eq 1.0` | F406 |
| G03 | `g03_claim_truth_single_leaf_rate` | `eq 1.0` | F406 |
| G03 | `g03_claim_text_parse_rate` | `eq 1.0` | F406 |
| G04 | `g04_unpaired_overlap_count` | `eq 0` | F407 |
| G04 | `g04_composition_integrity_valid` | `eq true` | F409 |
| G04 | `g04_causal_pair_contract_rate` | `eq 1.0` | F408 |
| G04 | `g04_causal_mutation_quota_deviation_max` | `le 1` | F408 |
| G04 | `g04_language_pair_contract_rate` | `eq 1.0` | F409 |
| G04 | `g04_paired_distribution_deviation_max` | `le 1` | F409 |
| G04 | `g04_pair_fold_group_rate` | `eq 1.0` | F409 |
| G05 | `g05_ere_depth_contract_rate` | `eq 1.0` | F410 |
| G05 | `g05_ere_provenance_quota_valid` | `eq true` | F410 |
| G05 | `g05_ere_composition_provenance_valid` | `eq true` | F410 |
| G05 | `g05_ere_sample_necessity_margin_min` | `ge 0.0` | F411 |
| G05 | `g05_ere_event_flip_margin_min` | `ge 0.0` | F411 |
| G06 | `g06_cps_pstar_depth_unique_optimal_rate` | `eq 1.0` | F412 |
| G06 | `g06_cps_valid_suboptimal_rate` | `eq 1.0` | F414 |
| G06 | `g06_cps_invalid_count_diversity_rate` | `eq 1.0` | F413 |
| G06 | `g06_cps_invalid_length_relation_rate` | `eq 1.0` | F413 |
| G06 | `g06_cps_valid_suboptimal_quota_valid` | `eq true` | F414 |
| G06 | `g06_cps_distractor_reason_rate` | `eq 1.0` | F413 |
| G06 | `g06_cps_none_match_rate` | `eq 1.0` | F414 |
| G06 | `g06_cps_none_ratio_valid` | `eq true` | F414 |
| G06 | `g06_cps_composition_capability_valid` | `eq true` | F412 |
| G07 | `g07_surface_assignment_rebuild_exact` | `eq true` | F415 |
| G07 | `g07_answer_choice_template_deviation_max` | `le 1` | F415 |
| G07 | `g07_candidate_action_position_deviation_max` | `le 1` | F416 |
| G07 | `g07_causal_joint_label_deviation_max` | `le 1` | F415 |
| G07 | `g07_structured_feature_evaluable_coverage_min` | `ge 0.90` | F418 |
| G07 | `g07_structured_parser_coverage_min` | `eq 1.0` | F418 |
| G07 | `g07_structured_heuristic_excess_max` | `le 0.0` | F417 |
| G08 | `g08_grouped_cv_protocol_valid` | `eq true` | F419 |
| G08 | `g08_train_fit_heldout_protocol_valid` | `eq true` | F419 |
| G08 | `g08_fold_pair_group_rate` | `eq 1.0` | F419 |
| G08 | `g08_statistical_model_valid_rate` | `eq 1.0` | F419 |
| G08 | `g08_answer_shortcut_excess_max` | `le 0.0` | F420 |
| G08 | `g08_claim_shortcut_excess_max` | `le 0.0` | F420 |

`margin` 定义为 `actual - required_min`，`excess` 定义为 `accuracy - allowed_max`；必须从保存的原始分子/分母复算，不能只写裁剪后的零。`g04_unpaired_overlap_count` 同时展开 semantic 与 surface 两个子计数并相加；`g02_runtime_and_stream_provenance_valid` 明确包括 Python/torch/transformers/tokenizers、Git/diff 和每条 named stream seed，不能只验证其中一项。G01 的 positive target Gate 固定指 G02–G08，不包含自己，从而不存在自计数循环。

R0 中 G01 由完整 positive/fault/metamorphic matrix 最后计算，不形成循环依赖。R1 以后 G01 改为验证：冻结 registry hash、已通过上一阶段的 evidence-seal hash，以及主设计层另行创建的阶段授权文件三者一致。执行 agent 不得生成或修改授权文件。

### 17.3 R0：独立 reference pack 与双向杀伤矩阵

R0 不生成候选训练数据。它使用 `tests/v2_r1r_v4/fixtures/` 下的 audit-only reference specification；reference builder 只能使用 Python 标准库，禁止 import production generator、renderer、audit 或 simulator。语义答案、teacher delta、claim 真值和 pair diff 由手写 expected values 给出，不能调用待测代码回填。

reference specification 不是手写 1,440 条重复 JSON，也不是第二个随机 generator。它固定由三层组成：`semantic_archetypes` 保存少量完整 AST、initial world、逐步 expected trace、answer 和 claim/pair truth；`instance_renamings` 只做 nonce entity/value/action 的一一替换，并对 expected 字段执行同一纯替换，禁止计算新语义；`surface_assignment_basis` 显式保存有限正交 block、每个 split 的 block repetition/row order 和 quota target，分配 archetype、template、label permutation、choice mask、candidate/action order。builder 只能按声明顺序展开 block 并校验配额，不得根据 simulator 结果、答案、audit 输出或启发式准确率搜索/修正 assignment，也不得用一个 modulo/RNG 同时决定多个变量。每个 archetype 至少有一个不经 renaming 的手写 canonical test，audit simulator 必须独立复现其完整 trace。

reference builder 在临时目录展开完整 reference artifact。ERE split 精确为 `train/validation/composition_ood/length_ood/entity_ood/language_ood/causal_pairs`，CPS 精确为 `train/validation/composition_ood/horizon_ood/distractor_ood/language_ood/causal_pairs`；不得新增别名或把 pair 混入普通 split。每族固定为 train `180`、validation 及四个正式 OOD split各 `90`、causal pair records `90`，另有 language render pairs `90`。这些数同时整除九个答案标签、五个 CPS candidate 位置、六分之一 NONE 和五类 ERE provenance，所有配额都可构造为计数差不超过一，而不是依赖随机波动。

最终 known-good R0 assessment 必须八项全真。随后 fault harness 对 reference artifact 的副本施加以下故障，并通过公开 audit CLI 重跑；只测 helper 不算：

| Fault | 注入 | 精确失败 Gate |
| --- | --- | --- |
| `F401` | `matrix_row_missing`、`metamorphic_row_missing`、`actual_false_set_forged`、`kill_map_forged` | G01 |
| `F402` | `snapshot_missing`、`input_seal_mismatch`、`manifest_schema_count_hash_missing`、`runtime_or_stream_provenance_missing` | G02 |
| `F403` | `model_view_extra_answer_ast_teacher`、`model_view_extra_role_validity` | G02 |
| `F404` | `tokenizer_revision_forged`、`token_count_forged`、`source_over_1024` | G02 |
| `F405` | `answer_replay_mismatch`、`teacher_trace_mismatch`、`state_delta_mismatch`、`budget_mismatch` | G03 |
| `F406` | `ere_claim_kind_missing`、`cps_claim_kind_missing`、`claim_label_flip`、`claim_pair_two_leaf`、`claim_text_unparseable` | G03 |
| `F407` | `undeclared_semantic_overlap`、`undeclared_surface_overlap` | G04 |
| `F408` | `causal_two_leaf`、`certificate_endpoint_wrong`、`answer_not_flipped`、`mutation_quota_broken` | G04 |
| `F409` | `language_opening_only`、`language_syntax_quota_broken`、`composition_signature_polluted`、`paired_distribution_broken`、`pair_fold_split` | G04 |
| `F410` | `ere_depth_illegal`、`ere_provenance_overquota`、`ere_composition_provenance_collapsed` | G05 |
| `F411` | `ere_sample_necessity_broken`、`ere_event_flip_broken` | G05 |
| `F412` | `cps_pstar_depth_one`、`cps_unique_optimum_broken`、`cps_dependency_chain_broken`、`cps_composition_capability_missing` | G06 |
| `F413` | `cps_invalid_count_low`、`cps_failure_reason_diversity_low`、`cps_invalid_length_relation_missing`、`cps_distractor_reason_low` | G06 |
| `F414` | `cps_valid_suboptimal_missing`、`cps_valid_suboptimal_quota_broken`、`cps_none_skeleton_mismatch`、`cps_none_ratio_broken` | G06 |
| `F415` | `answer_label_fixed`、`choice_mask_fixed`、`template_fixed`、`causal_joint_label_fixed`、`assignment_proof_mismatch` | G07 |
| `F416` | `correct_candidate_position_fixed`、`first_action_definition_position_fixed` | G07 |
| `F417` | `ere_static_literal_shortcut`、`cps_goal_only_shortcut`、`cps_first_definition_shortcut` | G07 |
| `F418` | `source_parser_missing_result`、`source_parser_coverage_low`、`feature_evaluable_coverage_low` | G07 |
| `F419` | `train_heldout_alias`、`copied_statistic`、`pair_fold_split`、`empty_vocabulary`、`single_class_fold` | G08 |
| `F420` | `claim_text_truth_leak`、`full_source_answer_leak` | G08 |

每个 F id 是一个固定 case group；表中反引号标出的每个 case 都必须独立复制 reference、独立调用公开 CLI，并得到同一个精确 false Gate set。只有组内所有 case 都通过，才把该 F id 计为 `1`，因此 `20/20` 不允许用一个复合破坏同时遮住多个漏检。非 G02 case 由独立 test sealer 重算 input hash，使 fault 只攻击目标语义；G02 case 故意不重封。每个 case 的实际 false Gate set 必须与表中精确相等，不接受“至少命中”。ledger 中每个 required metric 必须在 positive reference 上得到有限值，并至少被其登记 fault 的一个 case 杀伤；metric kill coverage 必须为 `1.0`。

F401 不读取或递归运行真实 fault harness：reference specification 另含一个 hand-authored `completed-matrix-fixture`，其中 G02–G08 positive 为 `7/7`、F401–F420 rows 为 `20/20`、coverage 为 `1.0`、metamorphic 为 `4/4`、import boundary 为 true。F401 的四个 case 只在该 fixture 副本上调用公开 `assess-r0` 并要求仅 G01=false。真实 R0 的执行顺序固定为：先算 positive G02–G08，再运行 F402–F420 与 metamorphic/import，随后运行非递归 F401 fixture，最后写真实 matrix 并计算 G01；`assess-r0` 绝不隐式调用 `run-fault-matrix`。

R0 还必须通过四个非故障 metamorphic test：

- artifact 整体移动目录后仍从相对路径复算通过；
- README、结果文档或当前工作树后来变化，不影响 sealed artifact；
- 对 in-memory rows 改变遍历顺序，不改变排序后的 metric core；
- 同一 reference specification 重建两次，除 run metadata 外的 immutable 文件逐字节一致。

R0 同时检查 import boundary：reference builder、fault harness、independent sealer 和 metamorphic harness 都只能 import Python 标准库且不能 import `src`；audit 不 import `tests` 或 generator；R0 CLI 只能用 `sys.executable` 子进程调用 harness，不能把 reference expected values import 到 evaluator；generator 在 R0 根本不存在；共同语义只允许位于 `common/`。只有 known-good 8/8、20 个 fault group 精确命中、metric kill coverage 1.0、四个 metamorphic test 和 import boundary 全部通过，R0 才可 `passed=true`。

### 17.4 Sealed artifact 与可复算 provenance

v4 artifact 把“运行时不可变证据”和“运行后项目状态文档”分开，并用两层 seal 消除“audit 验证一个包含自身的 seal”这一循环。不可变部分固定为：

```text
manifest.json
data/
language_render_pairs.jsonl
source_snapshot/
input-seal.json
audit-core.json
fault-matrix.json
assessment.json
evidence-seal.json
```

`build-reference` 完成 manifest、全部 data、language pairs 和 source snapshot 后，先生成 `input-seal.json`；G02 只验证这层输入封印、schema/model view/tokenizer/provenance，不读取未来的输出文件。`audit-core.json` 只保存确定性 metric，不含时间、cwd 或当前环境噪声；`fault-matrix.json` 保存 F401–F420、四个 metamorphic、import boundary、positive/kill coverage 和规范化后的 artifact-relative 命令/exit code；`assessment.json` 从 registry、audit-core 与 fault-matrix 精确重算。最后由 `assess-r0` 生成 `evidence-seal.json`，哈希 input seal、audit-core、fault-matrix 和 assessment；它创建后立即只读验证，作为 R0 顶层条件，但不反向写入被封印文件。`run-metadata.json` 可保存开始/结束时间、绝对临时路径和调用环境，但不进入语义 conjunction，避免自引用或位置噪声。

两层 seal 的职责不可合并：`input-seal` 使 G02 可在 audit 前无环验证输入，`evidence-seal` 使主设计层可在 audit 后验证完整证据。任何输入在 `input-seal` 后变化都使 G02 失败；任何 audit/fault/assessment 在 `evidence-seal` 后变化都使最终 seal 验证失败。assessment 不允许通过“先写假结果、后用 seal 自证”改变 Gate。

source snapshot 必须包含当前 v4 设计、当前阶段执行命令、CLI、package、测试、`pyproject.toml`/`uv.lock` 和 reference specification。README、`docs/DIRECTORY_REFERENCE.md`、`docs/next-stage-test-plan.md`、结果报告及主验收文档属于 post-run 状态，不进入 immutable source set。manifest 仍记录 Git HEAD、dirty、`git status --porcelain=v1 -z` hash、`git diff --binary` hash和逐文件 hash，但 sealed artifact 的重放只读取 artifact 内 snapshot，不比较后来变化的工作树。所有进入 immutable 文件的路径均为 artifact-relative POSIX path；cwd、临时目录、时间和进程 id 只能进入 run metadata。

G02 还必须验证：

- Python、torch、transformers、tokenizers 版本；
- `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` tokenizer class、revision、special-token config；
- 每条 source 的 actual token count 精确相等且不超过 `1024`；
- named stream derivation algorithm/version 与每条 episode stream seed；
- data 文件 hash、record count、schema 和 generator/contract version；
- model view 精确只含 `example_id/source_text/reasoning_budget/valid_choice_mask`，不得递归出现 answer、AST、teacher、claim、span、role、state、validity 或 certificate。

生成开始时 active source hash 必须与主任务提供的设计/命令 hash 一致；artifact sealed 后，复算从 snapshot 运行。snapshot replay 把 input artifact 复制到新临时目录，只用 snapshot 中的 CLI/package 重新生成 audit-core、fault-matrix、assessment 和 evidence-seal；四者必须与原件逐字节一致。当前仓库变化只影响是否能开始新运行，不得追溯性使旧 artifact 失败。

### 17.5 G03/G04：语义、监督、split 与 pair

G03 对每条 record fresh replay simulator，并逐项核对 semantic answer、局部字母 answer、teacher trace、state/resource delta 与 reasoning budget。claim evaluator 与 claim constructor 分离，不 import generator helper：

- 每条 ERE 精确包含 attribute value、relation existence、condition truth 三个 positive/negative pair，共六条；
- 每条 CPS 精确包含 prefix legality、resource-or-cumulative-cost、fact truth、goal-or-final-constraint 四个 pair，共八条；
- 每个 pair 只改变一个 predicate leaf，两个 fresh truth 必须相反，claim text parse coverage 为 `1.0`；
- 无法构造 prefix legality 或任一 required kind 时，record 在生成阶段失败，不能少采样后用比例阈值放行。

G04 负责全部跨记录关系。未声明 pair 的 train/heldout/heldout 之间 semantic 与 surface overlap 都必须为零；composition OOD signature 与 train/validation 保留 signature 交集为零，同时组成原语在 train 中出现。causal pair 必须共享 template、surface/candidate/action order、label mapping 和 choice mask，AST 精确单叶变化，certificate from/to 与端点一致，fresh semantic 和字母答案都翻转。

ERE causal mutation 家族和 CPS 的 initial-resource/precondition/effect/cost 家族在 R2/R3 各占 `0.20–0.30`；R0/R1 因 pair 数较小，四或五类计数差不得超过一。language audit pair 使用同一 AST 的 train/OOD renderer，删除 opening 后正文仍不同，三类句法变化计数差不超过一；正式 validation 与 language OOD 的 recipe、长度、entity/candidate、provenance、NONE 和 failure profile 分布计数差不超过一。所有 pair 在统计 fold 中不可拆分。

### 17.6 G05：ERE 保留的语义构造

v4 可以移植 v3 已独立通过的 simulator 与 ERE semantic builder 逻辑，但必须进入新 package，不保留兼容 wrapper。每条 AST 仍使用 `SET/COPY/SWAP/IF/LINK/UNLINK/FOREACH_LINKED`，并满足：

- train/validation/entity/language/composition 必要 spine 长 `3–7`，length OOD 长 `8–20`；
- train/validation/length/entity/language 的 initial-copy、condition-true、condition-false、swap-source、relation-neighbor 五类 provenance 计数差不超过一；composition 至少覆盖三类且最大占比不超过 `0.35`；
- train/validation/entity/language 的 sample-level all-events-necessary rate不低于 `0.90`、event-level flip rate不低于 `0.95`；length OOD 两项都为 `1.0`；
- 每个声明必要事件都由 audit 独立 no-op/delete 后 fresh replay，不能信任 generator 的 dependency 布尔值；
- causal pair 只按 G04 的单叶/翻转合同评价，不把其 mutation-specific dependency 当成普通 split provenance/必要性阈值。

ERE structured heuristic 归 G07；G05 只判断任务确实需要多步状态执行，不再混入表面平衡。

### 17.7 G06：CPS 依赖图与逐记录 hard negative

CPS 必须重写，禁止移植 v3 的五角色候选数组。production constructor 从 goal/final constraint 反向建立 prerequisite/effect/resource dependency graph，再从 `P*` 应用注册的局部 transformation 生成候选。transformation registry 至少覆盖 dependency swap、prerequisite deletion、wrong precondition、wrong effect、resource shortage、relation direction、final-constraint destruction、redundant cost 和 budget overflow。每个 candidate 保存 audit-only derivation diff，但 source、model view、label 与 surface schedule不得读取 canonical role。

每条 non-NONE 五候选 record 必须逐条满足：

- `P*` 在 base split 长 `2–5`、horizon 长 `6–10`，fresh replay 合法且是唯一最低成本答案；
- 至少一个 valid-suboptimal；
- 三个 invalid candidate 具有至少三种不同 fresh failure reason；
- invalid 中至少各有一个与 `P*` 等长、一个更短、一个更长；
- valid-suboptimal 的长度关系由独立 split quota 决定，每个 split 的 equal/longer 各占 `0.40–0.60`；
- 最短、最长、raw cost、precondition-only、goal-only、budget-only 或 final-constraint-only 都不能成为稳定答案规则。

distractor OOD 八候选至少覆盖四种 invalid reason。NONE 从同一 `recipe × P*_length × candidate_count` skeleton 配对构造，所有 candidate 失败、至少三种原因，candidate length multiset 与配对 non-NONE 完全相同；NONE 比例按精确 quota 分配为六分之一，计数取最接近整数。

audit 对每条 record 计算上述条件，禁止用 split 级 failure reason 并集替代 per-record conjunction。train/validation 覆盖 fact chain、resource replenish/consume、fact/relation precondition、final constraint、cost 和 budget；四个 composition signature 只在 composition OOD 组合出现。

### 17.8 G07/G08：确定性表面分配与捷径审计

semantic pool 完成后，surface allocator 把分配建模为确定性 quota/matching 问题，而不是独立随机洗牌。相同 seed 与 semantic pool 必须得到唯一 assignment proof。每个 split 内下列计数最大值与最小值之差不超过一：

- 九个 answer label；
- 九个 label 在 valid choice mask 中的激活次数；
- renderer template；
- non-NONE 正确 candidate 的表面位置；
- `P*` 首 action 在真实 action-definition domain 中的位置；
- causal base/flip 联合答案 label，且同一 pair 两端 label 不同。

candidate/action 位置在声明 strata 内平衡；小 stratum 按 manifest 中冻结的 merge hierarchy 合并，不能临时改变。source-visible feature 的单项与二阶组包括 template、token/character bucket、recipe、provenance、entity/rule/event/action/candidate count、`P*` 长度、plan-length multiset、首末 candidate 长度、首 action 定义位置、`action_count × candidate_count` 和 `recipe × P*_length`。正式结论只用于 `n>=20` 的组，但每个 feature family 至少 `90%` 记录必须落在可评估组，否则 G07 失败。

G07 的 source-only parser 不得读取 AST、answer、role 或 certificate。ERE 必须实现 initial value、first/last literal/rule/event、last mention、ignore events、fixed IF branch、ignore relation direction；CPS 必须实现 first/last candidate、first-definition、shortest/longest、raw cost、precondition/goal/budget/final-constraint only。每项 parse coverage 必须为 `1.0`，准确率不高于该 split 平均 chance `+0.10`。

G08 使用 deterministic grouped 5-fold CV；causal/language/claim pair 按 id 同组。必须分别保存 split-local CV 和“只在 train 拟合一次、逐 heldout 评估”的模型、fold/group 数、fit/eval split、confusion counts 与 feature vocabulary hash。至少评估 majority、question-only、length/count、template/position、claim-only unigram/length、full-source unigram和 character n-gram。answer heuristic 上限为平均 chance `+0.10`；claim truth 上限为 `0.60`。任何 train/heldout 别名、复制数值、pair 拆分、空 vocabulary、单类 fold或算法异常都直接失败。

R0/R1 的统计实现固定使用 Python 标准库，不新增 scikit-learn/scipy 依赖。文本模型是 multinomial Naive Bayes：Laplace `alpha=1.0` 同时用于 class prior 和 feature likelihood；word tokenizer 为 lowercase 后的 `(?u)\b\w+\b`，character model 对 lowercase、连续空白折叠后的字符串取 `3–5` gram；vocabulary 只由当前 fit rows 建立并按 Unicode code point 排序，OOV 丢弃。categorical length/count/template/position 模型使用带 `alpha=1.0` 的条件多数；所有 label tie 按局部 label index 升序打破。answer prediction 只在该 record 的 `valid_choice_mask=true` labels 中取最大 posterior；split chance 精确为 `mean(1 / active_choice_count)`，不是固定写死 `1/9`。

fold builder 先按 pair/group id 聚合 label counts，再把 groups 按 `SHA256("r1r-v4-fold|" + group_id)` 排序，依次放入使“各 label fold count 最大偏差、总 row count 最大偏差、fold index”三元组字典序最小的 fold；不得按 row 拆组或用运行库默认 shuffle。每个 fold 的 vocabulary、prior 和统计只从另外四 fold fit；train-fit-heldout 只 fit 一次 train，禁止复用 split-local accuracy、vocabulary 或 model object。question-only 只能读取 source 中明确的 question span；template/position 只能由 source parser 提取，不能读取 template id、AST 或 answer。全部算法参数、fit row fingerprint、group-to-fold map、vocabulary hash 和 confusion matrix必须进入 evidence。

R2 即使八项全真，主设计层仍要做 registry 外的未知 source-only shortcut review；未知审计不由生成器作者自行签字。

### 17.9 直接替换的代码边界

R0 开始时先删除当前 flat v3 package、旧 P0 CLI 和旧 `tests/test_v2_r1r_*`，历史实现由 v3 artifact snapshot 保留。R0 只建立：

```text
src/yggdrasil_v2/r1_revalidation/
  common/
    schema.py
    simulator.py
    fingerprint.py
  audit/
    registry.py
    artifact.py
    replay.py
    pairs.py
    structure.py
    shortcuts.py
    assessment.py
  __init__.py
experiments/v2_r1_revalidation.py
tests/v2_r1r_v4/
  reference_builder.py
  fault_harness.py
  independent_sealer.py
  metamorphic_harness.py
  fixtures/
  test_common_semantics.py
  test_reference_positive.py
  test_fault_matrix.py
  test_metamorphic.py
  test_import_boundaries.py
```

R0 不得存在 `generate/`、`ere.py`、`cps.py`、`render.py`、model/cache/train 文件或旧命令 alias。reference builder 不 import `src`；audit 只 import `common`。主设计层验收 R0 并授权 R1 后，才允许新增 `generate/ere.py`、`generate/cps.py`、`generate/surface.py`、`generate/claims.py` 与对应新测试。

CLI 是严格阶段机。R0 只暴露 `build-reference`、`audit-reference`、`run-fault-matrix`、`assess-r0`；任何 `generate-p0`、`--mode preflight/formal` 或训练命令都必须不存在，而不是运行时静默 fallback。

### 17.10 R0–R3 与逐阶段授权

| 阶段 | 唯一目标 | 固定规模 | 当前权限 |
| --- | --- | --- | --- |
| R0 audit-reference | 建立八 Gate registry、known-good、20 faults、metamorphic、snapshot replay | reference train 180、各 heldout/pair 90 | **已授权设计后的首次执行；结束即停** |
| R1 generator-smoke | 直接实现 v4 ERE/CPS/renderer/claims，并验证逐记录 construction invariant | 每族 train 180、各 heldout/pair 90 | 未授权；需主设计层 R0 authorization |
| R2 adversarial-preflight | 完整八 Gate、正式阈值与未知捷径复核 | 每族 train 1024、各 heldout/pair 256 | 未授权；只允许一次，需 R1 authorization |
| R3 formal | 形成 P0-D 正式候选数据 | 每族 train 4096、各 heldout/pair 512 | 未授权；需 R2 主审与单独 authorization |

authorization 固定放在 `artifacts/v2-r1r/authorizations/`，包含上一阶段 evidence-seal SHA-256、当前设计/执行命令 hash、唯一允许的下一阶段和主设计层判决。执行 agent 不得创建、复制或编辑 authorization；缺失时 CLI 在创建输出目录前失败。

每个阶段使用新目录并拒绝覆盖。R0/R1 可在正式阶段运行前迭代单元测试，但每个阶段的 sealed artifact 只运行一次；sealed run 任一 Gate 失败后只分析和同步文档，不修复、不重跑。阶段通过也必须停止并唤醒主设计层。只有 R3 八项全真、artifact-internal replay 通过和主设计层未知捷径复核通过，才可能另行授权 P0-M。

### 17.11 证据边界

v4 首先修复实验的认识论边界，不提高 latent 架构成功概率。R0 只证明审计器能区分手写正确世界与指定错误；R1 只证明生产生成器能构造满足合同的小规模数据；R2 只证明一个 preflight seed 未触发已知/主审捷径；R3 才可能关闭 P0-D。

任何 P0-D 结果都只说明任务、teacher、claim 和接口足以开始训练。共享 Boundary/core 是否成立，仍必须由后续 ERE+CPS 行为、因果干预、K=1/K=8 比较和 matched Pareto 决定；不能把 reference pack、数据 Gate 或审计复杂度写成架构正证据。

## 18. P0-D v5：分层验证测量系统（历史合同）

### 18.1 判决与目标

v4 的方向错误不在于“要求独立 reference”，而在于把手写语义事实、第二套 simulator、1,440 条展开、八 Gate、70 个 fault case、统计 learner、metamorphic 和 snapshot replay 同时交给一个执行任务。执行层最终用自洽近似替代了关键算法，而三个浅层预测试没有暴露这一点。

v5 不降低最终 P0-D 要求，而是先分别证明各测量部件正确，再做一次完整集成。每一阶段只回答一个问题，使用独立 artifact，结束后必须停止并由主设计层复核。前一阶段的通过不自动授权后一阶段；执行 agent 永远不能创建 authorization。

v5 合同冻结时的唯一授权是 **R0A**。该阶段已经执行并被后续 v6/v7/v8 合同覆盖；下表“当前权限”只记录 v5 当时的权限，不代表 2026-08-01 的前向授权。v4 source implementation 当时在 R0A 直接删除，没有建立 wrapper、alias、schema reader 或兼容测试；v4 sealed artifact 与 v4 执行命令只保留为历史失败证据。

### 18.2 阶段划分

| 阶段 | 唯一问题 | 主要交付 | 当前权限 |
| --- | --- | --- | --- |
| R0A semantic oracle | 手写世界与独立 simulator 是否逐字段一致 | canonical fixture、公共 simulator、正负控、确定性报告 | **已授权；结束即停** |
| R0B structural audit | artifact/model-view/语义回放/split/pair/ERE/CPS 审计是否能接受正控并精确拒绝错误 | G02–G06 与对应 faults | 未授权 |
| R0C shortcut audit | source-only heuristics 与真实统计 learner 是否按已知数值工作 | G07/G08、toy numeric fixtures、对应 faults | 未授权 |
| R0D integrated reference | 已验证部件组合后是否仍能通过完整 reference/fault/metamorphic/replay | v11 八 Gate、F401–F420、一次 sealed integration | 已完成并有限 accepted；不是 production evidence |
| R1 generator smoke | production generator 能否构造小规模合格数据 | v13 实际为每族 train 288、各 heldout/pair 72 | **已完成并经主审有限接受；fixed-seed smoke，不是完整 P0-D** |
| R2 adversarial preflight | 较大样本是否暴露已知或主审捷径 | v14 实际为每族 train 1024、各 heldout/pair 512 | **已完成；8,192 records、双生成/报告、G01–G11 全通过** |
| R3 formal | 是否形成 P0-D 正式候选数据 | 每族 train 4096、各 heldout/pair 512 | **已完成并失败；仅 G09 false，禁止 P0-M** |

R0A–R0C 是测量部件资格验证，不宣称任何正式 Gate 通过。只有 R0D 才能形成完整 R0 判决；只有 R3 可能关闭 P0-D。

### 18.3 R0A 的冻结输入

主设计层直接交付并冻结：

- `tests/v2_r1r_v5/fixtures/oracle-spec.json`：14 个完整手写世界；
- `tests/v2_r1r_v5/oracle_validator.py`：只调用公共 API 并逐字段比较，不含任务语义；
- `tests/v2_r1r_v5/r0a_runner.py`：资格检查、报告与 seal 的唯一实现；
- `tests/v2_r1r_v5/cli_template.py`：R0A CLI 的精确字节模板；
- `tests/v2_r1r_v5/contract_guard.py`：写实现前的 frozen hash 检查；
- 五个 `test_*.py` 资格测试；
- `tests/v2_r1r_v5/frozen-inputs.json`：上述文件与本设计的 SHA-256 清单。

执行层不得编辑这些文件。任何 frozen hash 改变都在创建 artifact 前判为 `BLOCKED`。这次不让执行层根据 prose 自己发明 AST、expected trace、validator、fault 或 Gate。

oracle fixture 包含 7 个 ERE 与 7 个 CPS canonical cases。每条 case 直接保存完整 AST、expected answer、canonical final state、逐事件/逐 action trace；需要 prefix 语义的 case 还保存完整 prefix output。所有 expected values 都由主设计层手写，不由 simulator、builder、audit 或脚本生成。

ERE 必须覆盖 `SET/COPY/SWAP/IF/LINK/UNLINK/FOREACH_LINKED`、IF true/false、attribute query、relation query、argument binding 和 prefix execution。CPS 必须覆盖 fact/resource/attribute/relation precondition，add/remove fact、resource delta、set attribute、link/unlink effect，cost、budget、goal、final constraint、unknown action、无有效计划、相同成本并列、唯一最优和 prefix replay。

R0A 没有 reference builder、renaming、surface template、label mapping、claim、pair、dataset split、tokenizer 或统计 learner。这些不属于语义 simulator 的最小问题。

### 18.4 R0A 公共语义 API

R0A 只允许 `src/yggdrasil_v2/r1_revalidation/common/`，并精确暴露三个纯函数：

```python
simulate_ere(ast: Mapping[str, Any], prefix: int | None = None) -> dict[str, Any]
evaluate_cps(ast: Mapping[str, Any]) -> dict[str, Any]
replay_cps_prefix(ast: Mapping[str, Any], candidate_index: int, prefix: int) -> dict[str, Any]
```

返回值必须完全 JSON-serializable，不向调用者暴露 dataclass、set 或内部 state object。canonical state 固定为 `attributes/relations/facts/resources` 四个字段；所有 map key、relation pair 和 fact 都稳定排序。

ERE trace 每个 event 保存 `event_index/rule/arguments/executed_ops/delta_budget/state_before/state_after`。`IF:true` 与 `IF:false` 只记录真实执行分支；`FOREACH_LINKED:n` 保存实际邻居数，并按 target 字符串排序执行。`delta_budget` 只计实际执行的 mutating leaf operations，不计 IF/FOREACH 控制节点。

CPS candidate report 保存 `candidate_index/plan/legal/valid/total_cost/budget_ok/goal_satisfied/final_constraints_satisfied/failure_reasons/trace/final_state`。未知 action 或 precondition 失败时在该步前停止，不收取 cost、不应用 effect；合法执行后再独立判断 budget、goal 与 final constraint。没有有效 candidate 或最低成本并列时 `answer=null`、`unique_optimum=false`；只有唯一最低成本有效 candidate 才返回其 index。

任何未知 primitive、condition、effect、rule/action、参数缺失、重复定义、非法 prefix/index 或格式错误都必须抛出明确异常，禁止静默 NOOP 或默认值。

### 18.5 R0A 正控、负控与判定

positive validator 对 14 个 canonical cases 与全部 prefix expectations 逐字段 exact compare；不得只比较 answer。任何异常、缺字段、额外字段、顺序不稳定或非 JSON 值都失败。

fixture 另存 12 个手写 negative controls，分别破坏 expected answer/final state/trace/candidate 字段，或修改 AST 中的 SET literal、IF condition、UNLINK target、resource、cost、final effect和 tie cost。runner 必须证明每个 corruption 都导致 exact comparison 失败。负控只是机械复制和单路径替换，不计算新的 expected value。

R0A assessment 只有以下全部成立才 `passed=true`：

1. frozen-input hash 清单完整且不变；
2. 14/14 canonical exact；
3. 全部 prefix exact；
4. ERE/CPS 冻结能力覆盖完整；
5. 12/12 negative controls 被拒绝；
6. 相同输入连续评估两次的 canonical report bytes 相同；
7. simulator/CLI import boundary 通过；
8. manifest、oracle report、assessment 与 evidence seal 完整且 hash 可复算。

R0A 通过只证明公共语义实现能复现主设计层手写世界，并且 validator 能发现指定 corruption；不证明大规模 reference、audit Gate、generator、任务难度、数据无捷径或模型可训练。

### 18.6 R0A 代码与 artifact 边界

R0A 直接把运行面收缩为：

```text
src/yggdrasil_v2/r1_revalidation/
  __init__.py
  common/
    __init__.py
    simulator.py
experiments/
  v2_r1_revalidation.py
tests/v2_r1r_v5/
  cli_template.py
  contract_guard.py
  frozen-inputs.json
  oracle_validator.py
  r0a_runner.py
  fixtures/oracle-spec.json
  test_oracle_fixture.py
  test_common_semantics.py
  test_negative_controls.py
  test_import_boundaries.py
  test_cli_contract.py
```

必须删除 v4 `audit/`、旧 `common/schema.py`、旧 `common/fingerprint.py` 和整个 `tests/v2_r1r_v4/`；不得保留 import alias。R0A 禁止 `generate/`、renderer、artifact audit、model/cache/train/GPU、第三方依赖和网络访问。

唯一 CLI 只提供：

```text
python experiments/v2_r1_revalidation.py verify-oracle ...
```

唯一 sealed root 是 `artifacts/v2-r1r/p0d-v5-r0a-20260801-1/`，存在即 `BLOCKED`。预测试可在临时目录反复运行；sealed command 只能运行一次。失败后保留 partial root、停止、通知主设计层，不修复、不重跑。通过后同样停止；执行层不得创建 R0B authorization。

### 18.7 R0B–R0D 的边界

R0B 只接收已通过并由主设计层验收的 R0A simulator/evidence hash，再建立 artifact/model-view、语义回放、split/pair 和 ERE/CPS 结构审计。它只覆盖原 G02–G06 及对应 faults，不接触 surface/statistics。

R0C 使用独立、拥有手算预测与 confusion matrix 的小型 numeric fixtures实现 source-only parsers、条件多数、word NB、character n-gram、grouped fold 和 train-fit-heldout。它不得读取 production records 的 hidden metadata。只有 toy predictions、vocabulary hash、fold map、confusion counts 和 OOV/tie/empty/single-class faults 全部逐项一致，才可能验收 G07/G08 evaluator。

R0D 才把 R0A–R0C 已验收实现组合成完整 hand-authored reference、有限正交 surface expansion、八 Gate、F401–F420、metamorphic、import boundary 和 artifact-internal replay。R0D 不允许重新实现已有算法，只允许组合和验证；任何接口不匹配应失败，不得增加兼容层。

### 18.8 证据边界

v5 修复的是实验测量链，不提高 latent 架构本身的成功概率。R0A–R0D 的意义是确保后续 P0-D 失败能被解释为数据/任务问题，而不是审计器自洽或测试未覆盖。只有 P0-D 完成后，才进入同数据、同 teacher、同基座的 direct SFT、text-CoT SFT、K=1 latent 与 K=8 latent 比较；最终架构结论仍由跨任务行为、因果、最差 seed 和 matched 质量—成本决定。

## 19. 历史测量链状态：v11 R0D-integrated accepted

v11 结束时的有效边界是 **R0A、R0B、R0C 的有限测量部件已经分别资格化，完整集成、生产任务与模型仍未验证**。v7 以公共 simulator、858-cell operand lattice 和 27 项 registry 外探针关闭 R0A；v8 的机器 PASS 因五项登记表外 false negative 被主设计层拒绝，且其 artifact 保持只读历史证据。

v9 直接删除 v8 active audit/CLI/tests 源码并重建 R0B：资格剖面固定为 11 records、18 claims、2 causal pairs 和 2 reversible language pairs；schema 精确封闭；ERE query provenance 由 AST 与 fresh trace 的 typed dataflow 推导；CPS composition witness 由 action、candidate outcome 和 budget 独立推导；自然语言证据收缩为四种可逆、full-match grammar，不宣称开放域语言理解。唯一 formal root `artifacts/v2-r1r/p0d-v9-r0b-invariant-20260802-1/` 得到 G02–G06 `5/5`、adversary `48/48`、四类 family holdout `4/4`、metric kill `45/45`、metamorphic `6/6`，九项顶层 entry 与 45 个递归 evidence hash 均可复算。

主设计层另用未登记探针检验 extra claim、role swap、自报 provenance、nonsense language、ERE 初态答案捷径和 CPS fact shortcut；六项都由目标 Gate 拒绝，且后两项在上游 G03/G04 保持为真时分别只使 G05/G06 失败。因此 v9 可被接受为当前固定资格剖面内的 measurement-system qualification，而不是“测试写得更严所以架构成立”。

v10 没有扩张 R0B，而是直接切换 active CLI/tests 并新增独立 `learner/`。固定 profile 用手算预测、exact rational score、vocabulary/fold map、confusion matrix 与错误码资格化 source-only parser、条件多数、word/character multinomial NB、grouped five-fold、split-local CV 和 train-fit-heldout。唯一 formal 得到 G07/G08 `2/2`、adversary `39/39`、六 family holdout `6/6`、声明 metric kill `34/34`、metamorphic `5/5`；固定语义摘要和六组 registry 外探针通过，主设计层接受有限 R0C。

v11 只新增 `integration/` 编排，不修改三层 accepted 算法。12 个 hand-authored case 同时经过 simulator truth、answer-label binding、typed invariant、toy parser 与 word/char grouped learner；formal 得到 G01–G08 `8/8`、F401–F420 `20/20`、metric kill `19/19`、metamorphic `4/4`、artifact replay `1/1`。主审复算 144 文件直接或传递覆盖，并用 unknown `Answer:`、hidden model field、overlength source 三项外部探针验证 fail-closed，因此接受有限 R0D integrated measurement system。

设计探针同时暴露 v10 exact `Fraction` learner 的长度边界：较长 char 3–5 gram surface 会触发 Python 4300 位整数转字符串保护，v11 只资格化规范化 source `<=400`。qualification surface 也不是完整 ERE AST 的充分 renderer。因此 v11 当时不能自动运行 generator；必须先设计语义充分的 production renderer，并根据真实长度决定是否先做 learner-scaling qualification。该历史后继已由 v12–v17 与 P0-M v1–v5 承接；v7/v9/v10/v11 formal 仍禁止覆盖、改写或重跑。

## 20. production-data 历史状态：v12/v13 accepted，v14/v15 rejected

v12 已用完整可逆 controlled-natural-language grammar 替代 qualification surface，并把 model view 精确封闭为 `example_id/source_text/reasoning_budget/valid_choice_mask`。pinned Qwen tokenizer 的最长入口剖面为 984 tokens；compact exact-rational scorer 与 v10 短输入语义一致，并避免旧 Fraction 十进制报告失败。唯一 formal 与主审接受有限 production entry。

v13 在 accepted simulator/renderer/scorer 上生成 ERE/CPS 各 720、总 1,440 records。生产数据使用 episode-local nonce、split-local 精确均衡标签日程、alpha-invariant fingerprint、可重放单叶反事实、局部交叉配平 claim 和 pinned token budget。唯一 formal 的 G01–G11 全 true；62 文件 seal、两次独立生成/报告和 artifact replay 均一致。400 个 registry 外新 seed 结构探针通过，主设计层接受 **fixed-seed R1 generator smoke**。

该接受不改变第 1–15 节的模型或 Pareto Gate，也不把数据审计写成架构证据。v14 随后修复任意 root provenance、非九整除标签配平和 ERE 完整规模容量，并用每族 train 4,096、六个 heldout/pair 各 512 完成唯一 formal。14,336 records 的结构、因果、claim、token、overlap 与 replay Gate 通过，但 98-way Bonferroni-Wilson G09 有两个局部 char-NB upper 越线；全部 family aggregate 仍通过。正式状态因此是 machine-fail/main-review rejected，而不是“近似通过”。

v15 已按该边界先做独立 G09 qualification。heldout=1,536 的完整 98/14-cell 合成决策在 100,000 trials 下通过 null、局部 `+0.04`、单格 `+0.10` 与扩散 `+0.05` 操作特性；14 项 fault、3,264 个 exact 对照、single-bank/batch/progress 和 replay 也通过。但唯一 Q formal 的一次性速度比为 `4.7617x < 5x`，Q08 false，最终 `FAIL_G09_QUALIFICATION`；固定 fresh-seed root 未创建。该失败暴露的是把 correctness 与短时相对 wall-clock 混成一个 Gate 的合同问题，不允许事后接受 v15。

该后继条件已由第 21 节 v16 合同承接；本节继续只保存 v12–v15 的历史判决。

## 21. 已执行 production-data 合同：v16 runtime-Q → fresh-seed F

v16 不重写 v15，也不调整 G09 统计 ceiling。它把 v15 Q01–Q07/Q09 的 sealed 窄组件作为只读输入，另用 15 轮交替顺序 paired benchmark、完整 v14 G09/G10 prediction projection、240 秒绝对耗时、4.5 GiB 峰值工作集、single-bank/progress 与 18 项 assessor fault 资格化运行面。唯一 Q 的 Q01–Q09 全 true，主审接受该有限 runtime component。

Q PASS 后，seed `2026081602` 的唯一 26,624-record production formal 已执行。G09/G10、结构 replay、全量 regeneration 均通过，但 G05 因 307 条 `if_copy` ERE 的六值 witness 被 control overwrite 破坏而 false；G07 因动态资源名 `rules` 被 path-insensitive alpha renamer 当 schema key 而 false。最终 `FAIL_P0D_PRODUCTION`，两个 fixed root 均已消耗。该历史失败已由第 22 节的独立 v17 合同承接；v16 本身仍不得修补或重跑。

## 22. 已执行 P0-D v17：repaired production accepted

v17 没有重判 v16，而是先建立独立 repair qualification。ERE 路径加入 pattern override 后的六值可见域后置不变量、50,000-seed `if_copy` sweep、13,312-fingerprint 正式容量和 v16 307 条失败逐 seed 回归；alpha 路径让 fingerprint 与 transform 共享 path-aware grammar/domain 区分，并用 58 项 schema/reserved collision 和旧 transform fault kill 资格化。R01–R07 全 true，且 v16 已接受的 runtime/G09/G10 保护 AST hash 不变。

fresh seed `2026081702` 随后生成 26,624 records、1,536 causal pairs。唯一 production formal 的 G01–G11、全量 regeneration、artifact replay 与 watched-input immutability 全部通过；root seal 为 `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`。主设计层接受 P0-D production data + verifier，只授权独立 P0-M，不产生架构结论。

## 23. 已执行 P0-M v1–v5：training-path smoke accepted

P0-M 依次验证 cache、ERE/CPS 单任务 K=8、joint shared K=8、direct/text-CoT 和 100-step CUDA throughput。v1–v4 分别暴露并修复 cache batch 决策、BF16 mask overflow/baseline 全序列 logits、claim state–query compatibility、owner shortcut 与 claim 监督提前关闭；每次失败都保留 root 并直接切换新版本，没有降低 Gate或扩大 selection。

v5 以 interaction probe、真假 pair ranking、同族 owner contrast和贯穿训练窗口的 claim 梯度关闭 M01–M08。joint ERE/CPS answer 为 `1.0/1.0`，claim `0.90659`，owner-shuffle drop `0.37231`；probe 物理删除后答案不变；同 episodes 的 direct/text-CoT greedy answer 都为 `1.0`；100-step throughput 为 `71.706 examples/s`。最终 `PASS_P0M`、`p1_eligible=true`、`p1_started=false`。

接受范围只到训练通路 smoke。单一 fixed selection overfit 不能证明共享推理，owner contrast 也可能学习 episode identity；P1 必须使用 fresh data/model seeds、heldout 组合、因果干预、K=1/K=8 和 matched cost。该入口后来由第 24 节的 P1 v1 合同承接。

## 24. 已执行 P1 v1：fresh-data G09 rejected

P1 v1 已另立 single-seed falsification 合同并直接切换活动实现。正式 preflight 通过；fresh generator seed `2026081901` 以每族 train `8192`、validation/OOD/causal split 各 `1024` 生成 `28,672` records。唯一 data formal 的 G01–G08、G10、G11、全量 regeneration、artifact replay 和 watched-input immutability 均通过，但 G09 false，最终 `FAIL_P1_DATA`。

唯一失败 cell 为 `ERE/validation/full_text_char_3_5_nb`：accuracy `0.33203125`、chance `0.27099609`、98-cell Bonferroni-Wilson upper `0.38190468`，高于 `chance+0.10=0.37099609`。停止后的只读学习曲线显示同一 validation 上 train 从 1K 扩到 8K 时该指标总体上升，且去掉 choices 图例仍保留大部分信号；当前最强判断是 generator assignment/presentation 在合法答案 mask 条件下留下了可扩展的正文—label 相关性。

合同已正确停在 data Gate：`cache_authorized=false`，没有运行 cache、K=8、K=1、direct、text-CoT、因果干预或 P1 assessment。因此本次结果拒绝 P1 v1 输入，不构成 shared Boundary/recurrent core 的成功或失败证据。下一步应另立 generator-only 的 P1 v2 修复与正式规模 qualification，保持模型、训练和 G09 阈值不变；完整复核见 `docs/v2-r1r-p1-v1-data-failure-review.md`。

## 25. 已执行 P1 v2：4096 data accepted，cache infrastructure rejected

P1 v2 先用 100,000 trials 资格化 train/heldout `8192/4096` 的 98 source cells 与 14 aggregates；Q01–Q09 全 true，且同一公共 API 仍精确拒绝 P1 v1。fresh seed `2026082002` 随后生成 65,536 records；G01–G11、两次全量生成、artifact replay、watched-source 和与 label/metric 无关的 12×1024 模型子集全部通过。原敏感 ERE validation char-NB cell accuracy/upper 为 `0.29639/0.32034`，低于 `0.37087` ceiling。

cache 正式运行完成 28,672 source 与 191,144 claim entries，共 132+26 packed shards、20,563,413 FP16 hidden tokens。外层执行等待在四小时退出但未杀死子进程；子进程完成全部 banks 后，在 source audit progress callback 中得到 `OSError [Errno 22]`，正式 root 写入 `passed=false` 并以 `55C00FDF43757F5E4B958FA0E6C672AC033B3181D480859D7F314F731083E315` 封存。封存后的同一 full audit 在关闭 progress 输出时对两个 banks 全 entry 通过，证明内容有效并把根因定位到 execution telemetry。

正式合同仍停在 cache：没有运行 K=8、K=1、baselines 或 assessment，不能进入 P2。推荐另立 P1 v3 recovery，只读绑定 v2 cache seal，在全新 root 中资格化断开 stdout、双重 full audit 并复用 immutable banks；不得覆盖或重跑 v2。完整复核见 `docs/v2-r1r-p1-v2-cache-infrastructure-failure-review.md`。

## 26. 已执行 P1 v3：cache recovered，K=8 rejected

P1 v3 另立 frozen recovery 合同，没有覆盖或重判 v2 cache。telemetry qualification 的 T01–T06 全 true；cache recovery 的 R01–R09 全 true，两次 full-content audit 均覆盖 28,672 source 和 191,144 claim entries、canonical 相等、`failures=[]`，审计前后旧 root 不变。recovery 只授权训练只读复用，v2 cache 仍保持 `passed=false` 与原 seal。

随后唯一 K=8 formal 完成 2,400 updates、正式 validation/OOD/causal/intervention、probe stripping 与 evidence seal。K01–K06 false、K07/K09 true，K08 formal false；最终 `FAIL_P1_K8`。ERE/CPS validation 为 `0.24902/0.19043`，八个 OOD cells 全低于 `0.30`，causal flip 为 `0.00391/0`，zero/batch-shuffle/slot-permutation middle drop 约为零；claim accuracy `0.5`、owner-shuffle drop `-0.00195`。K08 的两个 hook shape 子项因审计器把 batch size 写死为 `2` 而误报，但即使修正也不改变 K01–K06 已失败的正式结论。

训练吞吐和数值均正常；最重要的新发现是冻结预算只提供 `38,400 / 16,384 = 2.34375` 次 episode 平均暴露，且 153,600 次 claim 观察少于 191,144 个唯一训练 claim。P0-M v5 只证明小集合高复用可记忆，没有资格化这一 full-scale 低复用训练。当前不能据此否定抽象 mixed latent core，但已否定“从 overfit smoke 直接用 2,400 updates 进入跨任务泛化”的训练合同。

K=8 FAIL 后，K=1、direct、text-CoT、assessment 和 P2 roots 均未创建。推荐另立 P1-LQ/P1 v4 training-qualification：先做 data scale × exposure ratio 学习曲线，并以 claim/owner-shuffle/heldout 的早期 Gate 判定机制是否启动；只有预算本身不足以启动时，才以等计算比较 joint-from-start 与通用 mechanism warm-start→joint。完整证据与边界见 `docs/v2-r1r-p1-v3-k8-failure-review.md`。

## 27. 已执行 P1 v4-LQ：B128 通过，S512 身份记忆失败

P1 v4-LQ 接受 v3 的核心归因：训练集 claim 诊断同样为随机，而同一模型/损失能在 P0-M 128 条固定 selection 上启动，因此需要先建立学习路径，不应直接把 full-scale 失败外推为抽象 mixed core 不成立。

活动实现保持 P1 v3 K=8 的 Boundary、匿名 slots、shared recurrence 和 latent-only readout不变，只引入通用训练课程。每族 train 通过不读取 heldout 的确定性平衡次序构造 `64 ⊂ 512 ⊂ 2048 ⊂ 8192`；四段使用同一参数连续扩展、每段重置 optimizer、claim 与跨 episode owner-contrast 辅助目标始终保留。

正式 preflight 与 B128 sealed PASS；S512 跑满 4,800 updates 后，训练 answer ERE/CPS 为 `1.0/1.0`、claim `0.962890625`、owner drop `0.408203125`，但 validation 只有 `0.234375/0.1884765625`，因此 sealed `FAIL_P1_LQ_SCALE512` 并停止，S2048/F8192/assessment roots 不存在。父任务只读诊断在未参与 S512 梯度的 train episodes 上得到 claim `0.498/0.506`、answer `0.336/0.195`，而见过 episodes 为 claim `0.939/0.965`、answer `1.0/1.0`；zero-state/zero-claim 约 `0.5`。根因是 owner-contrast 把真值未定义的跨 episode 组合强制为负，奖励 source–claim nonce 身份记忆。完整复盘见 `docs/v2-r1r-p1-v4-lq-failure-review.md`。

## 28. P1 v5：静态 paired claim 的正式失败

v5 直接删除 owner-contrast、owner-shuffle Gate 和旧 v4 CLI/tests，不修改部署 K=8 架构，也不降低 K01–K09。活动目标只保留同 episode answer CE 与同 episode、同 prefix 的 paired claim CE + ranking。长期不变量是：跨样本负例只有在组合真值有定义时才能用于训练；辅助机制必须在未参与梯度的 episode 上验收。

每族 8,192 条 train 由新 seed 按标签、pattern、reasoning budget 分层为 7,168 条 optimization 与 1,024 条永久隔离 audit。Q7168 使用 batch 32、最多 14,336 updates；必须在未见 audit 上达到 answer 每族 `0.50`、claim 每族 `0.65`、相对 zero-state drop 每族 `0.10`，且 optimization answer 每族 `0.75`。通过后才把 audit 并回 F8192，最多再训练 12,288 updates；validation 每族达到 `0.85` 后才运行原 K01–K09。完整合同与唯一命令见 `docs/v2-r1r-p1-v5-semantic-transfer-design.md`、`docs/v2-r1r-p1-v5-semantic-transfer-execution-command.md`。

唯一 Q7168 已跑满并 sealed FAIL：训练 answer ERE/CPS 为 `1.0/1.0`，隔离 audit answer 为 `0.4746/0.1846`，claim 为 `0.5/0.5`，zero-state 也约 `0.5`。F8192/assessment 未创建。正式失败与 post-hoc 梯度、小规模 claim-only、dense-claim 和 truth-flip 覆盖诊断共同把问题定位为规模化 predicate-state 信用对称，而非断图或单纯曝光不足。完整复盘见 `docs/v2-r1r-p1-v5-semantic-transfer-failure-review.md`。

## 29. P1 v6：causal temporal witness 机制资格与 assessment false negative

V6 不追求完整架构切面，而只判决可伴随架构生命周期的训练机制。它把同-prefix 的两个不同 predicate，替换为同一 predicate 在同 episode 相邻状态上的真值翻转，并与原 static signal 做等 batch、等四判断、同初始化、同 schedule 的比较。逻辑 prefix `p` 唯一映射到 source-conditioning 后的 `trajectory[p]`，内部运行 `reasoning_budget+1` 次以表示 T 次 transition 的 T+1 个状态；未引入 AST、oracle span、显式寄存器、task embedding 或 simulator state model input。

formal 固定每族 256 optimization + 128 audit、batch32、两臂各 3,200 updates。checkpoint 只按不早于 1,600 的 optimization seen 指标选择；audit 不参与选择。单臂 Gate 为 seen 每族 `0.70`、audit 每族 `0.65`、audit state-dependency drop 每族 `0.10`；temporal 另需 swapped-state drop 每族 `0.20`。static 若自己通过则优先；否则 temporal 只有在自身通过且对 static audit 每族优势至少 `0.10` 时可被选择。

formal 使用新 `SELECTION/MODEL/ORDER=2026082601/2026082617/2026082621`。前四阶段 sealed PASS；static audit ERE/CPS `0.4805/0.4868` 且 state drop 近零，temporal audit `0.7058/0.8053`、state drop `0.2058/0.3053`、swap drop `0.4117/0.6105`，预注册 temporal arm 通过。原 assessment 只因 persisted selection 的六个 integer count keys 写入 JSON 后成为 string、raw Python equality 返回 false 而 W604 FAIL；其余八 Gate 和 18,899-query content audit 均通过。原 root 保持 sealed FAIL。

## 30. P1 v6R：形式恢复 accepted，进入 integrated P1 设计

v6R 固定五个 v6 evidence-seal 与原 source snapshot，独立重放同一 selection。records、partition canonical、JSON round-trip 和 selection preimage SHA-256 全部相等；query ID、label 顺序和 count 三类篡改负控全部被拒绝；18,899 query entries/209,391 tokens 再审零 failure。R601–R610 全 true，正式 `PASS_P1_V6R_SELECTION_RECOVERY`。原 v6 assessment 不改判，只恢复 temporal-witness 机制资格。

下一步另立 integrated fresh-seed P1：不续训 v6 comparator checkpoint，在 full 8,192/族上联合 answer CE 与 temporal witness，先运行 shared K=8 及完整 validation/OOD/causal Gate，通过后才运行 K=1、direct SFT、text-CoT SFT 和 Pareto assessment。任何 K=8 首要 Gate 失败即停；当前仍为 `p1_passed=false`、`p2_started=false`。

## 31. P1 v7：integrated K=8 formal 失败与边界修正

v7 把 v6R 的机制结论放回完整 shared reasoner，但不把 comparator checkpoint 当初始化。full 16,384 个训练 episode 各出现 40 次，前 4,096 update 只训练 temporal pair，后 16,384 update 联合 answer CE；preflight 与 86,016-query cache sealed PASS，唯一 K=8 formal 跑满 20,480 updates。

validation ERE/CPS 为 `0.999023/0.851562`；K10 temporal accuracy 为 `0.766357/0.938965`，zero/swap drop 与 batch-shuffle state dependence 均通过。失败集中在 CPS OOD 与因果答案：四项主 OOD 为 `0.587891/0.754883/0.472656/0.214844`。正式 evaluator 把合法 `base/flip` 错写成 `base/counterfactual`；只读修正后 ERE/CPS pair flip 为 `0.705078/0.0078125`，仍然明确不通过。

梯度关系与干预共同支持：temporal objective 建立了过程表征，但 ordinary answer CE 在 final state 中学出近正交的表面决策路径。K04 zero-middle 在干预后仍能读取完整 source，主要测可恢复性；K06 T1 drop 是效率控制而非白皮书必要机制。两项不改写 v7 历史 Gate，却不再进入后继资格。v7 正式 `FAIL_P1_V7_INTEGRATED_K8`、三根不可重跑，K=1/baselines/P2 未运行。

## 32. P1 v8：causal-bridge 等训练预算资格合同

v8 只验证 answer path 的反事实闭包，不宣称完整 P1。每族 512 个旧 causal pair 经固定 hash 划为 384 optimization 与 128 sealed audit；ordinary arm 从 train 取 768 records/族。三臂均从同一 seed 初始化相同 K=8/T+1 模型，固定 batch32、3,072 updates、98,304 exposures、每条 record 精确 64 次。前 768 updates temporal-only，后 2,304 联训；每个 optimization episode 统一选择 3 个 witness，audit 为 2 个，总 query 10,240。

三臂变量被限制为：ordinary CE；同一 causal records 打散后的 unpaired CE；同一 causal records 的 base/flip 同批，加权 `0.5` 的双向 final-logit ranking。pair metadata 不进入前向，probe 在评估前物理删除。长度分桶 schedule 的 padding efficiency 为 `0.88313/0.88783/0.88862`。普通诊断每 cell 256 条；batch-shuffle 使用完整 1,024 条 length/horizon 并与同 cohort full 比较，预期有效覆盖 `1.0/0.99609`。

B01–B07 要求两族 pair flip `>=0.65`、causal raw `>=0.75`、temporal/state dependence、validation retention、probe strip、architecture/compute 全部通过；causal arm 相对 ordinary 两族还须各提升 `>=0.15`。正式 preflight/query-cache PASS，qualification sealed FAIL。三臂只通过 B01/B04/B07；ordinary train 复评 `1.0/1.0` 而 validation `0.25/0.1758`，paired 的 train CPS pair flip `0.5807` 明显高于 unpaired `0.1458`，audit 则仍为 `0.0078`。

根因不是 pair loss 不可优化，而是合同把 broad competence 与 causal binding 同时交给一个 768 条/族 scratch 数据集；causal 臂更完全没有 ordinary train，却被要求 retention。v8 保持正式 FAIL，但不构成架构否证。完整分析见 `docs/v2-r1r-p1-v8-causal-bridge-failure-review.md`。

## 33. P1 v8R：shared-checkpoint causal curriculum recovery

v8R 从 sealed v7 stripped checkpoint 开始，不再随机初始化基础能力。两个臂共享完全相同的 mixed batch：ordinary ERE/CPS 各 8 条，causal ERE/CPS 各 4 个完整 pair；总 batch32、3,072 updates。完整 8,192/族 ordinary train 每条恰好 3 次，v8 的 384 optimization pair/族每条恰好 32 次。

`replay_ce` 与 `replay_pair` 的初始化、样本、顺序、优化器和计算量完全一致，唯一差异是 pair-ranking 权重 `0/1`。本轮不创建 temporal probe，避免新随机辅助头再次与 answer objective 竞争；v7 temporal 正证据由 checkpoint 输入保留，训练后以 retention、causal flip、intervention 和完整性重新审计。

R01–R06 要求 zero-update v7 common start 成立、两臂同初始化/同 schedule/exact compute、paired 在 CPS optimization 上相对 CE 提升至少 `0.20` 且绝对 `>=0.70`、已揭示 audit CPS pair flip `>=0.30` 且相对最好控制提升 `>=0.20`、ordinary validation 相对 v7 每族下降不超过 `0.05`，以及 probe absence/source-gradient/architecture/seal 完整。该 audit 已被 v8 揭示，所以 PASS 只授权从未使用的 causal pair 建立 fresh P1 v9；不完成 P1，不进入 P2。

正式结果为 preflight PASS、qualification `FAIL_P1_V8R_CAUSAL_CURRICULUM`。`replay_pair` optimization ERE/CPS pair `0.9948/0.3281`，audit `0.8047/0.0078`，validation `0.9990/0.7480`；R01/R02 true，R03–R06 false。由于 shared competent start、完整 rehearsal、同 schedule 与同 mutation family 均已成立，失败不再归于 scratch 或 coverage。当前 pair loss 仍可分解为逐样本 hard-negative CE，冻结 core/readout 与线性 probe 又排除了“只换最后一层”的主要解释；剩余根因是 CPS final-decision state closure 未形成。

## 34. P1 v8D：causal-decision witness 机制资格

v8D 不继承 v8/v8R 权重，只从 sealed v7 stripped checkpoint 开始。训练期共享 probe 对 base/flip 的 `H_T` 使用同一自然语言 decision query 与相反真值；ERE/CPS 都覆盖局部答案标签，CPS 额外覆盖 unique optimum 与候选成本有向比较。全部 query 形成固定 34 条 surface，由同一 frozen Qwen final hidden 编码；query 不进入 core/readout，probe 在正式答案评估前物理删除。

probe-only 固定 Boundary/core/readout，训练 384 updates；joint 使用完整 ordinary rehearsal 与 384 causal optimization pair/族训练 3,072 updates，前 2,560 updates 冻结 answer readout，后 512 updates 解冻。损失由 answer CE、权重 `0.25` 的 coupled answer-logit difference-of-differences、权重 `1.0` 的 paired decision CE+rank 组成。D03 先要求隔离 audit 上的 decision judgment accuracy、zero-state 与 swapped-state dependence；D04 再要求答案真正消费该状态；D06 用 probe-only 差值、物理剥离与 FP32 source gradient 区分机制形成和测量假阴性。该 split 已揭示，PASS 只授权另立 fresh-seed P1 v9，不完成 P1/P2。

正式 preflight/query-cache PASS，qualification sealed `FAIL_P1_V8D_CAUSAL_DECISION_WITNESS`。joint audit decision ERE/CPS 为 `0.8887/0.5143`，CPS zero/swapped drop 仅 `0.0143/0.0286`；答案 audit ERE raw/pair 为 `0.8984/0.7969`，CPS 为 `0.375/0`，validation 仍为 `0.9990/0.8193`。post-stop 只读定位显示 Boundary 与 `H_T` 不但看见 perturbation，v8D 还把 CPS source delta 放大到 `H_T`，但跨 pair 线性诊断仍接近随机。正式分类是 `causal_perturbation_detected_but_semantic_reduction_not_formed`；完整复核见 `docs/v2-r1r-p1-v8d-causal-decision-witness-failure-review.md`。

## 35. P1 v8L：fixed-anchor causal-state ladder

v8L 是当前匿名 K=8 主线最后一个 development-mechanism qualification。从 sealed V7 stripped checkpoint 重启，不继承 v8D 权重；彻底删除可学习 ClaimProbe，以 fresh simulator replay 在正确 `H_t` 建立 CPS `cost trace → final cost → cost order → unique optimum → choice` 和 ERE `semantic transition → semantic final → choice` 梯子。

冻结 Qwen 与冻结 V7 Boundary 分别编码每组互斥 claim。两个 query 按 hash 排序后构造 `normalize(q_left-q_right)`，方向不读取 truth label；cache 含 2,121 raw queries、1,448 个去重 contrast directions 和 4,096 个 pair-level contrast instances，不拟合 audit statistics。预正式诊断先拒绝了 effective-rank `5.62` 的中心化单 query 方案；contrast 方案达到 `25.61`，96-update 不落盘 probe 又把固定 batch direction accuracy 从 `0.375` 提到 `0.594`。这些只属于设计预测试，不是 formal 结果。

正式路径为 384-update ladder bootstrap 和 3,072-update mixed joint closure。`final_norm` 始终冻结，最后 512 updates 只解冻 pooling query 与 answer head；checkpoint 参数集合与 V7 完全一致。L01–L07 同时要求 teacher/cache truth、exact compute、audit 五层 CPS state transfer、答案因果翻转、ordinary retention、无辅助架构完整性与 successor stop。PASS 只授权未见 seed 的 P1 v9；FAIL 终止当前 K=8 mainline，不再建立 v8 后继。完整合同见 `docs/v2-r1r-p1-v8l-causal-state-ladder-design.md`。

唯一 formal 的 preflight 与 anchor-cache sealed PASS，qualification 在 384-update bootstrap sealed `FAIL_P1_V8L_BOOTSTRAP`。ERE optimization/audit fixed-anchor direction 为 `0.7726/0.7109`，CPS 只有 `0.5682/0.5281`；CPS optimization 五层均未达到 `0.65`，所以 joint、答案迁移、ordinary retention 和最终 L03–L06 没有运行。三个 formal seals 为 `06BA4E18...5115`、`5F1AABBF...3C14`、`3332CD3D...FE75`，且没有 successor root。

只读分层复核发现，预注册 global anchor effective-rank Gate 没有单独资格化 CPS 数值 measurement：ERE semantic-transition/final effective rank 为 `39.89/33.76`，CPS cost-trace/final-cost 只有 `3.47/3.84`；按数值大小重定向后 shared-axis alignment 也只有 `0.340/0.380` 并包含反向方向。因此 v8L 关闭的是“匿名 shared K=8 + lexical metric teacher”的组合，不是 mixed core 或白皮书总否证。继续路线前必须另立 numeric/relation decision-power qualification 与 mixed/typed-core 架构合同；完整复核见 `docs/v2-r1r-p1-v8l-causal-state-ladder-failure-review.md`。

## 36. V8L 后的 P1 总合同：NR1 → H1 → F1

用户于 2026-08-17 授权从 V8L 结论继续尝试关闭 P1。该授权不追溯改写 `FAIL_P1_V8L_BOOTSTRAP`，也不允许补跑 bootstrap/joint、复用 fixed roots、降低 Gate 或建立 P1 v9。新路线直接切换为三个相互授权的阶段：

1. `P1-NR1` 只资格化 numeric/relation measurement system；
2. `P1-H1` 只在 NR1 PASS 后比较 anonymous shared core 与 task-independent mixed/typed core；
3. `P1-F1` 只在 H1 PASS 后以 fresh seed 运行 K=8/K=1/direct/text-CoT 完整 falsification、因果与成本 Gate。

NR1/H1 PASS 都不完成 P1；只有 F1 的完整 conjunction 才能设置 `p1_completed=true`。任一阶段 FAIL 都原样停止，不自动建立 successor，不用后验诊断改判。

H1 的目标 mixed state 遵守白皮书 `S_t=(A_t,H_t)`：`A_t` 只提供 opaque identity、类型、来源、阶段和合法寻址，`H_t` 保留连续语义 payload。handle/type/route 必须从 source content 学得；task id、oracle role/span、candidate index、答案、simulator state 和任务专属 transition 都不得进入 forward。numeric/relation expert 只能作为同宽 residual 上的 learned/content-routed 计算归纳偏置，不能成为程序旁路。shared 与 typed 两臂必须使用同 fresh data、同 teacher、matched active compute/token/FLOPs 与同答案头约束。

## 37. P1-NR1：numeric/relation measurement qualification

NR1 不加载 Qwen、不训练模型。numeric case 只含 opaque candidate handle 与 integer deltas；measurement 以迭代累计建立 prefix/final scalar，再以 strict minimum 导出 winner。relation case 只含 opaque handles、DAG direct edges 与无答案 query；measurement 以 fixed-point closure 导出 reachability。独立 oracle 分别使用 `itertools.accumulate` 和逐 source/query BFS，不得与 measurement 互相 import。另冻结由外部对话手算的 6 个 numeric、6 个 relation 与 11 类 fault target；measurement 与 reference 都必须匹配该 bundle，不能只在共享 generator 上互证。

正式 qualification/heldout 分别含 numeric `384/256`、relation `384/256`。heldout 的 candidate `6–9`、horizon `7/9/11`、delta magnitude `64–1024` 与 relation handles `9–16` 全部超出 qualification；fingerprint overlap 必须为零。关系集不再是固定 chain/four-query pattern，而是覆盖 branch、merge、多路径、多组件、irrelevant/redundant edge、4–8 query 与多种真值位置。正控覆盖共同平移、正比例缩放、候选/edge/query permutation、handle rename 与 transitive redundancy。fault matrix 的 target 由独立 oracle manifest 指定，注入器不得读取 measured winner/closure；全部 state detection 为 `1.0`，全部 required metric kill 与 decision-affecting fault kill 各 `>=0.80`。

N01–N07 同时关闭历史 seal、合同/source/Git/snapshot identity、schema/handle/import/fixture boundary、numeric、relation/topology、fault/holdout 和 process/transport/artifact integrity。CLI 只暴露唯一 formal 命令；runner 在任何 fixed root 写入前检查 root freshness，preflight 非 sealed PASS 时不会创建 qualification。PASS 状态只能是 `p1_h1_design_authorized=true`、`p1_completed=false`、`p2_eligible=false`；完整冻结合同见 `docs/v2-r1r-p1-nr1-numeric-relation-measurement-design.md`。

## 38. P1-NR1 正式结果与 H1 入口

NR1 唯一 formal 已执行一次。preflight 与 qualification 均 PASS，两根 seals 为 `1825282BF46C821BFD89636B58DCCDAD53CAAE1C9F226BA87B3CE8387B095DAF`、`DADBDDBF3B672702A4FFADE74504EAD2F941628D729B5A29B0B7D4DA18EF6FB3`；N01–N07 全 true。numeric/relation qualification/heldout 的注册 exact 指标和全部 metamorphic 均为 `1.0`，fingerprint overlap 为 0；11 类 formal fault detection 全为 `1.0`，所有 decision-affecting fault kill 为 `1.0`。详细主审见 `docs/v2-r1r-p1-nr1-main-review.md`。

该 PASS 只关闭 measurement surface，不证明任何模型学习或架构优越性。当前唯一新增授权是另立 H1：在 fresh data 和合格 measurement teacher 下，以 matched active compute/token/FLOPs 比较 anonymous shared core 与 content-routed mixed/typed core。H1 必须保留 frozen Qwen/learned Boundary 输入、禁止 task/oracle/candidate-index 路由与 simulator/答案旁路，并把 deployment forward 与 teacher/audit 完全分开。H1 PASS 只授权 F1；`p1_completed` 与 `p2_eligible` 继续为 false。
