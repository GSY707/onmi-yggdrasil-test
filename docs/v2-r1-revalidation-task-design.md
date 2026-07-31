# V2-R1R 跨任务重验证：冻结设计与执行合同

日期：2026-08-01

文档地位：R1 重验证的唯一实现级设计。本文由主设计层冻结模型、数据、监督、训练、评测和停止规则；执行 agent 只能忠实实现与运行，不能自行改变结构、降低 Gate 或把 smoke 升级为正式证据。2026-08-01 主设计层复核否决了 generator v1 的机器自判通过，现以第 16 节的 P0-D v2 修订覆盖与其冲突的旧实现细节。

证据状态：P0-D v1 已生成代码和 artifact，但因结构化捷径、split 污染、错误 claim 与审计缺口被主设计层判为失败诊断；它不是有效 P0 证据。P0-D v2 修订已冻结，P0-M、模型、cache 与训练继续禁止。A1.20D 只保留为历史机制正控制，不作为本轮初始化、数据源或实现基座。

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

## 16. P0-D v2 主设计层修订

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
