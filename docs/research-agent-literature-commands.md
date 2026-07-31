# Project-Yggdrasil V2 Google Deep Research 研究命令

日期：2026-07-14

本文提供专门面向 Google Gemini Deep Research 的可重复研究提示词和三天冲刺顺序。目的不是生成一份宽泛的“相关工作综述”，而是趁研究额度仍可用时，建立覆盖当前故障、后续架构和未来产品化问题的可核验资料库。研究结果是决策输入，不自动成为架构规范或实验事实。

路线状态说明（2026-08-01）：本文的 A1-A4 与部分 B5 标签是 2026-07-14 时的历史专题分组，不再代表当前阶段顺序。当前路线把主动 I/O、工作树/记忆、异步节点、离线专家晋升与整机评测统一聚合为 V2-C；这些命令仍可用于文献检索，但正式实验边界以 [`v2-c-hierarchical-full-duplex-agent-experiment.md`](v2-c-hierarchical-full-duplex-agent-experiment.md) 和 [`next-stage-test-plan.md`](next-stage-test-plan.md) 为准。

## 1. 当前问题与研究优先级

当前 A1.5 已证明 shared recurrent transition、Qwen hidden→latent 接口和 learned K=8 workspace 可以在 ordinary split 上训练成功，但没有通过 Gate：

- composition-heldout 的 final/state 仍约为 `0.2773/0`，说明未见 operation bigram 不能可靠组合；
- same-answer shuffle 暴露样本身份依赖，模型没有形成稳定的操作语义与状态绑定；
- length-heldout final answer 可达 `1.0`，但逐步 state full exact 只有 `0.6484`，说明最终答案与忠实状态轨迹分离；
- 旧路线中的 K/T sweep、弱过程监督和 step-level verifier RL 已经失败或发生策略塌缩，不能作为默认修复手段。

因此，研究优先级固定为：

1. operation-composition binding 与变量/角色绑定；
2. 长度外推与逐步状态忠实性；
3. frozen hidden 到多槽 latent workspace 的信息保真；
4. 因果过程监督、验证器和审计方法；
5. 连续潜变量推理的整体证据；
6. 任务与 split 如何真正测出组合泛化；
7. V2-A 通过后才需要的 Boundary-MoE 与多模态融合。

## 2. Google Deep Research 使用方式

每次运行时，把“共同前缀”和一个专题命令合并成一次 Deep Research 请求。上传以下四份文件，并保留 Google Search 来源：

- `docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md`
- `docs/next-stage-test-plan.md`
- `docs/v2-a1.5-latent-foundation.md`
- `docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md`

提交后先检查 Gemini 生成的研究计划，再点击开始研究。若计划没有覆盖原始论文、官方代码、负结果、复现和项目映射，先编辑计划，不要直接开始。每份报告完成后立即导出到 Google Docs，并另存一份 Markdown 或纯文本；账号是否长期保留历史报告不应成为资料保存前提。

报告命名统一为：`YGG-V2-D<1|2|3>-T<01..16>-<主题>-<YYYYMMDD>`。在一个总索引中记录报告名、专题、运行时间、附件版本和导出链接，避免后续综合时遗漏。

第一批建议分别运行专题一至六。第二批运行专题七及未来专题。第三批把前两批报告交给“反证命令”，专门寻找失败复现、相反证据和适用边界；最后把全部报告交给“综合决策命令”。

研究智能体再次运行同一专题时，必须附上上一轮报告，并使用本文的“增量复查命令”；否则多轮运行很容易重复返回同一批高引用论文。Google Deep Research 的日请求数和并发数按账号动态限制，以界面实际显示为准，不在本文假设固定额度。

### 2.1 三天冲刺顺序

以“今天”为 Day 1。不要等全部报告完成后才导出；每完成一份就立即留存。

| 档位 | 总运行数 | 三天安排 |
| --- | ---: | --- |
| 最低档 | 8 | Day 1 跑专题一、二、三、五；Day 2 跑专题四、六；Day 3 跑一次红队和一次综合。 |
| 推荐档 | 16 | Day 1 跑当前专题一至六；Day 2 跑专题七至十三；Day 3 跑两次独立红队和一次综合。 |
| 余量档 | 16 以上 | 先完成推荐档；剩余额度用于专题十四至十六、增量复查，或把证据冲突最大的专题交给第二个独立智能体重做。 |

同一天允许并发多少次由界面决定。若并发额度有限，优先让专题一、二、三和五先开始；这些报告最直接影响 A1.5 是否值得更换结构。

操作依据：[Gemini Apps Help：Use Deep Research](https://support.google.com/gemini/answer/15719111?hl=en)。官方说明当前支持在请求中上传文件、选择 Google Search/Drive 等来源、开始前编辑研究计划以及导出到 Google Docs；历史报告能否从 Recent 找回还依赖 Keep Activity 设置，因此本冲刺要求立即导出。

## 3. 所有专题共用的前缀

```text
你是 Project-Yggdrasil V2 的文献研究智能体。请进行实时、可核验的学术检索；不要仅依靠模型记忆。当前日期为 2026-07-14。

项目目标：比较显式文本思维链与连续 latent recurrence，并在成熟冻结文本基座之上建立可因果审计的多向量 latent workspace。只有 V2-A 通过质量—成本 Pareto、heldout、长度外推和因果忠实性门禁后，项目才进入多模态 Boundary-MoE。

当前实现摘要：冻结 Qwen3.5-2B hidden；P1 hidden→latent 接口；P2 使用 K=8 learned slots、共享 recurrent transition、每个 operation span 驱动一次状态更新，最终答案头只读取最终 latent。ordinary split final/state=1.0/1.0，但 composition-heldout 约为 0.2773/0，same-answer shuffle 失败；length-heldout final/state=1.0/0.6484。旧 K/T sweep、浅层 loss 调权和一次 step-level verifier RL 没有闭环，RL 还出现 policy collapse。

你的任务不是堆论文标题，而是找出能解释这些结果、并能转化为结构替换、训练目标或判伪实验的工作。遵守以下规则：

1. 优先原始论文、作者项目页、正式会议页面和官方代码仓库；综述只用于发现线索，不能替代原始来源。
2. 核验每篇论文的准确标题、作者、年份、venue、稳定链接和代码链接。无法核验的信息明确标为“未核验”，不要补全猜测。
3. 同时搜索奠基工作、2023—2026 的新工作、负结果、复现研究和相邻领域。不要只找名字包含 latent reasoning 的论文。
4. 严格区分三类证据：直接证据、相邻机制证据、推测类比。论文作者没有验证的结论不得写成已证实。
5. 对每项工作说明：研究问题、核心机制、训练信号、关键消融、数据与规模、成立结果、失败边界、与本项目的结构对应、可借用部分、不可直接外推部分。
6. 重点检查其提升是否来自更大模型、更多计算、teacher trace、答案旁路、额外数据或更宽状态，而不是所宣称的推理介质。
7. 不要把“增加 K/T、延长训练、轻调 loss 权重、加入普通 probe”当作主要建议，除非文献给出明确机制和判伪证据。
8. 所有设计建议必须落到最小实验：修改什么结构、删除/替换什么旧设计、固定哪些控制变量、与什么强基线比较、用哪些消融判定成功或失败。
9. 用中文写报告，术语首次出现时保留英文。不要把本项目当前 probe 写成架构已成立。

输出结构固定如下：

A. 核心判断：先用不超过 400 字回答这个专题最重要的结论。
B. 检索方法：数据库、关键词族、时间范围、纳入/排除标准。
C. 证据矩阵：至少 12 篇高相关工作；每篇一行，包含来源链接、证据类别、机制、关键结果、失败边界、与 Yggdrasil 的对应关系。
D. 深读：选择最有用的 5 篇，解释到足以指导架构或实验，不只复述摘要。
E. 冲突与负证据：列出结论互相冲突之处、失败复现、规模依赖和未解决问题。
F. 可执行候选：最多 3 个，按“预期信息增益/实现成本”排序。每个候选给出结构变化、训练目标、最小对照、因果消融、Gate 和停止条件。
G. 不建议采用：列出看似相关但会重复本项目已失败路径的方案及原因。
H. 可机器合并清单：最后给出 JSON 数组，每篇论文包含 title、year、url、code_url、evidence_class、topic_tags、project_use、confidence。
```

## 4. 第一批专题命令

### 专题一：操作组合、变量绑定与系统性泛化

```text
专题：如何让 recurrent latent workspace 学到可组合的 operation，而不是记住 operation bigram 或样本身份。

重点检索 systematic compositional generalization、variable/role binding、neural program execution、neural algorithmic reasoning、algebraic state update、equivariant slot/state representation、tensor-product or vector-symbolic binding、modular recurrent networks，以及在 unseen operator composition 上的失败研究。

必须回答：
1. operation 应如何表示为可复用的状态变换，而不是普通 learned embedding 或序列模式？
2. source、target、operation family 和 value 的角色绑定如何避免被位置或样本身份替代？
3. 哪些工作真正测试了“训练见过单个操作，但没见过相邻组合”的泛化？
4. 命名 register slots、无名 permutation-equivariant slots、显式关系边和函数式 delta update 各有什么证据？
5. 对当前 P0/P2，最值得直接替换的结构是什么？不要给兼容补丁；给出旧 operation-conditioning 路径应删除的边界。

把候选实验固定在同一 A1.5 schema 上，必须保留 ordinary/composition/length、same-answer shuffle、operation counterfactual 和逐步 state exact。
```

### 专题二：长度外推、循环深度与状态轨迹忠实性

```text
专题：为什么 recurrent 模型能给出正确 final answer，却不能保持逐步 state 正确；哪些结构能在更长未见轨迹上执行同一算法。

重点检索 length generalization、algorithmic extrapolation、weight-tied/looped Transformer、Universal Transformer、recurrent depth、neural execution engines、state consistency、teacher forcing/exposure bias、adaptive computation，以及对 intermediate state fidelity 的因果评估。

必须回答：
1. 哪些架构在训练短序列、测试长序列时真正保持逐步状态，而非只保持最终标签？
2. shared transition 的哪些归纳偏置有利于外推，哪些 positional/step embedding、normalization 或 readout 会制造长度 shortcut？
3. 如何区分“算法执行”与“模板答案映射”？
4. state CE、rollout consistency、cycle/semigroup consistency、multi-step prediction、scheduled sampling 或其他目标，各自有什么直接证据和副作用？
5. 应如何设计跨长度干预，使 final answer=1.0 但 state 错误不能通过 Gate？

给出一个只改变 transition/state objective 的最小实验，以及一个结构性更强但实现成本更高的直接替换实验。
```

### 专题三：冻结语言模型 hidden 到多槽 latent workspace 的信息保真

```text
专题：如何从冻结语言模型 hidden states 建立保留实体、角色、操作和关系的多槽 latent workspace。

重点检索 latent bottleneck、Perceiver/Resampler、Q-Former、set/slot representation、token compression、recurrent memory、entity-centric representation、object files、relational bottleneck、cross-attention interface，以及 frozen foundation model hidden 的适配研究。

必须回答：
1. 单次 pooled operation span 为什么可能保留 operation family 却丢失 source/target 绑定？
2. learned unnamed slots 何时形成置换等变集合，何时只是样本指纹缓存？
3. named slots、slot attention、entity tokens、relation tokens、key-value binding 与动态读源各有什么证据？
4. 如何在不允许答案旁路的情况下测量 hidden→latent 的信息充分性？线性 probe 为什么不够？
5. 对 P1/P2，哪些接口应该被直接替换，而不是在现有 token pooling 上继续叠 adapter？

候选必须包含信息保真 Gate：受控反事实、same-answer 不同程序、实体/角色置换、source-target swap、hidden span shuffle、latent slot permutation 和下游状态重算。
```

### 专题四：过程监督、验证器、因果审计与 RL 稳定性

```text
专题：如何监督和审计连续 latent trajectory，使中间状态具有因果意义，并避免 verifier/RL 学到奖励捷径或策略塌缩。

重点检索 process supervision、process reward models、step-level verifier、outcome vs process reward、causal representation learning、causal mediation/intervention、faithful chain-of-thought、probe reliability、latent decoder faithfulness、self-critical policy gradient 和 reward hacking/collapse。

必须回答：
1. 中间 state decoder 的高准确率在什么条件下仍可能只是可读相关性，而不是状态被主路径使用？
2. 什么干预能证明 trajectory 的某一步对后续状态和答案有方向一致的因果作用？
3. 稀疏 verifier reward、低熵策略和 self-critical baseline 为什么容易塌缩？有哪些经验证的稳定替代？
4. 对已知符号状态任务，监督确定性 transition、约束一致性或对比反事实是否比 policy gradient 更合适？
5. audit readout 如何避免事后合理化，且不成为训练时答案旁路？

不要默认推荐 RL。只有当论文证据表明 RL 解决了与本项目同构的问题时才把它列为候选，并给出 entropy、reward decomposition、off-policy/teacher mixing 和失败停止门槛。
```

### 专题五：连续潜变量推理与循环推理介质的总体证据

```text
专题：连续 latent recurrence 相对显式文本 CoT 是否已经形成可信的质量—成本或泛化优势，什么条件下成立，什么条件下失败。

重点检索 latent reasoning、continuous chain of thought、implicit reasoning、recurrent/looped Transformer、latent thoughts、hidden-state recurrence、test-time compute in latent space、token-free reasoning，以及公开复现或批评研究。搜索具体方法名，也搜索更早的循环深度和隐式迭代工作。

必须回答：
1. 哪些工作是真正连续递归，哪些只是压缩文本 token、蒸馏 CoT、隐藏 token 或增加内部计算层？
2. 是否使用成熟基座、是否冻结基座、是否需要 teacher CoT、是否有答案旁路？
3. 成本比较是否同时计入 latent transitions、状态宽度、KV/激活、decoder/audit 和训练成本？
4. 是否有同基座、同数据、同预算、多 seed、heldout 和长度外推证据？
5. 哪些结果能支持 V2-A，哪些结果只能作为相邻机制证据？

建立一张“方法拓扑表”，按推理介质、状态形状、transition、监督、输出、审计、成本口径和公开代码比较。最终判断必须允许结论为“现有文献仍不足以支持项目假设”。
```

### 专题六：组合泛化任务、split 与泄漏防护

```text
专题：如何设计一个不会被模板、样本身份、答案分布或局部前缀捷径破解的操作组合与长度外推基准。

重点检索 SCAN、COGS、CFQ、CLUTRR、PCFG/algorithmic tasks、neural program execution benchmarks、compositional split construction、counterfactual data、shortcut learning、same-label contrast sets、causal scrubbing，以及这些基准的已知缺陷和后续修订。

必须回答：
1. 当前“训练排除 swap→copy，composition 只含该 bigram”的设计还可能泄漏什么？
2. 如何构造 same-answer/different-program、same-program/different-answer、角色置换、操作重命名和最小对立样本？
3. prefix-N、n-gram、bag-of-operations、length、answer-frequency 和 sample fingerprint 基线应该怎样实现？
4. 哪些 split 能把组合泛化与长度泛化、词汇泛化、格式泛化分开？
5. 什么样的训练/测试矩阵足以证明一个 transition 是可组合的，而不是只对单个 heldout bigram 特化？

输出一个 A1.5 v2 数据合同草案，但不要直接宣布替换当前 schema。草案必须包含生成规则、泄漏审计、弱基线、反事实、Gate 和使候选架构失败的压力测试。
```

### 专题七：Boundary-MoE 与多模态边界融合（低优先级）

```text
专题：异构文本/视觉/动作专家如何在边界投影到统一 latent workspace，并与核心内部 FFN-MoE 分工。

这是 V2-B 储备研究，不得把建议写成当前 A1.5 的修复，也不得建议在 V2-A 未通过时提前实现。

重点检索 multimodal adapters、Perceiver Resampler/Q-Former、late/early fusion、mixture of modality experts、expert-choice/token-choice routing、multimodal conflict and missing-modality robustness、tool/action token interfaces、FFN-MoE routing 和负载均衡。

必须区分：
1. Boundary router 选择由哪个外部专家编码/输出，与 FFN router 选择 latent token 内部参数路径；
2. 原始专家 token、压缩 pump 和 residual/bypass 的信息保真；
3. 强制语言化、早期拼接和统一 latent 边界三类基线；
4. no-text、no-image、shuffled-image、冲突模态、缺失动作候选的因果门禁；
5. 模态路由收益与总参数/active compute 收益。

输出只需形成未来证据地图和 V2-B0/B1 的预注册问题，不给当前代码修改建议。
```

## 5. 未来阶段研究命令

以下专题用于提前建立资料储备，不代表当前可以跳过 A1.5 Gate 开始实现。除专题八和九外，报告不得给当前代码修改建议。

### 专题八：V2-A3 自然语言审计读出与因果忠实性

```text
专题：如何把连续 latent trajectory 按需读成自然语言审计记录，同时证明读出忠实而非事后合理化。

重点检索 faithful explanation、rationalization、causal scrubbing/mediation、concept bottleneck、latent-to-text decoder、amortized interpretability、chain-of-thought faithfulness、process probing 和 intervention-based interpretability。

必须回答：
1. audit decoder 只读取 H1…HT 时，如何避免从最终状态反推一个听起来合理的过程？
2. 训练 readout 所需的监督来自哪里，teacher trace 会带来什么虚假忠实性？
3. 删除、替换、交换某个 latent step 后，文本审计和任务结果应满足什么方向一致关系？
4. 哪些指标能超越 BLEU、语义相似度和 probe accuracy？
5. audit 是否应冻结 reasoner 后单独训练，何时允许低权重联合训练？

输出 V2-A3 的预注册实验合同，包括允许输入、禁止旁路、因果干预、忠实性 Gate、失败类型和最小人工审计方案。
```

### 专题九：V2-A4 的公平计算、Pareto 与正式统计

```text
专题：如何公平比较显式文本 CoT 与连续 latent recurrence 的质量、成本、稳定性和能力保留。

重点检索 test-time compute accounting、inference FLOPs、KV/activation memory、latency benchmarking、paired statistical tests、multi-seed deep learning evaluation、Pareto frontier、energy/cost measurement、capability retention 和 benchmark contamination。

必须回答：
1. text sampling、latent transitions、状态宽度、audit decoder、缓存 hidden、训练摊销分别如何计费？
2. 同参数、同 active compute、同 wall-clock、同质量四种比较口径各有什么偏差？
3. 3 seeds 是否足够，什么情况下需要 bootstrap、paired test 或更多 seeds？
4. 如何报告 best/latest checkpoint、失败运行、中断和超参选择成本，避免只挑最优结果？
5. 原始文本能力 retention 和 hidden bypass 应如何验证？

输出一份 architecture-formal 计量规范和结果表 schema，可直接用于未来 V2-A4 预注册，但不评价当前 A1.5 已通过。
```

### 专题十：不可单模态破解的文本—视觉—动作任务

```text
专题：为 V2-B 选择真正需要文本约束、视觉事实和动作候选共同参与的任务与数据。

重点检索 multimodal compositional reasoning、embodied/GUI agents、visual tool use、document agents、conflicting modalities、missing-modality robustness、counterfactual image-text pairs 和 action legality benchmarks。

必须回答：
1. 哪些公开任务能被 OCR、caption、语言先验或单一模态捷径破解？
2. 如何构造同文本不同图像、同图像不同约束、冲突模态、缺动作候选和非法动作 hard negatives？
3. 视觉先语言化、早期拼接和直接视觉 latent 应如何公平比较？
4. 哪些任务可在单卡/缓存特征条件下先做 surrogate，哪些必须用真实环境？
5. 如何同时评价任务成功、动作合法性、证据归因和成本？

输出 V2-B0 候选任务排名及数据权利、环境维护、算力和泄漏风险。最多保留两个主任务和一个备选。
```

### 专题十一：FFN-MoE、双层路由与主动专家调用

```text
专题：在 Boundary-MoE 之外，核心 FFN-MoE 和 READ/EMIT 专家调用如何提供可归因的条件计算收益。

重点检索 sparse MoE、expert-choice/token-choice routing、load balancing、router collapse、capacity factor、conditional computation、tool routing、adaptive retrieval、active perception、budgeted inference 和 abstention/recovery。

必须回答：
1. Boundary router、FFN router 和工具/输出调用策略为什么必须分开？
2. 小规模 4-expert top-1 MoE 在数据较小、专家异质性高时的主要失败是什么？
3. 如何设计 Boundary/Dense、Boundary/MoE、plain/Dense、plain/MoE 的 2×2 因果对照？
4. 从 teacher-specified READ/EMIT 过渡到自主调用时，如何处理空结果、超时、错误调用和恢复？
5. active compute 收益如何排除总参数和训练数据增益？

输出 V2-B3 静态 FFN-MoE 与 V2-C C3 主动系统 Boundary 的研究地图、路由日志 schema、collapse 指标、故障注入和停止条件；不得把二者混为同一层路由。
```

### 专题十二：高保真 I/O 专家、引用句柄与输出仲裁

```text
专题：latent core 如何驱动图像/文档/音频等高保真生成与编辑，同时允许受控 reference bypass，不把原始细节压进小 latent 瓶颈。

重点检索 multimodal generation/editing controllers、reference-conditioned generation、identity/content preservation、latent diffusion adapters、pointer/reference handles、structured constraints、tool output arbitration、calibration 和 provenance-aware generation。

必须回答：
1. intent latent、reference handle、constraints 和原始媒体应如何分离？
2. 什么是合法的高保真旁路，什么会成为绕过推理核心的答案通路？
3. no-source、wrong-source、shuffled-reference、foreground/background、identity 和 constraint fidelity 如何测？
4. 多个输出专家产生候选时，arbiter 如何校准、拒绝和保留来源？
5. 输出专家为什么不能自行调用输入专家，权限和成本记录如何保持？

输出 I/O 专家合同、候选表 schema、授权旁路边界和从文本/动作到图像编辑的扩展顺序。
```

### 专题十三：工作树、记忆树、Provider KV 与跨窗口恢复

```text
专题：把单回合 latent core 接入可恢复的长程认知运行时，而不把上下文窗口、工作记忆、长期记忆和 provider KV 混为一体。

重点检索 hierarchical task memory、episodic/semantic memory for agents、external memory、event sourcing、checkpoint/replay、retrieval for long-horizon agents、context compression、cache recomputation、multi-agent shared memory、provenance 和 memory poisoning。

必须回答：
1. 工作树应保存哪些充分状态，哪些短期 latent/KV 不应持久化？
2. 子分支向父级返回怎样的结论、证据、前提、不确定性和重读入口？
3. 跨进程恢复如何测目标保持、证据覆盖、重复工作、污染和遗漏？
4. provider KV 的缓存、重算和丢弃如何与语义恢复分开验证？
5. 多 Agent 应共享什么最小充分基线，如何避免复制错误或完整短期现场？

输出工作树/记忆树信息合同、故障注入矩阵、无工作树/完整上下文基线和长任务 Gate。不要假定“节点保留”等于 KV 无损恢复。
```

### 专题十四：离线专家晋升、持续模块化学习与资源自适应

```text
专题：如何根据真实能力缺口离线训练新 Boundary/FFN expert，并以可授权、可灰度、可回滚的方式加入系统；资源下降时如何显式退化。

重点检索 modular continual learning、progressive networks、adapter/expert addition、expert specialization、catastrophic forgetting、model merging limits、canary evaluation、safe model updates、anytime prediction、adaptive computation 和 graceful degradation。

必须回答：
1. 新专家如何在不修改身份、真理标准和旧专家的条件下增加能力？
2. 能力缺口提案、隔离训练、独立 heldout、安全回归、授权和 manifest 应如何闭环？
3. router 如何学会使用新专家而不造成旧能力回退或专家塌缩？
4. 读取、latent steps、审计和渲染预算如何独立控制？
5. 资源耗尽、降级、拒绝和回滚应怎样成为显式语义，而不是静默降质？

输出专家晋升协议、版本 manifest、canary/rollback Gate 和资源—质量曲线的预注册方法。
```

### 专题十五：完整架构场景、故障注入与 Agent 评测

```text
专题：如何用一条真实长程任务同时验证文本目标、视觉/文档证据、工具动作、工作树恢复、按需专家调用、高保真输出和因果审计。

重点检索 end-to-end agent benchmarks、long-horizon task evaluation、GUI/web/document agents、fault injection、recovery benchmarks、human intervention metrics、provenance evaluation 和 integrated multimodal systems。

请设计至少三个候选场景，并分析可复现环境、数据权利、自动 verifier、失败成本、运行时长和单卡可行性。最终只保留一个最适合 V2-C C5 的架构完整版场景，给出 direct、text-CoT、single multimodal model、Flat、Tree-sync 基线，以及 OOD、故障、恢复、成本和旁路审计。
```

### 专题十六：商用场景、安全、数据治理与单位经济

```text
专题：哪些垂直场景真正需要 Yggdrasil 的多模态、长任务、按需读取、审计或高保真输出优势；把研究系统变成产品还缺哪些安全与工程证据。

这是一项技术与市场联合研究，不要求当前实现产品。重点检索 2025—2026 的企业工作流、采购与部署报告、可靠性研究、agent security、prompt injection、malicious media、tool authorization、multi-tenant data governance、AI observability、serving economics、human-in-the-loop 和监管/许可要求。

必须回答：
1. 候选用户、购买者、现有工作流、失败成本和可程序验证目标是什么？
2. 普通强多模态模型或现有 agent 产品能否以更低成本完成？
3. 合法数据、模型/媒体许可、租户隔离、保留/删除、地域和审计要求是什么？
4. prompt injection、工具越权、恶意文档/媒体、供应链和密钥风险如何进入威胁模型？
5. 单位任务成本、人工接管率、延迟、SLO、毛利和客户续用应如何验证？

输出不超过三个候选垂直场景、淘汰理由和一份 P0/P1 证据缺口图。区分公开事实、行业估计和你的推断；涉及法规时必须链接当前官方来源并注明司法辖区和日期。
```

## 6. 第三批：增量、反证与综合命令

### 增量复查命令

```text
这是同一专题的第 N 轮增量研究。下面附有上一轮报告：

<PASTE_PREVIOUS_REPORT>

不要重做上一轮综述。先从上一轮 JSON 清单建立排除集，除非需要纠正元数据或补充关键负证据，否则正文不得重复已有论文。执行以下增量任务：

1. 对最关键 5 篇做 backward/forward citation snowballing；
2. 找 2024—2026 后续工作、正式出版版本、作者代码、issue、复现和失败报告；
3. 针对上一轮最强的 3 个机制，各找至少一个支持证据和一个反对/限制证据；
4. 补齐上一轮未核验元数据与缺失的训练规模、消融、成本或 heldout 结果；
5. 判断新证据是否改变候选排序。若没有改变，明确写“排序不变”及理由；
6. 输出新增论文 JSON 和一份去重后的总清单，不得把同一论文的 arXiv/会议版本算作两篇。
```

### 反证与红队命令

```text
你是独立的文献红队。不要继续支持已有设计，目标是找出下列研究报告中最可能错误、过度外推或忽略负结果的地方：

<PASTE_REPORTS_OR_SYNTHESIS>

请实时检索原始来源和复现证据，逐项检查：

1. 所谓组合泛化是否其实来自词汇、模板、答案分布或更大计算预算；
2. intermediate state/probe 是否只是可读而非被主路径因果使用；
3. latent reasoning 是否暗藏文本 teacher、离散 token、答案旁路或昂贵 decoder；
4. length extrapolation 是否只在单一算法、短长度或特殊位置编码上成立；
5. 方法是否依赖远大于本项目 8GB 本地预算的预训练/数据规模；
6. 是否存在作者未强调的失败 split、消融反转、复现失败或代码与论文不一致。

输出“被推翻、被削弱、仍成立、证据不足”四类结论。为仍成立的候选设计最强判伪实验；不要提出新的宽泛综述。
```

### 综合决策命令

```text
你是研究负责人。下面是多个专题研究智能体和红队的报告：

<PASTE_ALL_REPORTS>

请去重、核验冲突并把文献转成 Project-Yggdrasil V2 的研究决策。不得以投票或引用量代替机制判断，也不得把相邻证据升级成直接证据。

输出一份高密度中文决策报告：

1. 当前失败的最佳因果解释：operation composition、role binding、hidden→latent 信息保真、length state fidelity 各自的证据强度；
2. 文献共识、不一致和真正未知项；
3. 候选架构树，但最终最多保留两个下一轮实现方向；
4. 每个保留方向给出“直接切换合同”：要删除的旧模块、要新建的核心抽象、禁止保留的兼容路径、最小代码边界；
5. 预注册实验：数据、seeds、训练预算、强基线、weak baselines、ordinary/composition/length、same-answer shuffle、角色置换、反事实和 trajectory 干预；
6. Gate：final/state、相对 prefix/n-gram 基线、因果干预、成本和停止条件；
7. 资源判断：能否在单卡 8GB 上形成有意义的 surrogate probe，哪些主张必须等待更大算力；
8. 明确列出仍未解决的问题，以及为什么暂不进入 V2-A formal、V2-B 或 V2-C。

最后生成一个“实现交接块”，内容足以交给代码智能体，但不要直接写代码。若现有证据不足以选择结构，应明确建议先补哪一个判别实验，而不是强行选型。
```

## 7. 判断研究是否完成

一轮研究只有同时满足以下条件才算有用：来源可核验；直接证据与类比清楚分开；至少包含负证据；建议能够落成最小判伪实验；没有把扩大 K/T、训练时长或普通 loss 调权包装成架构修复；最终候选不超过三个。只返回论文清单、摘要拼接或没有失败条件的报告，应直接要求智能体按同一命令重做。

这些研究报告在经过人工综合和实验验证前，只应放在研究资料区，不应修改白皮书、Gate 状态或当前实验结论。
