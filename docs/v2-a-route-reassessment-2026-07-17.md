# V2-A 路线重审：混合 core 与完整 R1 收敛

日期：2026-07-17

状态：路线决策记录；A1.19H 已通过，A1.20B/A1.20C 均失败并触发停线，V2-A 当前不通过

## 1. 核心判断

从完整白皮书看，宏观顺序仍然正确：必须先完成 R1/V2-A 的推理介质 Pareto 与 audit Gate，才能进入多模态 Boundary-MoE、FFN-MoE、主动调用、I/O、工作树和完整系统集成。不能用后续组件掩盖 latent reasoning medium 尚未成立。

需要修正的是 A1.18B 后的微观顺序。白皮书要求语义推理主要在连续 `H_t` 中递归、每一步不经过词表采样和 token 回嵌；它同时明确允许离散控制事件、source handle、专家索引和阶段状态。因此“全连续寻址”不是白皮书硬门，A1.19R 不应决定整个 V2-A 生死。

A1.18B 证明固定实体槽、relation-addressed shared transition、soft closure 和每步密集状态信用分配可以形成稳定因果递归。A1.10/A1.11 则证明完全匿名、无结构的通用 reasoner 即使获得每步完整 state CE 仍 formal `0/3`。这组证据支持把对象持续性与关系寻址保留为 core 归纳偏置，而不是继续追求无身份 latent soup。

## 2. 合规的混合 core

候选运行状态写为 `S_t = (A_t, H_t)`：

- `H_t`：连续 key-value payload，承载判断、工作内容、证据摘要和输出意图；
- `A_t`：离散或 prototype-anchored 的对象/来源 handle、类型、阶段、权限、预算和读写地址；
- shared event-conditioned transition 根据 `A_t` 选择读写位置，但实际语义更新发生在 `H_t`；
- closure 与训练期每步密集预测继续保留，正式输出只能读取最终 `H_t`；
- Boundary/控制面可以分配合法 handle，但需要从文本理解的 source/target/query 角色必须由模型学得。

允许稳定 typed/addressable slots、离散 handle、prototype anchor 和结构化 READ/EMIT 控制。不允许标准答案、完整 teacher state 或任务语义藏进 `A_t`；不允许推理时 oracle state 注入；不允许 COPY/SWAP 等任务专用执行分支；不允许把全部语义 payload 每一步硬量化回任务标签。

## 3. 新执行顺序

### 3.1 A1.19H：可泛化混合 core

执行状态：已完成。H1/H2 overfit32、三组 fresh formal 与 formal 后 causal 均通过；机器分类 `generalized_hybrid_core_confirmed`。训练内实体数为 `N=2,3,4`，heldout `N=5` 与 `N=5+relation` 三 seed trajectory/final/answer 均通过。证据与边界见 `docs/v2-a1.19h-hybrid-core.md`。

最近实验不再以 learned continuous addressing 为唯一候选，而是把 A1.18B 的三寄存器 scaffold 推广为任务无关的 addressable workspace。为避免重复 A1.10 的多变量切换，A1.19H 分两个严格顺序子门：

1. `H1 opaque-handle`：实体数量、操作、数据和训练预算全部保持 A1.18B，不再让 slot index 等同 amber/cobalt/jade；每条样本随机分配 opaque handle，并要求 handle 与 slot 联合 permutation 后行为不变。
2. `H2 variable-cardinality`：只有 H1 formal `3/3` 后，才把实体数量从固定 3 扩展为训练内多种 `N` 与 heldout `N`；操作族、transition 参数和监督机制继续不变。

离散 handle 只提供身份持续性，连续 payload 与同一个 shared transition 完成实际更新。禁止 family-specific transition、固定前三槽语义和答案 metadata。soft continuous addressing 可以作为 H2 通过后的同预算消融，但不是前置 Gate。

每个子门都先过 overfit32，再做三个 fresh seed formal/causal。Gate 覆盖 OOD horizon、relation holdout、handle permutation/alias、同值不同实体、query swap、旧轨迹拒绝、disable recurrence 和辅助头物理删除；H2 另加 OOD entity count。任一子门失败都停止后续 full-text 与 Pareto。新增任务/关系族不在 A1.19H 混入，而留到 A1.21P 的跨任务 Pareto。

### 3.2 A1.20B：learned full-text boundary

执行状态：已停止，未通过。

A1.19H 通过后，冻结 core，用完整 Qwen hidden 训练 Boundary 输出连续 payload 与必要的离散 handle/type/control。环境本来就提供的 source identity、模态、权限和位置 metadata 可以直接使用；source/target/query、关系和状态内容不得由 oracle span/role tensor 注入。

先单独验证 pointer、role、handle geometry 和 task formal，再允许短程联合。Boundary 失败不得通过 hard re-embedding 兼容补丁掩盖；若最终选择 prototype anchoring，必须作为正式混合架构重跑多 seed/causal。

实际结果中，overfit32 严格通过，但 run-1 fixed-5000 的 2048 条 heldout validation trajectory/final/answer 仅为 `0.297363/0.409668/0.608887`，最差 cell trajectory 为 `0`。正式 split 前 eligibility 已失败，因此 formal cache/eval、causal、run-2/run-3 均未执行。

oracle 诊断把全部 Boundary 输出替换后，同一 frozen core 的 trajectory/final/answer 恢复 `1.0`；失败主要集中在 value 与 source/target entity binding，family sequence 次之。梯度审计进一步确认 state CE 无法穿过 threshold/argmax 进入离散控制 logits，当前控制只由 factorized local labels 训练。机器分类为 `full_text_entity_binding_and_program_extraction_failure`。详细证据见 `docs/v2-a1.20b-full-text-boundary.md` 与 `artifacts/v2-a/a1_20b/assessment-summary.json`。

### 3.3 A1.20C：分层编译器与 straight-through execution credit

执行状态：已停止，目标臂 overfit32 未通过。

A1.20C 保持 frozen Qwen full-token hidden、frozen A1.19H core、数据和 seed 不变，只正交引入 flat/hierarchical compiler 与 hard-local/straight-through execution credit。分层 compiler 的 token anchor target 只用于训练监督，不进入前向；straight-through 模式保持 hard mask/argmax forward，并让 soft surrogate 反向经过 frozen core。

目标臂 `hierarchical + straight-through` 的所有 entity/value/operation/query anchor sequence exact 均为 `1.0`，hard-forward state/answer logit 差为 `0`，core hash 不变。但 fixed-5000 overfit32 的 best trajectory/final-state/state-token/answer 为 `0.875/0.90625/0.964474/1.0`，四条 `N=4` 样本仍有 entity/operation count 与 pointer-validity 联动错误，严格 Gate 失败。

实际 checkpoint 梯度审计显示 state CE 与其余 local compiler objective 的全局 cosine 为 `-0.7366`，entity path 为 `-0.9680`，presence heads 为 `-0.9944`。此外 pointer local CE 先于 predicted entity mask 计算，而执行会用 hard entity mask 改写候选集合；current ST bridge 没有连续化这个 validity-set 决策。机器分类为 `anchor_localization_solved_but_execution_objectives_conflict`。完整证据见 `docs/v2-a1.20c-boundary-repair.md` 与 `artifacts/v2-a/a1_20c/overfit32/hierarchical__straight_through/failure-diagnostic.json`。

### 3.4 A1.21P：matched R1 Pareto

执行状态：未运行；被 A1.20C overfit32 停线门阻止。

完成 full-text hybrid core 后立即回到白皮书真正的 H1/H2：同成熟基座、同输入、同数据、同任务和计算匹配条件下，对比 direct、显式 text-CoT、`K=1` 与多向量 hybrid latent recurrence。至少覆盖当前状态机和一个新的可执行关系/规划任务族，core transition 不允许使用任务族专用参数。

必须报告多 seed、heldout、长度与实体数量 OOD、质量、延迟、KV/激活、latent transitions、训练/teacher 数据成本和无旁路干预。只有形成稳定质量—成本 Pareto 才继续；否则停在 V2-A。

### 3.5 A1.22A：audit readout 与 architecture-fidelity formal

执行状态：未运行；被 A1.20C overfit32 停线门阻止。

冻结或大部分冻结通过 Pareto 的 reasoner，训练只读 `H_1...H_T` 的自然语言 audit decoder。decoder 不看标准答案；关键 latent step 的删除、替换和打乱必须同步改变 readout 与任务行为，中间判断必须能预测后续 state/动作/答案。

A1.21P 与 A1.22A 同时通过，才能关闭路线图 R1/V2-A。此前的 trajectory state 指标和反事实干预是必要前置证据，但不等于自然语言 audit Gate。

## 4. 训练目标来源的地位

target source 不再是进入 Pareto 前的独立硬门。白皮书明确把成熟文本模型视为教师；可执行 trace、程序状态、verifier-filtered text teacher record 和训练期辅助头都是允许的数据/优化机制。必须记录其生成成本、覆盖率和错误率，并在部署前物理删除辅助路径。

零 teacher、自监督 next-state 和完全无过程标签训练可以继续作为效率研究，但不属于白皮书成立的必要条件。真正的禁止项是推理/部署时 teacher 或 oracle 绕过 latent core。

## 5. 完整白皮书顺序

R1/V2-A 通过后，原高层路线保持不变：V2-B/R2 验证文本、视觉和动作 Boundary experts；R3 验证 Dense 与 FFN-MoE 的同 active-compute 2×2；随后依次验证 I/O 专家、工作树/记忆树与恢复、离线专家晋升；最终 A4 在同一系统中关闭九项完整版条件。

## 6. 决策门

- A1.19H 失败：结构化诊断 core 不能推广，停止 full-text、Pareto 和 V2-B，重做 core 或终止当前分支。
- A1.19H 通过、A1.20B 失败：混合 Reasoner 证据保留，瓶颈在语义边界；V2-A 不通过。
- A1.20C 修复臂 overfit32 失败：不运行完整 2×2、formal、Pareto 或 audit；先修复 count/pointer-validity 信用与冲突梯度。
- full-text Boundary 通过、A1.21P 无 Pareto：记录为可运行但无介质优势，停止 V2-B。
- Pareto 通过、audit 失败：只能称任务模型有效，不能称 architecture-fidelity。
- Pareto 与 audit 都通过：关闭 R1，进入 V2-B；不因混合寻址降低证据等级。
