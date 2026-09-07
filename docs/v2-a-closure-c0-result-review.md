# V2-A Closure C0 结果复盘

日期：2026-08-23

正式身份：`V2-A-CLOSURE-C0-20260823-1`

机器终态：`FAIL_V2_A_CLOSURE_C0_READINESS`

## 1. 结论

C0 已按固定 root 单次完成并封存。它没有训练模型，也没有判定 latent 架构失败；它证明现有 ERE/CPS bank 与历史 smoke 还不足以开始一场能解释介质差异的 C1 训练。

八门中 C001、C002、C003、C005、C007、C008 通过，C004 与 C006 失败。直接含义是：数据文件、执行代数、split、封存、训练路径和历史证据都可复验，但 ERE 的一个公开题型存在确定性答案捷径，新的 matched compact-trace 主比较又缺少生成 trace 的 parser 与语义 verifier。因此 C0 `authorizes=nothing`；C1、四臂 Pareto、V2-B 与 V2-C 均未授权。

固定证据：

- root：`artifacts/v2-a/closure-c0-20260823-1/`
- result SHA-256：`3FC871BB9EDE76421F08C386E2BB1702878B35273FF5831A1D541C7CB6CC43B1`
- evidence-seal SHA-256：`462317E7BD97040AB901BFD79FA33B631898EA8BD9B75C4ECA8C83B191949972`
- seal：13/13 files 复验通过
- 运行：`training_started=false`、`optimizer_steps=0`、`model_writes=0`

## 2. Gate 结果

| Gate | 结果 | 解释 |
| --- | --- | --- |
| C001 | PASS | P0-D v17 seal、assessment、manifest、74 个 sealed files 与当前 generator/renderer/schema identity 全部一致。 |
| C002 | PASS | ERE 与 CPS 的 AST 分别为 rule/event/query 与 action/candidate/constraint 两套不同执行代数。 |
| C003 | PASS | 26,624 records、14 cells、答案/mask/teacher/claims、causal pairs 与全局 fingerprint 隔离完整。 |
| C004 | FAIL | 旧 shortcut/claim reports 本身通过，但新增 relation-query 条件语义支持与 source-only visible-legend oracle 同时失败。 |
| C005 | PASS | P0-M v5 assessment 与七个子 root seal 全部可复验；解释保持为 training-path smoke。 |
| C006 | FAIL | 公平四臂合同已冻结，但 compact trace 只有 formatter 和 final-answer parser，没有 trace parser、semantic replay、full-bank roundtrip 或 directed fault-kill。 |
| C007 | PASS | 七项历史 V2-A 机器证据均按固定 hash 复验，并完成复用/禁止外推分级。 |
| C008 | PASS | factorized H1 与 direction-geometry v2 仍为 `authorizes=nothing`；C1 明确排除 route/projection expert 与旧 checkpoint。 |

## 3. C004 为什么失败

问题只污染 ERE 的 relation-query 子组，不等于整套 ERE/CPS 可被捷径解掉。

v17 generator 的普通 relation record 固定先建立 `e1 -> e2` 关系，再查询同一关系，所以普通 record 恒为 TRUE；只有 causal-pair alternate 把查询目标换掉并形成 FALSE。题面同时公开 TRUE/FALSE 与本地答案标签的对应关系。一个严格只读 `source_text` 的规则只需识别布尔 choices，再选择 TRUE 标签：

| split | relation-query support | TRUE/FALSE | 条件准确率 | Gate |
| --- | ---: | ---: | ---: | --- |
| train | 342 | 342/0 | 1.000 | FAIL |
| validation | 128 | 128/0 | 1.000 | FAIL |
| language OOD | 128 | 128/0 | 1.000 | FAIL |
| causal pairs | 128 | 64/64 | 0.500 | PASS |

这个子组约占 ordinary ERE 的 8.33%，相对二元随机的整 split 理论抬升约 4.17 个百分点。它不会直接解掉其余 attribute records，却会让 relation 能力、最差子组和介质差异的解释失真。由于该不平衡由 generator 构造决定，换 seed 无法修复；下一份数据必须使用新的 generator identity，配平 relation TRUE/FALSE，并重过完整 production 与新增分层 shortcut Gate。

## 4. C006 为什么失败

旧 P0-M 只能证明三条训练通路能 overfit：direct/text 输入 `source_text`，latent 还额外读取 `reasoning_budget` 和 mask；direct 只看答案，text-CoT 看实际 compact trace，latent 又可看更密集的 claim/state target。它们不是公平的介质比较。

C0 已把新合同改为：

- 四臂 forward 只读相同 `source_text`；`reasoning_budget`、mask、family、AST、trace、claims 均不可作为模型输入；
- 主比较逐 record 使用相同 compact trace bytes、tokens、loss mask 和 exposure ledger；direct/latent 的 training-only trace decoder 在评测前物理删除；
- dense state/claim 只允许作为 `supervision-advantaged` sensitivity，不能支持介质优越性；
- 同时报告 equal-example 与 equal-GPU-hour 两个 matched slice；online Qwen encode 与 cache 构建必须计费；
- generation cap 固定为 512，只能由合同或 train/validation 冻结，不能读取 heldout teacher length。

但现有代码只有 `compact_teacher_trace` formatter，text-CoT generation 只抽取最终 `Answer:`。在 formatter→parser roundtrip、source simulator semantic replay、malformed rejection 和定向 fault-kill 完成前，trace 质量不可机器判定，因此 C006 必须失败。

## 5. 以前的 V2-A 实验有没有用

有用，但用途不是“接着旧 checkpoint 训练”。机器复用矩阵把历史证据分为三层：

| 历史证据 | 现在可复用什么 | 不能复用什么 |
| --- | --- | --- |
| A1.8 | T1–16 训练、长程 T20/T24、三 seed formal/causal 方法 | structured COPY/SWAP 结果不能替代 ERE/CPS；不含 learned full-text Boundary |
| A1.18B | 每个递归步的 dense credit、training-only auxiliary 物理剥离 | oracle symbolic state target 不能进入 matched 主比较 |
| A1.19H | 共享连续 recurrence、opaque addressing、N2–N5 与因果必要性测试；这是最强 core 先验 | 旧权重、COPY/SWAP 准确率和 family-specific 结构不能迁移为新架构结论 |
| A1.9 | Qwen cache integrity 与 hidden causal intervention 方法 | oracle spans/typed roles 不是 learned full-text Boundary |
| A1.20B/C | entity/value binding、pointer validity、compiler 与 execution-gradient 冲突诊断 | 两者都是失败定位，不是正向资格 |
| A1.20D | 双向 section inference 可作 mechanism positive control | post-stop continuation、同一 COPY/SWAP 代数，不是 fresh formal |
| A1.21P | K=1 实现思路、成本账本字段和 Pareto fail-stop | 旧 K=1 数值、五条在线样本、不公平 baseline 和未闭合 Pareto 均不可复用 |

因此，历史实验已经替我们排除了很多坏设计，并提供了 core、训练和审计部件；它们没有提供这轮缺失的任务资格或公平四臂结果。H1/WD 方向只作为“route/projection 局部修复没有形成架构收益”的负证据保留，不进入下一实现。

## 6. 下一步

下一轮仍是 C0 资格修复，不是 C1：

1. 直接创建新的 production generator identity，随机化并配平 ordinary ERE relation outcome；不修改或覆盖 v17。
2. 把 visible-pattern/type/legend oracle 纳入全 family × split 分层 shortcut audit；重新生成、回放和封存完整 bank。
3. 实现 compact trace parser、语义 replay、malformed rejection、定向 fault-kill 与全 bank roundtrip，冻结 trace metric。
4. 复验四臂 public-input/teacher exposure hash、equal-example/equal-GPU-hour 与 cache 成本合同。
5. 用新 single-use identity 重做 C0；只有全部 Gate 通过，才实现单 seed C1。

当前不应继续研究 projection、先写后删或 parameter-Jacobian/Fisher，也不应启动训练。C0 失败发生在架构比较之前，先修好尺子和赛道才有可解释的整体结果。

