# V2-A Closure C1U：Publicly-Grounded Gate-Free Workspace

更新日期：2026-09-01

## 1. 合同地位

C1U 是 C1T sealed FAIL 后的 fresh successor，不是对旧 root、checkpoint、source snapshot 或训练身份的修补。C1T 的 `fixed_4000.pt`、result、seal、lease 与 source identity 永久保持只读；C1U 使用独立 package、task seed、cache、source identity、roots 与 leases，不读取任何旧 model/optimizer state。

2026-09-01 用户以“没问题，按这个顺序做”授权按 Gate 顺序推进：C1U 设计与实现 → S0 zero-training 资格 → 仅在 S0 PASS 后冻结并执行 S1 Overfit32 → 仅在 S1 PASS 后冻结 matched learned K1/K8 S2。该授权不允许跳过 predecessor Gate，不授权重试、调阈值、选择 checkpoint、S3 single-seed formal、C2、V2-A PASS、V2-B 或 V2-C。

## 2. 修复目标与证据

C1T S1 的 sealed endpoint 为 overall `22/32`、factorial exact `1/8`，直接故障是第二次 operation 的 learned sigmoid gate 按组饱和关闭。post-stop 三臂 CUDA BF16 disposable screen 进一步区分了修复项：

| 非正式臂 | 固定 endpoint | 结果 |
| --- | ---: | --- |
| 原 sigmoid transition + direct counterfactual CE | 2,000 | `21/32`、factorial `0/8`，未通过 |
| gate-free direct transition + 原 support hinge | 1,000 | `32/32`、factorial `8/8`、双 contributor `32/32` |
| gate-free direct transition + direct counterfactual CE | 1,000 | 同样全过，没有显示相对上一臂的必要性 |

这些结果只用于选择 successor 机制，不是 formal 或 qualification evidence。C1U 的最小结构修复因此是删除 learned write gate；不得把 gate floor、bias clamp、推理时强制 gate=1、延长旧 schedule 或提高 loss 权重写成等价修复。

源码审计同时发现两项 public identifiability 缺口。旧 ERE public cards 没有声明两个 opaque value symbol 分别代表 bit 0/1；旧 CPS cards 没有把 source address 显式绑定到 query legend 的 `candidate:0/1`。这不是 sealed dead-gate 的近端原因，却会使单条 fresh-group 输入存在未公开语义桥。C1U 必须在重新编码 source cache 前修复，不能依靠 Overfit32 记住 group。

## 3. 公共输入合同

每条记录仍只向模型公开彼此独立编码的 `object_cards[K]`、`operation_cards[T]` 与 `query_card`。answer、family、factor、AST、simulator output、support slots、counterfactual indices 与 valid-choice mask 全部留在 model forward 外部。

C1U 新增以下公开、组内恒定且 factor-independent 的语义桥：

1. ERE 的第二张 operation card 和 query card 必须逐字包含 `Value legend: <zero-symbol> means bit 0; <one-symbol> means bit 1.`；四个 factorial cell 使用同一 legend。它公开 value domain，不公开当前 A/B value 或答案。
2. CPS 的两张 operation card必须分别包含 `Candidate index: 0` 与 `Candidate index: 1`，把 source address 显式绑定到 query choice semantics；四个 factorial cell 内不变。
3. public-only reference replay 只能读取公开 cards 的结构化同源字段，必须对 32 条记录逐条复现 simulator semantic answer 与 raw label。删除、交换或冲突的 value/candidate bridge 必须被 fault registry 杀死。

opaque names、对象排列、raw A–I label mapping 继续按 group 随机化。不得把 opaque value 直接改成字面 `0/1`，也不得在卡中加入当前 factor、正确答案或 AST-only state。

## 4. Gate-free target-only transition

`PartitionedBoundary` 保持逐卡 local encoder、无参数公开地址路由和 source/target/query exact registration。operation 是否存在只由 `operation_mask.any(tokens)` 形成的确定性 `active` 控制。

共享 transition 固定为：

```text
source = route_source(payloads)
target = route_target(payloads)
proposed_target = Operator(source, target, operation_state)
next_payloads = overwrite_registered_target(payloads, proposed_target) when active
```

Operator 的最后线性层只输出 `payload_width`。实现和参数表中禁止 `gate`、`gate_logit`、learned activity scalar 以及任何乘到 `proposed_target - target` 上的 sigmoid。非 target slots 必须逐位保持不变；inactive operation 必须是严格 identity。transition 仍共享跨 family、group 与 step 的同一组参数，不允许 step-specific executor、task-specific head 或 simulator primitive lookup。

## 5. Objective 与行为量具

第一版 C1U 保留 C1T 已审计的三项训练目标，以隔离架构修复：raw A–I full-answer CE 权重 1.0、no-core uniform KL 权重 0.5、同地址单因素 counterpart answer-margin hinge 权重 0.5/floor 0.5。direct counterfactual CE 不进入主合同，因为完整 factorial 下它的 forward 几乎重放 counterpart full CE，disposable screen 未证明必要性。

counterpart gradient ownership、stop-gradient 与 worst-edge aggregation只登记为后备 ablation，不得在 S0 后、S1 启动前临时加入。若 S1 失败，只能在新 successor 合同中重新资格，不得复用 C1U identity。

行为 Gate 继续使用 gauge-invariant margin `correct - logsumexp(all eight raw wrong)`、raw A–I top-1、逐 support flip、逐记录 two-contributor、no-core invariance/necessity 与 object permutation；均值 loss、route hit 或非零 state delta不能替代 exact/worst-cell Gate。

## 6. 分阶段路线

```text
C1T sealed FAIL（只读）
  -> C1U S0 preflight
  -> C1U S0 zero-training qualification
  -> 仅 PASS：冻结 S0 hashes 后编写/审计 C1U S1 合同
  -> C1U S1 fixed-budget Overfit32
  -> 仅 PASS：设计 matched learned K1/K8 S2
  -> 仅 S2 PASS：才讨论 S3；当前不授权
```

S0 必须资格化 public identifiability、fresh independent-card cache、gate-free deployment graph、target-only overwrite、no-core topology、permutation equivariance、operation-2 source-to-target/readout gradient connectivity、真实 CUDA BF16 finite backward、0 optimizer steps、0 model writes 与完整 seal。

S1 必须使用 S0 新 cache 和 fresh model seed；首版保持同 optimizer/LR/batch/clip 以隔离机制变化。S1 的固定 endpoint、预算、root、source/cache/schedule hashes 只能在 S0 PASS 后写入独立执行合同。

S2 必须使用未在 S1 出现的 opaque value/address/label identities，建立 matched learned K=1/K=8、functional-K、K1-null、owner swap/deletion 与完整 seal replay。Overfit32 PASS 不能替代这些泛化和机制证据。

## 7. 结论边界

C1U 当前只是获授权的 fresh repair chain。设计、代码测试、disposable smoke 或 S0 PASS 都不构成学习成功；S1 PASS 只证明固定 32-record overfit 闭环；只有后续 matched learned K1/K8 和更高阶段按各自合同通过，才可能推进 V2-A。
