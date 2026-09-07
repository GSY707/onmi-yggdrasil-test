# C1S SRW：S1、S2 与 single-seed formal 执行合同

## 1. 合同地位与直接切换

本文件与 `src/yggdrasil_v2/v2_a/closure_c1s_successor/contract.py` 共同构成 C1S Semantic-Routed Workspace（SRW）的冻结真源。它只定义 S1 Overfit32、S2 matched K=1 discovery 和 S3 single-seed formal；不自动启动任何阶段，也不授权 C2、V2-A PASS、V2-B 或 V2-C。

此前准备但从未启动的 candidate-choice ACW successor 已在领取 root/lease 前被否决：它把正确答案 choice 当 query owner，用循环 choice schedule 冒充任务计算，并把一个静态终态复制到十步。该实现没有形成可用的架构证据，且没有任何旧 successor root 或 lease 可复用。SRW 是新 identity 的直接切换，不是对旧实现的兼容补丁。

阶段链只有一条：

```text
sealed S0 PASS + 本轮用户授权
  -> S1 preflight -> S1（一次）
  -> 仅当 S1 PASS：S2 preflight -> S2（一次）
  -> 仅当 S2 PASS：S3 preflight -> S3 formal（一次）
```

任一阶段 FAIL、CRASH、INCOMPLETE、seal/source 漂移或授权范围不符，后序阶段保持 `NOT_RUN`。禁止重跑、换 seed、改阈值、延长 endpoint 或挑选 checkpoint。

## 2. SRW 实际检验的架构命题

模型 public forward 始终只有 `source_hidden` 与 `source_mask`。family、AST、answer、label mapping、semantic owner、state target、simulator output 和 split metadata 全部留在 loss/evaluator 外侧，不能进入部署图。

SRW 的状态是一个带地址的内容集合和输入条件 query。Boundary 只读 source 一次；之后十步共享 transition 只能通过 address-routed source/target binding 更新内容；最终答案只能从 query 选中的 final payload 产生。训练辅助头可以给 Boundary/transition 信用，但在部署前物理删除，删除前后 logits 必须一致。

本合同要证明的不是“有八个张量”或“slot cosine 不相同”，而是以下联合命题：

1. query、operation source 和 operation target 能恢复公开语义 owner；
2. 多个内容地址对最终 decision 有独立可干预贡献；
3. 关闭 recurrence，破坏 payload-operation、operation-step 或 target-route binding 会破坏 answer margin；
4. padding 地址的干预不应改变答案；
5. K8 在同数据、同更新、同参数预算的真实 K1 对照上取得每族可复现增益。

## 3. 公开、答案独立的地址合同

外生 `example_id` 哈希置换不进入本合同。只置换 target 会使任务不可学习；同时置换 Boundary pair 与 target 又与原 loss/gradient 等价，只会洗匀 owner 直方图，不能排除固定槽捷径。

SRW 改用模型从 source 可观察的 record-local 规范：

- CPS：candidate 与 `none` decision 按本记录实际出现的 `LABEL KEY` 行排名分配地址。label mapping 每记录变化，所以 decision 不固定在某个物理槽；地址来源公开且不读取正确 candidate。
- ERE：对象是真实 public entity，按实体公开名称的 Unicode NFKC + casefold 规范词典序形成地址，并以原始 UTF-8 作为稳定 tie-break；规范化碰撞 fail closed。它不读取 `example_id`、答案、label、trace 或 simulator target。query owner 是公开 attribute query entity 或 relation target。
- CPS operation 是当前 candidate 到 decision 的 candidate-level 更新；ERE operation 来自实际有序 event 及其实体读写关系。

target bank 必须报告 query/source/target 槽分布、query-owner×answer contingency、owner-only answer majority accuracy，以及按 family/split/object-count 分层的 query-slot contingency。supported train/validation 中 query 槽必须非退化，owner-only answer accuracy 不得超过冻结上限 `0.35`；对象数条件下猜 query slot 的多数准确率不得超过 `max(0.35, uniform-within-count + 0.10)`。这是一项泄漏审计，不把自然的 input-target 关联当随机噪声。

## 4. 真实时序目标与容量边界

state target 固定为 `[10, 8, 8]`，每一步拥有独立 feature mask 与 step mask，不允许把终态复制到全部 step。

- ERE 动态硬指标：`touched`、`changed`、`operation_source`、`operation_target`。
- CPS 动态硬指标：`processed`、`running_best`、`final_winner`。

其中 ERE 的 `query_semantic_match` 与 CPS 的 `final_winner` 含有 answer-derived 监督语义；它们只允许作为模型外、detached 的 loss/evaluator 标签，不能单独构成机制证据。尤其不得用 pooled state 或 `final_winner` 通过宣称时序机制成立；阶段 Gate 仍须同时满足答案独立 owner、非答案派生的动态 feature、operation 时序、answer-margin 干预、functional-K 与 matched K1。

除 pooled bit accuracy 外，每个动态 feature 必须同时拥有正负类、跨步 change coverage，并分别报告 balanced accuracy；S1 每族每项不低于 `0.95`，S2/S3 不低于 `0.80`。`operation_active` 同样按每族正负覆盖、跨步变化和 balanced accuracy 过同级 Gate；pooled accuracy 只作诊断。这样 present/stable/legal 等大量易预测 bit，或“十步中九步 active”的常量预测，都不能掩盖真正的时序失败。

workspace 容量为 8，但 raw answer head 保留 A-I 九类。ERE 9-entity 与 CPS 8-candidate+decision 的 9-object 记录只能作为 OOD behavior evidence：它们必须能正常物化和前向，但 `mechanism_supported=false`，所有 owner/state/causal auxiliary mask 为零，且不得出现在 S1/S2 或 validation mechanism Gate 中。任何 unsupported 非 OOD 记录都是 launch blocker。

ERE semantic object 与 CPS decision object 的 `object_label=-1`；只有具有公开 raw candidate label 的 CPS candidate 可接受 slot identity CE。全行 ignore 时 loss 必须是有限的零，不能产生 NaN。

## 5. 因果指标与反捷径控制

所有因果 Gate 使用 gauge-invariant answer margin：

```text
m(x) = logit(correct) - logsumexp(logits(wrong classes))
effect = m(base) - m(intervention)
```

原始 correct-logit drop 只作 diagnostic；给全部九类统一加常数不得产生因果 gain。functional-K 的逐内容贡献也使用 pre-recurrence content mean-replace 后的 answer-margin degradation，不能用全 logits L2 的 nuisance 变化过门。

注册干预包括：no-core、wrong-start、payload zero/shuffle、operation-state zero/shuffle、target-route shuffle、query/decision relevant replace、padding-slot irrelevant replace，以及 same-payload/different-address duplicate control。no-core 除 margin drop 外还要求 raw top-1 不高于 `0.25`；padding 干预的 margin 变化绝对值不得超过 `0.10`。K8 每条记录至少两个 content contributor，平均 effective contributors 不低于 2。

slot permutation 只作为下游 address-content pair 等变 sanity，不被解释为 Boundary 已消除固定 slot specialization。duplicate control 的第一个 read 使用模型实际 learned query，第二个 read 才使用 exact alternate address 作为结构仪器。

## 6. S0 证据的严格角色

S0 root 固定为 `tmp/v2-a-closure-c1s-s0-preflight-20260828-1`。S1 preflight 必须重放 result/seal/full tree，并显式核对当前 `closure_c1s/model.py` 与 S0 snapshot SHA 相同。授权也必须逐字等于 `one independent C1S S1 Overfit32 implementation and single-use run only`，同时确认 S0 的 `never_authorizes` 仍包含 S2 与 single-seed formal；只看 `s1_status=AUTHORIZED_NOT_RUN` 不足以消费 S1 身份。

S0 K1 null 是零更新的解析 synthetic mean-replace 仪器，不是训练过的 K1 endpoint；S0 CUDA smoke 也只运行 K8。因此 S1 只训练一个 K8 arm，并把 S0 记录为 `STRUCTURAL_INSTRUMENT_ONLY`、`learned_k1_control=false`、`multi_address_qualified=null`。真正 learned K1 对照只在 S2 出现。

## 7. S1：Overfit32 因果资格

固定 identity/root：

- `V2-A-CLOSURE-C1S-SRW-S1-OVERFIT32-20260829-1`
- `artifacts/v2-a/closure-c1s-srw-s1-overfit32-20260829-1`

S1 使用新 SRW selector，从 train 选择 32 条、每族 16 条，并与 S2 完全互斥。只训练 fresh K8；batch 8（每族 4），固定 4,000 updates，endpoint `fixed_4000`，无 checkpoint selection。每一步计算全部注册 causal loss。

PASS 必须同时满足：32/32 和每族 16/16 answer exact；公开答案独立 query；query/source/target/operation/state 资格；每个动态 feature 的 coverage 与 balanced accuracy；多内容 functional-K；duplicate address/content；query-relevant pre-recurrence content mean-replace 必须破坏 answer margin、padding-slot irrelevant mean-replace 必须保持 answer margin，以及其余全部反捷径干预；slot/strip invariance；结构化 S0 单槽仪器重放。这里注册的是 mean-replace，不是物理 delete，两者不得在报告中混称。任一失败即封存并停止，不得把 Overfit32 写成泛化证据。

## 8. S2：真实 K8 与 matched K1

固定 identity/root：

- `V2-A-CLOSURE-C1S-SRW-S2-DISCOVERY-20260829-1`
- `artifacts/v2-a/closure-c1s-srw-s2-discovery-20260829-1`

每族固定 3,072 train、512 eval，排除 S1；formal validation/OOD/causal 保持 untouched。K8 与 K1 使用相同 cache、记录、caller order、schedule SHA、batch、optimizer、seed scheme、4,608 updates 和 6 epochs；参数量必须相等。当前主机只有一张可用 CUDA RTX 4070，两臂严格串行，不使用 DDP。

K1 必须实际训练并报告 `VALID_SINGLE_SLOT_CONTROL`。多地址专属指标对 K1 为 `NOT_APPLICABLE`，即 `multi_address_applicable=false`、`multi_address_qualified=null`；不得硬编码为 FAIL，也不得因 K1 准确率高而判对照无效。

S2 PASS 要求 K8 在 ERE、CPS 分别达到 point `0.75`、Wilson lower `0.70`；K8-K1 paired answer gain 在 overall、ERE、CPS 均至少 `+0.05` 且 cluster-bootstrap lower `>0`；K8 两族的 SRW ownership、动态 state、answer-margin interventions 和 functional-K 全过门；每臂不超过 4 GPU-hour。

## 9. S3：single-seed formal

固定 identity/root：

- `V2-A-CLOSURE-C1S-SRW-S3-SINGLE-SEED-FORMAL-20260829-1`
- `artifacts/v2-a/closure-c1s-srw-s3-single-seed-formal-20260829-1`

S3 仅运行 fresh K8 SRW。full train 为 8,192 条（每族 4,096），固定 6,144 updates、`fixed_6144` endpoint、无 checkpoint selection。Gate 顺序如下：

1. validation behavior；
2. OOD behavior（含明确的 9-object behavior-only cells）；
3. causal-pair behavior 与双向 semantic answer-margin degradation；
4. SRW ownership、动态 state、functional-K 与 answer-margin interventions；
5. hidden intervention；
6. recurrence intervention；
7. training auxiliary strip；
8. accounting/source/seal integrity。

validation/OOD 继续使用冻结的旧 C1 behavior 定义；causal-pair、hidden 与 recurrence 则全部使用 successor-native 报告，并显式传入 S3 bootstrap seed。causal-pair 同时要求 behavior 和两个 semantic 方向的 margin Gate，hidden/recurrence 只以 gauge-invariant margin degradation 判决；旧 raw-accuracy causal Gate 不进入结果，也不保留兼容 subgate。任一 Gate FAIL 后后续全部 `NOT_RUN`。即使 S3 PASS，结论也仅覆盖该 source snapshot、seed、预算和 split 的一次 formal；不自动授权复制、Pareto 或后续架构阶段。

## 10. Preflight、性能和证据完整性

每段拥有独立 preflight root/lease；preflight PASS 只授权对应阶段一次启动。S1 preflight 物化并 seal 全 26,624 条 target bank 和 split ledger，执行 full cache/source audit、全量 semantic/capacity/address/dynamic coverage 审计，以及 100-step disposable CUDA benchmark。写盘后必须重新读取 target bank/ledger，复算内部 canonical payload digest、外层 gzip/ledger SHA、record count、selector preimage 和 split semantics；`target_bank_integrity` 与 `cache_integrity` 必须记录实际审计状态，不能以 `NOT_REQUIRED` 通过。该 benchmark 使用与 S1 相同的每步 causal 热路径，模型/顺序 seed 只取冻结 contract 字段，权重不保存；结果须分别记录 disposable optimizer steps 与恒为零的 formal-stage steps。

所有正式 stage launch，以及 S2/S3 preflight，都重新哈希 S1 target-bank gzip 与 split ledger、重算 selector/assignment、重放 C1 cache result/seal/full tree，并复验 cache 到 immutable C0R source identity。source identity 包含 successor、共享模型、C1 contract/cache/data/evaluator/trace target、C0R trace 和 simulator/renderer 的直接依赖；任何漂移在训练前 fail closed。

训练统一使用 AdamW、BF16、batch 8、boundary/transition/head LR `1e-4/2e-4/3e-4`、weight decay `0.01`、clip `1.0`、256-step warmup 后 cosine。device preflight 必须把 `torch.cuda.is_bf16_supported()` 纳入 hard PASS，而不是仅记录；不允许静默改用 FP16/FP32。runner 必须从 contract 显式构造 `TrainSpec`、`LossWeights` 与 `CausalMargins` 并写出等值审计，S1/S2/S3 的 bootstrap seed 也必须在每个 evaluator 调用点显式传递，禁止依赖可漂移的默认参数。

当前单 GPU 热路径冻结 CPU intra-op 2、inter-op 1、pinned caller-order cache transfer。正式运行由独立子代理监控 GPU、进程、root 与最终 seal；父任务直接等待子代理完成，不做高频轮询。性能优化只能在领取一次性 root 前完成，且不得改变样本顺序、effective batch、loss 或更新数。

## 11. 失败和结论边界

以下任一情况均 fail closed：split/selector 漂移，unsupported 记录进入机制训练，公开地址退化为答案代理，dynamic feature 无 change coverage，K8/K1 schedule 或预算不等，非有限值，CUDA fallback，旧 checkpoint/optimizer 加载，forward 字段越界，source hash 漂移，意外模型写入，root/lease 冲突，seal 缺失，或进程非零退出。

禁止用 loss、slot cosine、attention entropy、raw correct-logit drop、query owner usage 直方图或 pooled state accuracy 单独宣称 SRW 成功。只有按阶段冻结的 behavior、ownership、真实时序、反捷径、matched K1 和 evidence integrity 联合通过，才获得该阶段明示的下一步资格。
