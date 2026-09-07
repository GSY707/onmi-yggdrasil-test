# V2-A Closure C1S SRW：S1 失败归因诊断结果复盘

日期：2026-08-31

身份：`V2-A-CLOSURE-C1S-SRW-S1-FAILURE-ATTRIBUTION-20260831-1`

终态：`CRASH_V2_A_C1S_S1_FAILURE_ATTRIBUTION`

## 1. 核心判断

唯一 preflight 完整通过，但唯一 diagnosis 在 D004 temporal readout 的 inner-fold 支持度检查处 fail-closed。旧 root 已 seal，不能修改、补算或重跑。因此本轮没有 D004、D005 或完整 failure-attribution 结论；D001–D003 是同一 sealed root 中在异常前完整写出的部分诊断证据，只能按各自边界解释。

崩溃不是模型训练失败，也不是“所有 temporal target 都没有 record-macro 支持”。真实原因是小样本哈希 inner-fold 没有按 record-level class support 分层：CPS `running_best` 有 12/16 条记录同时含正负 target，另 4 条为全负；在 outer fold 2 的 inner fold 2 中，唯一记录恰好是全负的 `cps-train-1140`，该 cell 的 valid record 数为 0，于是严格检查抛出 `inner readout fold has no record with both target classes`。

这证明当前 split 构造合同不适用于每族仅 16 条的 record-macro readout。它不支持把 D004 写成 `LATENT_NOT_FORMED`、`READOUT_INSUFFICIENT` 或任何其它科学分类。

## 2. 封存与执行边界

preflight root：`tmp/v2-a-closure-c1s-srw-s1-failure-attribution-preflight-20260831-1/`

- status：`PASS_C1S_S1_FAILURE_ATTRIBUTION_PREFLIGHT`
- result SHA-256：`31242B430CFB99A3675F98A2BAA12A0A4E9179C4379FA174FD11B8BB151902C0`
- seal SHA-256：`E065BBF949A18C7950A9AF019E8BF9E962F82364E63C70FD179C82EE8A54CBD3`
- wall：245.56 秒
- 只授权这一次 diagnosis

diagnosis root：`artifacts/v2-a/closure-c1s-srw-s1-failure-attribution-20260831-1/`

- result SHA-256：`58F45CC0D1C315E3A3DBCAF33D319679C8F10ABAC08F629F1390CC8FC7FD6163`
- seal SHA-256：`AF45A857E5C674FE6B7F31E7C5931937D73D83F555102329A729138F13814242`
- seal replay：86/86 matched
- optimizer steps：0
- model writes：0
- authorizes：`nothing`
- `v2a_passed=false`

此终态不授权修后重跑、S2、S3、single-seed formal、新训练、C2 或 V2-A PASS。

## 3. D001：objective 与 Gate 不一致已确认

CPS 与 ERE 都分类为 `OBJECTIVE_GATE_MISMATCH_CONFIRMED`。

- no-core Gate：训练 hinge 在端点 active fraction 为 0、梯度为 0；它只要求 full 与 no-core 的 margin 差超过 0.5，并不要求 no-core accuracy 接近随机。
- functional multi-address Gate：没有直接 loss，active fraction 与梯度均为 0。
- dynamic-state Gate：loss 仍有效，CPS/ERE 梯度 norm 分别约 0.0765/0.1155；但训练的是 masked micro-BCE，Gate 是逐族逐 feature balanced accuracy，两者不等价。

因此 S1 的 32/32 answer 并不能推出 recurrence necessity 或 functional K。当前 objective 对前两项机制 Gate 没有持续、等价的训练压力，这是已封存的直接失败归因。

## 4. D002：Boundary 直答与 core enhancement 并存

两族 A 轴均为 `MIXED`，不能写 recurrence necessary。

| family | full | H0 | core-only | final-delta |
|---|---:|---:|---:|---:|
| CPS | 16/16 | 12/16 | 13/16 | 12/16 |
| ERE | 16/16 | 16/16 | 16/16 | 9/16 |

H0 query-owner leave-one-out 的 margin-drop bootstrap lower 在 CPS/ERE 分别约 3.686/5.840，说明 Boundary 的 owner payload 确实承载答案信息。与此同时 full 相对 H0/core-only 仍有显著 margin 增益，说明 core 会增强置信度，但准确率证据不支持它是答案成立的必要路径。

答案置换 control 在两族都有效：固定 full logits 的观测准确率为 1.0，置换 null mean 约 0.150/0.141，margin 右尾 p 均约 `1e-4`。因此 D002 不是评分器错位制造的假信号。

## 5. D003：当前任务反事实不足以证明多地址必要性

CPS 与 ERE 的 B 轴均为 `INCONCLUSIVE`。

CPS 的 registered certificate 在 16/16 记录中都包含 5 个 content object；但 12 条可评估的 source-simulator deletion answer-support 全部只有 arity 1，另 4 条因 baseline NONE/tie 等边界不可评估。于是 certificate 与 answer causal support 在 12/16 条上冲突，不能把“执行图可达 5 个候选”解释成“答案必须依赖多个候选”。ERE 没有合法的对象删除反事实，16/16 都不能评估 answer support。

模型侧描述结果也更接近单槽/复制捷径：

| family | copy-all 保留答案 | 至少两个 contributor | mean contributor count |
|---|---:|---:|---:|
| CPS | 15/16 | 1/16 | 0.4375 |
| ERE | 16/16 | 6/16 | 1.375 |

这些数字不能独立升级为 `SINGLE_SLOT_COPY`，因为 task-side multi-object causal premise 没有成立；但它们足以说明当前 C1S S1 并未提供多地址功能必要性的正证据。

## 6. D004 崩溃的修复原则

若未来另立新身份，只能把本轮作为 predecessor negative evidence，不能复用旧 root 或 lease。新的 temporal 合同至少需要：

1. 在每个 family/feature 内先标记每条 record 是否同时含正负 target；
2. outer/inner fold 对这一 support 标记做确定性分层或受约束分配，保证每个评分 cell 有注册的最小 valid-record 数；
3. split 在 preflight 前冻结，并在不读取 latent/prediction 的纯 target audit 中证明 coverage；
4. 如果 16 条记录不足以满足所有 feature 的 nested cross-fit，必须降低 fold 数或扩大独立 diagnosis sample，而不是把 micro observation 当作 record-macro 替代品；
5. source replay 只能称为同一 `materialize_target` 的 source-AST consistency regeneration，不是独立 semantic oracle。

新身份能否启动需要单独授权。本复盘本身不授权 successor。
