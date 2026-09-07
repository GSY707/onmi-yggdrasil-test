# V2-R1R R1G v13：production generator smoke 冻结设计

日期：2026-08-09

合同版本：`r1r-r1g-v13-generator-smoke`

前置判定：v12 `production entry accepted`。本阶段第一次在 accepted simulator、v12 production renderer 和 scalable learner 上生成小规模、跨 split 的 ERE/CPS production-shaped 数据。它回答“generator 是否值得扩到完整 P0-D”，不回答模型是否可学，不启动 P0-M、cache、GPU 或训练。

## 1. 规模、版本与直接切换

每个 family 独立生成：

| split 类别 | 每个 split 的 records |
| --- | ---: |
| train | 288 |
| validation | 72 |
| 四个非语言能力 OOD 中适用的三个 + language OOD | 每个 72 |
| causal_pairs | 72 records，即 36 对 |

ERE 为 `validation/composition_ood/length_ood/entity_ood/language_ood/causal_pairs`；CPS 为 `validation/composition_ood/horizon_ood/distractor_ood/language_ood/causal_pairs`。每族总计 720，合计 1,440 records。smoke generator seed 固定 `20260809`，但所有 record 使用可追溯派生 seed；本规模不替代高层合同的完整 P0 `4096/512`。

新增 `production/fingerprint.py`、`production/generator.py` 与 `production/generator_audit.py`；v12 renderer/scalable/schema 和 accepted common/audit/learner/integration 保持只读。v13 活动 CLI/test 准备完成后删除 `tests/v2_r1r_v12/`，v12 fixed root/source snapshot 永久保留。

## 2. 数据构造

### 2.1 ERE

train/validation 从五类结构按固定配额构造：copy/set causal chain、IF-controlled copy、SWAP-source chain、existing-relation FOREACH、relation query。至少 90% train/validation 的独立 typed dependency depth `>=3`；relation-query 浅例只占剩余诊断配额。

`composition_ood` 只使用 train 中未出现的组合拓扑：`LINK→FOREACH_LINKED→COPY`，或 `SWAP→IF→COPY`。`length_ood` 的 query dependency depth `>=8`；`entity_ood` 固定 6 或 8 entities。每条 attribute 记录提供四个 episode-local value choices，relation query 提供 TRUE/FALSE；正确 semantic choice 被均匀映射到 A–I，choice 行顺序独立随机。

每条记录都由一个指定必要字段产生 counterfactual：改变 first causal SET value、IF condition source、relation mutation 或 query-relevant event argument后，fresh simulator 答案必须改变。causal pair 共享 nonce names、grammar、choice mapping 和除单点字段外的全部 AST，两个 label 必须不同。

### 2.2 CPS

先构造唯一合法最低成本 `P*`，再派生五条 train/validation candidate：P*、合法但更贵的同效果计划、顺序错误、缺 prerequisite/resource、达到局部目标但违反 final constraint。每六条目标生成一条 `NONE`；NONE 必须是所有 candidate 无效，而不是并列最优。

`composition_ood` 强制同时出现 unlock/resource/final-constraint/cost-budget 依赖；`horizon_ood` 的最优 plan 长度为 6–8；`distractor_ood` 使用 8 candidates + NONE，并加入未被正确计划使用的 actions。所有 split 都必须有 valid-suboptimal 和至少两类 hard-negative 原因。

非 NONE 的 counterfactual 只改变 P* 的一个 action cost，使更贵的同效果 plan 成为唯一最优；NONE counterfactual 只改变一个初始 resource，使 P* 从 precondition failure 变为唯一最优。causal pair 的候选顺序、nonce names、grammar 和 choice mapping保持不变。

### 2.3 record 与 view

record 至少包含高层合同第 3.2 节字段，并增加 `record_seed/template_id/token_count/semantic_answer/generation_metadata/provenance`。`program_ast/label_mapping/teacher_trace/training_claims/causal_certificate/simulator_output` 只属于 audit/training view。

正式 `model_view` 仍精确为：

`example_id/source_text/reasoning_budget/valid_choice_mask`

teacher trace 必须逐字节等于 fresh simulator trace。claim 使用结构化 predicate；positive/negative 只换 value/entity/candidate/resource，不把固定的 `true/false/not` 词当标签提示。audit 必须用 fresh prefix replay重算 claim truth。

## 3. Fingerprint、语言与长度

production semantic fingerprint 使用 alpha-invariant graph canonicalization。CPS action definition、candidate set、goal/final conjunction、precondition conjunction和初始事实/关系集合按无序边处理；candidate plan、ERE events、rule primitive 和 CPS effect 顺序保持有序。必须通过 candidate/action/definition permutation 正控，同时对真实 event/plan 顺序改动敏感。

所有非声明 pair 的跨 split semantic/surface overlap为 `0`，family 间也为 `0`。main dataset 的 language OOD 使用 `indirect_v1`，其余 split 只用 `plain_v1/reordered_v1`；文本中不得出现 split/template/seed/answer 标记。

每条最终 source 都用 pinned Qwen revision 计数。token count 必须 `<=1024`，不得静默截断；常规 train/validation 的 p99 目标 `<=900`，靠近上限的复杂例只允许出现在 length/horizon/distractor OOD。超长候选必须拒绝并以新 record seed 重生，不得删除必要语义字段。

## 4. Shortcut 与 claim 审计

在 train 使用 grouped folds、在每个 heldout 使用完整 train fit，评估：majority、first-valid-label、last-choice-line、token-length bucket、question/objective-only word NB、full-text word NB、full-text char-3–5 NB。所有 learner 只读取 model-view source/mask；不得读取 family-specific AST、answer mapping 或 generator metadata。

每个 family/split 的 Gate 上限是该 split `mean(1/valid_choice_count) + 0.10`。label 频率 train 相对 1/9 最大偏差 `<=0.03`，72-record split `<=0.07`。claim word/char NB 在 heldout 的 accuracy 都必须 `<=0.60`；positive/negative 总数和每个 kind 的极性严格平衡。

smoke 结果若某个 shortcut 超标，只说明 surface/data 构造失败；不能因为“模型将来可能忽略”而放行，也不能删除该 baseline。

## 5. v13 Gate

| Gate | 判定 |
| --- | --- |
| G01 | v7/v9/v10/v11/v12 seals 与 accepted shared/production-entry runtime byte identity 全通过 |
| G02 | 14 split 文件、每族 720/总 1440、record/schema/model-view/provenance/seed/version 完整 |
| G03 | renderer source-only roundtrip、fresh simulator answer/trace、label binding和 claim replay全部 100% |
| G04 | 未声明 pair 的 semantic/surface overlap 0；36+36 causal pairs 单点修改、共享 surface factors、答案翻转 100% |
| G05 | ERE depth/necessity/primitive/query/composition/entity/length 配额全部满足且由 audit 独立派生 |
| G06 | CPS unique optimum/NONE、valid-suboptimal、hard-negative reason、composition/horizon/distractor 配额全部满足且独立派生 |
| G07 | train/heldout answer-label balance 达标；choice/candidate/action order 的正向 permutation equivariance 成立 |
| G08 | language template 分离、禁止 marker、pinned tokenizer无截断且 train/validation p99 `<=900` |
| G09 | 七个 source-only shortcut 在每个 family/split 都不高于 mean-random +0.10 |
| G10 | claim truth、kind/polarity balance与 claim-only word/char NB `<=0.60` |
| G11 | deterministic regeneration、artifact read-only replay、source snapshot、input/root seal 与 fixed-root protocol 全通过 |

机器状态只能是 `PASS_R1G_SMOKE` 或 `FAIL_R1G_SMOKE`。任一 Gate 失败即停止；保留 dataset 和失败证据，不扩规模，不进入 P0-D formal/P0-M。

## 6. Artifact 与后继边界

唯一 formal root：

`artifacts/v2-r1r/r1g-v13-generator-smoke-20260809-1/`

formal 前允许在自动清理的临时 root 反复做预生成、审计和修复。formal root 已存在时必须拒绝；attempt 在生成前创建；formal 返回后禁止修改、覆盖或换 suffix 重跑。

若机器 PASS 且父任务主审 accepted，只能说明 R1 generator smoke 值得扩为完整 production P0-D。完整 4096/512 生成与 formal audit、P0-M、Qwen cache、Boundary/core、GPU 和训练都需要新的独立合同；v13 不自动授权。
