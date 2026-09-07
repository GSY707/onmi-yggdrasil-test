# V2-A Closure C1S SRW：S1 时序归因修复诊断合同

状态：新身份已冻结设计，尚未消费 preflight 或 diagnosis

日期：2026-08-31

身份：`V2-A-CLOSURE-C1S-SRW-S1-TEMPORAL-ATTRIBUTION-REPAIR-20260831-1`

## 1. 核心判断

前一轮失败归因诊断已在 D004 的 nested readout 内层折叠处 fail-closed，并以 `CRASH_V2_A_C1S_S1_FAILURE_ATTRIBUTION` 封存。它不是可以修复后重跑的开发目录。新身份只补做缺失的 D004R 与 D005R：判断 non-answer temporal target 是否充分、latent 是否形成了跨记录可读的时序分量，以及冻结 state decoder 是否只是读出不足。旧 D001–D003 只作为 hash-pinned 的部分证据引用，不重新测量，也不被包装成本身份的新结果。

这仍是只读诊断。它不训练、不选 checkpoint、不修改模型或旧 root，终态永远 `authorizes=nothing`，也不能授权 S2、S3、single-seed formal、C2 或 V2-A PASS。

## 2. 旧崩溃为何不能只改一行

旧实现先按 record ID 做普通哈希分折，再在每个 inner heldout cell 计算 record-macro balanced accuracy。`CPS.running_best` 的 16 条记录中只有 12 条同时含正负 target，4 条为全负；某个 inner cell 恰好只含一条全负记录，因此有效 record 数为零。若只把异常跳过、退回 observation-micro，或换一个恰好不崩的 seed，就会让选参和评分依赖偶然样本构成，破坏原合同的 record-macro 含义。

新合同把“折叠是否可评分”提升为 preflight 前置事实：在读取任何 latent、prediction 或 endpoint 输出之前，先从 sealed target bank 标记每个 `family × feature × record` 是否同时含正负标签，再生成 feature-specific 的确定性受约束折叠账本。账本一经 preflight seal，真实读出与全部 null 必须共用它。

## 3. 身份与封存边界

固定 preflight root：
`tmp/v2-a-closure-c1s-srw-s1-temporal-attribution-repair-preflight-20260831-1/`

固定 diagnosis root：
`artifacts/v2-a/closure-c1s-srw-s1-temporal-attribution-repair-20260831-1/`

任一 root 或 sibling lease 已存在时必须在写入前拒绝。旧诊断 root、旧诊断 preflight、S1 root、S1 preflight、C1 cache 与 C0R source tree 全部只读；新 preflight 与 diagnosis 都要重放各自 seal、固定文件 hash、source identity 和当前外部输入。旧诊断的 result/seal 以及 D001、D002、D003 文件具有合同内固定 SHA-256。

旧 root 的 D001–D003 可以在 D005R 中作为 `sealed_predecessor_partial_context` 摘要，但必须同时写明：它们不是本身份重算的测量，且旧诊断没有 D004/D005 结论。

## 4. Target-only 折叠账本

硬特征仍只包括：

- CPS：`processed`、`running_best`；
- ERE：`touched`、`changed`、`operation_source`、`operation_target`。

`CPS.final_winner` 与 `ERE.query_semantic_match` 是 answer-derived auxiliary，只可列入 excluded 字段，不得进入 hard evidence。

每个特征独立定义 record eligibility：在该记录的有效 mask 内至少出现一个正标签和一个负标签。只有 eligible record 能进入 fit、inner selection、outer score 与 null score；ineligible record 必须完整列出其正负计数，但不能混入 observation-micro 来补足 record-macro。

折叠固定为 4 个 outer folds 和每个 outer-train 内 3 个 inner folds。eligible record 先按注册 salt 的 SHA-256 排序，再 round-robin 分配。每个 outer heldout cell 与每个 inner heldout cell 至少 3 条 eligible records；每个 inner train 也必须含足够 eligible records且同时具有两类标签。当前 target-only 只读核验预期为：

| family / feature | eligible | outer 每折 | 各 outer-train 的 inner 每折 |
| --- | ---: | ---: | ---: |
| CPS / processed | 16 | 4 | 4 |
| CPS / running_best | 12 | 3 | 3 |
| ERE / 四个硬特征 | 各 16 | 4 | 4 |

这些数值必须由新 preflight 从 pinned target bank 重新生成并 seal，不能把本文表格当成执行输入。任何 cell 不满足合同即 preflight FAIL，不得加载 endpoint。

## 5. D004R 测量

模型只以 `source_hidden/source_mask` 前向，使用 S1 固定 update-4000 endpoint，`model.eval()`、`torch.no_grad()` 和已资格化的 CUDA BF16 路径。只收集自然 rollout 的 H0、H1…H10、source-only temporal `state_features`、冻结 decoder logits 与 evaluator-side target；不再执行 D002 的十一种 answer-path counterfactual。

主读出为 in-memory、record-balanced 且在每条记录内 class-balanced 的 truncated ridge：每条 eligible record 先获得相同总权重，再在该记录的正负 observation 间各分一半，避免长 trace 在拟合阶段压过短 trace。rank 为 `8/16/32`，正则为 `1e-4/1e-3/1e-2/1e-1/1`；超参数只能由 frozen inner folds 的 record-macro balanced accuracy 选择，outer heldout 只评分。observation-micro 只作附属统计，不能覆盖 record-macro。

同一折叠账本执行三种 full-fit null：

1. `time_shuffle`：逐记录固定非零时序平移 state；
2. `temporal_mean`：逐记录把时间均值复制到所有 step；
3. `within_record_target_shuffle`：只在同一 eligible record 内打乱标签，保持该记录正负计数、mask、fold eligibility 与记录身份不变。

另对固定 natural cross-fit prediction 做 10,000 次 within-record target score-null，用于估计评分 chance；它不能替代 full-fit null。

每个 feature 统一报告：target 正负总量、跨 step 变化数、eligible/ineligible record、outer/inner coverage、active-step relative motion、cross-fitted fixed-channel record-macro AUC、natural ridge record-macro BA、冻结 decoder record-macro BA、三种 full-fit null、natural-minus-strongest-null、score-null p 值与全部 inner selections。natural、frozen、fixed-channel 和三个 fit-null 还必须封存逐 eligible-record 的 target bit、prediction/score 与记录内 BA/AUC，使后续审计能在不保存巨大 latent tensor的前提下复算 record-macro 指标和 score-null。

source replay 只能命名为 `source_ast_materialize_target_regeneration_same_implementation`。它证明 sealed source 与 target bank 在同一实现下相符，不能声称独立 semantic oracle。

## 6. D005R 分类

分类按 family 独立完成，overall 只汇总。阈值在 preflight 前冻结：active relative motion `0.05`，fixed-channel record-macro AUC `0.70`，crossfit ridge 与 frozen decoder record-macro BA `0.80`，natural 超 strongest full-fit null `0.05`，score-null `p <= 0.05`。

- `TARGET_INSUFFICIENT`：任一 hard feature 的正负量、跨步变化、source replay 或 fold coverage不足；
- `LATENT_NOT_FORMED`：target 充分，但 motion、跨记录 separation 和 readout 都不足，且表现与注册 null 相容；
- `READOUT_INSUFFICIENT`：motion、separation 与 crossfit readout 均成立且超过 null，但冻结 decoder 未达到 `0.80`；
- `MIXED`：上述证据只在部分 feature 或部分标准成立；
- `INCONCLUSIVE`：没有可归入失败类别的完整证据，或冻结 decoder 本身已经充分。

不得给 composite PASS。D005R 必须显式引用旧 D001–D003 的 sealed context，并说明新身份只补全 axis C；无论分类是什么，`authorizes=nothing`、`v2a_passed=false`。

## 7. 执行顺序

CLI：`experiments/v2_a_closure_c1s_s1_temporal_diagnosis.py`

1. `preflight`：独占 preflight lease；运行新旧专项测试；全量重放 target bank、68 GB cache/source pin、S1 与旧 diagnosis seal；生成并 seal target-only fold ledger；检查 CUDA BF16 与 diagnosis root/lease 缺席。
2. 独立审计 preflight 的 source closure、fold coverage、旧 root pin、null 定义与一次性边界。
3. 只有用户再次明确授权消费 diagnosis 时，才允许 `run-diagnosis`；它领取新 diagnosis lease，执行 D004R/D005R，终端再次重放所有输入并 seal。

本轮“另立身份”不等于允许自动消费一次性 diagnosis。实现与测试完成后，必须先给出 launch-readiness 结论与剩余不确定性。
