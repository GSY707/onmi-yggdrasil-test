# V2-R1R P0-D v14：完整 production 数据冻结设计

日期：2026-08-09

合同版本：`r1r-p0d-v14-full-production`

## 1. 核心判定与边界

v13 只证明固定 seed 下的 1,440-record generator smoke 可行。本阶段第一次回答：accepted simulator、production renderer、因果证书、fingerprint 与 source-only shortcut 测量链，能否构造一份足以进入后续 P0-M 讨论的完整 ERE/CPS production 数据。

本阶段只资格化数据与测量，不训练模型，不生成 Qwen hidden cache，不运行 GPU，不判断 recurrent core、Boundary 或完整 V2 架构成立。正式 PASS 只允许父任务另行审阅是否设计 P0-M；不自动创建训练授权。

v14 直接替换活动 generator/audit/CLI/test 合同。v13 formal artifact 与其中的 source snapshot 永久保留；活动 v13 CLI alias 与 tests 在 v14 adversarial preflight 通过后删除，不保留双轨入口。

## 2. 数据规模与 seed

每个 family 的正式文件固定为：

| split | records |
| --- | ---: |
| train | 4,096 |
| validation | 512 |
| composition OOD | 512 |
| length/horizon OOD | 512 |
| entity/distractor OOD | 512 |
| language OOD | 512 |
| causal_pairs | 512 records，即 256 对 |

ERE 与 CPS 各 7,168 条，总计 14,336 条、512 个因果 pair、14 个 JSONL。正式 root seed 固定为 `2026081402`，唯一 formal root 固定为：

`artifacts/v2-r1r/p0d-v14-full-production-20260809-1/`

在正式命令之前运行一个 seed 为 `2026081401` 的 adversarial preflight：每族 train 1,024，其余六个 split 各 512，共 8,192 条。preflight 使用自动清理的临时 root，不是正式证据。正式 seed 在合同冻结后不得因 preflight 指标改变。

root seed、family/split 派生 seed、unit index、generation attempt 和 record seed 必须同时进入 provenance。audit 必须从公开公式逐项重算，manifest 与每条 record 不得写固定默认常量冒充调用参数。

## 3. 生成域与容量修复

v13 ERE 生成域不能直接放大：容量探针在 train 第 1,544 条附近即可能耗尽 alpha-invariant 语义。v14 不降低跨 split 全局 uniqueness Gate，而直接扩大合法 episode 状态域：

- ERE 保持 train/validation 的 4–5 entity 与 entity OOD 的 6/8 entity 分离；
- 每个 episode 使用 3 个 nonce attribute 和 6 个 nonce value；
- 第一个 attribute 是主要 query/transition 状态，第二个提供 branch/control 与 claim witness，第三个形成真实可见但通常与 query 无关的状态干扰；
- 四个 value 构成 attribute-query 的局部答案域，另两个只允许作为初始状态干扰值，不能成为隐藏标签或 metadata shortcut；
- 所有属性和值仍完整渲染到 source，并由相同 simulator、parser、fingerprint 和 model-view 边界处理。

这是任务分布扩展，不是 uniqueness padding：不得向 fingerprint 注入不渲染的 nonce、record id、seed 或 metadata；不得把 surface identity 当 semantic identity。完整容量测试必须在 4,096/512 配额与全局 seen set 上通过，并记录重采样总量与最大 attempt。

CPS 保持 v13 已接受的唯一最优、NONE、valid-suboptimal、五类 hard negative、composition、6–8 horizon 与八候选 distractor 构造，不以 ERE 扩展为理由改变 CPS 难度。

## 4. 记录、因果与语言不变量

record/model-view 字段、renderer/parser exact roundtrip、fresh simulator answer/trace、claim prefix replay、single-leaf counterfactual、alpha-invariant semantic fingerprint、surface fingerprint、choice/action/candidate permutation 与禁止 marker 规则继承 v13，且全部在 14,336 条上重算。

标签日程允许配额不能被 9 整除：每个 split 的 A–I 计数只能相差 0 或 1，余数标签由 split seed 独立洗牌决定。causal pair 的两个标签必须不同，且总体计数仍只差 0 或 1。choice 中正确项的位置按局部 choice 数分层，不能由 label schedule 或结构 index 推导。

所有未声明 pair 在全数据范围内 semantic/surface overlap 都为 0。每个 causal pair 共享 grammar、nonce、choice mapping 与除一个叶字段外的 AST；fresh replay 必须翻转答案。teacher/AST/answer/label mapping/certificate/claim 不得进入 model view。

Qwen tokenizer 固定为 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`，只允许本地加载。每条 source `<=1024` tokens；ERE/CPS train 与 validation p99 `<=900`，禁止截断或删减必要语义。

## 5. Shortcut 与统计判定

七个 source-only baseline 保持不变：majority、first-valid-label、last-choice-line、token-length bucket、question/objective-only word NB、full-text word NB、full-text char-3–5 NB。train 使用按 causal group 的 5-fold out-of-fold 预测；六个 heldout 使用同一个完整 train fit。learner 只读取 source、mask 和公开 tokenizer 长度。

v14 预注册两层同时判定，避免 v13 的 72-record × 98-cell 离散噪声，同时不降低 effect ceiling：

1. **逐 cell 上界**：98 个 `family × split × baseline` cell 各报告 accuracy、mean random、point excess 与 one-sided Wilson upper bound。置信水准按 Bonferroni `alpha=0.05/98`。每格必须同时满足 point accuracy 与 simultaneous upper bound均 `<= mean_random + 0.10`。
2. **family 聚合上界**：对每个 `family × baseline` 合并六个 heldout，共 14 格；按 Bonferroni `alpha=0.05/14` 计算 one-sided Wilson upper bound。每格必须 `<= pooled mean_random + 0.05`。

Wilson 是冻结的审计判定，不被解释为模型独立同分布的理论证明。它的作用是用固定样本量和 familywise correction 把“一个 record 越线”与持续 leak 区分开。不得在 formal 后改 alpha、cell 数、聚合范围或 ceiling。

NB 实现允许新增稀疏/批量计算路径，但必须在正控上与 v12 accepted dense fit + exact rational ordering逐 prediction 一致；只能优化计算，不能改变 tokenization、Laplace smoothing、mask、tie-break 或 fold semantics。

claim 的每个 kind/pair 继续正负严格平衡。24 个 `family × heldout × word/char` claim cell 使用 Bonferroni `alpha=0.05/24` 的 one-sided Wilson upper bound，必须 `<=0.60`；同时保留 point accuracy。

## 6. Gate

| Gate | 正式判定 |
| --- | --- |
| G01 | v7/v9/v10/v11/v12/v13 seal 与所有声明只读上游 byte identity 通过 |
| G02 | 14 文件、14,336 条、4,096/512 配额、schema/version/model-view、root-to-record provenance 全部精确 |
| G03 | 14,336/14,336 source roundtrip、fresh simulator output、answer/label、teacher trace 与全部 claim replay 一致 |
| G04 | 全局 semantic/surface overlap 0；512 causal pair 单叶、共享表面因素与答案翻转 100% |
| G05 | ERE primitive/query/depth/composition/entity/length 配额及 3-attribute/6-value 生成域由 audit 独立派生 |
| G06 | CPS unique optimum/NONE、suboptimal、hard negative、composition/horizon/distractor 配额独立派生 |
| G07 | 每 split 标签计数差 `<=1`；choice/candidate/action/definition 与 alpha/permutation 正控通过 |
| G08 | language template 分离、marker 禁止、pinned tokenizer 无截断，train/validation p99 `<=900` |
| G09 | 98-cell simultaneous shortcut 上界和 14-cell family 聚合上界全部通过；预测完整且稀疏路径等价资格有效 |
| G10 | claim truth、kind/pair 极性平衡与 24-cell simultaneous claim 上界全部通过 |
| G11 | preflight 双运行确定性已通过；formal full regeneration、report/replay、只读输入、source snapshot 与 evidence seal 全部一致 |

机器状态只能是 `PASS_P0D_PRODUCTION` 或 `FAIL_P0D_PRODUCTION`。任一 Gate false、异常、超时、中断或 artifact 协议错误都使唯一 formal attempt 停止；不得修补 formal root、覆盖、换 suffix 或重跑。

## 7. 执行顺序与停止规则

1. unit/equivalence/capacity/performance tests 可在本地反复运行；
2. adversarial preflight 必须在两个独立临时 root 生成、完整审计并得到 byte-identical dataset/report；失败时只允许在 v14 写入范围内分析修复，再从空临时 root 重跑；
3. preflight PASS 后删除活动 v13 tests/CLI alias，运行 contract guard；
4. 只调用一次固定 formal 命令；返回后无论结果立即停止写 formal artifact；
5. 父任务只读复算 seal、未知反例和高层边界，再同步主审与仓库索引。

本合同不授权 P0-M、cache、model、optimizer、GPU、训练、架构成功结论或 integrated-system 表述。
