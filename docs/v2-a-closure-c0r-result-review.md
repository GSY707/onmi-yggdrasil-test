# V2-A Closure C0R 结果复盘

日期：2026-08-24

数据资格身份：`V2-A-CLOSURE-C0R-DATA-TRACE-20260824-1`

readiness 身份：`V2-A-CLOSURE-C0R-20260824-1`

机器终态：`PASS_V2_A_C0R_DATA_TRACE_QUALIFICATION` / `PASS_V2_A_CLOSURE_C0R_READINESS`

## 1. 核心结论

C0R 已用新身份关闭旧 C0 的两个 blocker：C004 的 ordinary ERE relation-query 公开答案捷径，以及 C006 的 compact trace 不可严格解析和语义复验。正式 data/trace 资格的 D001–D012 与 readiness 的 C001–C008 均通过；两个正式阶段都没有加载 checkpoint、启动训练、创建 optimizer 或写模型。

因此当前只授权 **C1 single-seed implementation and eligibility**。这不是 V2-A 架构通过：四臂训练结果仍不存在，`v2a_passed=false`，C2 matched Pareto、C3 自然语言审计、V2-B 和 V2-C 均未授权。

## 2. 两个旧 blocker 如何关闭

### 2.1 C004：新数据身份消除 relation visible-pattern 捷径

C0R 不修改旧 v17 bank，而是建立 `v2-a-c0r-relation-balanced-v1` 新 generator identity。ordinary ERE relation-query 按 relation ordinal 交替生成 TRUE/FALSE base；causal pair 仍保持同 mapping/template/seed 下的一真一假。新 record 与 manifest schema 分别为 `yggdrasil.v2-a.c0r.record.v1` 和 `yggdrasil.v2-a.c0r.dataset-manifest.v1`。

严格只读 `source_text` 的 visible-legend oracle 在四个预注册 relation cell 上均回到随机率：

| cell | support | TRUE/FALSE | oracle accuracy | 单侧 Wilson upper |
| --- | ---: | ---: | ---: | ---: |
| ERE/train | 342 | 171/171 | 0.50 | 0.56016 |
| ERE/validation | 128 | 64/64 | 0.50 | 0.59717 |
| ERE/language_ood | 128 | 64/64 | 0.50 | 0.59717 |
| ERE/causal_pairs | 128 | 64/64 | 0.50 | 0.59717 |

这只说明旧 C004 的已知确定性捷径被关闭；不等于所有未知捷径在数学上不存在。后续 C1 仍必须沿用 source-only、分层 heldout、causal pair 与 fingerprint 审计。

### 2.2 C006：compact trace 成为可冻结的监督与评测对象

新 `CT1` trace 使用 canonical JSON payload 与显式 `Answer:`，strict parser 拒绝 duplicate key、非 canonical JSON、非法类型/顺序/index、非有限数和 schema 偏差。每条 trace 都由公共 simulator 从 source AST 独立 replay；测试同时覆盖 malformed registry 与针对 ERE/CPS 每个语义字段的定向 fault mutation。

正式全 bank 结果为：

- formatter/parser roundtrip：`26,624/26,624`；
- source-simulator semantic replay：`26,624/26,624`；
- directed fault-kill：`394,278/394,278`；
- malformed rejection：`7/7`；
- tokenizer token cap：`512`，max/p50/p95/p99 为 `389/202/349/366`。

因此 matched compact trace 现在可以作为四臂共同的 training-only supervision 与生成评测对象，而不再只是一个无法验证语义的字符串摘要。

## 3. 正式证据与单次身份

data/trace 正式 root 为 `artifacts/v2-a/closure-c0r-data-trace-20260824-1/`。26,624 条记录和 D001–D012 全部通过：

- `result.json` SHA-256：`B2502F66CB5D9ECEB2CA0547CB280CC5346D1A24D2F8901F94E617E1B258FFF9`；
- `evidence-seal.json` SHA-256：`5AFB37AB5950D04298FED4D2DAFA78102260A728101A5EDE4972D1EC0EEBB570`；
- seal 逐文件复验：`42/42`。

C0R readiness 正式 root 为 `artifacts/v2-a/closure-c0r-20260824-1/`。旧 C0 FAIL、旧 seal、新 data/trace seal、fairness contract、历史复用矩阵与 H1/WD 排除边界均被重新核验，C001–C008 全部通过：

- `result.json` SHA-256：`391C846D2150FD27D2016A79A9C360F3CB4E1455EC71240720F772CF58A28D5E`；
- `evidence-seal.json` SHA-256：`161FBB367DEE40D18CDD23521E7DA9EBA3257EE8BF57C1FA194886E43D140BA4`；
- seal 逐文件复验：`25/25`。

一次性 preflight 曾因审计器把 manifest schema 错当 record schema，仅在 D002 失败；该失败 root 保持封存，没有覆盖或重跑。修复后先在同一 26,624-record 封存数据上只读重算 D002，再唯一启动正式 data/trace 命令。旧 C0、preflight、P0-D v17 与 P0-M v5 均未修改。

## 4. C0R 证明了什么

C0R 证明的是“尺子与赛道已具备资格”：ERE/CPS 是两套可执行代数，新 bank 的身份、split、causal pair 和已知捷径审计可复验；四臂只读同一 `source_text`；共同 compact trace 可严格解析、语义 replay；equal-example 与 equal-GPU-hour、公平曝光和完整成本账本已经进入冻结合同。

C0R 没有证明任何模型能学会 ERE/CPS，也没有证明 K=8 优于 K=1、latent 优于 direct/text-CoT，更没有证明 route/projection、先写后删或 parameter-Jacobian/Fisher 是正确架构。历史 A1.8/A1.18B/A1.19H 仍只提供 core 与训练方法，A1.20B/C/H1-WD 仍只提供负向诊断；任何旧 checkpoint 都不能进入 C1。

## 5. 下一步：C1，而不是继续局部修复

下一轮应直接冻结独立 C1 合同并实现一个 fresh single-seed architecture-eligibility run。C1 的任务是判断候选多向量 latent 路径能否在新 bank 上形成真实的 heldout、OOD、causal 与 state/trace 能力，同时通过 no/shuffled hidden、recurrence、slot/trajectory 等必要性干预。模型前向仍只能读取 `source_text` 派生的冻结 Qwen hidden，不得读取 family、route、AST、teacher trace、reasoning budget 或旧 checkpoint。

只有单 seed C1 全 Gate 通过，才允许增加另外两个 fresh seed；只有三 seed C1 通过，才进入 C2 的 direct/text-CoT/K=1/K=8 matched Pareto。若 C1 失败，应按其预注册 Gate 停止并回到整体架构归因，不恢复 H1/WD 或“先写后删”路线。
