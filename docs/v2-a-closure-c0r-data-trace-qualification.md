# V2-A Closure C0R：数据与 Trace 资格修复合同

日期：2026-08-24

身份：`V2-A-CLOSURE-C0R-20260824-1`

状态：实现中；正式命令尚未消费。

## 1. 目标与边界

上一轮 `V2-A-CLOSURE-C0-20260823-1` 已以 `FAIL_V2_A_CLOSURE_C0_READINESS` 封存，失败项只有 C004 与 C006。C0R 是直接切换的新 successor，不修改、覆盖或重跑旧 C0，也不修改 P0-D v17、P0-M v5 或其 artifacts。

C0R 只关闭两个准备缺口：第一，生成一个新身份的 ERE/CPS bank，使 ordinary ERE relation-query 的语义答案配平；第二，用严格、可逆、可重放的 compact trace 取代旧 P0-M 的有损字符串摘要。C0R 不训练模型，不评价 latent 架构，不产生四臂 Pareto，也不授权 C2、C3、V2-B 或 V2-C。

## 2. 新数据身份

数据资格身份固定为 `V2-A-CLOSURE-C0R-DATA-TRACE-20260824-1`，正式 seed 为 `2026082401`，规模保持每族 train `4096`、每个 validation/OOD/causal cell `1536`。新 generator、record 与 manifest identity 分别为：

- `v2-a-c0r-relation-balanced-v1`
- `yggdrasil.v2-a.c0r.record.v1`
- `yggdrasil.v2-a.c0r.dataset-manifest.v1`

旧 v17 production generator 作为逐哈希固定的 immutable substrate；新 package 负责新的 outcome schedule、record identity、manifest、public-forward boundary 和 qualification。不得修改旧 substrate 源码。

ordinary ERE 的题型调度保持不变：只有 train、validation 与 language OOD 中 `index % 12 == 0` 的记录是 relation-query。对这些记录，用 `relation_ordinal = index // 12` 交替生成 TRUE/FALSE base；query 本身固定，counterfactual 只改变第二个 LINK 的 target，因此仍是单 leaf、严格 answer flip。composition、length、entity OOD 和全部 CPS 不改变构造；causal-pair 仍在同一 mapping/template/seed 下包含一正一负。

不得通过隐藏 TRUE/FALSE legend、改变 relation 占比、降低 shortcut 阈值或更换随机 seed 来掩盖失败。

## 3. Compact trace v1

新 trace 使用唯一 canonical 文本：

```text
CT1 <canonical-json>
Answer: <A-I>
```

JSON 顶层只允许 `f` 与 `t`。`f` 是 `ERE` 或 `CPS`；`t` 是严格定长数组序列：

- ERE event：`[event_index, rule, sorted_argument_pairs, executed_ops, delta_budget]`；
- CPS candidate：`[candidate_index, plan, legal, valid, total_cost, budget_ok, goal_ok, final_constraints_ok, failure_reasons, steps]`；
- CPS step：`[position, legal, failure, cost]`。step action 由同一 candidate 的
  `plan[position]` 唯一恢复，不在 target 中重复编码。

parser 拒绝 duplicate key、非 finite、额外字段、错误类型、bool-as-int、非 canonical JSON、未知 family、错误顺序、trailing bytes 和 Answer/payload 合同外文本。semantic verifier 只在离线 evaluator 中读取 `program_ast` 与 `label_mapping`，重新调用 source simulator 并逐字段比较；这些字段不得进入模型 forward。四臂 public forward 的唯一内容字段仍是 `source_text`。

全 bank 的每条记录必须同时满足 formatter→parser→formatter byte roundtrip、fresh simulator semantic replay、canonical serialization、answer binding 和 `<=512` target tokens。每条记录还接受一个确定性的 schema-valid 单点 trace fault；所有 fault 必须被 verifier 拒绝。另有固定 malformed suite，禁止自动修复或 answer-only 降级。

## 4. 数据与 trace Gates

数据资格报告固定为 D001–D012：

| Gate | 判据 |
| --- | --- |
| D001 | 旧 P0-D v17 prerequisite、seal 与固定 substrate source hash 可复验。 |
| D002 | 新 manifest、record schema/provenance、26,624 records、全局唯一 ID、精确 public-forward view 与生成统计一致。 |
| D003 | source parse/rerender、fresh simulator、teacher trace、answer、fingerprint、certificate 与 claims 全量 replay。 |
| D004 | 全局 semantic/surface fingerprint 隔离；每个 causal pair 单 leaf、同 surface factors、答案翻转。 |
| D005 | ERE 域、primitive、dependency、composition、length/entity 与 counterfactual 必要性保持资格。 |
| D006 | CPS optimum/NONE、hard negatives、composition、horizon/distractor 保持资格。 |
| D007 | local label balance、alpha/permutation/choice invariance 通过。 |
| D008 | language template、forbidden marker 与 Qwen tokenizer 长度通过。 |
| D009 | 原 production source-only shortcut 全 cell Gate 通过。 |
| D010 | claim balance 与 heldout claim-NB Gate 通过。 |
| D011 | relation visible-pattern 新 Gate 通过。 |
| D012 | compact trace strict parse、全 bank roundtrip/replay、token cap、malformed rejection 与 directed fault-kill 全通过。 |

D011 保持旧 C0 冻结阈值：每个被测 relation cell support 至少 `128`、TRUE/FALSE 各至少 `64`、dominant semantic mass 不高于 `0.80`；严格 source-only visible-TRUE oracle 的 Bonferroni 单侧 Wilson upper 不得超过该 cell 随机率 `+0.10`。causal relation pair 另要求逐 pair 一真一假。

## 5. C0R Gates 与授权

C0R 复验旧 C0 seal、新 data/trace seal、固定 fairness contract 和 D001–D012，并保持 C001–C008 的原语义：

- C001–C004 改由新 bank 的 D001–D011 提供证据；
- C005 仍把 P0-M v5 限定为 training-path smoke；
- C006 要求 D012 与四臂公平合同同时通过；
- C007 复用旧封存的历史证据矩阵；
- C008 继续排除 H1/WD route/projection 与旧 checkpoint。

全部通过时状态为 `PASS_V2_A_CLOSURE_C0R_READINESS`，只授权 C1 single-seed implementation/eligibility。任一输入缺失为 `INCOMPLETE_V2_A_CLOSURE_C0R_READINESS`；输入完整但 Gate 失败为 `FAIL_V2_A_CLOSURE_C0R_READINESS`。

## 6. 单次身份与停止规则

固定 roots：

- data/trace：`artifacts/v2-a/closure-c0r-data-trace-20260824-1/`
- data/trace lease：`artifacts/v2-a/closure-c0r-data-trace-20260824-1.preflight-lease.jsonl`
- C0R：`artifacts/v2-a/closure-c0r-20260824-1/`
- C0R lease：`artifacts/v2-a/closure-c0r-20260824-1.preflight-lease.jsonl`

正式 CLI 不接受 seed、阈值、输入 root 或输出 root 覆盖。root 或 lease 已存在即拒绝。数据资格只有 D001–D012 全通过才允许启动 C0R；C0R 任一 Gate 失败即封存停止。两阶段都不得加载 checkpoint、创建 optimizer 或写模型。
