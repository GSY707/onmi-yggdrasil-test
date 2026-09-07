# V2-R1R P0-D v9 R0B-invariant 设计

日期：2026-08-02  
阶段目标：资格化一个由有限 profile、typed derivation 与 adversary lattice 共同约束的 G02–G06 审计器  
证据等级：measurement-system qualification；不是生产数据、模型、训练或架构证据

## 1. 核心判断

v8 不是因为 fault 数量不足而失败，而是把 `metric kill coverage` 误当成了书面不变量覆盖。新的 R0B 不继续增加同类登记 fault，而是改变审计表示：

1. qualification bundle 的有限 profile 由 runtime audit 固定，不能通过修改 manifest count 自我扩容；
2. claim、causal、language、ERE provenance、CPS composition 都由独立计算得到的关系判定，不允许输入 certificate 自报 witness path；
3. fault harness 从不变量的变换等价类生成 adversary lattice，覆盖 missing、extra、duplicate、swap、redirect 与 surface-preserving semantic break；
4. 自然语言范围收缩为两族可逆 grammar。R0B 只证明 parser 能把受控 train/OOD render 还原为同一 AST projection，不宣称能审计任意自然语言释义；
5. source provenance 以 bundle 内 snapshot/seal 为真源。当前 worktree 只在 formal build guard 中检查，不参与 sealed bundle 的后续 replay Gate。

阶段名固定为 **R0B-invariant**。v8 artifact 和实现只作 rejected diagnostic；v9 直接替换当前 audit/CLI/tests，不提供兼容 alias。R0C、R0D、R1 generator、P0-M、模型、cache、训练与 GPU 不进入本轮。

## 2. 上游与不可修改边界

上游资格仍固定为 v7 `R0A-lattice accepted`：

- artifact：`artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1/`；
- evidence seal：`794CB9518F094EE6CF99800AF0B5848D22A7746DAA5E76A84489F3F8201987F0`；
- simulator 三文件保持 v8 formal 时的 accepted SHA-256，v9 不得修改：
  - `src/yggdrasil_v2/r1_revalidation/__init__.py`；
  - `src/yggdrasil_v2/r1_revalidation/common/__init__.py`；
  - `src/yggdrasil_v2/r1_revalidation/common/simulator.py`。

允许的 runtime 形状：

```text
src/yggdrasil_v2/r1_revalidation/
  __init__.py
  common/
    __init__.py
    simulator.py
  audit/
    __init__.py
    artifact.py
    replay.py
    language.py
    pairs.py
    provenance.py
    structure.py
    invariant.py
experiments/
  v2_r1_revalidation.py
```

`audit/__init__.py` 只导出 `audit_invariant_bundle`。旧 `audit_structural_bundle`、`structural.py` 与 v8 CLI 命令全部删除。audit 只能 import `common` 的三个公共 simulator API；不能 import tests、control spec、adversary harness、runner、fixture id 或 expected digest 表。不得新增 generator/model/cache/train/renderer/compatibility package。

## 3. Qualification profile

输入树保持小而封闭：

```text
manifest.json
records.jsonl
causal-pairs.json
language-pairs.json
composition-controls.json
input-seal.json
source_snapshot/<manifest 精确声明的文件集>
```

runtime audit 固定以下 profile，而不是信任 manifest 自报：

| 对象 | 精确要求 |
| --- | --- |
| records | 11；ERE 7、CPS 4 |
| role | `ere_core=5`、`cps_rich=1`、`cps_none=1`、`causal_member=4` |
| claims | 18；9 个 required kind 各一对 |
| claim polarity | 每 pair 精确一条 `positive/label=true` 与一条 `negative/label=false` |
| causal pairs | 2；ERE/CPS 各 1 |
| language pairs | 2；ERE/CPS 各 1 |
| ERE provenance | 五类各 1 |
| CPS rich/NONE | 各 1；NONE ratio 精确 `1/2` |

record 顶层 key set、model-view key set、claim key set、pair key set与 role-specific certificate key set 都必须 exact；未知 key、缺 key、重复 id、重复 pair id、空字符串 id 均 fail-closed。manifest count/hash 既要匹配实际文件，也要匹配固定 profile。

## 4. 自包含 provenance 与 tokenizer

`input-seal.json` 精确哈希五个逻辑输入文件与 snapshot 内全部文件。manifest 记录 snapshot map、snapshot-set digest、Python/transformers/tokenizers 描述和 pinned tokenizer：

```text
Qwen/Qwen3.5-2B
revision = 15852e8c16360a2fea060d615a32b45270f8a8fc
class = Qwen2TokenizerFast
add_special_tokens = false
max_source_tokens = 1024
```

audit 独立加载本地精确 revision、复算每条 record 与 language render 的 token count。Git HEAD/status/diff 只由 formal runner 写入 run metadata 并由 frozen build guard 证明；bundle audit 不读取当前仓库 Git 状态。这样 sealed bundle 在结果文档同步后仍可得到相同报告。

## 5. G02–G06 原子不变量

### G02：artifact、profile、view、tokenizer

1. exact tree/input seal；
2. manifest schema、actual counts/hash 与固定 profile exact；
3. snapshot set/hash/digest exact；
4. runtime descriptor 与 named streams well-formed/exact；
5. record/model-view/claim/pair schema exact；
6. model view exact-field rate `1.0`；
7. recursive forbidden key count `0`；
8. tokenizer identity exact；
9. token recount rate `1.0`；
10. max token `<=1024`。

### G03：fresh replay 与 claim algebra

1. ERE/CPS teacher digest rate 各 `1.0`；
2. answer/index/mask/reasoning-budget/candidate trace contract rate `1.0`；
3. claim total 精确 18；
4. required kind 每种精确两条且 family 正确；
5. claim id、pair id 全局唯一，pair size 精确 2；
6. `positive↔true`、`negative↔false` 极性 rate `1.0`；
7. fresh truth 与 label rate `1.0`；
8. pair 只改变 predicate 一个 leaf；
9. frozen grammar text 可完全 parse 回 predicate。

### G04：fingerprint、causal 与可逆 language binding

1. semantic/surface fingerprint 独立复算 rate `1.0`；
2. 未声明 semantic/surface overlap count `0`；
3. 两个 causal pair 均为 AST single-leaf diff，端点、fresh answer flip、mask/mapping/fold exact；
4. mutation family count deviation `<=1`；
5. language pair profile 精确 ERE/CPS 各 1；
6. 每个 train/OOD text 必须被且只被一个 frozen reversible grammar 完全解析；
7. parser 输出的 family/semantic projection 必须与引用 record 的 AST-derived projection完全相等；
8. train/OOD projection 相同、variant 不同、去 header 后正文不同、surface fingerprint 不同；
9. causal/language fold-group contract 与 composition component contract exact。

可逆 grammar 只有：

- `ERE-ACTIVE-V1` / `ERE-PASSIVE-V1`：编码 initial attribute、IF predicate、then/else SET 和 query；
- `CPS-FORWARD-V1` / `CPS-REORDERED-V1`：编码 initial fact、action precondition/effect/cost、budget、goal 和 candidate plan。

parser 使用 full-match line grammar，不接受额外行、遗漏行、自由文本或自报 span。production renderer 不在 R0B；R0D 必须单独资格化其每个 template。

### G05：ERE typed provenance

audit 从 AST 与 fresh trace 自己执行 typed dataflow：initial attribute 建立 origin；SET/COPY/SWAP 传播 payload；IF 记录独立复核的 predicate outcome；LINK/UNLINK 建立 relation edge provenance；FOREACH_LINKED 将实际 edge、neighbor 与 effect 绑定。输入 certificate 不再包含 witness path。

五个 ERE core 必须：

1. role count 精确 5、event depth `>=3`；
2. `necessary_event_indices` 等于完整 event index set；逐 event 空 rule ablation 后 answer 均变化；保存的 `ablation_answers` 与 fresh 结果逐项相等；
3. query payload provenance 能由 typed derivation 完整重建；
4. derived provenance class 与声明 class 相等，并精确覆盖 `initial-copy/condition-true/condition-false/swap-source/relation-neighbor` 各一次；
5. condition branch 必须同时验证 predicate state、trace marker、所选 branch 与 payload effect；relation provenance 必须把 LINK edge、真实 neighbor、FOREACH effect 和 query chain 连通；
6. ordinary/composition provenance quota 从 derived class 计算，最大 share `<=1/3`。

public report 保存 derived witness digest 与结构化失败原因，但不把 witness 放进 model view。

### G06：CPS typed composition

rich/NONE certificate 只保存 role、P*、valid/invalid index 与 expected reason/length；不保存 composition witness paths。audit 从 AST 与 fresh candidate outcomes 派生：

1. unique minimum P* 且 plan depth `>=3`；
2. 至少一个 valid-suboptimal；
3. 至少五个 invalid candidate；
4. `precondition/unknown_action/budget/final_constraint/goal` 五类齐全；
5. invalid length 覆盖 `< / = / > P*`；
6. rich/NONE AST 只有一个 budget leaf 不同，移除该 leaf 后 skeleton 完全相同，NONE fresh valid count `0`；
7. rich/NONE role/count/ratio exact；
8. 从 action precondition/effect、candidate plan、budget 与 outcome 派生 `unlock_resource_goal`、`same_goal_different_final`、`resource_cost_budget`、`early_fact_destruction` 四类 witness；
9. canonical-role leak 在全部 record 的 source/model view 递归扫描，count `0`。

## 6. Adversary lattice，而非 fault 清单拟合

formal 仍保留每个 metric 至少一个定向 killer，但主要资格证据改为四个独立变换格：

1. **profile lattice**：对 record/claim/pair 做 remove、extra、duplicate-id、duplicate-pair、role swap、label-role decouple、kind-family redirect；
2. **language lattice**：对每个 grammar slot 做同类型值替换、跨 family record redirect、缺行、加行、乱序、nonsense、只保持 surface fingerprint schedule 的语义破坏；
3. **provenance lattice**：改变 declared class、ablation answer/index、插入 answer-neutral event、破坏 IF control 与 relation edge/effect chain；所有 teacher/count/hash/seal 等非目标派生字段独立刷新；
4. **CPS lattice**：分别破坏 P*、suboptimal、五类 invalid、length、NONE single leaf 和四种 derived composition，同时保持其他 Gate 可复算。

每个变换 case 必须记录 target invariant family、实际 false Gate set、killed metric set 与输入 byte diff。除 G02 seal case 外都重算 manifest/input seal，不能借连带 G02 失败冒充命中。lattice harness 不 import runtime audit；runtime audit 不 import harness。

本合同冻结为 45 个 raw metric、44 个定向 case 和 4 个不参与 metric mapping 的 family holdout，共 48 个 adversary case。四个 holdout 分别属于 profile、language、provenance 与 CPS；它们只证明对应变换族被拒绝，不能帮助 45/45 metric kill coverage。

正式门槛：

- 每个 lattice case 至少杀伤其目标 metric；
- 单目标 case 的 false Gate set 必须精确；组合 holdout 允许多个 Gate false，但必须包含所有目标 Gate；
- 每个原子 metric 至少被一个定向 case 杀伤；
- 每类变换至少保留一个未用于定向 metric mapping 的 holdout case；
- main review 仍必须从公共 API 构造 registry 外 case；任何漏检都拒绝 v9，不在原 formal 上修补重跑。

## 7. Positive metamorphic

六类合法不变换固定为：JSON object key 顺序、record row order、claim pair row order、全局 alpha-renaming、source whitespace normalization、bundle 离线 relocation/replay。所有 metamorphic 都必须走公共 API，并比较完整 report canonical bytes；变换前后的输入都必须保持只读。

## 8. Public API、CLI 与 formal artifact

公共 API：

```python
audit_invariant_bundle(root: str | Path) -> dict[str, Any]
```

返回 JSON-only，输入 bytes 不变，任意 missing/malformed/duplicate/unknown input 都返回 `passed=false` 报告而不是泄漏未捕获异常。

CLI 只提供：

```text
audit-invariant --bundle <dir>
preflight-invariant
seal-invariant
```

formal fixed root：

```text
artifacts/v2-r1r/p0d-v9-r0b-invariant-20260802-1
```

成功 root 精确包含：

```text
attempt.json
control-bundle/
invariant-report.json
adversary-ledger.json
coverage-ledger.json
metamorphic-ledger.json
assessment.json
run-metadata.json
evidence-seal.json
```

root 必须在任何求值前以 `mkdir(exist_ok=False)` 占用。evidence seal 覆盖除自身外全部递归文件。固定 root 存在即 BLOCKED。

## 9. PASS_R0B_INVARIANT conjunction

只有以下全部成立才能机器 PASS：

1. v7 accepted evidence 与 simulator hash exact；
2. v9 design/command/tests/harness/runner/CLI frozen manifest exact；
3. positive G02–G06 与全部 raw metric 为真；
4. 四个 adversary lattice 全部达到目标 Gate/metric contract；
5. metric kill coverage `1.0`，每个变换 family 有独立 holdout；
6. 六类 metamorphic 全通过；
7. API 两次报告确定、输入不变、malformed fail-closed；
8. bundle 在 current worktree 文档漂移后仍可从内置 snapshot 重放同一报告；
9. import/file/CLI boundary 与旧 v8 删除检查通过；
10. formal attempt-before-evaluation、artifact tree 与 evidence seal exact；
11. `authorization_created=false`，没有 R0C/R0D/R1/P0-M/model/train artifact。

机器 PASS 只代表 R0B-invariant 候选，仍需父任务独立 main review。

## 10. 停止规则

formal 前可以修改 v9 新设计、tests、harness 和实现，但每次设计语义改变都必须重新冻结全部 hash、重新从 red 状态验证。只有主设计层在 preflight 与 registry 外 holdout 后认为已无已知反例，才允许调用一次 `seal-invariant`。

formal 一旦调用，无论 PASS、FAIL、异常或中断都立即停止。不得修补、覆盖、换 suffix 或重跑；任何 main-review false negative 都把 v9 判为 machine-pass/main-review rejected，并要求另立新合同。R0C 及以后始终需要单独授权。
