# V2-R1R P0-D v8 R0B-structural 冻结设计

日期：2026-08-01  
阶段目标：资格化 G02–G06 structural audit，使后续 generator 不能用自报字段、自算 expected 或浅层配额骗过数据 Gate  
证据等级：measurement-system qualification；不是生产数据、模型、训练或架构证据

## 1. 核心判断

v7 已经通过主设计层验收。当前公共 simulator 对冻结 schema 的语义、非法输入与 operand 作用域取得了足够资格证据；继续做 v8 R0A 只会把一个局部切面推向无意义的完美化。下一步必须换层，直接验证依赖该 simulator 的结构审计器。

R0B 不恢复 v4 的 1,440-record reference generator，也不让执行层再实现第二套语义引擎。它使用主设计层冻结的小型 **qualification bundle**：

1. accepted v7 simulator 是唯一语义 replay 引擎，三个公共文件 hash 必须逐字节不变；
2. 正控只包含足以逐项杀伤 G02–G06 的手写 ERE/CPS 小世界、pair、claim、provenance 与 hard-negative certificate；
3. 每个负控是冻结的声明式 JSON patch 或完整负向 record replacement，不允许 fault harness 自行推理如何破坏；
4. 正控、41 个定向负控、4 个 metamorphic control 和 metric-to-fault kill ledger 全部走同一个公共 `audit_structural_bundle()` 路径；
5. R0B 只证明审计机制能测，不宣称任何生产数据已经满足这些 Gate。

阶段名固定为 **R0B-structural**。G07/G08 的真实统计 learner 属于 R0C；production generator 与完整 integration 属于 R0D/R1，均不进入本轮。

## 2. 上游资格与不可修改边界

R0B 的上游资格固定为：

- v7 主审判决：`R0A-lattice accepted`；
- v7 artifact root：`artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1/`；
- v7 evidence-seal SHA-256：`794CB9518F094EE6CF99800AF0B5848D22A7746DAA5E76A84489F3F8201987F0`；
- accepted simulator SHA-256 由 v8 frozen manifest 从当前三个文件读取并冻结：
  - `src/yggdrasil_v2/r1_revalidation/__init__.py`；
  - `src/yggdrasil_v2/r1_revalidation/common/__init__.py`；
  - `src/yggdrasil_v2/r1_revalidation/common/simulator.py`。

这三个文件在 v8 中只读。R0B 只能新增新的 `audit/` package、直接替换 CLI，并在执行切换时删除 `tests/v2_r1r_v7/`。不得把 audit 逻辑塞进 simulator，不得保留 v7 CLI alias，不得读取 v7 fixture/cell id 来特判。

允许的 runtime 形状只有：

```text
src/yggdrasil_v2/r1_revalidation/
  __init__.py                 # accepted v7 bytes，不改
  common/
    __init__.py               # accepted v7 bytes，不改
    simulator.py              # accepted v7 bytes，不改
  audit/
    __init__.py
    artifact.py               # G02
    replay.py                 # G03
    pairs.py                  # G04
    structure.py              # G05/G06
    structural.py             # 唯一编排与 public report
experiments/
  v2_r1_revalidation.py       # v8 frozen CLI template
```

不允许 `generate/`、`shortcuts.py`、model、cache、train、renderer 或 compatibility module。

## 3. qualification bundle

### 3.1 输入树

公共审计器只接受一个目录，输入树精确为：

```text
manifest.json
records.jsonl
causal-pairs.json
language-pairs.json
composition-controls.json
input-seal.json
source_snapshot/<manifest 声明的精确文件集>
```

`input-seal.json` 哈希前五个逻辑输入文件及 snapshot 内每个文件；不哈希自身，也不包含 audit 输出。任何 missing、extra、count、hash、schema 或 snapshot 漂移都归 G02。审计器不得自动修复、重封或忽略未知文件。

`records.jsonl` 的每条 record 精确包含：

```text
schema_version
example_id
family
split
source_text
qwen_token_count
model_view
program_ast
answer_semantic
answer_index
label_mapping
teacher_output_sha256
claims
semantic_fingerprint
surface_fingerprint
fold_group
structure_certificate
```

`model_view` 只允许：

```text
example_id
source_text
reasoning_budget
valid_choice_mask
```

禁止字段按递归 key 检查，至少包括 answer、AST、teacher、claim、span、role、state、validity、certificate、candidate role 和 derivation。只把顶层 key 改名或嵌入列表不能绕过。

### 3.2 tokenizer 与 provenance

tokenizer 固定为：

```text
model_id = Qwen/Qwen3.5-2B
revision = 15852e8c16360a2fea060d615a32b45270f8a8fc
class = Qwen2TokenizerFast
add_special_tokens = false
max_source_tokens = 1024
```

只能从本机精确 revision snapshot 以 `local_files_only` 加载；不得联网、fallback 到其他 revision、regex 估算或静默截断。manifest 记录 Python、transformers、tokenizers 版本与实际 tokenizer class；audit 独立 recount 每条 source，并要求 stored count 精确相等。

manifest 还必须包含 Git HEAD、dirty flag、status/diff hash，以及 `semantic/surface/labels/claims/pairs` 五条 named stream 的独立 seed。R0B qualification bundle 本身不随机生成语义，但仍验证未来 generator 必须提供的 provenance 形状和独立 stream 身份。

## 4. 主设计层冻结语义控制

### 4.1 ERE 五类 provenance

五个 positive record 都有 3 个有序事件；audit 不信任 `necessary=true`，而是逐事件替换为同参数空 rule 后 fresh replay，要求每次最终 answer 改变。

| case | provenance | 结构 | answer | accepted simulator output SHA-256 |
| --- | --- | --- | --- | --- |
| `ere_initial_copy` | `initial-copy` | 三段 COPY 链 | `amber` | `3E81ECD1A268F773929C9AD8236D6143E9DC24A8EC281A52BDF9A735E8E98BE3` |
| `ere_condition_true` | `condition-true` | SET guard → IF true/COPY → COPY | `jade` | `C56173DCDE33CB3C0AF9153158C344BD3F0B90EC8574AEAA478F91D0D3662F34` |
| `ere_condition_false` | `condition-false` | SET guard → IF false/COPY → COPY | `jade` | `76DC276B299481473D8ABF5EDF473FC1658E7C0A8884636E08F76E74578CDE11` |
| `ere_swap_source` | `swap-source` | SWAP → COPY → COPY | `blue` | `758AF8A0E68CE19166A3397690B05FCDF81E84DAEAC462D0F389D107775177AB` |
| `ere_relation_neighbor` | `relation-neighbor` | LINK → FOREACH/COPY → COPY | `gold` | `B0377F54018DDC1F317F0F99FC4033864A5EF15B41E9DB372026419C4138DF5D` |

每条 case 的三个 ablation answer 已冻结：前四类依次回退到各自非目标 literal，relation case 三次都回退为 `gray`。五类计数必须精确 `1/1/1/1/1`；composition control 至少含三类且最大占比不高于 `1/3`。provenance certificate 必须给出实际 op/predicate/path，audit 对 AST 和 executed trace 复核，不能只统计字符串标签。

### 4.2 CPS rich / NONE skeleton

`cps_rich` 冻结为：

- actions：`unlock`、`charge`、`finish`、`detour`、`unsafe_finish`；
- P*：`[unlock, charge, finish]`，depth `3`、cost `3`、唯一最优；
- valid-suboptimal：`[detour, unlock, charge, finish]`，cost `5`；
- 五个 invalid candidate 分别形成 `precondition`、`unknown_action`、`budget`、`final_constraint`、`goal`；
- invalid plan length 同时有 `<`、`=`、`>` P*；
- answer `0`，accepted simulator output SHA-256 `8BC1489C41B1991E62871DDC95B384717919EE3E6CA1E9065AD7AF2983FD6D67`。

`cps_none` 与它共享 actions、candidate plan/order、goal 和 final constraint，唯一 AST leaf 变化是 `budget: 5 → 2`。所有 candidate 均 invalid，answer `null`，output SHA-256 `040B01454944A0BC43ADFE669ED799ED25B4AE70BDF6E3804A97B29848B83D36`。audit 必须 fresh replay 两边，并在移除唯一声明 leaf 后比较 skeleton；不能信任 `is_none` 字段。

rich AST 同时提供四个 composition witness：unlock→resource→goal、同 goal/不同 final constraint、resource × action cost × budget、破坏后续必要 fact。每个 witness 都保存 AST path；audit 验证 path 指向正确 kind/condition/effect 与实际 candidate outcome。

### 4.3 causal / language pair

ERE causal pair 只修改 `/initial_state/attributes/flag/mode` 的 `open ↔ closed`，answer `allow ↔ deny`，两边 output SHA-256 分别为 `24B61C0C924623886697617206C302A8C5F01F233BA9EE97A3F39C155F488C0D` 与 `D1BBDA915D376293FF6182EA263EE3E0A9481AF668130010F72D7FED51001704`。

CPS causal pair 只修改 `/actions/0/cost` 的 `1 ↔ 3`，budget 固定为 `2`，answer `0 ↔ null`，两边 output SHA-256 分别为 `E1C15CEF9160DEBF2ED0CC1EB0A100CB133D3AFFC8CFCF19D184F29C47980BF3` 与 `688ECC45D7C977711BFE572744D2F103A041D1AAAEBC5764A6DF1E7487C121B9`。

language pair 对同一 AST 提供 train/OOD 两个完整 source。删除首行 opening 后正文仍须不同；冻结 feature 分别使用 `active_passive` 与 `clause_order_reversed`，并各自包含可复核的两侧文本 span。pair 两边必须共享 semantic fingerprint、choice mask、label mapping 与 fold group；surface fingerprint 必须不同。

### 4.4 claim 控制

ERE 必须各有一对 `attribute_value`、`relation_exists`、`condition_truth`；CPS 必须各有一对 `prefix_legality`、`resource_value`、`cost_value`、`fact_truth`、`goal_status`、`final_constraint_status`。每对 positive/negative：

- 只改变 predicate JSON 的一个 leaf；
- fresh replay truth 相反；
- text 由冻结 grammar 独立 parse 回完全相同 predicate；
- label 与 fresh truth 一致；
- pair id、family、prefix/candidate reference 合法。

共 9 个 kind、18 条 claim，正负精确 `9/9`。claim text 不进入 model view。

## 5. G02–G06 原子 conjunction

### G02 artifact / model view / token provenance

1. input seal 与 exact input tree；
2. manifest schema、record/pair counts 与 file hashes；
3. snapshot file set 与 hash；
4. runtime、Git 与五条 named stream provenance；
5. model view exact-field rate `1.0`；
6. recursive forbidden field count `0`；
7. tokenizer id/revision/class/special-token config exact；
8. token recount match rate `1.0`；
9. max source token `<=1024`。

### G03 semantic replay / teacher / claim

1. ERE replay output digest rate `1.0`；
2. CPS replay output digest rate `1.0`；
3. answer、trace/delta/reasoning budget 与 candidate report contract rate `1.0`；
4. 9 个 required claim kind 全部存在；
5. claim truth 与 positive/negative balance rate `1.0`；
6. single-leaf claim pair rate `1.0`；
7. claim text parse rate `1.0`。

### G04 overlap / causal / language / composition / fold

1. semantic/surface fingerprint 独立复算率 `1.0`；
2. 未声明 semantic 与 surface overlap 均为 `0`；
3. ERE/CPS causal pair single-leaf、endpoint、semantic/local answer flip rate `1.0`；
4. mutation family count deviation `<=1`；
5. language AST identity、正文变化、feature span、paired surface schedule rate `1.0`；
6. pair fold group rate `1.0`；
7. composition witness 与 component availability contract rate `1.0`。

### G05 ERE structural difficulty

1. depth `>=3` rate `1.0`；
2. necessary event declaration完整且逐事件 ablation answer flip rate `1.0`；
3. provenance witness AST/trace contract rate `1.0`；
4. 五类普通 provenance 精确平衡；
5. composition 至少三类且最大占比 `<=1/3`。

### G06 CPS structural difficulty

1. P* depth `>=3`、valid、unique minimum；
2. 至少一个 valid-suboptimal；
3. 至少五个 invalid candidate；
4. failure reason 五类齐全；
5. invalid length 同时覆盖 `< / = / > P*`；
6. NONE pair 单叶、skeleton、fresh no-valid contract；
7. frozen NONE ratio exact；
8. 四个 composition witness 全部可从 AST/outcome 复核；
9. distractor certificate 不被 model view/source-visible canonical role 引用。

报告必须保存每个原子 metric 的 raw numerator/denominator 或 raw count；Gate 只能从这些原始值计算。不存在“写 `passed=true` 但没有底层分子/分母”的合法报告。

## 6. 定向 fault matrix

冻结 fault case 共 `41` 个，每个 case 都在 qualification bundle 的独立副本上应用声明式 patch，再调用同一个公共 `audit_structural_bundle()`；`audit-structural` CLI 另以正负各一个 bundle 验证它只是该 API 的无状态包装。这样 tokenizer 与 schema 只在单个 preflight 进程内加载一次，不用 41 次 Python 冷启动改变所测逻辑。除 G02 故意攻击 seal 的 case 外，fault harness 使用独立标准库 sealer 重算 input seal；每个 case 的 expected false Gate set 必须精确等于一项，不能用其他 Gate 的连带失败冒充命中。

| Gate | fault ids | 目标 |
| --- | --- | --- |
| G02 | `S201–S209` | seal/snapshot、manifest/provenance、model view、tokenizer/count/limit |
| G03 | `S301–S307` | ERE/CPS replay、teacher、claim label/single-leaf/text/kind |
| G04 | `S401–S409` | semantic/surface overlap、causal、language、fold、composition |
| G05 | `S501–S506` | depth、event declaration/necessity、provenance witness/quota/composition |
| G06 | `S601–S610` | P*、unique optimum、valid-suboptimal、invalid count/reason/length、NONE、composition/distractor |

`fault-spec.json` 必须为每个 id 保存：target gate、patch list、是否重封、expected false Gate set、expected killed metric ids。patch engine 只实现通用 `replace/add/remove/copy-record`；不存在按 fault id 写分支的 `mutate()`。预测试逐行验证 path 存在、patch 后 JSON schema 可读且每行确实改变 bytes。

metric kill coverage 必须为 `1.0`：第 5 节每个原子 metric 至少被一个登记 fault 的 raw value 杀伤。Gate kill 不能代替 metric kill。

## 7. metamorphic、独立性与公共 API

四个 positive metamorphic control：

1. JSON object key 顺序变化不改变 report；
2. 无语义 record 行顺序变化不改变 report；
3. 全局一一 alpha-renaming 不改变 semantic fingerprint/G03/G05/G06；
4. source 连续空白规范化不改变 normalized surface fingerprint。

公共 API 固定为：

```python
audit_structural_bundle(root: str | Path) -> dict[str, Any]

__all__ = ["audit_structural_bundle"]
```

返回值必须 JSON-only，输入目录和已加载对象不变。audit package 只能 import `common` 公共 simulator API，不得 import `tests`、control spec、fault spec、runner 或 expected digest 表。control materializer、fault harness、patch engine 和 independent sealer 不得 import `src`；token recount reference 与 runtime audit 分别独立加载 pinned tokenizer，不共享 helper。

CLI 与 frozen template 字节一致，只提供：

```text
audit-structural --bundle <dir>
preflight-structural
seal-structural
```

旧 `preflight-lattice` / `seal-lattice` 必须不存在。

## 8. formal protocol 与 artifact

`preflight-structural` 在临时目录 materialize positive bundle，运行 positive audit、41 faults、4 metamorphic、import/file/hash guard 和两次 deterministic report，可反复执行；固定 formal root 必须不存在。

`seal-structural` 无参数，固定 root：

```text
artifacts/v2-r1r/p0d-v8-r0b-structural-20260801-1
```

formal 先用 `mkdir(exist_ok=False)` 占用 root 并写 `attempt.json`，再执行任何 bundle materialization/audit。成功 root 精确包含：

```text
attempt.json
control-bundle/
structural-report.json
fault-matrix.json
coverage-ledger.json
assessment.json
run-metadata.json
evidence-seal.json
```

evidence seal 覆盖除自身外的全部文件和 `control-bundle/` 递归文件，保存相对路径与 SHA-256。root 已存在时所有 formal 调用均 `BLOCKED`。

## 9. PASS_R0B_STRUCTURAL conjunction

只有以下同时成立才能得到机器 `PASS_R0B_STRUCTURAL`：

1. v7 accepted artifact/evidence 与三个 simulator file hash exact；
2. v8 frozen design/command/spec/tests/runner/CLI manifest 全部未漂移；
3. positive bundle 的 G02–G06 全真，所有原子 raw metric 达标；
4. 41/41 fault 的实际 false Gate set exact；
5. metric kill coverage `1.0`，无未登记或重复 fault id；
6. 4/4 metamorphic 通过；
7. 两次 report、fault matrix 与 coverage ledger bytes 确定；
8. import boundary、公共 API、文件边界与输入不变通过；
9. tokenizer 使用本地 pinned revision，recount/limit exact，无网络/fallback；
10. formal attempt-before-evaluation、fixed root 与成功 artifact tree exact；
11. evidence seal 的全部递归 hash 可复算；
12. `authorization_created=false`，没有 R0C/R0D/R1/P0-M/model/train artifact。

机器 PASS 仍需父任务独立验收。父任务必须构造 registry 外的新鲜 model-view 嵌套泄漏、renamed semantic overlap、额外无效 ERE event、不同 CPS hard-negative/NONE skeleton 等黑盒负控；任何一项漏检都拒绝 R0B，而不是在原版本补测试重跑。

## 10. 停止与证据边界

预测试失败只允许修改 v8 新增 audit modules 和 CLI；需要修改 simulator、design、control/fault spec、expected digest、threshold、tests、runner 或 fixed root 才能通过时状态为 `BLOCKED`，返回主设计层。

`seal-structural` 一旦调用，无论成功、失败、异常或中断都立即停止。不得修改实现、删除/覆盖 artifact、换 suffix 或重跑。R0B 通过只说明 G02–G06 的测量机制取得资格；R0C、R0D、R1 generator、P0-M、模型、cache、训练/GPU 和架构结论仍等待另行合同。
