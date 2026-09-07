# V2-R1R P0-D v6 R0A-strict 冻结设计

日期：2026-08-01  
阶段目标：证明公共 ERE/CPS simulator 不仅能复现手写语义，还会在执行前完整拒绝非法 AST  
当前权限：只允许直接切换并执行 R0A-strict；R0B 及以后全部未授权

## 1. 核心判断

v5 的 14/14 canonical、4/4 prefix、12/12 corruption 与 sealed hash 是有效的窄证据，但六项主审探针证明 simulator 会把缺参数或未使用的坏定义静默解释成合法输入。R0B 依赖 simulator 形成语义回放真值，所以不能带着这个缺口继续。

v6 不改变 ERE/CPS 任务语义，不增加数据、模型、训练、审计 Gate 或架构组件。它只把 R0A 的公共输入合同变成完整可执行测试：

1. 先验证整棵 AST；
2. 任一节点非法立即抛出冻结错误；
3. 全部验证通过后才允许执行状态更新；
4. valid 输入仍逐字段复现原手写 expected；
5. formal 调用固定路径、无人工 hash/path 参数，并在评估前占用唯一 attempt。

这仍是测量系统资格验证，不是 P0-D、模型或 latent 架构证据。

## 2. 保留与直接删除

保留：

- v5 sealed artifact 与主设计层验收文档，作为历史诊断；
- v5 的 14 个手写世界、4 个 prefix 和 12 个 corruption 的语义值，复制到独立 v6 fixture；除 schema/contract/provenance 元数据外，AST 与 expected 不变；
- 三个公共函数签名和原有合法输入输出语义。

直接删除：

- 整个 `tests/v2_r1r_v5/` 活跃测试目录；v6 不保留 wrapper、pytest ignore、旧 CLI alias 或 compatibility mode；
- v5 `verify-oracle` CLI；
- 任意 audit/generator/model/cache/train/GPU 入口。

v6 当前实现面精确为：

```text
src/yggdrasil_v2/r1_revalidation/
  __init__.py
  common/
    __init__.py
    simulator.py
experiments/
  v2_r1_revalidation.py
tests/v2_r1r_v6/
  cli_template.py
  contract_guard.py
  strict_validator.py
  strict_runner.py
  frozen-inputs.json
  fixtures/oracle-spec.json
  fixtures/invalid-ast-matrix.json
  test_fixture_contract.py
  test_invalid_matrix_contract.py
  test_strict_semantics.py
  test_import_boundaries.py
  test_cli_contract.py
  test_runner_protocol.py
```

## 3. 公共 API 与错误合同

公共 API 仍精确为：

```python
simulate_ere(ast: Mapping[str, Any], prefix: int | None = None) -> dict[str, Any]
evaluate_cps(ast: Mapping[str, Any]) -> dict[str, Any]
replay_cps_prefix(ast: Mapping[str, Any], candidate_index: int, prefix: int) -> dict[str, Any]
```

`common.__all__` 必须按此顺序精确为：

```python
["evaluate_cps", "replay_cps_prefix", "simulate_ere"]
```

所有 AST 错误都抛 `ValueError`，消息精确为：

```text
AST_VALIDATION|<code>|<json_pointer>
```

错误必须指向第一个违反冻结 schema 的节点。输入在成功或失败时都不得被修改。validator 对类型、code、JSON pointer 与 input bytes 逐项比较，不接受“抛了任意异常”作为通过。

## 4. 精确 AST schema

所有 object 都使用 exact fields；缺字段与额外字段都失败。所有 name/entity/attribute/relation/fact/action 字段均为非空且非纯空白字符串。bool 不得冒充 int。attribute value 与 SET/set_attribute/attribute_equals 的 value 为非空且非纯空白字符串。初始 resource 为非负整数；cost、budget、resource_at_least amount 为非负整数；resource_delta delta 为任意整数。

### 4.1 共同 state

```text
state = {attributes, relations, facts, resources}
attributes: {entity: {attribute: string}}
relations: {relation: [[source, target], ...]}
facts: [string, ...]
resources: {resource: nonnegative-int}
```

relation pair 与 fact 不得重复；所有 map/list 结构必须是实际 JSON-compatible 值。

### 4.2 ERE

根字段精确为 `{initial_state, rules, events, query}`。

```text
rules: {rule_name: {params, primitives}}
event: {rule, arguments}
attribute query: {kind, entity, attribute}
relation query: {kind, relation, source, target}
```

primitive exact fields：

| op | fields |
| --- | --- |
| `SET` | `op,target,attribute,value` |
| `COPY` | `op,source,target,attribute` |
| `SWAP` | `op,left,right,attribute` |
| `LINK` / `UNLINK` | `op,relation,source,target` |
| `IF` | `op,predicate,then,else` |
| `FOREACH_LINKED` | `op,relation,source,effect` |

ERE predicate 只允许：

- `attribute_equals = {kind,entity,attribute,value}`；
- `relation_exists = {kind,relation,source,target}`。

`$arg:name` 必须引用当前 rule 的已声明 param；`$neighbor` 只允许出现在 `FOREACH_LINKED.effect` 子树。validator 必须递归检查未使用 rule、IF 未执行分支和 FOREACH effect，然后才执行 event。event rule 必须存在，arguments key 集合必须与 params 精确相等。

ERE `rules` 是 Python `Mapping`，public API 接收时重复 key 已不可表达。因此 v6 明确删除“重复 rule key 必须检测”这条不可执行要求，不伪造测试。重复 param 仍必须拒绝；原始 JSON fixture 则由 duplicate-key loader 单独拒绝。

### 4.3 CPS

根字段精确为 `{initial_state, actions, candidates, budget, goal, final_constraints}`。

```text
action = {name, cost, preconditions, effects}
candidate = {plan}
```

action name 不得重复。candidate plan 中的未知 action 是冻结的任务语义：该 candidate 在相应 step 以 `unknown_action` 非法停止，不是 AST validation error。

condition exact fields：

| kind | fields |
| --- | --- |
| `fact_true` / `fact_false` | `kind,fact` |
| `resource_at_least` | `kind,resource,amount` |
| `attribute_equals` | `kind,entity,attribute,value` |
| `relation_exists` | `kind,relation,source,target` |

effect exact fields：

| kind | fields |
| --- | --- |
| `add_fact` / `remove_fact` | `kind,fact` |
| `resource_delta` | `kind,resource,delta` |
| `set_attribute` | `kind,entity,attribute,value` |
| `link` / `unlink` | `kind,relation,source,target` |

validator 必须检查全部 action，包括没有 candidate 使用的 action；也必须检查全部 goal/final constraint。`replay_cps_prefix` 在检查 candidate index/prefix 前先完成整棵 AST 验证。

## 5. 冻结矩阵与正证据

`oracle-spec.json` 保留：

- 14 个 canonical worlds；
- 4 个 prefix expectations；
- 12 个 expected/input corruption；
- ERE/CPS 原完整能力覆盖。

`invalid-ast-matrix.json` 冻结 245 个独立探针，覆盖：

- root/state exact fields、类型、非法值、重复 pair/fact；
- 每一种 ERE primitive、predicate、event、query、placeholder 与 prefix；
- IF 未执行分支、未使用 rule、FOREACH effect 的 eager validation；
- 每一种 CPS condition/effect、action/candidate/budget/goal/final constraint；
- 未使用 action、重复 action name、replay 对未使用坏定义的 eager validation；
- bool/int 混淆、非法 index/prefix、错误 type/code/pointer 和输入不变性。

matrix 只描述 base case、JSON-pointer mutation、调用参数与预期异常，不计算任何语义 expected。

## 6. CLI、preflight 与唯一 formal attempt

唯一 CLI 只有两个无业务参数命令：

```text
python experiments/v2_r1_revalidation.py preflight-strict
python experiments/v2_r1_revalidation.py seal-strict
```

`preflight-strict` 可反复运行全部冻结检查，但不得创建 formal root。它用于实现迭代，不是 evidence artifact。

`seal-strict` 不接受 output、fixture、manifest、hash 或 suffix 参数。runner 内固定唯一 root：

```text
artifacts/v2-r1r/p0d-v6-r0a-strict-20260801-1
```

formal 调用先以 `mkdir(exist_ok=False)` 占用该 root，并立即写 `attempt.json`，然后才运行 frozen validation、semantic/strict matrix 与 import boundary。root 一旦存在，任何再次调用都 `BLOCKED`。因此错误参数不会再形成“到底算不算 formal”的争议；正式调用本身也没有可抄错参数。

## 7. PASS_R0A_STRICT conjunction

只有以下全部成立才通过：

1. frozen manifest 内全部文件 hash 一致；
2. 14/14 canonical、4/4 prefix、12/12 corruption；
3. 245/245 strict errors 的 type/code/path 精确；
4. semantic coverage 与 strict coverage 均 exact；
5. 连续两次完整 report bytes 相同；
6. 所有成功/失败调用都不修改输入；
7. common/package/CLI/import boundary exact，v5 tests 不存在；
8. exact formal root、attempt-before-evaluation 与单次占用成立；
9. `attempt.json`、`manifest.json`、`strict-report.json`、`assessment.json`、`run-metadata.json`、`evidence-seal.json` 齐全；
10. evidence seal 中前五个文件 hash 全部可复算；
11. `authorization_created=false`，没有 R0B 或任何后续产物。

R0A-strict 通过只证明 semantic oracle 与其输入边界合格。它不形成 G02–G08、P0-D、generator、模型、训练或架构通过。

## 8. 停止规则

预测试与 preflight 失败时，只允许修改四个实现文件；冻结输入有问题则 `BLOCKED`，不得放宽 expected。

`seal-strict` 一旦调用，无论成功、失败、异常或中断都停止。保留 root，不改代码、不删 artifact、不换 suffix、不重跑。通过也不自动授权 R0B；只有主设计层独立复核后才能另行决定。
