# V2-R1R P0-D v5 R0A 执行命令

日期：2026-08-01  
执行层：独立 Luna max 任务  
唯一权限：直接切换并执行 R0A semantic oracle；R0B 及以后全部禁止

## 1. 唯一目标与状态

你只实现三个公共语义函数，使它们逐字段复现主设计层已经写好的 14 个 canonical worlds 与 4 个 prefix worlds，并让 12 个冻结 corruption 全部被机械 validator 拒绝。你不设计任务、不修改 expected values、不建立 reference builder、audit、generator、dataset 或训练代码。

只允许三种结束状态：

- `PASS_R0A`：冻结 conjunction 全真且唯一 sealed artifact 完整；
- `FAIL_R0A`：sealed command 已运行，但 assessment 任一项失败；
- `BLOCKED`：冻结输入/hash、依赖、路径或 artifact 前置不满足，因而没有启动 sealed command。

无论通过或失败都立即停止。`PASS_R0A` 不授权 R0B；你不得创建 authorization、继续 G02–G08、恢复 v4 audit 或接触模型/GPU。

## 2. 开始前必须完整读取

按顺序读取：

1. `AGENT.md`；
2. `README.md` 与 `docs/DIRECTORY_REFERENCE.md`；
3. `docs/v2-r1-revalidation-task-design.md` 顶部状态、第 1–3、7–15 节与当前第 18 节；
4. `docs/v2-r1r-p0-v4-main-review.md`；
5. 本执行命令；
6. `tests/v2_r1r_v5/frozen-inputs.json` 列出的每个文件；
7. 当前 `src/yggdrasil_v2/r1_revalidation/`、`experiments/v2_r1_revalidation.py` 与 `git status --short`。

父任务会在 delegation 中提供三个 SHA-256：

```text
DESIGN_SHA256=<parent supplied>
EXECUTION_SHA256=<parent supplied>
FROZEN_INPUTS_SHA256=<parent supplied>
```

先用 `Get-FileHash -Algorithm SHA256` 检查设计、本命令和 frozen manifest，再运行：

```powershell
python tests/v2_r1r_v5/contract_guard.py `
  --manifest tests/v2_r1r_v5/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>
```

任一不符立即 `BLOCKED`；不写任何文件，不试图“修正”冻结输入。

## 3. 冻结文件：绝对禁止编辑

`tests/v2_r1r_v5/frozen-inputs.json` 中每个路径均由主设计层冻结。尤其禁止编辑：

- `docs/v2-r1-revalidation-task-design.md`；
- `tests/v2_r1r_v5/fixtures/oracle-spec.json`；
- `oracle_validator.py`、`r0a_runner.py`、`contract_guard.py`；
- `cli_template.py`；
- 五个 `test_*.py`。

不得根据当前 simulator 输出回填 expected values，不得跳过/xfail/monkeypatch 测试，不得捕获错误后返回固定通过，不得在 source 中读取 fixture 的 expected block。

## 4. 允许和必须进行的直接切换

保护所有既存 dirty changes；禁止 `reset/checkout/clean/stage/commit/push`。只用 `apply_patch` 编辑或删除文件。

必须删除：

```text
src/yggdrasil_v2/r1_revalidation/audit/
src/yggdrasil_v2/r1_revalidation/common/fingerprint.py
src/yggdrasil_v2/r1_revalidation/common/schema.py
tests/v2_r1r_v4/
```

必须直接替换：

```text
src/yggdrasil_v2/r1_revalidation/__init__.py
src/yggdrasil_v2/r1_revalidation/common/__init__.py
src/yggdrasil_v2/r1_revalidation/common/simulator.py
experiments/v2_r1_revalidation.py
```

其中 public CLI 必须与 `tests/v2_r1r_v5/cli_template.py` **逐字节相同**。不要自行重写 CLI；按冻结模板直接替换。最终 common 目录只能有 `__init__.py` 与 `simulator.py`。

禁止新增其他 source/test/helper，禁止保留 v4 wrapper、alias、schema reader、compatibility test 或旧命令。禁止第三方依赖、网络、模型、tokenizer、cache、GPU、dataset、surface、claim、pair、audit 与统计 learner。

## 5. simulator 的精确实现要求

只实现：

```python
simulate_ere(ast, prefix=None)
evaluate_cps(ast)
replay_cps_prefix(ast, candidate_index, prefix)
```

API 与语义以设计第 18.4 节和冻结 fixture 的完整 expected objects 为准。实现必须通用解释 AST；禁止按 `case_id`、实体名、action 名或 expected answer 写分支。`simulator.py` 不得读取 `tests/`、fixture 或环境变量。

关键不可自行猜测的规则：

- 输出必须是纯 JSON 值，字段集合与 fixture exact；
- state 始终保存 `attributes/relations/facts/resources`，map、pair 和 facts 稳定排序；
- ERE IF 只执行一个分支；trace 写 `IF:true` 或 `IF:false`；
- FOREACH target 按字符串排序，trace 写 `FOREACH_LINKED:n`；
- ERE `delta_budget` 只计执行到的 mutating leaf；
- CPS precondition/unknown action 失败发生在 cost/effect 之前；
- candidate 的 goal/final 布尔从终态独立计算，但非法执行的 `failure_reasons` 只写执行错误；
- 合法 candidate 再记录 `budget`、`goal`、`final_constraint`；
- 只有唯一最低成本 valid candidate 才有整数 answer；无 valid 或最低成本并列时 answer 为 `None` 且 `unique_optimum=False`；
- duplicate rule/action、未知 op/condition/effect、缺参数、非法 prefix/index 必须抛异常，不能默认 NOOP。

## 6. 预测试：可修实现，不可改冻结输入

在 sealed root 不存在时反复运行以下命令，顺序固定：

```powershell
python tests/v2_r1r_v5/contract_guard.py `
  --manifest tests/v2_r1r_v5/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>

python -m pytest -q tests/v2_r1r_v5
python -m compileall -q src/yggdrasil_v2/r1_revalidation experiments/v2_r1_revalidation.py tests/v2_r1r_v5
python experiments/v2_r1_revalidation.py --help
git diff --check
```

资格条件精确为：guard exit `0`；pytest 全部通过且不是零收集；compileall/help/diff-check exit `0`；help 只有 `verify-oracle`；v4 audit/tests 和 forbidden source 均不存在；sealed root 仍不存在。

预测试失败时只允许修改四个实现文件。若必须修改冻结文件才能通过，状态是 `BLOCKED`，不是自行放宽合同。

## 7. 唯一 sealed command

固定 artifact：

```text
artifacts/v2-r1r/p0d-v5-r0a-20260801-1
```

执行前最后一次检查该路径不存在，再运行且只运行一次：

```powershell
python experiments/v2_r1_revalidation.py verify-oracle `
  --fixture tests/v2_r1r_v5/fixtures/oracle-spec.json `
  --output artifacts/v2-r1r/p0d-v5-r0a-20260801-1 `
  --design-doc docs/v2-r1-revalidation-task-design.md `
  --execution-doc docs/v2-r1r-p0d-v5-r0a-execution-command.md `
  --frozen-inputs tests/v2_r1r_v5/frozen-inputs.json `
  --expected-design-sha256 <DESIGN_SHA256> `
  --expected-execution-sha256 <EXECUTION_SHA256> `
  --expected-frozen-sha256 <FROZEN_INPUTS_SHA256>
```

不要用 shell 重定向伪造输出，不要手工创建 artifact。runner exit `0` 且 `assessment.json.passed=true` 才是 `PASS_R0A`。exit 非零或任一 assessment check false 即 `FAIL_R0A`。

sealed command 开始后，任何失败都不得改代码、改 fixture、删除/覆盖 artifact 或换 suffix 重跑。保留 partial root并停止。

## 8. PASS_R0A 的完整 conjunction

必须独立复核 artifact 内：

- `manifest.json`、`oracle-report.json`、`assessment.json`、`evidence-seal.json`、`run-metadata.json` 全部存在；
- positive `14/14`；prefix `4/4`；negative `12/12`；
- ERE/CPS coverage exact；
- `deterministic_report_bytes=true`；
- import boundary true；
- frozen input validation true；
- evidence seal 中三个文件 hash 与当前 artifact bytes 相同；
- `authorization_created=false`；
- 没有 R0B、audit、generator、数据、模型或训练产物。

不要把“pytest 通过但 sealed assessment 失败”写成 PASS，也不要把 R0A 解释为 G02–G08 或 P0-D 通过。

## 9. 结束、文档与唤醒

执行任务不修改 README、结果、阶段计划、目录索引或设计文档。主设计层在独立验收后统一同步，避免执行层把实现自评写成项目真源。

结束前报告：

1. `PASS_R0A/FAIL_R0A/BLOCKED`；
2. 实际 pytest 数量与 sealed command exit code；
3. artifact 路径和 assessment conjunction；
4. 完成项与未完成项；
5. 未运行 R0B/R0C/R0D/R1/P0-M/模型/训练；
6. 未 stage/commit/push。

最后一次工具调用必须向父任务发送：

```text
[P0D_V5_R0A_WAKE] status=<PASS_R0A|FAIL_R0A|BLOCKED>; child=<thread id>; artifact=<path or none>; failed=<check ids or none>; summary=<不超过120字>
```

回传后再输出 final 并结束。不得用 hook、轮询文件或自行启动下一阶段。
