# V2-R1R P0-D v6 R0A-strict 执行命令

日期：2026-08-01  
执行角色：纯实现与一次性运行层  
唯一权限：直接切换、通过 v6 preflight、运行一次 `seal-strict`  
禁止：设计变更、fixture/test 修改、R0B 及以后、模型/训练/GPU

## 1. 开始前读取

完整读取：

1. `AGENT.md`；
2. `README.md`、`docs/DIRECTORY_REFERENCE.md`、`docs/next-stage-test-plan.md`；
3. `docs/v2-r1r-p0-v5-r0a-main-review.md`；
4. `docs/v2-r1r-p0d-v6-r0a-strict-design.md`；
5. 本执行命令；
6. `tests/v2_r1r_v6/frozen-inputs.json` 列出的全部文件；
7. 当前 `src/yggdrasil_v2/r1_revalidation/`、CLI、v5 tests 和 `git status --short`。

主设计层会在委派消息中提供 `DESIGN_SHA256`、`EXECUTION_SHA256` 与 `FROZEN_INPUTS_SHA256`。开始时只读核对三项 hash，并运行：

```powershell
python tests/v2_r1r_v6/contract_guard.py `
  --manifest tests/v2_r1r_v6/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>
```

任一不符立即 `BLOCKED`，不得写文件。这里的 expected hash 只用于启动前 guard，不进入 formal 命令。

## 2. dirty worktree 与文件权限

保护全部既存变更，禁止 reset/checkout/clean/stage/commit/push。不得修改任何文档、v5/v6 artifact 或 v6 frozen input。

只能修改：

```text
src/yggdrasil_v2/r1_revalidation/__init__.py
src/yggdrasil_v2/r1_revalidation/common/__init__.py
src/yggdrasil_v2/r1_revalidation/common/simulator.py
experiments/v2_r1_revalidation.py
```

只能删除：

```text
tests/v2_r1r_v5/
```

tracked text source/test 的编辑与删除必须使用 `apply_patch`。源码删除后若 v5 目录只剩 `__pycache__`/空目录，可在先 `Resolve-Path` 并核对其精确位于当前 workspace 后，仅对 `tests/v2_r1r_v5` 做 literal filesystem cleanup；不得清理 `.tmp`、artifact 或其他目录。

不得新增 helper/source/test。public CLI 必须与 `tests/v2_r1r_v6/cli_template.py` 逐字节相同。不得保留 `verify-oracle` alias 或 v5 compatibility。

## 3. 实现要求

先实现完整 AST validation，再复用或整理状态执行。不能靠当前执行路径顺便校验。

建议内部顺序：

```text
validate common state
validate every ERE rule primitive/predicate/event/query or every CPS action/condition/effect/candidate/goal/final
validate placeholders/references
validate call index/prefix
deep-copy canonical state
execute
return JSON-only result
```

所有错误精确为：

```text
ValueError("AST_VALIDATION|<code>|<json_pointer>")
```

不得读取 fixture/matrix/tests/expected，不得出现 case/control/probe id 特判，不得 import 第三方库。所有输入在成功或失败后字节等价。

ERE rules 保持 mapping；不实现无法表达的重复 rule 检测。CPS candidate 中未知 action 保持 `unknown_action` semantic failure，不能改成 schema error。

## 4. 可迭代预测试

formal root 必须不存在。按顺序反复运行：

```powershell
python tests/v2_r1r_v6/contract_guard.py `
  --manifest tests/v2_r1r_v6/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>

python -m pytest -q tests/v2_r1r_v6
python -m compileall -q src/yggdrasil_v2/r1_revalidation experiments/v2_r1_revalidation.py tests/v2_r1r_v6
python experiments/v2_r1_revalidation.py --help
python experiments/v2_r1_revalidation.py preflight-strict
git diff --check
```

资格条件：guard、全部 pytest、compileall、CLI help、preflight、diff-check 均 exit `0`；preflight 报 14/14、4/4、12/12、245/245；v5 tests 与禁止路径不存在；formal root 仍不存在。

预测试失败只修改四个实现文件。需要放宽 frozen expected/test 才能继续时状态为 `BLOCKED`。

## 5. 唯一 formal command

最后只读确认以下路径不存在：

```text
artifacts/v2-r1r/p0d-v6-r0a-strict-20260801-1
```

然后只运行一次、且命令没有任何附加参数：

```powershell
python experiments/v2_r1_revalidation.py seal-strict
```

runner 会先占用 root 并写 `attempt.json`，再评估。调用后不论 exit code 都不得修改实现、删除/覆盖 artifact、换 suffix 或重跑。

exit `0` 且 `assessment.json.passed=true` 才是 `PASS_R0A_STRICT`。exit 非零、缺文件、任一 check false 或中断都是 `FAIL_R0A_STRICT`。formal 前发现 root 已存在则 `BLOCKED`。

## 6. 结束与回传

无论结果如何立即停止；禁止运行 R0B/R0C/R0D/R1、P0-M、模型、cache、训练或 GPU。

只读汇报：

1. `PASS_R0A_STRICT` / `FAIL_R0A_STRICT` / `BLOCKED`；
2. pytest 数量、preflight counts、formal exit；
3. formal root 与六个文件是否齐全；
4. evidence hashes 是否匹配；
5. 实际修改/删除文件；
6. 未运行的后续阶段。

结束前主动唤醒父任务，格式：

```text
[P0D_V6_R0A_STRICT_WAKE] status=<PASS_R0A_STRICT|FAIL_R0A_STRICT|BLOCKED>; child=<thread id>; artifact=<path or none>; failed=<check ids or none>; summary=<不超过120字>
```

回传成功后再结束子任务。
