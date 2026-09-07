# V2-R1R P0-D v7 R0A-lattice 执行命令

日期：2026-08-01  
执行角色：纯实现与一次性运行层  
唯一权限：直接切换到 v7、通过 preflight、执行一次固定 `seal-lattice`  
禁止：设计变更、冻结输入修改、R0B 及以后、模型/训练/GPU

## 1. 开始前完整读取

1. `AGENT.md`；
2. `README.md`、`docs/DIRECTORY_REFERENCE.md`、`docs/next-stage-test-plan.md`；
3. `docs/v2-r1r-p0-v6-r0a-strict-main-review.md`；
4. `docs/v2-r1r-p0d-v7-r0a-lattice-design.md`；
5. 本执行命令；
6. `tests/v2_r1r_v7/frozen-inputs.json` 列出的全部文件；
7. 当前四文件 runtime/CLI、`tests/v2_r1r_v6/` 和 `git status --short`。

父任务会提供 `DESIGN_SHA256`、`EXECUTION_SHA256`、`FROZEN_INPUTS_SHA256`。任何写入前先只读核对三项，并运行：

```powershell
python tests/v2_r1r_v7/contract_guard.py `
  --manifest tests/v2_r1r_v7/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>
```

任一不符立即回传 `BLOCKED`，不得写文件。

## 2. dirty worktree 与精确权限

保护全部既存变更。禁止 reset、checkout、clean、stage、commit、push；不得修改任何文档、v4/v5/v6/v7 artifact 或 v7 frozen input。

只能修改：

```text
src/yggdrasil_v2/r1_revalidation/__init__.py
src/yggdrasil_v2/r1_revalidation/common/__init__.py
src/yggdrasil_v2/r1_revalidation/common/simulator.py
experiments/v2_r1_revalidation.py
```

只能删除：

```text
tests/v2_r1r_v6/
```

tracked text 的修改/删除使用 `apply_patch`。若 v6 目录存在大量 untracked frozen 文件，可先 `Resolve-Path`，确认绝对路径精确位于当前 workspace 的 `tests/v2_r1r_v6`，再用 PowerShell `Remove-Item -LiteralPath <verified-absolute-path> -Recurse -Force`；不得清理 `.tmp`、artifact、其他 test 或其他目录。

不得新增 runtime/helper/test。CLI 必须与 `tests/v2_r1r_v7/cli_template.py` 逐字节一致，不得保留 `preflight-strict`、`seal-strict` 或任何旧 alias。

## 3. 精确实现任务

当前 v6 已通过 14/4/12、legacy 245/245 和 valid lattice 350/350。不要重写 simulator，也不要改 ERE/CPS 合法语义。

唯一已知语义缺口位于：

```python
def _validate_ere_query(value: Any, path: str) -> None:
```

在完成 query object exact-field 与 kind validation 后，五个 operand 必须使用与 rule operand 相同的 `_operand` validator，并显式传入：

```python
params=set()
neighbor_allowed=False
```

即：

- attribute query：entity、attribute；
- relation query：relation、source、target。

不得只特判 `$neighbor` / `$arg` 字符串；不得读取 lattice spec、fixture、test、cell id；不得修改 `_resolve`、执行语义、state schema、CPS unknown-action 语义或错误格式。

然后把公共 CLI 直接替换为 frozen v7 template。`__all__` 继续精确为：

```python
["evaluate_cps", "replay_cps_prefix", "simulate_ere"]
```

## 4. 可迭代预测试

正式 root 必须不存在。按顺序执行，可在失败后只修改四个允许文件：

```powershell
python tests/v2_r1r_v7/contract_guard.py `
  --manifest tests/v2_r1r_v7/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>

python -m pytest -q tests/v2_r1r_v7
python -m compileall -q src/yggdrasil_v2/r1_revalidation experiments/v2_r1_revalidation.py tests/v2_r1r_v7
python experiments/v2_r1_revalidation.py --help
python experiments/v2_r1_revalidation.py preflight-lattice
git diff --check
```

全部命令必须 exit `0`。preflight counts 必须精确为：

```text
positive              14/14
prefix                 4/4
corruption            12/12
legacy_strict        245/245
lattice_invalid      508/508
lattice_acceptance   350/350
coverage_cells       858/858
```

同时确认：

- `tests/v2_r1r_v6/` 不存在；
- CLI 只含 `preflight-lattice` / `seal-lattice`；
- v7 fixed formal root 仍不存在；
- 没有 R0B/R0C/R0D/R1/P0-M/model/train artifact。

需要修改任何 frozen input 或降低 count 才能继续时立即 `BLOCKED`。

## 5. 唯一 formal command

最后只读确认下列路径不存在：

```text
artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1
```

然后只运行一次：

```powershell
python experiments/v2_r1_revalidation.py seal-lattice
```

命令不得带任何附加参数。runner 会先占用 root 并写 `attempt.json`，再评估。调用后无论 exit code 都不得修改实现、删除/覆盖 artifact、换 suffix 或重跑。

exit `0`、`assessment.json.passed=true` 且固定 root 精确七文件才是机器 `PASS_R0A_LATTICE`。其他情况均为 `FAIL_R0A_LATTICE`；formal 前 root 已存在则为 `BLOCKED`。

## 6. formal 后只读验收并停止

只读检查：

1. 七文件集合精确；
2. evidence seal 覆盖并匹配前六文件；
3. assessment 全 conjunction；
4. manifest 中 design/execution/frozen/oracle/legacy/lattice/四实现文件 hash；
5. `authorization_created=false`；
6. 没有任何 R0B 或后续 artifact。

禁止运行 R0B/R0C/R0D/R1、P0-M、模型、cache、训练或 GPU；禁止编辑 README、结果、计划或目录索引，这些由父任务主审后同步。

结束前主动唤醒父任务：

```text
[P0D_V7_R0A_LATTICE_WAKE] status=<PASS_R0A_LATTICE|FAIL_R0A_LATTICE|BLOCKED>; child=<thread id>; artifact=<path or none>; failed=<check ids or none>; summary=<不超过120字>
```

回传成功后结束子任务。

