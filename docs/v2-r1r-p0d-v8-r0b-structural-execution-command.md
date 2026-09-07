# V2-R1R P0-D v8 R0B-structural 执行命令

日期：2026-08-01  
执行角色：纯实现与一次性 formal 运行层  
唯一权限：实现冻结的 G02–G06 audit、通过 v8 preflight、执行一次固定 `seal-structural`  
禁止：修改 accepted simulator、冻结输入/测试/阈值、R0C 及以后、generator、模型/训练/GPU

## 1. 开始前完整读取与核对

按顺序完整读取：

1. `AGENT.md`；
2. `README.md`、`docs/DIRECTORY_REFERENCE.md`、`docs/next-stage-test-plan.md`；
3. `docs/v2-r1r-p0-v7-r0a-lattice-main-review.md`；
4. `docs/v2-r1r-p0d-v8-r0b-structural-design.md`；
5. 本执行命令；
6. `tests/v2_r1r_v8/frozen-inputs.json` 列出的全部文件；
7. 当前 accepted simulator、CLI、`tests/v2_r1r_v7/` 与 `git status --short`。

父任务会给出 `DESIGN_SHA256`、`EXECUTION_SHA256`、`FROZEN_INPUTS_SHA256`。任何写入前必须先逐项核对，并运行：

```powershell
python tests/v2_r1r_v8/contract_guard.py `
  --manifest tests/v2_r1r_v8/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>
```

随后只读核对：

- v7 artifact/evidence seal 存在且 hash 与 frozen manifest 一致；
- 三个 accepted simulator 文件 hash 与 manifest 一致；
- v8 fixed formal root 不存在；
- pinned Qwen tokenizer snapshot 在本机精确 revision 路径存在；
- 没有 R0C/R0D/R1/P0-M/model/train artifact。

任一不符立即回传 `BLOCKED`，不得写文件。

## 2. dirty worktree 与精确权限

保护所有既存变更。禁止 reset、checkout、clean、stage、commit、push；不得删除 `.tmp-*`、历史 artifact、文档或其他实验。

只能新增或修改：

```text
src/yggdrasil_v2/r1_revalidation/audit/__init__.py
src/yggdrasil_v2/r1_revalidation/audit/artifact.py
src/yggdrasil_v2/r1_revalidation/audit/replay.py
src/yggdrasil_v2/r1_revalidation/audit/pairs.py
src/yggdrasil_v2/r1_revalidation/audit/structure.py
src/yggdrasil_v2/r1_revalidation/audit/structural.py
experiments/v2_r1_revalidation.py
```

只能删除：

```text
tests/v2_r1r_v7/
```

三个 accepted simulator 文件绝对禁止修改：

```text
src/yggdrasil_v2/r1_revalidation/__init__.py
src/yggdrasil_v2/r1_revalidation/common/__init__.py
src/yggdrasil_v2/r1_revalidation/common/simulator.py
```

不得修改任何 `tests/v2_r1r_v8/`、v8 文档、v7 artifact 或项目状态文档。CLI 必须与 `tests/v2_r1r_v8/cli_template.py` 字节一致；不保留 v7 alias、wrapper 或 compatibility mode。

若 `tests/v2_r1r_v7/` 是 untracked 大目录，删除前用 `Resolve-Path` 确认绝对路径精确位于当前 workspace 的该目录；只删除这一目录。

## 3. 实现分工

### 3.1 `artifact.py`：只负责 G02

实现 sealed bundle 的 fail-closed 读取与 G02 raw metrics：

- exact input tree、input seal、manifest/data/snapshot count 与 SHA-256；
- runtime/Git/named-stream provenance；
- model view 四字段 exact 检查与递归 forbidden key 计数；
- 从本机 `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/<revision>` 以 `local_files_only=True` 加载 tokenizer；
- model id、revision、class、`add_special_tokens=false`、实际 token recount 与 max 1024；
- records、causal pairs、language pairs、composition controls 的严格 schema 读取。

tokenizer 在同一进程内缓存一次。禁止联网、fallback、regex token count、silent truncate。G02 失败不能阻止其他结构文件仍可读时继续 G03–G06；报告必须能形成 exact false Gate set。

### 3.2 `replay.py`：只负责 G03

只通过 accepted public API：

```python
simulate_ere
evaluate_cps
replay_cps_prefix
```

fresh replay 每条 record，按 canonical JSON bytes 计算 output SHA-256，并核对 stored digest、answer、local label、reasoning budget 与完整 candidate/trace outcome。不得读取 case id 或把 frozen digest 写进 runtime。

claim evaluator 与 text parser 分开：

- ERE 三类 claim 从 `simulate_ere(ast, prefix)` 的 state 求真；
- CPS 六类 claim 从 `replay_cps_prefix` state/cost/legal 与 goal/final condition求真；
- 每个 pair 的 predicate JSON 必须只差一个 leaf，fresh truth 相反；
- text 必须按冻结九种 grammar parse 回同 kind/predicate；
- required kind、9/9 正负、pair id 与引用范围全部复核。

任何 replay/parse 异常转成对应 raw failure，不得 crash、跳过 denominator 或把异常当负例通过。

### 3.3 `pairs.py`：只负责 G04

独立实现与 `control_materializer.semantic_fingerprint()` 规范等价、但不 import tests 的有限-schema WL graph fingerprint；schema key 与动态 map key必须按路径区分，至少正确处理 rules、state maps 与 event arguments。surface fingerprint 固定为连续空白折叠、首尾 trim 后 SHA-256。

检查：

- stored fingerprint 独立复算；
- 未声明 semantic/surface overlap；
- causal pair 的 AST single-leaf diff、changed path、from/to、semantic/local answer flip、choice/label/fold 一致；
- mutation family count deviation；
- language pair AST identity、去首行后正文不同、feature span 位于对应正文、两侧 surface fingerprint 不同、fold 一致；
- composition controls 中 ERE required component op 真正在 ordinary control AST 出现，CPS required witness 名存在。

不得依赖文件行顺序或 JSON object key 顺序。

### 3.4 `structure.py`：只负责 G05/G06

G05 对 `role=ere_core` 的 record：

- 直接从 AST 计算 event depth，不信任 certificate；
- necessary indices 必须精确等于全部 event index；
- 对每个 event 建立同参数、空 primitives 的临时 rule，fresh replay 后 answer 必须改变，并核对 frozen ablation answer；
- provenance witness 的 path 必须指向正确 op/predicate/branch，executed trace marker 必须真实出现；
- ordinary 五类必须各一次；composition ids 至少三类且最大占比 `<=1/3`。

G06 对 `role=cps_rich/cps_none` 的 record：

- 从 fresh candidate report 复算 P* depth、valid、unique minimum；
- 验证 valid-suboptimal、至少五个 invalid、五类 failure reason、`lt/eq/gt` plan length；
- rich/NONE 两 AST 必须只在 certificate path 差一个 leaf，去掉该 leaf 后 skeleton 完全一致，NONE 边 fresh valid count 为零；
- NONE ratio 用 bundle 的真实 rich/NONE 数复算；
- 四个 composition witness path 必须指向匹配的 condition/effect/cost/budget，并与实际 candidate outcome一致；
- source/model view 中出现 `PSTAR`、`canonical candidate role` 等 canonical role 泄漏时计数失败。

不得只验证 certificate 字符串存在；所有 certificate 都是待核对主张。

### 3.5 `structural.py` 与 `__init__.py`

唯一 public API：

```python
audit_structural_bundle(root: str | Path) -> dict[str, Any]
__all__ = ["audit_structural_bundle"]
```

编排 G02–G06，并读取 `source_snapshot/tests/v2_r1r_v8/fixtures/metric-registry.json` 是禁止的；registry 不在 audit runtime 的输入。37 个 metric id 与阈值属于公开合同，可作为 runtime 常量实现，但不得读取 control/fault spec、case id、expected digest 或 fault id。

返回 JSON-only 且输入不变，schema 固定包含：

```text
schema_version
passed
gates: {G02..G06 -> bool}
metrics: {37 metric ids -> {value, numerator, denominator, passed, failures}}
failures
```

每个 metric 保存 raw value；rate 必须有整数 numerator/denominator，bool/count 也保存可复算原始字段。Gate 是本 Gate 原子 metric 的 conjunction；顶层是五 Gate conjunction。

## 4. 预测试闭环

先把 CLI 字节替换为 frozen template并删除 v7 tests。然后只在七个允许文件内迭代，按顺序执行：

```powershell
python tests/v2_r1r_v8/contract_guard.py `
  --manifest tests/v2_r1r_v8/frozen-inputs.json `
  --expected-sha256 <FROZEN_INPUTS_SHA256>

python -m pytest -q tests/v2_r1r_v8
python -m compileall -q src/yggdrasil_v2/r1_revalidation experiments/v2_r1_revalidation.py tests/v2_r1r_v8
python experiments/v2_r1_revalidation.py --help
python experiments/v2_r1_revalidation.py preflight-structural
git diff --check
```

全部 exit `0`。preflight 最后一行必须为：

```text
PASS_R0B_STRUCTURAL_PREFLIGHT positive=5/5 faults=41/41 metamorphic=4/4 metric_kill=37/37
```

并确认：

- CLI 的 `audit-structural` 对 positive bundle exit `0`，fault bundle exit `1`且仍输出 JSON report；
- 三个 accepted simulator hash 未变；
- `tests/v2_r1r_v7/` 不存在；
- v8 formal root 仍不存在；
- 没有 `shortcuts.py`、generate/model/cache/train 路径或后续 artifact。

需要改 simulator、fixture、fault、metric、test、runner、digest、threshold 或 fixed root 才能继续时立即 `BLOCKED`；不得降低 Gate 或在 audit 中特判控制 id。

## 5. 唯一 formal command

最后只读确认以下路径不存在：

```text
artifacts/v2-r1r/p0d-v8-r0b-structural-20260801-1
```

然后只调用一次：

```powershell
python experiments/v2_r1_revalidation.py seal-structural
```

不得带 output、suffix、hash 或其他参数。runner 会先占用 root 并写 `attempt.json`，再 materialize/audit。调用后无论 exit code 都不得修改实现、删除/覆盖 artifact、换 suffix 或重跑。

exit `0`、stdout `PASS_R0B_STRUCTURAL`、assessment `passed=true`、41/41 fault、4/4 metamorphic、37/37 metric kill 且 root 精确八个顶层 entry 才是机器通过。其他均为 `FAIL_R0B_STRUCTURAL`；调用前 root 已存在则为 `BLOCKED`。

## 6. formal 后只读验收并停止

只读检查：

1. artifact 顶层 entry 与 control-bundle 输入树 exact；
2. evidence seal 覆盖除自身外全部递归文件且 hash 一致；
3. assessment、structural report、fault matrix、coverage ledger conjunction；
4. v7 evidence、accepted simulator、v8 design/command/frozen inputs 与实现 hash；
5. `authorization_created=false`；
6. 没有 R0C/R0D/R1/P0-M/model/train artifact。

禁止运行 R0C/R0D/R1、P0-M、模型、cache、训练或 GPU；禁止编辑 README、结果、计划或目录索引，由父任务验收后同步。

结束前主动唤醒父任务：

```text
[P0D_V8_R0B_STRUCTURAL_WAKE] status=<PASS_R0B_STRUCTURAL|FAIL_R0B_STRUCTURAL|BLOCKED>; child=<thread id>; artifact=<path or none>; failed=<gate/fault/metric ids or none>; summary=<不超过120字>
```

回传成功后结束子任务。
