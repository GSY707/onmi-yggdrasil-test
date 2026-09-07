# V2-R1R P0-D v4 R0 audit-reference 执行命令

日期：2026-08-01  
执行层：独立低能力执行任务  
唯一规范：`docs/v2-r1-revalidation-task-design.md` 第 17 节  
本任务权限：只实现并运行 R0；R1 及 production generator 禁止

## 1. 唯一目标和最终状态

你的唯一目标是把当前 v3 P0 package 直接替换为 v4 R0 audit-reference implementation，证明审计器能接受 hand-authored known-good reference，并能让 F401–F420 精确击中冻结 Gate。完成一个 sealed R0 artifact 后立刻停止并通知主设计层。

最终状态只能是：

- `PASS_R0`：known-good 8/8、20 faults 精确命中、metric kill coverage 1.0、四个 metamorphic 和 import boundary 全通过；
- `FAIL_R0`：sealed R0 已运行，但任一冻结条件失败；
- `BLOCKED`：设计/命令 hash 不符、规范矛盾、固定依赖不可用，或同一实现根因连续三次仍不能通过预运行测试。

`PASS_R0` 不是 P0-D pass，不授权 R1、generator、preflight、formal 或训练。

## 2. 开始前只读核对

任何写入前按顺序完整读取：

1. `AGENT.md`；若实际存在 `AGENTS.md` 也读取；
2. `README.md` 与 `docs/DIRECTORY_REFERENCE.md`；
3. `docs/v2-r1-revalidation-task-design.md`，重点是第 17 节；
4. `docs/v2-r1r-p0-v3-main-review.md` 与 `docs/v2-r1r-p0-result.md`；
5. 本执行命令；
6. 当前 `src/yggdrasil_v2/r1_revalidation/`、`experiments/v2_r1_revalidation.py` 和全部 `tests/test_v2_r1r_*`；
7. `git status --short`。

父任务 prompt 必须提供设计文件和本命令的 SHA-256。用 `Get-FileHash -Algorithm SHA256` 核对；缺少 hash 或任一不符立即 `BLOCKED`。保护全部既有 dirty changes，禁止 reset、checkout、clean、stage、commit、push。

唯一 Python：

```powershell
.\.venv\Scripts\python.exe
```

在任何删除/写入前验证冻结 tokenizer 只读可用；不得下载模型权重：

```powershell
.\.venv\Scripts\python.exe -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B', revision='15852e8c16360a2fea060d615a32b45270f8a8fc', use_fast=True, local_files_only=True); assert type(t).__name__ == 'Qwen2Tokenizer' and t.is_fast; print(type(t).__name__)"
```

非零退出立即 `BLOCKED`。当前合同不允许通过联网下载、换 revision、换 tokenizer 或放宽 token Gate 继续。

## 3. 直接切换和文件权限

R0 必须直接替换，不建立 `v4` 平行 package，不保留 v3 wrapper、alias、schema reader 或 compatibility test。

允许删除并替换：

- 当前 `src/yggdrasil_v2/r1_revalidation/` 下全部 v3 Python 文件；
- `experiments/v2_r1_revalidation.py`；
- `tests/test_v2_r1r_contract.py`、`tests/test_v2_r1r_audit.py`、`tests/test_v2_r1r_data.py`。

允许新建的实现只限第 17.9 节列出的 `common/`、`audit/`、R0 CLI 和 `tests/v2_r1r_v4/`。允许在 sealed run 后同步 README、`docs/v2-r1r-p0-result.md`、`docs/next-stage-test-plan.md` 与 `docs/DIRECTORY_REFERENCE.md`。

禁止：

- 修改第 17 节或本执行命令；
- 新建 `generate/`、`ere.py`、`cps.py`、`render.py`、model/cache/train/baseline；
- 创建或修改 `artifacts/v2-r1r/authorizations/`；
- 读取旧 v1/v2/v3 JSONL 作为 reference；
- 运行 GPU、加载 Qwen 模型权重或生成 production data；
- 删除任何历史 artifact；
- 把被删除 v3 代码复制到 archive 或另一个兼容目录。

删除文件使用 `apply_patch`；不要用递归 shell 删除。R0 common simulator 可以从 v3 snapshot 重新审阅后移植已验证语义，但文件和 API 必须按 v4 layout 重建。

## 4. 严格实现顺序

### 4.1 先重建 common semantic reference

先建立：

```text
src/yggdrasil_v2/r1_revalidation/common/schema.py
src/yggdrasil_v2/r1_revalidation/common/simulator.py
src/yggdrasil_v2/r1_revalidation/common/fingerprint.py
```

schema 只接受 `.v4`。simulator 覆盖 ERE 七个原语和 CPS precondition/effect/resource/cost/budget/final constraint；每种语义先写 hand-authored expected-world test。测试 expected value 不得由 simulator 自己生成。

### 4.2 冻结 registry 与 metric schema

`audit/registry.py` 必须静态声明第 17.2 节八个 Gate 和全部 required metrics。每个 metric 都有 clause、evaluator、类型、比较器、阈值、positive fixture 和 fault ids。registry validator 必须拒绝：

- Gate 顺序/集合变化；
- metric 缺失、重复或额外；
- 任一 Gate 没有 positive 或 fault；
- 任一 metric 没有 fault；
- 空 evaluator、无阈值或未知 comparator。

`audit/assessment.py` 只能从 registry 和 metric report 重算 Gate；不得接受调用方传入 `passed=true`。

### 4.3 独立 reference builder

`tests/v2_r1r_v4/reference_builder.py` 只能 import Python 标准库。它从 `tests/v2_r1r_v4/fixtures/reference-spec.json` 展开第 17.3 节固定规模，不 import `src`，不调用 simulator、renderer、generator 或 audit。

reference specification 精确使用第 17.3 节三层：hand-authored `semantic_archetypes`、纯一一替换的 `instance_renamings`、显式有限正交 `surface_assignment_basis`。每个 archetype 保存 AST、canonical surface、initial world、expected answer、完整 teacher trace、claim truth 和 pair diff；basis 保存 block rows、每个 split 的 repetition/row order 和 quota target，分配 template、label permutation、choice mask、candidate/action order。builder 只能按声明展开并校验，不得调用 simulator、根据答案/audit/heuristic 搜索 assignment、用一个 modulo 或 RNG 联动多个变量。每个 archetype 都有不经 renaming 的 canonical expected-world test；reference 展开两次的 immutable bytes 必须一致。

CLI 不得 import reference builder；`build-reference` 必须用 `sys.executable` 子进程调用当前 source tree 的 builder，检查 exit code 后再由 `audit/artifact.py` 建 snapshot 和 input seal。snapshot replay 时同一相对寻址必须自动落到 snapshot 中的 `tests/v2_r1r_v4/reference_builder.py`。

### 4.4 Artifact、snapshot 与 seal

`audit/artifact.py` 实现：

- input/data/source snapshot hash；
- actual Qwen tokenizer recount；
- immutable/post-run source 分离；
- `input-seal.json` 与 `evidence-seal.json` 两层无环封印；
- `audit-core.json`、`fault-matrix.json` 和 `assessment.json` 的 canonical serialization；
- artifact-relative replay；
- current workspace drift 不影响 sealed replay。

`build-reference` 只生成输入、snapshot 和 `input-seal.json`；G02 只验证 input seal。`assess-r0` 最后生成并立即复验 `evidence-seal.json`，它哈希 input seal、audit-core、fault-matrix 和 assessment，不能反向修改这些文件。snapshot source set 只能包含第 17.4 节列出的设计、命令、实现、测试和依赖文件。不得把 README、目录索引、阶段计划、结果或主验收文档加入 immutable set。

### 4.5 G02–G08 evaluator

按 Gate 各自模块实现，不可合并成一个返回常量的大函数：

- G02：`artifact.py`；
- G03：`replay.py`；
- G04：`pairs.py`；
- G05/G06：`structure.py`；
- G07/G08：`shortcuts.py`；
- Gate conjunction：`assessment.py`。

audit 可以 import `common`，不能 import reference builder 或未来 generator。source-only parser 的函数签名只接收 `source_text`；测试必须证明传入 AST/answer 不可达。

G08 严格实现第 17.8 节的标准库 multinomial Naive Bayes、条件多数、tokenizer、grouped fold 和 tie-break；不得安装/新增 scikit-learn、scipy 或调用不固定默认值的第三方 learner。测试必须包含 OOV heldout、pair 不拆分、train/heldout object/fingerprint 不同、空 vocabulary fail-closed 和同输入逐字节相同报告。

### 4.6 Fault、metamorphic 与 import boundary

F401–F420 按第 17.3 节保存固定 case group；每个反引号 case 都有独立 patch specification 和 `expected_false_gates`。F402–F420 的每个 case 对 reference input 做独立临时副本；除 G02 case 外，使用 test-only independent input sealer 更新 hash。它们用 `sys.executable` 和运行中 CLI 的 `Path(__file__)` 调用公开 `audit-reference`，保证 snapshot replay 不会回落到当前工作树；每个 case 的实际 false set必须精确等于 expected set。

F401 必须使用 reference spec 中 hand-authored `completed-matrix-fixture`，它不调用真实 fault runner；四个 F401 case 分别破坏 matrix row、metamorphic row、actual false set 和 kill map，再调用公开 `assess-r0`，只允许 G01=false。真实顺序固定为：positive G02–G08 → F402–F420 → metamorphic/import → F401 synthetic fixture → 写真实 matrix → `assess-r0` 计算 G01。`assess-r0` 只读取已有 audit-core/fault-matrix，严禁内部调用 `run-fault-matrix`。

`fault_harness.py`、`independent_sealer.py`、`metamorphic_harness.py` 与 reference builder 一样只能 import 标准库，不能 import `src`；CLI 只能以子进程调用它们。independent sealer 必须自己实现 canonical JSON、相对路径枚举和 SHA-256，不能调用 `audit/artifact.py`。import-boundary test 同时做 AST import scan 和隔离进程实际 import trace。

四个 metamorphic test 和 import graph test 按第 17.3 节逐项保存机器结果。不要只写 pytest assertion；sealed R0 的 `fault-matrix.json` 必须包含完整 group/case matrix、每项 artifact-relative 规范化命令/exit code、metric positive/kill map 和失败明细。一个 F id 只有组内全部 case 精确命中才计为 pass。时间、cwd、绝对临时路径与 pid 只能进入 `run-metadata.json`。

## 5. R0 CLI 的精确表面

`experiments/v2_r1_revalidation.py --help` 只能出现：

```text
build-reference
audit-reference
run-fault-matrix
assess-r0
```

参数固定为：

```text
build-reference
  --spec <reference-spec.json>
  --output <R0_ROOT>
  --seed 20260801
  --design-doc <path>
  --execution-doc <path>

audit-reference
  --root <R0_ROOT>
  --output <audit-core.json>

run-fault-matrix
  --root <R0_ROOT>
  --output <fault-matrix.json>

assess-r0
  --root <R0_ROOT>
```

未知命令、`generate-p0`、`--mode`、`preflight`、`formal` 或已有输出目录必须非零退出。CLI 不得创建 authorization。

## 6. 预运行测试与唯一 sealed run

可以反复运行 pytest/临时目录测试；这些不是 sealed R0。先依次执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\v2_r1r_v4
.\.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation experiments\v2_r1_revalidation.py tests\v2_r1r_v4
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
git diff --check
```

任一失败先修实现并重新运行预测试。同一根因连续三次失败则 `BLOCKED`。

预测试全部通过后，唯一 sealed root 固定为：

```text
artifacts/v2-r1r/p0d-v4-r0-20260801-1
```

若该目录已存在，不覆盖；停止 `BLOCKED` 交主设计层决定新 suffix。正式命令严格为：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py build-reference --spec tests\v2_r1r_v4\fixtures\reference-spec.json --output artifacts\v2-r1r\p0d-v4-r0-20260801-1 --seed 20260801 --design-doc docs\v2-r1-revalidation-task-design.md --execution-doc docs\v2-r1r-p0d-v4-r0-execution-command.md
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py audit-reference --root artifacts\v2-r1r\p0d-v4-r0-20260801-1 --output artifacts\v2-r1r\p0d-v4-r0-20260801-1\audit-core.json
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-fault-matrix --root artifacts\v2-r1r\p0d-v4-r0-20260801-1 --output artifacts\v2-r1r\p0d-v4-r0-20260801-1\fault-matrix.json
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py assess-r0 --root artifacts\v2-r1r\p0d-v4-r0-20260801-1
```

四条中任一非零，或 final assessment 任一条件 false，状态 `FAIL_R0`。保留 partial/sealed root，不修复、不重跑、不改 suffix。

### Snapshot replay

`assess-r0` 成功后，新建一个随机名临时目录，只复制原 artifact 的 `manifest.json`、`data/`、`language_render_pairs.jsonl`、`source_snapshot/` 和 `input-seal.json`。设置 `PYTHONPATH=<临时根>/source_snapshot/src`，只调用 `<临时根>/source_snapshot/experiments/v2_r1_revalidation.py`，依次在临时根生成 `audit-core.json`、`fault-matrix.json`、`assessment.json` 和 `evidence-seal.json`。不得复制原来的四个派生文件，也不得 import 当前工作树。

随后逐文件比较原 artifact 与临时重放的这四个文件，SHA-256 和 bytes 都必须相同；只允许未封印的 `run-metadata.json` 不同。恢复原 `PYTHONPATH`，保留 replay 命令与比较结果到原 artifact 的 run metadata，但不得修改任何已被 evidence seal 覆盖的文件。该 replay 失败即 `FAIL_R0`，不得修改原 artifact 或重跑 sealed R0。

## 7. PASS_R0 的完整 conjunction

只有以下全部成立才是 `PASS_R0`：

1. known-good final assessment 精确 8/8；
2. F401–F420 共 20 项实际 false set 与 expected 完全一致；
3. required metric positive coverage 和 kill coverage 都为 1.0；
4. 四个 metamorphic test 全真；
5. import boundary 全真，且文件系统不存在 generator/model/cache/train/v3 compatibility；
6. actual tokenizer、model view、input seal、evidence seal 与 snapshot replay 全真；
7. reference 数量精确为每族 train 180、各 heldout/causal 90、language pairs 90；
8. `pytest`、`compileall`、CLI surface 和 `git diff --check` exit code 0。

不要把“测试通过但 R0 assessment 失败”写成 PASS。

## 8. 结果文档同步

sealed run 后才同步：

- `docs/v2-r1r-p0-result.md`：R0 命令、artifact、八 Gate、fault/metamorphic、PASS/FAIL；
- `docs/next-stage-test-plan.md`：R0 状态、R1 是否授权；
- `docs/DIRECTORY_REFERENCE.md`：当前 v4 package/tests/artifact 路径；
- `README.md`：一句当前状态与明确未完成项。

不得修改设计和本命令，不得创建 R1 authorization。文档同步不应使 snapshot replay 失败；若失败，说明 G02 实现违反合同，状态改为 `FAIL_R0`。

## 9. 最终检查与报告

最终重新运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\v2_r1r_v4
.\.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation experiments\v2_r1_revalidation.py tests\v2_r1r_v4
git diff --check
git status --short
```

final 必须按顺序报告：

1. `PASS_R0/FAIL_R0/BLOCKED`；
2. 删除的 v3 文件和新建的 v4 文件；
3. tests/compile/diff exit code；
4. sealed artifact 与 input/evidence seal SHA-256；
5. 八 Gate 明细；
6. 20 faults、metric kill coverage、四个 metamorphic、import boundary；
7. snapshot replay；
8. 已完成和未完成；
9. 明确 R1/generator/preflight/formal/P0-M/model/train 均未运行。

## 10. 结束前唤醒父任务

全部写入和验证完成后，最后一次工具调用必须向父任务 `019f5ac3-5b49-7cb2-af67-51509231a14a` 发送：

```text
[P0D_V4_R0_WAKE] status=<PASS_R0|FAIL_R0|BLOCKED>; child=<thread id>; artifact=<path or none>; failed=<gate/fault/check ids or none>; summary=<不超过120字>
```

回传成功后再输出 final 并结束。回传工具失败时，final 第一行写 `[P0D_V4_R0_WAKE_FAILED]` 和原始错误；不得使用 hook、文件轮询或自动开始 R1。
