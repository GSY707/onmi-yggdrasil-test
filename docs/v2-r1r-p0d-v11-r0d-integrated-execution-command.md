# V2-R1R P0-D v11 R0D-integrated 执行合同

日期：2026-08-02  
唯一 formal root：`artifacts/v2-r1r/p0d-v11-r0d-integrated-20260802-1/`  
唯一成功状态：`PASS_R0D_INTEGRATED`

## 1. 权限与任务

本轮只允许把 active R0C CLI/tests 直接切换为 v11 R0D integration，新增 `integration/` 编排层，构建固定 12-case bundle，并执行一次 sealed R0D formal。v7 `common/`、v9 `audit/`、v10 `learner/` 与三棵 prior formal artifact 必须只读且逐文件复算。

禁止兼容 alias、旧测试并存、hidden label adapter、第二套 simulator/learner、production generator、R1–R3、P0-M、Qwen/Boundary/core/cache、GPU 或训练。不得修改、覆盖或重跑 v7/v9/v10 formal root。

## 2. 直接切换结果

active CLI 只能包含：

```text
audit-integrated --bundle <path>
preflight-integrated
seal-integrated
```

active tests 只能是 `tests/v2_r1r_v11/`；`tests/v2_r1r_v10/` 的 `.py/.json` 全部删除，历史由 v10 formal source snapshot、文档与 artifact 保留。不得保留 v10 command alias 或兼容入口。

## 3. 冻结前预测试

按顺序执行：

```text
python tests/v2_r1r_v11/contract_guard.py
python -m pytest -q tests/v2_r1r_v11
python -m compileall -q src/yggdrasil_v2/r1_revalidation experiments/v2_r1_revalidation.py tests/v2_r1r_v11
python experiments/v2_r1_revalidation.py --help
python experiments/v2_r1_revalidation.py preflight-integrated
git diff --check
```

要求：guard 通过；pytest 全通过；compileall exit 0；help 只暴露三个 v11 command；preflight 输出 `PASS_R0D_INTEGRATED_PREFLIGHT` 且 positive `8/8`、fault `20/20`、metric kill `19/19`、metamorphic `4/4`、replay `1/1`；diff check exit 0。任何失败都停止，不能调用 formal。

`frozen-inputs.json` 必须在预测试前生成并封印 design、execution contract、CLI、integration runtime、fixture、materializer、sealer、fault/metamorphic/replay/runner、guard 与 tests。guard 还必须全树复算 v7/v9/v10 evidence seal，并证明 active `common/audit/learner` 与 v10 source snapshot逐字节一致。

## 4. 唯一 formal

预测试全部通过且 fixed root 不存在后，只执行一次：

```text
python experiments/v2_r1_revalidation.py seal-integrated
```

不得预创建 root，不得重定向到别的 root，不得在失败后修代码重跑，不得创建 `-2` 或日期 suffix。runner 必须先原子声明 root 并写 `attempt.json`，再 materialize、audit、fault、metamorphic、import、snapshot replay 与 assessment。

formal root 精确十项：

```text
attempt.json
integration-bundle/
integrated-report.json
fault-ledger.json
coverage-ledger.json
metamorphic-ledger.json
replay-ledger.json
assessment.json
run-metadata.json
evidence-seal.json
```

exit `0`、stdout 精确含 `PASS_R0D_INTEGRATED`、assessment `passed=true`、全部 conjunction 成立且 evidence seal 可独立复算，才是机器 PASS。其他均为 FAIL 或 BLOCKED。

## 5. Formal 后停机与主设计层验收

formal 命令结束后立即停止实验。若失败，保留 artifact，报告 false Gate、fault、metric 或 replay 项，不进入任何后续阶段。若通过，主设计层另行只读复算：formal root children、attempt-before-evaluation、递归 evidence hashes、prior tree identity、positive report、20 fault exact sets、19 metric coverage、4 metamorphic、snapshot replay、import boundary 与授权位。

通过也只允许登记为 `R0D integrated measurement-system accepted`。production generator、R1–R3、P0-M、模型与训练仍需新的设计审查和单独授权。
