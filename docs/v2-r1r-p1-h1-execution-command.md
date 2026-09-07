# V2-R1R P1-H1 唯一执行命令

日期：2026-08-17

状态：runner/workflow 已实现并由 before-mutation readiness 锁定。共享 nonlinear feature trunk + routed final state-write projection 的预登记 `2026081761/2026081762` full-budget screen 已正常完成，但 H05、wrong-route H06 与 conditional-write necessity 同时失败，`passed=false`、`authorizes=nothing`。因此 `2026081763/2026081764` calibration 禁止创建，secondary floors 与合同 hashes 不得冻结，本页 formal 命令必须继续返回 before-mutation refusal；当前实现不得正式启动。

## 1. 唯一命令

从仓库根目录、项目 `.venv` 执行且只执行一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-h1
```

不提供独立 preflight、cache、train、evaluate、retry、resume、seed、root、device、threshold 或 F1 参数。launcher 在创建 transport 前先复算 threshold/hash/history/CLI/source/process/fixed-root readiness；未冻结状态只允许返回 `REFUSE_P1_H1_BEFORE_MUTATION`，不得创建 transport 或 formal root。readiness 通过后才固定创建 transport，绑定 PID/PPID/argv/cwd/stdout/stderr；任何 fixed path 已存在时必须 fail closed。

## 2. Fixed paths

```text
artifacts/v2-r1r/p1-h1-preflight-20260817-1
artifacts/v2-r1r/p1-h1-development-20260817-1
tmp/p1-h1-transport-20260817-1
```

preflight root 保存三组正式 package、全部 causal transforms、独立 32-record smoke 共 `23,456` records 的 full-text frozen-Qwen token cache，以及合同/source/Git/process/CLI/test/compute/leakage audit。它还登记 history、两组已消耗非正式身份、当前 screen/calibration、formal、smoke 七域的 package/fingerprint identity 与两两零重叠证明，但 screen/calibration records 不进入正式 cache。development root 保存 `2026081891/2026081892`、800-update overfit32 smoke，三 seed × 两 arm 的 fixed-final checkpoints、schedule/metric/intervention/compute ledgers、result、source snapshot 与 seal。

## 3. 顺序与停止

1. launcher 在任何 root 写入前验证两根 root 与 transport 全新、当前进程链唯一、H1/F1/v9/P2 无并行或 successor；
2. preflight 复算 NR1 sealed PASS、冻结合同 hash、source identity、测试、五域 fresh-data/leakage identity、Qwen cache 与 GPU/磁盘；
3. preflight 若不是当前 source identity 下的 sealed PASS，停止且不得创建 development root；
4. development 先运行两类各 16 条、batch 32、800 updates、answer/trace/route `0.95/0.90/0.99` 的独立 overfit32/gradient/strip smoke；失败即封存停止；
5. 按固定 seed 顺序运行 shared/mixed 各 4000 updates；不得提前停止、重排、重启或查看 heldout 选 checkpoint；
6. 在剥离前运行 supported/heldout/causal/metamorphic/route/source/recurrence intervention；shared route controls 必须在 logits/state/trajectory/trace/answer 上 hash-identical，mixed 的七类变形须逐 transform 与逐 family 报告 answer/route consistency；compute audit 还必须证明两臂每条记录都执行相同的 `H=384` 公共 FFN、同一个完整 `H=384` nonlinear feature trunk 与恰好一个 `H=384 -> D=256` state-write projection，未选 projection 无执行/梯度且两臂 active parameters/FLOPs 相同；随后物理剥离 trace head并执行 reload equivalence；
7. 计算 H01–H08，立即写 result、metadata、source snapshot 与 evidence seal；
8. 无论 PASS/FAIL 都退出，不创建、设计内启动或排队 F1。

任何 Gate FAIL、异常、CUDA OOM、source/Git 漂移、cache hash 漂移、进程/transport 异常或 seal 失败都属于本次唯一 H1 结果。禁止删除 root 后重跑、改变阈值、seed、batch、模型、数据、cache 或命令后补跑。

## 4. 合法后继

只有 `PASS_P1_H1_MIXED_CORE_DEVELOPMENT` 且主设计层独立复算通过，才允许在后续工作中编写 F1 合同。它不等于 F1 启动授权，更不等于 `p1_completed=true`。
