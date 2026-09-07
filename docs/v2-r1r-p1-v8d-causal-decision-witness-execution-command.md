# V2-R1R P1 v8D causal-decision witness 执行合同

日期：2026-08-11

## 1. 唯一正式命令

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-decision-witness
```

该命令在同一个进程中按顺序创建：

1. `artifacts/v2-r1r/p1-v8d-decision-witness-preflight-20260811-1`
2. `artifacts/v2-r1r/p1-v8d-decision-witness-query-cache-20260811-1`
3. `artifacts/v2-r1r/p1-v8d-decision-witness-qualification-20260811-1`

任一 root 已存在时不得删除、覆盖或换名；任一阶段非 PASS 时立即返回非零，后序 root 必须不存在。禁止分别调用分段命令后拼接成第二次 formal。

## 2. 执行纪律

- 只使用项目 `.venv\Scripts\python.exe`；系统 Python 不得用于 formal。
- 启动前核对三根、transport 与同名进程不存在，记录 cwd、git dirty fingerprint、GPU 和完整命令。
- 正式命令只启动一次；不得因 crash、Gate FAIL、接近阈值或 telemetry 警告而重启、补跑、改权重、换 checkpoint 或缩减数据。
- query cache、qualification 和全部 artifact 创建后都立即 seal；已 seal 文件只读。
- qualification 无论 PASS/FAIL 都不得创建 fresh v9、K=1、direct、text-CoT、assessment 或 P2。

## 3. 结束回传

结束时回传：

```text
[P1_V8D_DECISION_WITNESS_WAKE] status=<PASS|FAIL_PREFLIGHT|FAIL_QUERY_CACHE|FAIL_QUALIFICATION|CRASH>; failed_stage=<...>; artifact=<last-root>; later_roots_absent=<true|false>; summary=<关键判决>
```

摘要必须包含三根 status/seal、source identity、query/catalog/partition/schedule hash、probe-only 与 joint decision-witness 的 ERE/CPS accuracy/zero-drop/swapped-drop、答案 optimization/audit/validation、CPS gain、probe strip、FP32 gradient、compute/GPU、唯一启动证据及后继 root absence。
