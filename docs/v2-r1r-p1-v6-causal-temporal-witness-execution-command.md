# V2-R1R P1 v6 causal-temporal-witness 执行合同

日期：2026-08-11

## 唯一命令

在仓库根目录只启动一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-ctw
```

命令固定执行 `preflight → witness-audit → query-cache → mechanism-compare → assessment`。已存在且 seal 有效的 PASS root只读接受；已存在 non-PASS root、任何新失败或异常都立即返回非零，不重启、不覆盖、不进入后序。

正式身份固定为 `SELECTION_SEED=2026082601`、`MODEL_SEED=2026082617`、`DATA_ORDER_SEED=2026082621`；旧开发 seeds 已因读过非 formal audit 而退役。两臂各固定 3,200 updates，seen Gate 为每族 `0.70`，audit/因果/优势 Gate 以设计文档为准。执行层不得引用或恢复已删除的临时 dry-run cache。

## Fixed roots

- `artifacts/v2-r1r/p1-v6-ctw-preflight-20260811-1`
- `artifacts/v2-r1r/p1-v6-ctw-witness-audit-20260811-1`
- `artifacts/v2-r1r/p1-v6-ctw-query-cache-20260811-1`
- `artifacts/v2-r1r/p1-v6-ctw-mechanism-compare-20260811-1`
- `artifacts/v2-r1r/p1-v6-ctw-assessment-20260811-1`

执行层不得修改源码、测试、合同、README、目录索引或任何旧 artifact；不得改变 seed、选择、query 文本、updates、batch、Gate 或 arm 顺序；不得补跑单臂、继续 v5、启动 integrated P1/P2。

## 回传

结束后只读复算最后 root seal，确认失败点后的 roots 不存在，并回传：

```text
[P1_V6_CTW_WAKE] status=<PASS_CTW|FAIL_<STAGE>>; failed_stage=<none|preflight|witness-audit|query-cache|mechanism-compare|assessment>; artifact=<last_root>; later_roots_absent=<true|false>; summary=<coverage、cache、static/temporal train-audit-state metrics、selected mechanism、seal 与停止原因>
```

PASS 后也必须停止，等待父任务另立 integrated P1 v7。
