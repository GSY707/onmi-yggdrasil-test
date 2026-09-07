# V2-R1R P1 v6R selection-replay 恢复执行合同

日期：2026-08-11

## 唯一命令

在仓库根目录只启动一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-ctw-recovery
```

命令固定执行 `preflight → selection-recovery-assessment`。已存在且 seal 有效的 PASS preflight 可只读接受；任一 existing non-PASS root、新失败或异常都立即停止，不覆盖、不重启、不补跑。

## Fixed roots

- `artifacts/v2-r1r/p1-v6r-selection-recovery-preflight-20260811-1`
- `artifacts/v2-r1r/p1-v6r-selection-recovery-assessment-20260811-1`

执行层不得修改源码、测试、合同、README、目录索引或任何 v6 artifact；不得删除/重封 v6 assessment；不得把 Python raw mapping equality 恢复为 Gate；不得跳过 full query content audit 或负控；不得启动 integrated P1/P2。

该命令只进行 CPU/磁盘重放与验收，不使用 GPU 训练。结束后只读复算最后 root seal，确认 P1 v7/P2 roots 不存在，并回传：

```text
[P1_V6R_WAKE] status=<PASS_SELECTION_RECOVERY|FAIL_<STAGE>>; failed_stage=<none|preflight|selection-recovery-assessment>; artifact=<last_root>; later_roots_absent=<true|false>; summary=<五个 v6 seal、raw/canonical/roundtrip、selection hash、六个 key normalization、三负控、query 18899 content audit、temporal selection、recovery seal>
```

PASS 后也必须停止；由父任务独立验收并另立 integrated P1 合同。
