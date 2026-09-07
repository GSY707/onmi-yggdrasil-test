# V2-R1R P1 v5 语义迁移执行命令

日期：2026-08-11

合同：`docs/v2-r1r-p1-v5-semantic-transfer-design.md`

## 1. 唯一正式命令

在仓库根目录只启动一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-transfer
```

命令固定执行 `preflight → qualification7168 → full8192 → assessment`。已存在且 seal 有效的 PASS root 只读接受；不存在的 root 创建一次。已存在的 non-PASS root、任一新失败或异常都立即返回非零，不重启、不覆盖、不进入后序阶段。

## 2. 固定 roots

```text
artifacts/v2-r1r/p1-v5-transfer-preflight-20260811-1
artifacts/v2-r1r/p1-v5-transfer-qualification7168-20260811-1
artifacts/v2-r1r/p1-v5-transfer-full8192-20260811-1
artifacts/v2-r1r/p1-v5-transfer-assessment-20260811-1
```

禁止手工创建这些目录。stdout/stderr 传输日志只允许写入 `tmp/p1-v5-transfer-transport/`，不属于正式判决证据。

## 3. 执行边界

执行者只读确认工作目录、GPU、磁盘、无同名 formal process，以及四个 roots 全部不存在或为 sealed PASS。不得修改 P1 v2/v3/v4 artifact，不得因 dirty worktree reset/clean 用户文件。preflight 固定活动源码 identity；此后任何受监视文件变化都必须拒绝后序阶段。

## 4. 结束回传

结束后只读复算最后 root 的 seal，确认失败点后的 roots 不存在，并回传：

```text
[P1_V5_TRANSFER_WAKE] status=<PASS_TRANSFER|FAIL_<STAGE>>; failed_stage=<none|preflight|qualification7168|full8192|assessment>; artifact=<last_root>; later_roots_absent=<true|false>; summary=<transfer、validation、formal Gate、吞吐与停止原因>
```

若 PASS，只表示 fresh-seed 完整 P1 successor 获得授权；不得自行启动 successor 或 P2。若失败，不得修改代码后复用同一 root。
