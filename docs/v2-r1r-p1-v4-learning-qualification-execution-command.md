# V2-R1R P1 v4-LQ 执行命令

日期：2026-08-11

合同：`docs/v2-r1r-p1-v4-learning-qualification-design.md`

## 1. 唯一正式命令

在仓库根目录执行且只执行一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-lq
```

该命令依次处理：

```text
preflight
→ bootstrap128
→ scale512
→ scale2048
→ full8192
→ assessment
```

已存在且 seal 有效的 PASS root 只读接受；不存在的 root 创建一次。若 root 已存在但不是 sealed PASS，命令返回非零，不重启、不覆盖。任一新阶段失败后立即停止，后序 roots 必须不存在。

## 2. 固定 roots

```text
artifacts/v2-r1r/p1-v4-lq-preflight-20260811-1
artifacts/v2-r1r/p1-v4-lq-bootstrap128-20260811-1
artifacts/v2-r1r/p1-v4-lq-scale512-20260811-1
artifacts/v2-r1r/p1-v4-lq-scale2048-20260811-1
artifacts/v2-r1r/p1-v4-lq-full8192-20260811-1
artifacts/v2-r1r/p1-v4-lq-assessment-20260811-1
```

禁止手工创建这些目录，禁止把输出写入 P1 v2/v3 roots。stdout/stderr 的传输日志只允许放在 `tmp/p1-v4-lq-transport/`，不属于正式判决证据。

## 3. 执行前检查

执行者只读确认：工作目录正确、没有正在运行的同名 formal process、GPU 可用、磁盘余量满足 preflight、六个 fixed roots 均不存在或为 sealed PASS。不得因 dirty worktree 删除、reset 或覆盖用户文件；活动源码 identity 由 preflight snapshot 固定。

## 4. 结束回传

结束后只读复算最后已创建 root 的 seal，确认失败点以后的 roots 不存在，并回传：

```text
[P1_V4_LQ_WAKE] status=<PASS_LQ|FAIL_<STAGE>>; failed_stage=<none|preflight|bootstrap128|scale512|scale2048|full8192|assessment>; artifact=<last_root>; later_roots_absent=<true|false>; summary=<核心指标与停止原因>
```

若 `status=PASS_LQ`，只表示可另立 fresh-seed 完整 P1 合同；不得自行启动 P1 successor 或 P2。若失败，不得修改代码后复用同一 root。
