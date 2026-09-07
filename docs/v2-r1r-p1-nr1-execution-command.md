# V2-R1R P1-NR1 数值/关系测量资格执行合同

日期：2026-08-17  
状态：合同冻结前最终预测试；正式命令尚未启动

## 唯一正式入口

工作目录固定为仓库根。预测试、冻结 hash、V8L 三根 seal、无外部 formal/successor 进程、两根 fixed root 与 fixed transport 均不存在、H1/F1/v9/P2 successor absence 全部确认后，只能启动一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-nr1
```

CLI 只暴露这一入口，不暴露可拼接的 preflight/qualification 子命令。入口在任何正式 root 写入前同时检查两根 root 和 transport；任一已存在时返回 `REFUSE_P1_NR1_SINGLE_USE`，不修改现存证据。

固定位置：

- preflight：`artifacts/v2-r1r/p1-nr1-preflight-20260817-1`
- qualification：`artifacts/v2-r1r/p1-nr1-qualification-20260817-1`
- transport：`tmp/p1-nr1-transport-20260817-1`
- seed：`2026081703`

CLI 自身以 exclusive-create 建立 transport，在同一 Python 进程内把 stdout/stderr 重定向到 `stdout.log` / `stderr.log`，并写入包含 PID、PPID、argv、cwd 与 attempt id 的 `launch.json`。preflight 和 qualification 都枚举系统进程，要求所有 `run-p1-nr1` 匹配项只位于当前祖先进程链，且不存在 H1/F1/v9/P2 进程；同时机械核对 transport 环境、非 TTY 重定向与 launch PID。

## Fail-closed 顺序

启动顺序固定为：

1. launcher 在无 root/transport 时创建 transport；
2. runner 再次确认两根 root 均不存在，采集 source/Git identity 后才创建 preflight root；
3. preflight 运行冻结 hash、V8L seal、预测试、CLI、进程/transport、successor 与 source identity Gate，并复制 source snapshot；
4. `run_stage` 写 result/metadata，复算 post-action source/Git/snapshot identity，封存并立即 `verify_seal`；
5. 只有 sealed `PASS_P1_NR1_PREFLIGHT` 与当前 source identity 完全一致时，才允许创建 qualification root；
6. qualification 封存 N01–N07，runner 再做 post-action identity 与即时 seal verify；随后命令结束，不自动启动 H1/F1/P2。

preflight 非 PASS、crash、seal/source/Git/process/transport 异常时，qualification root 必须不存在。qualification 非 PASS 时，H1/F1/v9/P2 roots 必须不存在。不得重启、补跑、换 root、换 seed、改 case 数、删 fixture/fault、改阈值或覆盖 artifact。

## 允许的正式前动作

只允许读取当前文档、源码和 sealed artifacts；运行 `python -m pytest -q -p no:cacheprovider tests/v2_r1r_p1_nr1`、`compileall`、CLI `--help`、静态 hash/source identity/fixed-root/process absence 检查；以及不写 fixed root/transport 的内存级或系统临时目录 probe。这些动作不构成 formal，也不能用于正式后调参。

## 结束回传

进程退出后只读复算已创建 root 的 evidence seal，读取 fixed transport，并报告：

```text
[P1_NR1_WAKE] status=<PASS|FAIL_PREFLIGHT|FAIL_QUALIFICATION|CRASH>; failed_stage=<...>; artifact=<last-root-or-none>; later_roots_absent=<true|false>; summary=<高密度验收>
```

摘要必须包含唯一 launch/process/transport 证据、预测试、V8L 三根 status/seal、两根 NR1 status/seal/source/Git/snapshot identity、手工 fixture hash、case/fingerprint/topology 数量、qualification/heldout numeric/relation 指标、metamorphic、每类 fault 的 required metric 与 decision kill、运行成本、successor absence 及 `p1_h1_design_authorized/p1_completed/p2_eligible`。
