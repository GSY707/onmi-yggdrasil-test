# V2-R1R P1 v8 causal-bridge 执行命令

## 1. 唯一正式命令

从仓库根目录只启动一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-causal-bridge
```

命令固定执行 preflight、query-cache、qualification。已有 sealed PASS 的本合同上游阶段只读接受；已有 non-PASS 同名根、源码身份变化或当前阶段失败都会立即退出。不得第二次调用、删除 root、覆盖证据或补跑子命令。

## 2. 固定根

```text
artifacts/v2-r1r/p1-v8-causal-bridge-preflight-20260811-1
artifacts/v2-r1r/p1-v8-causal-bridge-query-cache-20260811-1
artifacts/v2-r1r/p1-v8-causal-bridge-qualification-20260811-1
```

运行前三根必须全部不存在。失败阶段之后的根不得创建，任何 P1 v9、controls 或 P2 root 都不得创建。

## 3. 长命令 transport

独立 Codex 执行任务只允许一次隐藏启动：

```powershell
$stdout = 'tmp\p1-v8-causal-bridge-transport\run.stdout.log'
$stderr = 'tmp\p1-v8-causal-bridge-transport\run.stderr.log'
Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList @('experiments\v2_r1_revalidation.py','run-p1-causal-bridge') -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
```

transport 只负责启动一次、观察同一进程树、在结束后只读验收并回传；不得修改代码、文档或 artifact，不得重启或拆开补跑。

## 4. 只读验收与回传

验收三个 root 的 `result.json`、`run-metadata.json`、全量 evidence seal 和一致 source identity；确认分区、selection、query cache、三臂初始化、3072 updates/臂、98,304 exposures/臂、B01–B07、相对 decision power、选择优先级、probe strip、GPU telemetry 与后序 roots absence。

回传：

```text
[P1_V8_CAUSAL_BRIDGE_WAKE] status=<PASS_UNPAIRED|PASS_PAIRED|PASS_ORDINARY_NO_BRIDGE|FAIL_PREFLIGHT|FAIL_QUERY_CACHE|FAIL_QUALIFICATION>; failed_stage=<none|preflight|query-cache|qualification>; artifact=<last_root>; later_roots_absent=<true|false>; summary=<三臂关键指标、Gate、seal、唯一启动证据>
```
