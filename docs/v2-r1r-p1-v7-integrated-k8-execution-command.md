# V2-R1R P1 v7 integrated K=8 执行命令

## 1. 唯一正式命令

从仓库根目录只启动一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-integrated
```

该命令按固定顺序执行 preflight、query-cache、K8。已有 sealed PASS 上游根可以只读接受；已有 non-PASS 同名根、源码身份变化或任一当前阶段失败都会立即退出。不得第二次调用正式命令，不得删除、覆盖或补写固定根。

## 2. 固定根

```text
artifacts/v2-r1r/p1-v7-integrated-preflight-20260811-1
artifacts/v2-r1r/p1-v7-integrated-query-cache-20260811-1
artifacts/v2-r1r/p1-v7-integrated-k8-20260811-1
```

运行前三个根必须全部不存在。preflight FAIL 时后二者不得创建；query-cache FAIL 时 K8 根不得创建。K8 无论 PASS/FAIL，都不得创建控制组或 P2 根。

## 3. 允许的 transport

若由独立 Codex 执行任务托管长命令，只允许一次隐藏启动，并保存 stdout/stderr：

```powershell
$stdout = 'tmp\p1-v7-integrated-transport\run-p1-integrated.stdout.log'
$stderr = 'tmp\p1-v7-integrated-transport\run-p1-integrated.stderr.log'
Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList @('experiments\v2_r1_revalidation.py','run-p1-integrated') -WorkingDirectory (Get-Location) -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
```

transport 只负责启动一次、观察同一进程树、只读验收和结束前回传；不得重启、补跑子命令、改代码、改文档或改 artifact。

## 4. 只读验收

结束后逐根确认：

- `result.json` 的 `passed/status/gates`；
- `evidence-seal.json` 与根内文件的全量一致性；
- `run-metadata.json` 的 source snapshot identity 在三阶段一致；
- query-cache 的 selection SHA-256、86016 query、全量 content audit；
- K8 的固定 update=20480、episode exposure=655360、checkpoint 无 `claim_probe.*`；
- K01-K10 的逐项值、temporal validation、干预 drop、GPU/吞吐与 finite；
- 失败阶段之后的根不存在，P2 未启动。

回传格式：

```text
[P1_V7_INTEGRATED_WAKE] status=<PASS_K8|FAIL_PREFLIGHT|FAIL_QUERY_CACHE|FAIL_K8>; failed_stage=<none|preflight|query-cache|k8>; artifact=<last_root>; later_roots_absent=<true|false>; summary=<关键 Gate、指标、seal、唯一启动证据>
```
