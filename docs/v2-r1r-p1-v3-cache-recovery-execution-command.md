# V2-R1R P1 v3 cache-recovery 执行合同

日期：2026-08-11

状态：**single-use；固定顺序；失败即停；恢复后继续原 P1 Gate；P1 PASS 后才由父任务另立 P2。**

## 1. 允许的直接切换

活动实现从 P1 v2 直接切换到 P1 v3：修改 `src/yggdrasil_v2/r1_revalidation/p1/`、package init 与 `experiments/v2_r1_revalidation.py`；删除 `tests/v2_r1r_p1_v2/`，新建 `tests/v2_r1r_p1_v3/`。删除 power/data/cache build CLI 与未使用的活动实现，不保留 v2 alias、兼容 reader 或双版本测试入口。v2 由 frozen docs、sealed source snapshot 和 artifacts 保留；v3 verifier 对 v2 schema 的读取是固定输入验证，不是活动兼容面。

允许同步 `README.md`、`docs/DIRECTORY_REFERENCE.md`、`docs/next-stage-test-plan.md`、`docs/v2-r1-revalidation-task-design.md`、`docs/v2-r1r-p0-result.md` 和独立结果/复核文档；状态文档不进入 frozen active source。

## 2. 启动前验证

```powershell
.venv\Scripts\python.exe -m pytest -q tests\v2_r1r_p1_v3
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation experiments\v2_r1_revalidation.py
git diff --check
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
```

随后只运行一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py preflight-p1
```

preflight 必须验证 frozen v3 docs、活动 source identity、v2 accepted power/data、v2 failed cache 的固定 hashes、CUDA/磁盘和全部 v3 roots absence。失败立即停止。

## 3. 正式顺序

每条命令只允许调用一次；前一条 sealed PASS 才允许下一条：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py qualify-recovery-p1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py recover-cache-p1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-latent --slots 8
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-latent --slots 1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-direct
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-text-cot
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py evaluate-p1
```

`run-p1` 只能接受已经存在且 sealed PASS 的 v3 preflight/telemetry/recovery root，并从首个不存在的训练阶段继续；existing non-PASS root 或新失败立即停止。

长命令允许由独立 Codex 任务启动，但必须使用本目录、项目 Python 和上述唯一参数。stdout/stderr 可以重定向到 root 外的 transport log；这不改变 formal 命令身份。不得因调用端等待超时再次启动同一阶段，必须先检查进程与 fixed root。

## 4. 失败与后继

- telemetry FAIL：停止，不运行 recovery；
- recovery FAIL：停止，不训练；
- K=8 FAIL：停止，不运行 K=1/baselines；
- 任一对照或 assessment 不完整：不进入 P2；
- P1 PASS：父任务同步主设计层验收，另立完整 P2 matched-Pareto 合同后启动；P1 CLI 自身不得越级。
