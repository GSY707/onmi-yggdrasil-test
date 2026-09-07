# V2-R1R P1 v2 执行合同

日期：2026-08-10

状态：**single-use；固定顺序；任一 Gate 失败立即停止；P1 PASS 后才由父任务另立 P2。**

## 1. 允许的直接切换

活动实现直接从 P1 v1 切换到 P1 v2：修改 `src/yggdrasil_v2/r1_revalidation/p1/`、`experiments/v2_r1_revalidation.py` 和 package init；删除 `tests/v2_r1r_p1_v1/`，新建 `tests/v2_r1r_p1_v2/`。不得保留 v1 CLI alias、兼容 schema、兼容 reader 或并行测试入口。v1 由 sealed source snapshot、冻结文档与失败 artifact 保留。

允许同步 `README.md`、`docs/DIRECTORY_REFERENCE.md`、`docs/next-stage-test-plan.md`、`docs/v2-r1-revalidation-task-design.md`、`docs/v2-r1r-p0-result.md` 和独立结果/主审文档；状态文档不进入 frozen active source。

## 2. 启动前验证

使用项目环境：

```powershell
.venv\Scripts\python.exe -m pytest -q tests\v2_r1r_p1_v2
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation experiments\v2_r1_revalidation.py
git diff --check
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
```

随后只运行一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py preflight-p1
```

preflight 必须验证上游 accepted seals、P1 v1 失败 result/seal、冻结设计/执行 hash、CUDA/8 GiB 级 GPU、磁盘和所有正式 roots absence，并封存 active-source identity。失败立即停止。

## 3. 正式顺序

每条命令只允许调用一次；前一条 sealed PASS 才允许下一条：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py qualify-p1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py generate-p1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py cache-p1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-latent --slots 8
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-latent --slots 1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-direct
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-text-cot
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py evaluate-p1
```

也可在 preflight PASS 后运行 `run-p1`；runner 只能接受已经存在且 sealed PASS 的同版本 root，遇到 existing non-PASS root 或新失败立即停止。

## 4. 失败与后继

- Q FAIL：停止，不生成 fresh data；
- D FAIL：停止，不建 cache；
- cache FAIL：停止，不训练；
- K=8 FAIL：停止，不运行 K=1/baselines；
- 任一对照不完整：不创建 PASS assessment；
- P1 PASS：同步主设计层验收，然后另立 P2 设计/执行合同并启动；不得把 P1 训练时间或离线 accuracy 自行写成 Pareto。

