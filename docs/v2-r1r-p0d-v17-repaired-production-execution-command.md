# V2-R1R P0-D v17 执行命令

## 1. 固定边界

只执行 `docs/v2-r1r-p0d-v17-repaired-production-design.md`。v7–v16 artifact 全部只读。不得降低 Gate、修改统计阈值、覆盖固定 root、用 nonce 黑名单代替 path semantics，或在 P0-D main review 之前进入 P0-M。

## 2. Preflight 与直接切换

使用仓库 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_v17 -q
.\.venv\Scripts\python.exe -m py_compile experiments\v2_r1_revalidation.py src\yggdrasil_v2\r1_revalidation\production\*.py tests\v2_r1r_v17\*.py
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
```

必须确认活动 CLI 仅含 `qualify-repairs`、`seal-production-data`、`audit-production-data`，版本/seed/root 与设计一致，并删除旧 `tests/v2_r1r_v16/*.py`。

## 3. 唯一 repair qualification

仅运行一次：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py qualify-repairs
```

只有 assessment 为 `PASS_REPAIR_QUALIFICATION`、R01–R07 全 true、evidence seal 复验通过，才能执行正式数据。

## 4. 唯一 fresh formal

仅运行一次：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py seal-production-data
```

结束后必须读取 assessment、generator/structure/shortcut/claim report、replay ledger、run metadata 与 evidence seal。若任一 G01–G11 false，保留失败 artifact 并以新版本/新 seed 修复；不得重跑同一 root。

## 5. 汇报

报告 repair Gate、正式 Gate、规模、seed、生成/审计/重生时间、peak memory、关键统计余量、seal hash，以及 completed/not-completed。只有父任务 main review 明确接受 P0-D 后，才可冻结 P0-M 合同；本命令本身不运行训练。
