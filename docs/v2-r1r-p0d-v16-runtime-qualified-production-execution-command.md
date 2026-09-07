# V2-R1R P0-D v16 执行命令

日期：2026-08-10

本文件是 `docs/v2-r1r-p0d-v16-runtime-qualified-production-design.md` 的唯一执行合同。先 Q、后 F；任一阶段失败立即停止。

## 1. 允许范围

允许直接切换以下活动面：

- `src/yggdrasil_v2/r1_revalidation/production/`
- `experiments/v2_r1_revalidation.py`
- `tests/v2_r1r_v16/`
- v16 设计、命令、主审、结果、路线与 `docs/DIRECTORY_REFERENCE.md`

v16 preflight 全通过后删除 `tests/v2_r1r_v15/`；不得保留兼容 CLI、alias 或双活动合同。v7/v9/v10/v11/v12/v13、v14、v15 artifact 全部只读。不得创建或运行 P0-M、模型、cache、GPU 或训练入口。

## 2. Preflight

在仓库根目录执行：

```powershell
$env:PYTHONPATH = "src"
& ".venv\Scripts\python.exe" -m pytest tests\v2_r1r_v16 -q
& ".venv\Scripts\python.exe" -m compileall -q src\yggdrasil_v2\r1_revalidation\production tests\v2_r1r_v16 experiments\v2_r1_revalidation.py
git diff --check
```

preflight 必须验证 fixed roots、seed、版本、v15/v14 引用、runtime assessor fault matrix、非正式 capacity seed、CLI 直接切换和上游 seal。任一命令失败不得执行 Q。

## 3. 唯一 Q 命令

确认 `artifacts/v2-r1r/p0d-v16-runtime-qualification-20260810-1/` 不存在后，只执行一次：

```powershell
$env:PYTHONPATH = "src"
& ".venv\Scripts\python.exe" experiments\v2_r1_revalidation.py qualify-runtime
```

允许的唯一 Q root 是：

`artifacts/v2-r1r/p0d-v16-runtime-qualification-20260810-1/`

命令返回非零、进程中断、root 不完整或 assessment 非 `PASS_RUNTIME_QUALIFICATION` 时立即停止。不得删除、补写、覆盖或用 `-2` 重跑；不得创建 F root。

## 4. 唯一 F 命令

只有 Q assessment、evidence seal 与只读复算全部通过后，确认 `artifacts/v2-r1r/p0d-v16-full-production-20260810-1/` 不存在，再只执行一次：

```powershell
$env:PYTHONPATH = "src"
& ".venv\Scripts\python.exe" experiments\v2_r1_revalidation.py seal-production-data
```

允许的唯一 F root 是：

`artifacts/v2-r1r/p0d-v16-full-production-20260810-1/`

F 使用 root seed `2026081602`。命令返回后无论 PASS/FAIL 均停止；不得换 seed、suffix 或修改 Gate 后续跑。

## 5. 父任务验收

正式进程结束后只读执行：

1. 复算 root child set、全部文件 SHA-256 与 evidence seal；
2. 对 Q 复核 v15/v14 引用、15 轮 paired 数据、98/14/24 投影、完整路径时间、峰值工作集和 fault matrix；
3. 对 F 复核 26,624 条配额、G01–G11、regeneration/replay、Q reference 与 source snapshot；
4. 检查 formal seed 未在 preflight 中进入 builder；
5. 写主审并同步 `docs/v2-r1r-p0-result.md`、`docs/v2-r1-revalidation-task-design.md`、`docs/next-stage-test-plan.md`、`README.md` 与 `docs/DIRECTORY_REFERENCE.md`。

正式 artifact 一旦封印即只读。机器 PASS 只表示合同内 production 数据/测量成立，不自动授权后续训练。
