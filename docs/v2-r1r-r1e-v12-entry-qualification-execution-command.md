# V2-R1R R1E v12 执行合同

日期：2026-08-09

本文件只允许执行 `r1r-r1e-v12-entry-qualification`。设计真源为 `docs/v2-r1r-r1e-v12-entry-qualification-design.md`；任何降低 Gate、缩短语义文本、修改 v7/v9/v10/v11 accepted runtime/artifact 或进入 generator/model/train 的行为都违反合同。

## 1. 允许写入

- `src/yggdrasil_v2/r1_revalidation/production/`
- `experiments/v2_r1_revalidation.py`（直接切换为 v12 CLI）
- `tests/v2_r1r_v12/`
- 本阶段两份 v12 文档及完成后的主审/索引/结果同步
- 唯一 formal root `artifacts/v2-r1r/r1e-v12-entry-qualification-20260809-1/`

禁止修改 `common/`、`audit/`、`learner/`、`integration/` 和所有既有 formal roots。v12 预测试通过后删除活动 `tests/v2_r1r_v11/`；v11 已封印 source snapshot 不受影响。

## 2. 预测试

在仓库根目录使用项目 Python：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_v12
.\.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation\production experiments\v2_r1_revalidation.py tests\v2_r1r_v12
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py preflight-entry
git diff --check
```

预测试输出只写临时目录并自动清理。任一项失败时不得 formal；先在 v12 允许写入范围修复并重新跑完整预测试。

## 3. 唯一 formal

只有预测试全部通过后，执行一次：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py seal-entry
```

命令内部固定 root，不接受替代路径。它必须先原子创建 `attempt.json`，然后生成 qualification bundle、运行 G01–G08、fault/metamorphic/replay、写 assessment 和 seal。root 已存在时必须非零退出。

formal 返回后无论结果如何都停止本合同，不修改 artifact、不换 suffix 重跑、不运行 generator。父任务只允许只读主审：复算 hash、重放 sealed snapshot、调用公共 API 做 registry 外反例，并写独立 main review。

## 4. 后继授权

只有机器 `PASS_R1E_ENTRY` 且主设计层 accepted，才允许父任务另立并执行 R1 generator smoke。该后继授权仍不包含 P0-M、模型、cache、GPU 或训练。
