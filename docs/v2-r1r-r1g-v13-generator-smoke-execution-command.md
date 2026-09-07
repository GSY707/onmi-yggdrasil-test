# V2-R1R R1G v13 执行合同

日期：2026-08-09

设计真源：`docs/v2-r1r-r1g-v13-generator-smoke-design.md`。本合同只执行 1,440-record production generator smoke，不执行完整 P0-D、P0-M、模型或训练。

## 1. 允许写入

- `src/yggdrasil_v2/r1_revalidation/production/fingerprint.py`
- `src/yggdrasil_v2/r1_revalidation/production/generator.py`
- `src/yggdrasil_v2/r1_revalidation/production/generator_audit.py`
- `production/__init__.py`
- `experiments/v2_r1_revalidation.py`（直接切换 v13）
- `tests/v2_r1r_v13/`
- v13 文档、主审和仓库索引/结果同步
- 唯一 formal root `artifacts/v2-r1r/r1g-v13-generator-smoke-20260809-1/`

禁止修改 accepted `common/audit/learner/integration` 与 v12 renderer/scalable/schema；禁止修改任何既有 formal root。v13 完整预测试通过后删除活动 `tests/v2_r1r_v12/`。

## 2. 预测试顺序

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_v13
.\.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation\production experiments\v2_r1_revalidation.py tests\v2_r1r_v13
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py preflight-generator-smoke
git diff --check
```

preflight 必须在临时 root 生成两次并得到 byte-identical dataset/report，再运行 G01–G11。任一失败时只在 v13 允许写入范围分析、修复并从空临时 root 重生；不得降低规模、阈值或移除 baseline。

## 3. 唯一 formal

所有预测试和 direct-switch guard 通过后，只执行一次：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py seal-generator-smoke
```

命令内部固定 formal root。返回后无论 PASS/FAIL 都停止：不修改 dataset/artifact，不覆盖，不换 suffix，不运行完整 P0-D 或 P0-M。父任务只做只读 seal/replay、公共 API registry 外反例与主设计层验收。
