# V2-R1R P0-D v14 完整 production 执行合同

日期：2026-08-09

设计真源：`docs/v2-r1r-p0d-v14-full-production-design.md`

## 1. 允许写入范围

- `src/yggdrasil_v2/r1_revalidation/production/generator.py`
- `src/yggdrasil_v2/r1_revalidation/production/generator_audit.py`
- `src/yggdrasil_v2/r1_revalidation/production/p0_shortcut.py`
- `src/yggdrasil_v2/r1_revalidation/production/__init__.py`
- `src/yggdrasil_v2/r1_revalidation/__init__.py`
- `experiments/v2_r1_revalidation.py`
- `tests/v2_r1r_v14/`，并在 preflight PASS 后删除 `tests/v2_r1r_v13/`
- v14 设计、执行、结果、主审及 README/计划/目录索引同步
- 唯一 formal root `artifacts/v2-r1r/p0d-v14-full-production-20260809-1/`

`common/`、`audit/`、`learner/`、`integration/`、production renderer/schema/fingerprint/scalable 与所有既有 formal root 均只读。不得创建 v13 兼容 alias。

## 2. 预测试与 adversarial preflight

先运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_v14
.\.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation\production experiments\v2_r1_revalidation.py tests\v2_r1r_v14
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py preflight-production-data
```

预测试必须覆盖：非 9 整除 label schedule、paired label、任意 root provenance 重算、3-attribute/6-value 容量、renderer/fingerprint 不变量、稀疏 NB 对 accepted dense/exact scorer 的 prediction identity、Wilson/Bonferroni 边界、formal root single-use 与 CLI 直接切换。

preflight 固定 seed `2026081401`，每族 train 1,024、其余 split 各 512。命令内部生成两个独立临时数据集并完整审计；dataset 与 report bytes 必须一致。任一 Gate 失败不得进入 formal。

## 3. 唯一正式命令

仅当预测试和 preflight 全部通过，且固定 formal root 不存在时，调用且只调用一次：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py seal-production-data
```

命令内部固定 seed `2026081402`、每族 train 4,096、其余 split 各 512、总计 14,336 条；禁止 CLI 覆盖 seed/count/root。attempt 必须在生成前创建。正式命令必须生成完整报告、shortcut/claim/structure 报告、replay ledger、source snapshot、run metadata、assessment 与 evidence seal。

## 4. 硬停止

formal 返回后，无论 PASS、FAIL、异常或中断，立即停止：

- 不修改、删除、覆盖或补写 formal root；
- 不换 suffix 或 seed 重跑；
- 不因最窄统计 cell 修改 alpha、Wilson 公式、聚合范围或 threshold；
- 不进入 P0-M、Qwen cache、模型、GPU 或训练；
- 父任务只做只读 seal/replay/未知 shortcut 与主设计层验收。

formal PASS 仅表示完整 production P0-D 数据候选获得机器资格；只有父任务主审 accepted 后，才可另立 P0-M 合同。
