# V2-R1R P0-D v15 执行合同

日期：2026-08-10

设计真源：`docs/v2-r1r-p0d-v15-g09-qualified-production-design.md`

## 1. 允许写入范围

- `pyproject.toml` 与由依赖声明机械同步的 `uv.lock`
- `src/yggdrasil_v2/r1_revalidation/production/g09_decision.py`
- `src/yggdrasil_v2/r1_revalidation/production/g09_qualification.py`
- `src/yggdrasil_v2/r1_revalidation/production/p0_shortcut.py`
- `src/yggdrasil_v2/r1_revalidation/production/generator_audit.py`
- `src/yggdrasil_v2/r1_revalidation/production/generator.py`
- `src/yggdrasil_v2/r1_revalidation/production/__init__.py`
- `src/yggdrasil_v2/r1_revalidation/__init__.py`
- `experiments/v2_r1_revalidation.py`
- `tests/v2_r1r_v15/`，并在 Q PASS 后删除活动 `tests/v2_r1r_v14/`
- v15 设计、执行、结果、主审及 README/计划/目录索引同步
- 唯一 Q root `artifacts/v2-r1r/p0d-v15-g09-qualification-20260810-1/`
- 唯一 F root `artifacts/v2-r1r/p0d-v15-full-production-20260810-1/`

`common/`、`audit/`、`learner/`、`integration/`、production renderer/schema/fingerprint/scalable、v14 sealed root 与所有既有 formal root 均只读。不得创建 v14 兼容 CLI alias 或复制旧测试作为双轨入口。

## 2. 预测试

只允许使用非 F seed 的合成 fixture、临时 root 和 v14 sealed artifact 的只读固定抽样：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_v15
.\.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation\production experiments\v2_r1_revalidation.py tests\v2_r1r_v15
git diff --check
```

预测试必须覆盖 Wilson 上下界、98/14-cell schema、null/local/diffuse 场景、故障矩阵、fast-vs-legacy exact 随机模型和 tie fallback、single-bank 编排、进度事件、Q/F root single-use、Q-before-F guard、CLI 直接切换及 4,096/1,536 配额。预测试不得生成 root seed `2026081502` 的任何 record。

## 3. 唯一 Q 命令

固定 Q root 不存在时，调用且只调用一次：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py qualify-g09-decision
```

命令固定 NumPy PCG64 seed `2026081501`、100,000 trials、train 4,096、heldout 1,536，并对只读 v14 sealed sample 完成 scorer identity/performance qualification。命令必须生成 attempt、decision-power、fault、scorer、performance、replay、assessment、run metadata、source snapshot 与 evidence seal。

Q 返回非零、assessment 非 PASS、seal 不一致或 root 已存在时立即停止；不得创建 F root，不得覆盖 Q 或换 suffix 重跑。

## 4. 唯一 F 命令

仅当固定 Q root 的 assessment、evidence seal 和 source snapshot replay 全部有效，且固定 F root 不存在时，调用且只调用一次：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py seal-production-data
```

命令内部固定 root seed `2026081502`、每族 train 4,096、其余六个 split 各 1,536、总计 26,624 条；CLI 不提供 seed/count/root override。attempt 必须在生成前创建。formal 必须生成完整 generator/shortcut/claim/structure 报告、进度 ledger、regeneration/replay ledger、Q 引用、source snapshot、run metadata、assessment 与 evidence seal。

## 5. 硬停止

F 返回后，无论 PASS、FAIL、异常或中断，立即停止：

- 不修改、删除、覆盖或补写 Q/F formal root；
- 不换 suffix 或 seed 重跑；
- 不修改 alpha、Wilson 公式、98/14 拓扑、effect ceiling 或 1,536 配额解释当前结果；
- 不进入 P0-M、Qwen cache、模型、GPU、训练或架构结论；
- 父任务只做只读 seal/replay、未知 shortcut 与主设计层验收。

F PASS 只表示 fresh-seed production P0-D 候选获得机器资格；只有父任务 main review accepted 后，才可另立 P0-M 合同。
