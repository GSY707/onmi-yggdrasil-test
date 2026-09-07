# V2-R1R P0-D v9 R0B-invariant 执行合同

日期：2026-08-02  
固定 formal root：`artifacts/v2-r1r/p0d-v9-r0b-invariant-20260802-1/`

## 1. 适用范围

本合同只资格化 G02–G06 measurement audit。它不授权 R0C、R0D、R1、P0-M、模型、训练、cache 或 GPU 实验。v7 simulator 是只读上游；v8 formal artifact 是 rejected diagnostic，不得覆盖或补跑。

## 2. formal 前条件

formal 前允许重复执行：

```powershell
python -m pytest -q tests/v2_r1r_v9
python -m compileall -q src/yggdrasil_v2/r1_revalidation tests/v2_r1r_v9 experiments/v2_r1_revalidation.py
python experiments/v2_r1_revalidation.py preflight-invariant
```

还必须人工确认：

1. fixed root 不存在；
2. `contract_guard.py` 对 frozen inputs 返回成功；
3. v7 evidence seal 与三个 accepted simulator 文件 SHA-256 精确；
4. v8 test tree、`structural.py` 和旧 CLI 已被直接删除；
5. 公共 API 的 registry 外反例、malformed input、输入只读与离线 replay 已通过；
6. `git diff --check` 通过，且没有进入未授权阶段。

## 3. 唯一 formal 命令

满足全部前条件后，只允许执行一次：

```powershell
python experiments/v2_r1_revalidation.py seal-invariant
```

runner 必须先用 `mkdir(exist_ok=False)` 占用 fixed root，再写 `attempt.json`，之后才允许 materialize 或 audit。fixed root 已存在时只能返回 `BLOCKED`。

## 4. Gate 与停止规则

成功必须同时满足：positive G02–G06、四类 adversary lattice、metric kill coverage、六类 positive metamorphic、确定性/只读/fail-closed、精确 artifact tree 与 evidence seal。

formal 一旦开始，无论 PASS、FAIL、异常或中断都立即停止；不得修补、覆盖、改 suffix 或重跑。机器 PASS 仍只产生 main-review candidate。registry 外出现任何 false negative，v9 必须登记为 `machine-pass、main-review rejected`，后续另立合同。
