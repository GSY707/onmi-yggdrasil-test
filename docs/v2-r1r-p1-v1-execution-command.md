# V2-R1R P1 v1 执行命令

状态：**single-use 正式执行合同；任一阶段失败立即停止；P2 未授权。**

工作目录固定为 `C:\Users\24408\Documents\onmi yggdrasil test`，解释器固定为项目 `.venv\Scripts\python.exe`。禁止使用系统 Python，禁止覆盖任一正式 root。

## 1. 预测试

预测试只验证合同、实现、缓存格式与小张量数值，不产生 P1 证据：

```powershell
.venv\Scripts\python.exe -m pytest -q tests\v2_r1r_p1_v1
.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation\p1 experiments\v2_r1_revalidation.py
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py preflight-p1
git diff --check
```

任一命令失败不得创建正式 data root。

## 2. 正式阶段

每条命令必须看到前序 root 的有效 evidence seal；已有同名 root 时拒绝运行。

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py generate-p1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py cache-p1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-latent --slots 8
```

若 K=8 返回非零，立即停止。只有 K=8 sealed PASS 后才运行：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-latent --slots 1
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-direct
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py train-text-cot
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py evaluate-p1
```

`run-p1` 只按上述顺序编排；它可以读取并复验已经完成的 sealed PASS，但不能重跑或覆盖任何 stage：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1
```

## 3. 停止规则

- data G01–G11 失败：保留 fresh-data root，停止 cache/training；
- cache C01 失败：保留 cache root，停止 K=8；
- K=8 K01–K09 失败：保留 checkpoint、progress、干预与 seal，停止所有对照；
- K=1 或文本对照不完整：不创建 PASS assessment，停止 P2；
- `PASS_P1_SINGLE_SEED` 也只产生 `p2_eligible=true`、`p2_started=false`。

失败后修复必须另立版本、root 与合同；不得重用本文件的 single-use 命令。

