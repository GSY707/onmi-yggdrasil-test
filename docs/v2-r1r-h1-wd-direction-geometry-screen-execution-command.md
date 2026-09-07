# V2-R1R H1-WD R0–R4 direction-geometry 执行合同

## 1. 唯一入口

唯一命令为：

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1r_h1_wd_direction_geometry.py run-h1-wd-direction-geometry
```

CLI 只接受 `run-h1-wd-direction-geometry`，并只调用
`screen.run_direction_geometry_screen(repo_root)`。实际 orchestration 位于新的隔离
模块 `h1_wd_direction_geometry/screen.py`，不得回填 consumed 的旧 WD 模块。

## 2. 只读 preflight

启动前必须验证：

- source checkpoint、cache manifest 和 causal target bank 存在；
- source checkpoint SHA-256 与 target bank SHA-256 完全匹配合同；
- token-cache manifest 与三个 frozen array hash 匹配；
- train/heldout 为 `4096/1024`，每条记录 16 sites，record order 不变；
- 输出 root `artifacts/v2-r1r/h1-wd-direction-geometry-screen-20260823-1/`
  尚不存在；
- 旧 causal/overlap-residual root 不被写入、删除、覆盖或作为 checkpoint 输入。

任何失败立即停止，不产生后继训练或 formal root。

现有 train ID 为连续前段、heldout ID 为后段且中间缺 1536 条，不能默认 IID random
split。该 gap 不得偷偷并入训练或评分；当前 bank 又没有 family/class 标签，因此
本次结果必须明确 `family_group_ood_measured=false`。

## 3. 执行顺序

```text
preflight → R0 global → R1 route-conditioned → R2 frozen-feature
→ R3 Jacobian/Fisher → R4 null/randomness audit → final result
```

所有计算均为 frozen inference、feature extraction、统计和线性代数。禁止
optimizer、`backward` 后参数更新、model/checkpoint write、target regeneration、
Qwen encoding、route training 或 family/task/answer supervision。

## 4. 输出要求

输出只允许保存 preflight、输入 hash、R0–R4 metrics、bootstrap/permutation null、
Jacobian/Fisher diagnostics、residual classification 和最终 result。不得保存模型、
optimizer、梯度更新或 successor checkpoint。最终字段必须包含：

```text
scope = NONFORMAL_H1_WD_DIRECTION_GEOMETRY_SCREEN_ONLY
authorizes = nothing
h1_qualified = false
p1_completed = false
f1_authorized = false
p2_authorized = false
```

heldout 是历史 finite split，不得声称 IID 泛化；record-cluster bootstrap 只描述
该 split 内不确定性。任何 R0–R4 PASS 都不能恢复旧 decision-causal root 或授权
H1/P1/F1/P2。

## 5. fail-stop

任一 Gate、hash、finite、coverage、route-permutation、record-permutation、split-half
或 heldout integrity failure 都是终态。禁止删除 output 后重跑、改 lambda、改
epsilon、换 sketch、换 split、降低 Gate、启动 calibration 或 formal successor。
