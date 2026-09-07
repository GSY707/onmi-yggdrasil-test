# V2-R1R P1-H1-WD 决策因果非正式执行命令

> 终态：本命令已唯一执行并在 W Gate 以
> `FAIL_WD_DECISION_CAUSAL_WRITE_FIT` 正常停止。固定 root 已消耗，以下命令只作
> 审计追溯，禁止再次调用。结果见
> `docs/v2-r1r-p1-h1-wd-decision-causal-failure-review.md`。

本命令只执行 `NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY`。它不调用旧
`run-p1-h1`，不复用 overlap-residual WD checkpoint，不创建 calibration、
formal、F1 或 P2 root。

## 固定身份

```text
source:
  artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/mixed-deployment.pt
  sha256 = 112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D
  package = A52C52225921A0F834A43031A9F4E595B301D7B99AACD2419882D41198486DA4
  cache = .../p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/token-cache

new output:
  artifacts/v2-r1r/p1-h1-wd-decision-causal-screen-20260821-1/

forbidden input:
  artifacts/v2-r1r/p1-h1-wd-overlap-residual-screen-20260821-1/
```

source checkpoint、token cache、package identity 和 record order 必须只读。
新 output root 在启动前必须不存在；若 root 已存在，命令必须 fail closed，
不能删除后重跑。

## 启动前检查

runner 必须按以下顺序完成并记录 manifest：

1. 核对 source checkpoint SHA-256、package identity 和 cache audit；
2. 核对 forbidden WD root 不被读取，且新 output root 不存在；
3. 在同一 predecessor route 上复演 source logits，route replay 必须等价；
4. 用真实 common-off margin drop 构造 target，检查 target 全部 finite、正向
   request fraction `>=0.10`、realized request ratio `>=0.20`；
5. 以 target microbatch `4` 做 CUDA smoke，峰值显存必须 `<6 GB`；
6. 将 train/heldout target 固化为 `causal-target-datasets.pt`，W/D/J 不得在
   batch `32` 上重新求 VJP；固化 manifest 后才允许创建训练 checkpoint。

## 唯一命令

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1r_h1_wd_decision_causal.py run-h1-wd-decision-causal
```

固定预算为 W `800`、D `800`、J `1200` updates，batch `32`、target
microbatch `4`。causal arm 使用正 margin-VJP，matched control 使用相同范数
的负 margin-VJP；两臂共享 seed、sample order、route replay 和 schedule。

J 只使用 ordinary answer CE + two-path causal-target allocation-lock；不得启用
teacher model/logit KL、family classification、route supervision 或
projection-only objective。ordinary answer CE 使用模型自由路由；冻结 source
route schedule 只服务 allocation-lock 的因果 replay，不作为 answer dispatch 或
route CE target。router 与 upstream 在 J 中冻结，J 后两臂 free-route agreement
均须 `>=0.90`。

## 运行状态与停止

runner 必须把预检、target statistics、W/D 两臂进度、J 进度和最终 Gate 写入：

```text
run-state.json
training-events.jsonl
result.json
```

W/D 任一臂 fit Gate、两臂 rollout agreement、最终 common/projection 消融、
Shapley 分工或负 VJP directional-control Gate 失败，都必须原样停止
并保留 root。禁止重跑、换 seed、调 `0.5` margin fraction、放宽 `0.25` site
norm cap、调 schedule、降低 Gate 或启动 formal successor。

最终必须报告：两臂 W/D heldout transfer-normalized nMSE、target
finite/positive/request ratio、
GPU smoke、route replay equality、causal common-off drop、projection-off
source gain、common residual energy、route-replay Shapley C/P shares 及 paired
CI、causal-control heldout gain 及 family regressions。负 VJP control 只检验
方向性，不得写成 shared-only 与 routed 架构收益证明。

无论 PASS 还是 FAIL，本屏的机器声明均为：

```text
scope = NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY
authorizes = nothing
H1/P1/real-text qualification = false
F1/P2 = not authorized
```
