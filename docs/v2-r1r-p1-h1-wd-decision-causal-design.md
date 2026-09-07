# V2-R1R P1-H1-WD 决策因果写入—公共残差化非正式设计

日期：2026-08-21  
证据级别：`NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY`

## 1. 目的与边界

上一轮 overlap-residual screen 证明了一个有限事实：把 `common` 与
`projection` 的隐藏状态重合方向搬运过去，可以在冻结轨迹上保持函数，但没有
让 projection 获得对答案真正有因果作用的能力。本轮因此改写“可转移分量”的
定义：不再按隐藏向量余弦分解，而按真实答案 margin 对 common 的消融影响来
请求 transfer，再用该 margin 的反向传播方向决定 transfer 的向量方向。

这仍是同一个模型：共享 attention、共享 FFN、冻结 route 的 routed feature
trunk、selected projection 和同一个 answer head。projection 只接收被测得的
决策相关增量，不能单独成为第二个完整模型。结果无论 PASS/FAIL 都不能资格化
H1、完成 P1、证明真实文本无标签路由，也不授权 F1/P2。

## 2. 只读 predecessor 与隔离 root

唯一输入是旧 factorized screen 的 stripped predecessor：

```text
artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/mixed-deployment.pt
SHA-256 = 112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D
package = A52C52225921A0F834A43031A9F4E595B301D7B99AACD2419882D41198486DA4
cache = .../p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/token-cache
```

新实验只允许写入：

```text
artifacts/v2-r1r/p1-h1-wd-decision-causal-screen-20260821-1/
```

上一轮 `p1-h1-wd-overlap-residual-screen-20260821-1` 的任何 checkpoint 都是
本合同禁止输入；本轮不 warm-start、不读取、不覆盖。source checkpoint、cache、
package identity 和 record order 全程只读，运行后必须重新核验 hash。

## 3. 决策因果 transfer target

### 3.1 真实 margin drop

对 predecessor 的同一路由、同一答案 token，定义正确答案 margin：

\[
m(x)=\ell_{y^*}(x)-\log\sum_{k\ne y^*}\exp \ell_k(x)
\]

分别运行 common-on 与真实 common-off，得到：

\[
d_m(x)=\max\bigl(0,\;m_{on}(x)-m_{off}(x)\bigr).
\]

这不是 family/task 分类标签，也不是教师模型输出；它只使用普通答案 token
的标量 margin 和模型自身的真实 common 消融。请求 projection 吸收的 margin
只取一半：

\[
d_{req}(x)=0.5\,d_m(x).
\]

### 3.2 scalar-margin VJP 方向

在每个 routed projection site `i`，对标量 `m(x)` 求相对于该 site selected
projection 输出的 VJP：

\[
v_i=\frac{\partial m}{\partial P_i},\qquad
u_i=\frac{v_i}{\max(\lVert v_i\rVert,\epsilon)}.
\]

`u_i` 是该 site 的决策方向；它不读取 family target，也不从另一个模型蒸馏
logits。把 `d_req` 按各 site 的一阶可见度分配为 `d_i`，然后选择最小的
非负幅度 `a_i` 使 `v_i·(a_i u_i)` 接近 `d_i`，同时强制：

\[
\lVert\Delta P_i\rVert=\lVert a_i u_i\rVert
\le 0.25\,\lVert C_i\rVert,
\qquad \Delta P_i=a_i u_i.
\]

因此每个 site 的 transfer 既有真实 margin 依据，又有独立的 common-norm
上限。train/heldout target 必须按 microbatch `4` 一次性物化到 CPU，并封装为
不含 family/task target 的 `causal-target-datasets.pt`；W/D/J 都从这份冻结
数据集按 record order 取值，不得在 batch `32` 上重新求 VJP。target manifest
必须记录有限性、正向请求比例、每 site 请求量和最终实现量；实现量与请求量的
比率低于 `0.20` 时直接 Gate FAIL，而不是事后解释为“梯度太小”。

### 3.3 causal arm 与 matched control

causal arm 的 W target 为 `P0 + ΔP`，D target 为 `C0 - ΔP`。matched
control 使用完全相同的 route、样本、site norm 和 schedule，但以负 VJP：

```text
causal:  ΔP_i = +a_i u_i,   C_D = C0 - ΔP
control: ΔP_i = -a_i u_i,   C_D = C0 + ΔP
```

两臂在 predecessor frozen trajectory 上都满足 `C_D + P_W ≈ C0 + P0`，
差别只在 transfer 是否沿着答案 margin 的因果方向。这是必要的 matched
control：若两臂表现相同，不能把任何因果 drop 误认为架构收益。

## 4. 训练阶段

W（800 updates）只训练 routed feature trunk 和 projection；common、attention、
router、upstream 与 answer head 冻结。causal/control 两臂都用 batch 32，目标
构造使用 target microbatch 4，以把 VJP 峰值显存控制在 6 GB 以下。
W 的 fit 与训练损失按 `||(P-P0)-ΔP||²/||ΔP||²` 检查转移增量本身，不能用
完整 `P0+ΔP` 的能量稀释误差后误判为“已写入”。

D（800 updates）固定各自的 W checkpoint，只训练 shared FFN 拟合对应的
`C_D` target；route、upstream、trunk 和 projection 冻结。两臂都必须通过 heldout
common transfer-fit 与 rollout retention。D target 使用 W 实际写入的
`P_W-P0`，而不是假定 W 已精确实现理想 `ΔP`。

J（1,200 updates）对 causal arm 与 matched control 使用同一 answer-only
continuation schedule。只解冻 common、routed projection 分支和 answer head；
router/upstream 保持冻结。损失是 ordinary answer CE 加 two-path
allocation-lock：在冻结 source route replay 上同时约束
`P-P0≈±ΔP` 与 `C-C0≈∓ΔP`。它只读取前述因果 target，不读取 predecessor
logits 或 family/task target；冻结 schedule 只用于可比的因果 replay，不作为
route CE target，也不注入 ordinary answer CE。answer CE 继续使用模型自由路由，
以避免把 predecessor 的逐记录 route 变成教师标签；不使用 teacher KL、route
CE、family CE 或 projection-only loss。router 本身始终冻结，最终两臂相对 source
的 free-route agreement 还必须分别至少为 `0.90`，否则机制 Gate FAIL。

## 5. 冻结 Gates 与 fail-stop

启动前必须通过 predecessor identity、cache/package audit、route replay
等价性、output root 不存在和 CUDA 检查。target 还必须满足：所有值 finite，
正向请求 site 比例至少 `0.10`，实际实现请求比率至少 `0.20`，target microbatch
GPU smoke 峰值小于 `6 GB`。

W/D 两臂各自对转移增量归一化的 heldout MSE 都必须不超过 `0.08`；D 阶段两臂的
rollout agreement 必须至少 `0.90`。否则原样封存并停止，不进入 J 或干预评估。
J 后两臂的 free-route agreement 也都必须至少 `0.90`，以防自由 answer CE 的
route 漂移偷换固定-route transfer/Shapley 证据。

最终机制 Gate 要求 causal arm 同时满足：

1. common-off 正确答案 drop 至少 `0.02`；
2. projection-off 相对 source 的 effect gain 至少 `0.02`；
3. common residual energy 位于 `[0.50, 0.95]`；
4. route-replay 两玩家 Shapley 中，projection share 相对 source 和 matched
   control 的增益都至少 `0.05`，其 CI 下界大于零；
5. common 与 projection 两个 Shapley component share 都至少 `0.20`。

Shapley 只在固定 route replay 上计算，避免把 route 改变误计为组件能力。对
`C/P` 两个玩家，使用所有加入顺序的平均边际贡献；projection share 为
`|φ_P|/(|φ_C|+|φ_P|+ε)`，并对 paired heldout records 做 CI。

独立 directional-control Gate 要求 causal arm 相对负 VJP matched control 的
heldout gain 至少 `+0.05`、CI 下界大于零、任何 family regression 不超过
`0.02`。该 Gate 不能由人为制造的 common-off/projection-off drop 替代；它只
证明 VJP 正方向相对反方向是否有用，不是 shared-only 与 shared+routed 的架构
对照，因此不能单独证明架构收益。

transfer-mechanism Gate 与 directional-control Gate 任一 FAIL，总状态都必须是
FAIL。任何 Gate FAIL 都是终态：不得删除 root 重跑、调 transfer fraction、改
schedule、降低阈值、启动 formal successor。所有结果固定为
`authorizes=nothing`。

## 6. 证据解释

本屏若通过，只说明“答案 margin 驱动的、受 norm 限制的 projection transfer”
值得另立 fresh 研究；仍不说明真实文本里可无标签地发现任务分类路由。若失败，
应根据 target finite/realization、W/D fit、causal retention、Shapley 分工和
matched-control gain 的具体 Gate 分离是方向估计失败、分配失败还是方向收益
缺失，不能把一个低层 failure 写成 H1/P1 结论。

## 7. 唯一运行结果

唯一命令已执行并以 `FAIL_WD_DECISION_CAUSAL_WRITE_FIT` 正常停机。target Gate
全部通过：train/heldout positive fraction 为 `0.97900/0.57715`，realized/request
为 `0.85606/0.93106`，CUDA peak 均为 `0.32335 GB`。但 W heldout
transfer-increment nMSE 为 causal `1.00450`、negative-VJP control `0.99247`，
远高于 `0.08`。完整输出 nMSE 虽只有约 `0.0047`，但这只是 transfer energy 仅占
完整 target energy 约 `0.468%` 造成的零学习假象。

合同按 Gate 原样停止，未保存 W checkpoint，D/J、干预、Shapley 与方向对照均未
执行。root 已消耗，禁止重跑或调参；完整归因与封存 hash 见
`docs/v2-r1r-p1-h1-wd-decision-causal-failure-review.md`。
