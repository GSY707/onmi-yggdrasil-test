# V2-R1R P1-H1-WD 重合写入—公共残差化非正式设计

日期：2026-08-21  
证据级别：`NONFORMAL_WD_MECHANISM_SCREEN_ONLY`

## 1. 目标与边界

本实验不重跑、修补或改判已封存的 factorized H1 screen。它只回答一个新的机制问题：在保留单一 `shared FFN + shared nonlinear routed features + selected projection` 模型的前提下，能否先把公共 FFN 中已经与 selected projection 同方向的状态更新写入 projection，再把相同更新从公共 FFN 中删除，使正常函数保持、公共路径仍然必要，同时提高 projection 的正常路径作用。

projection 不承担完整 FFN，公共 FFN 不被删除；因此本实验不是两个完整模型或互斥 full experts。结果无论 PASS/FAIL 都不资格化 H1、P1、F1、P2，也不证明真实文本中的无标签路由。

## 2. 冻结起点

唯一 predecessor 是失败 screen 的 stripped mixed checkpoint：

```text
artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/mixed-deployment.pt
SHA-256 = 112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D
```

token cache、screen package identity `A52C5222…486DA4`、全部 record order 与旧 checkpoint 只读复用。新实验只能写入：

```text
artifacts/v2-r1r/p1-h1-wd-overlap-residual-screen-20260821-1/
```

旧 deployment checkpoint 已物理删除 trace head，因此训练只使用 answer target、冻结 predecessor transition 与 self-distillation，不重新创建 trace head，也不使用 family route CE。

## 3. 操作性能力差

对 predecessor 在每个 layer/step 的 attention 后状态 (a)，记录：

\[
C_0(a)=\text{shared\_ffn}(a),\qquad
P_0(a)=P_{r_0}(\Phi(a))
\]

其中 (r_0) 是 predecessor 自己产生的 route。对每条记录，把公共输出在已有 projection 方向上的正向最小二乘系数定义为：

\[
\alpha(a)=\operatorname{clip}\left(
\frac{\langle C_0(a),P_0(a)\rangle}{\|P_0(a)\|^2},
0,0.75\right)
\]

写入目标与转移量为：

\[
P_W(a)=P_0(a)+S(a),\qquad S(a)=\alpha(a)P_0(a)
\]

删除阶段的公共目标为：

\[
C_D(a)=C_0(a)-\left(P_W(a)-P_0(a)\right)
\]

因而在冻结 predecessor trajectory 上：

\[
C_D(a)+P_W(a)\approx C_0(a)+P_0(a)
\]

这是冻结状态坐标中的操作性分解，不宣称得到了唯一的语义能力边界。本轮只迁移与已训练 projection 正向重合的部分；预审测得平均余弦约 `0.46`、正向可解释 common 能量约 `0.23`，各层/route 的全局最小二乘系数约 `0.43–0.50`，所以不会把完整公共 FFN 写入每个 projection。

## 4. 训练阶段

### W：先写

冻结 boundary、attention、router、answer head 与 shared FFN，只训练共享 routed feature trunk 和两个 final projections。训练数据是 predecessor 在线生成但坐标固定的 transition dataset；不读取 `family_targets`。固定预算为 800 updates、batch 32、learning rate `1e-4`。

### D：后删

冻结 W checkpoint 的 routed feature trunk、projections、router 与全部上游，只训练 shared FFN 拟合 (C_D)。目标中的 predecessor state、(C_0)、(P_0)、(P_W) 全部 stop-gradient，避免 moving target。固定预算同为 800 updates。

### J：自由 rollout 联合修正与 matched control

D 通过后，WD 模型与未经 W/D 的 predecessor control 使用完全相同的无 family schedule，各自进行 1,200 updates。两臂只解冻 shared FFN、routed feature trunk、projections 和 answer head；router 保持冻结。损失为 answer CE 加 predecessor-logit KL。该比较用于区分“人为重参数化造成的因果 drop”与真正的 heldout 优势。

## 5. Gate 与停止规则

W heldout normalized MSE 必须不高于 `0.08`；失败即封存并停止。D heldout common normalized MSE 必须不高于 `0.08`，predecessor prediction agreement 不低于 `0.90`，common residual energy 必须处于 `[0.50,0.95]`；任一失败即停止，不进入 J。

最终 mechanism Gate 同时要求：

1. W/D fit Gate 通过；
2. WD 相对 predecessor heldout accuracy 下降不超过 `0.02`；
3. common-off answer drop 至少 `0.02`，排除 projection-only；
4. projection-off effect 相对 predecessor 至少增加 `0.02`；
5. common residual energy 仍处于 `[0.50,0.95]`；
6. 所有训练阶段不读取 family route target；
7. predecessor checkpoint hash 运行后不变。

architecture-benefit 是独立 Gate：WD 相对 matched continuation control 的 heldout macro gain 至少 `+0.05`、paired CI 下界大于零且任一 family 回退不超过 `0.02`。人为构造的 wrong-route/disable-projection 因果变化不能代替该 Gate。

所有机器状态均为 `authorizes=nothing`。本轮继承旧 router 的历史路由，只验证 write-delete 机制；即使得到正信号，也必须另立 fresh、无人工 route label 的真实文本/新数据实验。

## 6. 已执行结果

唯一命令已于 2026-08-21 正常完成，固定 root 已消耗。机器终态为：

```text
FAIL_H1_WD_NONFORMAL_MECHANISM
authorizes = nothing
```

W/D fit、free-rollout retention、common residual energy、无 family target 和 predecessor hash 不变均通过。最终失败项为 `common_remains_necessary=false` 与 `projection_effect_increased=false`；WD 相对 matched control 的 `+0.00488` 增益及跨零 CI 也使 architecture-benefit Gate 失败。该结果不修改本页冻结合同；详细指标与根因解释见 `docs/v2-r1r-p1-h1-wd-failure-review.md`。
