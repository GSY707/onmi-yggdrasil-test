# V2-R1R P1-H1-WD decision-causal 非正式失败复盘

日期：2026-08-21  
证据级别：`NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY`  
机器终态：`FAIL_WD_DECISION_CAUSAL_WRITE_FIT`  
机器授权：`authorizes=nothing`

## 1. 核心判断

本轮没有失败在 target 构造、显存、梯度消失或 J 阶段奖励不足，而是更早失败在
W 的 heldout 增量拟合：按真实 common-off answer-margin drop 与
projection-output VJP/Fisher 构造出的逐记录 `ΔP`，不能在冻结预算内被当前共享
参数的 routed projection 函数泛化实现。

这一区分很重要。output-space VJP 说明“若能独立修改某条记录、某个 site 的
projection 输出，哪个方向会提高答案 margin”；它并不自动说明存在一个全局共享
参数更新 `Δθ_P`，可以同时对所有记录实现这些输出变化。当前 W 实际需要满足：

\[
P_{\theta+\Delta\theta}(h_x)-P_\theta(h_x)\approx\Delta P^*(x),
\]

而真正可达的一阶空间是 `J_θ(x)Δθ`。本轮直接把每条记录的 output VJP 当作监督
target，尚未先把它投影到跨记录一致的 projection-parameter Jacobian 子空间。这是
当前最有证据支持的设计缺口；但单次 frozen-budget 结果仍不能完全分开“参数空间
不相容”与“现预算优化不足”，因此不得写成绝对不可学习定理。

## 2. 预检与 target Gate 均通过

唯一输入仍是原 factorized screen 的 immutable mixed deployment：

```text
artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/mixed-deployment.pt
SHA-256 = 112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D
```

普通 forward 与固定 route replay 的 logits、route、trajectory 位级一致，最大
logit 误差为 `0`。train/heldout target 所有 materialized common、projection、
transfer、VJP、margin 均 finite；family/task target 未进入 target、schedule 或
训练目标。

| target 指标 | train | heldout | Gate |
| --- | ---: | ---: | ---: |
| records | 4,096 | 1,024 | — |
| positive request fraction | 0.97900 | 0.57715 | ≥0.10 |
| realized/request | 0.85606 | 0.93106 | ≥0.20 |
| CUDA peak GB | 0.32335 | 0.32335 | <6.0 |
| sites per record | 16 | 16 | — |

因此 norm cap 没有把 target 信号抹掉。target 数据集已经一次性固化为
`causal-target-datasets.pt`，大小 `2,685,312,779` bytes，SHA-256 为
`DFC8F08776CE56EFB8022BF35B7C3DD72F8F1C53C95DBF0DD28F593DC2BFD503`。

## 3. W Gate 的直接失败证据

两臂使用完全相同的 4,096 条训练记录、schedule 与 exposure；每条记录出现
`6–7` 次。两臂都完成全部 800 updates，没有 NaN、OOM 或 crash。记录到的当前
minibatch transfer-nMSE 在 update 200/400/600/800 分别约为：

```text
causal  +VJP: 0.4283 / 0.5060 / 0.3736 / 0.4363
control -VJP: 0.4280 / 0.5017 / 0.3705 / 0.4353
```

梯度范数持续非零，四个记录点约位于 `1.47–3.89`；这排除了“没有训练信号”。
但 heldout 增量拟合几乎等于零学习起点：

| W heldout 指标 | causal +VJP | control -VJP | Gate |
| --- | ---: | ---: | ---: |
| full-target nMSE | 0.0047067 | 0.0046475 | 仅诊断 |
| transfer-increment nMSE | **1.00450** | **0.99247** | ≤0.08 |
| transfer energy | 63,972.09 | 63,972.09 | matched |
| realized/request | 0.93106 | 0.93106 | ≥0.20 |

transfer energy 只占完整 target energy 的 `0.4683–0.4686%`。若 projection 在
heldout 完全保持 `P0`，旧式 full-target nMSE 本来就会约为 `0.004686`；实测
causal 的 `0.004707` 恰好等于该零学习基线乘以 `1.00450`。所以若继续沿用完整
`P0+ΔP` 能量归一化，本轮会被错误写成 W fit PASS。新增的增量 Gate 在这里成功
阻止了假阳性。

## 4. 根因分层

可以排除的解释包括：

- target 非 finite 或 common-off signal 太稀疏：两个 split 均通过 Gate；
- `0.25×||common||` cap 过紧：只读 post-stop 几何复算显示 cap 饱和的 site-record
  比例仅为 train `0.1151`、heldout `0.0367`；
- VJP 只落在单一 site：Fisher effective sites 平均为 train `12.19`、heldout
  `9.39`；
- 正负号实现错误：两臂同能量、同 exposure，学习曲线与 heldout failure 高度
  对称；
- J 奖励不足：J 根本没有启动。

最可能的失败链是：逐记录 output VJP target 高度异质且分散在多个 recurrent
site；同一 routed trunk/projection 参数必须同时保持 `P0` 并拟合这些小而
sample-conditioned 的向量。只读几何复算中，各 site 的 normalized transfer
方向均值范数平均只有 train `0.2059`、heldout `0.1729`，说明方向并非一个可由
全局常量偏移吸收的公共轴。minibatch loss 有响应而 heldout nMSE 仍约为 `1`，
更符合“局部追随、跨记录不泛化”，而不是奖励强度不足。

这些 post-stop 几何量只用于解释，不是预登记 Gate，不能据此追溯改判。

## 5. 已接受、已拒绝与未测量

本轮接受的有限证据是：真实 common-off margin drop 可以稳定产生 finite、显存
可控且大部分可实现的一阶请求；固定 route replay 与无 family/task target 数据流
成立；transfer-increment Gate 能识别完整输出 nMSE 隐藏的零学习假象。

本轮拒绝的是更具体的命题：当前 routed projection 参数化在 W800 合同下可以把
raw per-record output VJP/Fisher target 泛化写入 projection。

由于 W Gate fail-stop，以下内容全部没有测量：D 的公共残差化、J continuation、
common/projection 消融、固定-source-route Shapley、正 VJP 相对负 VJP 的方向收益。
因此不能说“先写后删的 parameter-reachable 版本已失败”，也不能从本轮推断
shared+routed 架构优于或劣于 shared-only。projection 从未被训练成单独模型。

## 6. 后继研究条件

当前 root 已消耗，禁止删除后重跑、延长 W、调学习率、降低 `0.08` Gate、缩小
target 或直接进入 D/J。若未来另立 fresh 研究，首先应在训练前资格化
“parameter-reachable causal transfer”，而不是再次使用 raw output VJP：

1. 仍以真实 common-off answer-margin drop 定义需要搬运的能力幅度；
2. 对 projection 参数求跨记录 Jacobian/Fisher，显式处理不同记录梯度冲突；
3. 把理想 `ΔP*(x)` 投影到联合可达空间 `J_θ(x)Δθ`，或冻结一个 heldout 可预测的
   低秩 basis；
4. 在任何 W 训练前，用独立 heldout reachability/predictability Gate 证明该 target
   不是逐样本 oracle 向量；
5. 负方向臂仍只作方向性 control，不冒充 shared-only 架构基线。

这是一条新实验身份，不由本轮授权。

## 7. 封存与完整性

固定 root：

```text
artifacts/v2-r1r/p1-h1-wd-decision-causal-screen-20260821-1/
```

`result.json` SHA-256 为
`9A574022E68C161BBF2142D22E3055E3945704C50988E12C107B82DDCBBF5CB4`。
运行正常退出码 `0`，总耗时 `1606.16 s`，无 `crash.json`。W Gate 失败发生在任何
checkpoint 保存之前，因此没有 write/delete/joint checkpoint，也没有 formal、
calibration、F1 或 P2 后继 root。source checkpoint hash 与旧 overlap-residual
`result.json` hash `BEE1978D…BC15E` 均保持不变。
