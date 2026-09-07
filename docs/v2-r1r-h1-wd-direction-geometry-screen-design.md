# V2-R1R H1-WD R0–R4 direction-geometry 非正式设计合同

日期：2026-08-23  
证据级别：`NONFORMAL_H1_WD_DIRECTION_GEOMETRY_SCREEN_ONLY`  
机器授权：`authorizes=nothing`

## 1. 目的与边界

本 screen 只检查旧 H1-WD decision-causal target 是否存在跨记录可复用的方向结构，
以及这些方向是否落在当前 routed feature/projection 参数的一阶共同可达空间内。
它不训练模型、不执行 optimizer step、不写 checkpoint、不重建 causal target，也不
修改、删除或重跑旧 root。

旧 decision-causal root 已 consumed。它的 `causal-target-datasets.pt` 只作为冻结
输入；旧 W 的 `1.00450/0.99247` heldout transfer-increment nMSE 保持原判决，不由
本 screen 追溯改判。R0–R4 全部通过也只授权“另立 parameter-reachable causal
transfer 研究”的设计讨论，不能授权 H1、P1、F1、P2、calibration 或真实文本路由。

## 2. 允许输入与证据边界

唯一输入 allowlist、固定 hash、禁止操作和输出 root 由
`src/yggdrasil_v2/r1_revalidation/h1_wd_direction_geometry/contract.py` 定义。
输入包括 immutable predecessor checkpoint、token cache、旧 causal target bank
及其只读 metadata。禁止读取旧 overlap-residual checkpoint，禁止调用旧
decision-causal CLI，禁止使用 family/task/answer supervision。

target bank 是 prepared train/heldout order 的历史 bank，不是新抽样数据。其
heldout 不能自动被解释为 IID 总体；报告必须写明 finite historical split、record
cluster resampling 和 heldout distribution risk。

## 3. 统一对象与指标

每个 site 为 `(layer, step)`，共 16 个 site；每条记录的 transfer target 是
`y[x,site] = ΔP*`，shape `[8,256]`。primary metric 是不含 `P0` 能量稀释的
transfer-increment normalized MSE：

```text
sum(||prediction - target||²) / max(sum(||target||²), epsilon)
```

同时报告 cosine、解释能量、site/route 分布和 record-cluster bootstrap CI。16 个
site 不能当作独立样本。主能量分解只沿 record 轴求均值，保留完整
`[site,slot,256]` 坐标；把 site/slot 一起池化成单个 256 维 centroid 只回答另一
个输出空间问题，禁止用它制造“全局方向接近零”的结论。

## 4. R0–R4

### R0：global centroid

在 train 上估计不依赖 route 的完整 record target centroid，heldout 只应用该
train-fitted tensor。R0 是全局公共方向 baseline，不是机制通过门。

### R1：route-conditioned centroid

在 train 上按 `(site, frozen predecessor route)` 估计 centroid，与 R0 做 paired
heldout 对比，并执行保持 route 计数的 route-permutation null。route 只能作为历史
模型条件键，不能写成 route supervision 或 route causal proof。每个 route cell
需要预登记最小样本数；缺失 cell 直接记录 coverage failure。

### R2：frozen-feature heldout predictability

从 immutable predecessor checkpoint 与 token cache 只读重放 target 构造前状态，
主特征是 frozen routed feature trunk 输入到 final head 的输出；attention state 与
common private hidden 是 target-before 敏感性对照。仅用 train 拟合固定 ridge，
heldout 只评估一次。bank 中保存的 common/projection baseline output 只能标为
`frozen_output_proxy_only`；尤其 projection output 与 margin/VJP target 相邻，不能
冒充 input-predictable 证据。margin、VJP、target norm、positive-request mask 及其
派生量一律不得进入 X。不得把新的神经网络训练伪装成 predictability screen。

### R3：projection-parameter Jacobian/Fisher reachability

参数块预注册为旧 W 实际解冻的 `routed_feature_trunk + all routed_projections`；
router、upstream、common FFN 和 answer head 排除。冻结 predecessor 参数点，计算
共享 `Δθ` 的 Jacobian/Fisher 一阶输出，解一次 ridge/Fisher 最小范数问题，在
heldout 直接评估。不得保存参数更新或 checkpoint。必须报告 train/heldout
increment NMSE、error ratio、effective rank、condition/Fisher diagnostics 和
projection-only 对照。全 bank 精确解利用 final routed head 对参数的仿射性，并在
同 layer/route 的八个 step 与全部 slot 间共享同一更新；trunk+head 解使用固定
route-balanced train/heldout 子样本和 matrix-free JVP/VJP/CG。后者冻结 capture 的
attention state，因此只叫 local projection-path Jacobian；本 screen 不声称测量了
projection 改变后经未来 recurrent state 回流的 full recurrent Jacobian。

### R4：unexplained/randomness/方向鬼故事审计

v1 的 hard null 集合冻结为：within-route 与 unrestricted record-ID pairing
permutation、route-ID permutation、整体与 route-conditional record sign flip、保持
row norm/coordinate scale 的 matched-Gaussian spectrum、train–heldout alignment 以及
heldout split-half alignment。对 feature/target 的 pairing permutation 在 score/fit
问题中是同一置换零假设，分别重复命名不会增加独立证据。lambda grid 的 train-inner
validation 全量报告，但 v1 不以 heldout 选择 regularization。

negative-VJP 是旧 causal arm 的训练对照，不在冻结 bank 中重新生成；重新构造会违反
本 screen 的 target-regeneration 禁令。site-balanced null、第二独立 sketch、完整
block-covariance matched null、whitened/Fisher residual null 与 full-J permutation
登记为未测边界，不得暗称已完成。低 predictability 只能叫
`unexplained_under_this_screen`；六项 residual tests 全部未拒绝时也只能写
`consistent_with_current_registered_nulls_only`，不能写成随机性证明。

## 5. 推荐 Gate

固定合同值为：route cell 最小 128 records、record bootstrap 2000、轻量 score/
route/sign permutation 10000、完整 fit-null 128、matched-spectrum 与 alignment null
各 1024、route-conditional sign null 2048、
qualified heldout increment NMSE `≤0.08`、heldout/train error ratio `≤1.5`、null
p-value `≤0.01`。六个 residual tests 另以 Bonferroni `0.01/6` 判定，split-half
alignment 必须进入 random-compatible Gate。fit-null 次数与 score-null 不混称；
within-route 与 unrestricted record-ID permutation 必须分别报告；rank/lambda 只能由
train 内层 split 固定。这些是 reachability qualification floor，不是旧 root 的补救阈值。

R0/R1 可提供描述性结果；R2/R3 要成为后续 transfer 资格，必须同时优于相应
permutation null，且真实 effect 的 record-bootstrap CI 下界大于零。R4 的任一主
要 null control 胜出，或 split-half 方向不稳定，总状态 FAIL。任一输入 hash、
finite、coverage、null 或 heldout integrity failure 都 fail-stop。

## 6. 结论限制

通过不等于 shared+routed 优于 shared-only，不等于 projection 已获得任务能力，
不等于 nonlinear training 可达，不等于 H1/P1 qualification。结果始终
`authorizes=nothing`，输出不得包含 optimizer state、model checkpoint 或 successor
root。
