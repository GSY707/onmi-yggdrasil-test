# V2-R1R H1-WD R1–R4 direction-geometry v2 设计合同

日期：2026-08-23  
身份：`H1-WD-DG-V2-20260823-1`  
证据级别：`NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_ONLY`  
机器授权：`authorizes=nothing`

## 1. 核心目标

v2 的完成条件是取得一套身份有效、完整封存的 R1–R4 测量，而不是强迫某一层显著。
无信号、R3 不可达、R4 与当前 null 相容或仍有 residual structure，都是有效科学结果；
只有输入、重放、finite、coverage、参数只读或 artifact integrity 失败才是执行 crash。

v2 是 v1 crash 的新 successor identity，不是旧 root 重跑。v1 root
`artifacts/v2-r1r/h1-wd-direction-geometry-screen-20260823-1/` 及其 sibling logs
以逐文件 SHA-256 固定为 immutable predecessor。旧 decision-causal target bank、source
checkpoint 和 token cache 仍只读，旧 W `1.00450/0.99247` 不追溯改判。

## 2. Replay identity 修复

旧 target bank 由 microbatch `4` 物化；v1 使用 batch `128`，全 split 尾部产生
`7.96914e-05/9.50396e-05` common/projection 漂移并超过 `2.5e-05` Gate。v2 不放宽
容差，而直接冻结两种 batch geometry：

- trajectory replay microbatch 固定为 `4`；
- 已捕获 CPU feature 与 target 的后续纯分析 batch 固定为 `128`。

正式 output root 创建前，程序必须加载冻结模型并以 microbatch `4` 完整重放 train
4096 与 heldout 1024。它报告 split/global max、RMS、relative RMS、batch-max
percentiles、finite 与 coverage；任一 max 超过 `2.5e-05` 时 root 不创建。正式流程随后
在同一冻结模型、同一进程、同一 geometry 上重复 capture，并再次保留 replay Gate。
首批 spot check 不再构成资格证据。

为使“root 前失败”和单次 identity 语义同时成立，唯一命令会在任何硬 Gate 前以 exclusive
create 写入 sibling append-only lease：
`artifacts/v2-r1r/h1-wd-direction-geometry-screen-v2-20260823-1.preflight-lease.jsonl`。
lease 一旦存在，该 identity 即已消费；root 前失败只向 lease 追加 crash 事件，仍不创建
output root，也不得用同一 identity 再试。

## 3. R1–R4 对象

R0 仍是 train global centroid baseline，但本轮交付从 R1 开始显式编号。所有主对象保留
每条 record 的 16 site × 8 slot × 256 output 坐标；site 不能作为独立样本。

### R1：route-conditioned direction

只用 train 按 `(site, predecessor route)` 拟合 centroid，heldout 应用一次。报告相对 R0
的 record-micro、record-macro、site-macro 能量和 route-ID permutation。route 是冻结
条件键，不是训练监督或因果证明。

### R2：target-before predictability

主特征为冻结 routed feature trunk 输出；attention state 与 common-private hidden 是
target-before 对照，bank common/projection 仅为 output proxy。margin、VJP、target norm、
positive mask、answer/family/task target 及其派生量不得进入 X。lambda 只由 train-inner
split 选择；heldout 只评分。

### R3：shared-parameter reachability

full-bank exact 解只覆盖 shared final routed heads，并在 layer/route 的全部 step、slot、
record 间共享同一参数方向；sampled matrix-free 解覆盖 routed feature trunk + 两个 heads，
冻结 capture attention state，因此只称 local projection-path Jacobian/Fisher，不称 full
recurrent Jacobian。两者都报告 train/heldout transfer-increment nMSE、error ratio、
Fisher/Gram、CG、finite-difference 与有效秩信息，不写入参数或 checkpoint。

### R4：方向鬼故事与 residual

v2 注册以下完整控制：

- within-route 与 unrestricted record-ID score null；
- target permutation 与显式 feature permutation fit-null，R2/R3 分开重 fit；
- exact record bootstrap 与 record-cluster sketch bootstrap；
- record-resampled、site 等权的 site-macro bootstrap；
- 四个固定 lambda 的 heldout regularization stability；
- 两套独立固定 random sketch，全部 score/sign/spectrum/alignment 控制重复；
- global 与 route-conditional record sign flip；
- matched norm/diagonal-covariance spectrum、train–heldout alignment、heldout split-half；
- 显式 `target -> -target` negative-VJP sign-reversal audit。

negative target 在线性方向几何中应得到系数和预测的精确符号反转；它用于揭示几何的
中心对称性，不能冒充 answer-direction 因果对照。两套 sketch 共 12 个 residual tests，
统一使用 Bonferroni `0.01/12`。只有两套都不拒绝时才允许写
`consistent_with_both_registered_sketch_null_families_only`，仍不称随机性证明。

site-macro 的分母合同不要求每个 `record×site` 严格非零。每个阶段先以 heldout 全体记录
的 baseline energy 注册 `>1e-30` 的 active sites；aggregate 为零的 site 明示为 inactive。
bootstrap 若某次 record 重采样丢失任一已注册 active site 的分母支持，该次标为 undefined，
并使 signal Gate 不通过，但流程继续产出 R1–R4，不能把合法零能量误报成 infrastructure crash。

## 4. 硬 Gate 与科学标签

硬 Gate 包括：固定输入和 v1 archive hashes、root absence、CUDA、全量 batch-4 replay、
finite/coverage、model digest、source/target post-run hashes 与无 optimizer/model writes。
lease 前若发现 root/lease 已存在，只是拒绝重复入口；exclusive lease 创建后，该 identity 已
消费。root 前硬 Gate 失败向 lease 写
`CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_PRE_ROOT`；root 后失败写
`CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`。两者都不允许同 identity 重跑。

硬 Gate 通过且 R1–R4 全部写出时，顶层状态固定为
`COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`；另以 `scientific_status` 原样区分
R2/R3 qualification 与 R4 residual。R2/R3 signal 要同时满足 exact record CI、
site-macro CI、两套 sketch 的 score/target-fit/feature-fit null、negative-sign symmetry
和至少 `3/4` lambda heldout gain 为正。R3 qualification 另要求 exact-head 与 sampled
local-J heldout nMSE `<=0.08`、heldout/train ratio `<=1.5`。

## 5. 重试与授权边界

每个 frozen identity 只允许一次真实 attempt；attempt 以 exclusive lease 成功创建为起点，
而不是以 output root 创建为起点。lease 与 root 均不得删除、覆盖或续跑。用户
本轮“直到拿到 R1–R4”的授权只覆盖最多三个因基础设施或实现 crash 而另立的新 identity；
每个 successor 必须保留前序 root、使用新代码/root/manifest 并重新通过全量 preflight。
科学上的 no-signal/FAIL 已经是完成结果，不授权换 seed、调阈值或继续挑结果。

无论结果如何，本合同不训练模型，不恢复旧 H1/WD，不授权 calibration、H1 formal、F1、
P1、P2 或真实文本路由。
