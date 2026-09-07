# Project-Yggdrasil V2 研究工作区

本仓库当前只保留 Project-Yggdrasil V2 的架构真源、路线决策、V2-A 分层实验、最近测试计划和远期 V2-C 系统实验合同。A1.19H/A1.20D 提供 mixed latent core 与 full-text mechanism 的强诊断证据，A1.21P 因任务同构、baseline 不公平和 formal 缺口停止。V2-R1R 随后完成 P0-D/P0-M、P1 v1–v8L、NR1、H1 与 H1-WD 的机制排除链；所有 routed projection 后继均为 `authorizes=nothing`。Closure C0R 已资格化 relation-balanced 数据与 strict compact trace，并只授权了 C1。旧 C1 在 G007 fail-stop；独立 C1R 随后以 fresh answer-only Stage A 排除 trace exposure/shared-gradient 作为充分原因，但唯一 formal 又在首个行为门 G004 失败。直接切换后的 C1S Addressed Content Workspace 唯一运行 S1 Overfit32 后 sealed FAIL：答案虽为 `32/32`，no-core 仍为 `28/32`、同记录至少两个因果贡献者仅 `7/32`。后续时序诊断确认 target、latent motion 与 readout 均存在，因而问题不是“没有时序表征”，而是完整 source hidden 让 H0 保留答案捷径且任务没有证明真实双对象因果阶数。C1T Causally-Partitioned Workspace 随后以独立 public cards、注册地址和 target-only transition 切断完整 source hidden 旁路；唯一 S0 sealed PASS 后，fresh S1 Overfit32 已按冻结 schedule 唯一运行到 `fixed_4000`，但以 overall answer `22/32`、factorial exact `1/8`、CPS/ERE two-contributor `8/16` 与 `4/16` 在 R103/R105 sealed FAIL。post-stop 归因已把主故障定位为多组共享训练中的 operation-2 dead-gate：完整 factorial hinge 在正损失下发生梯度抵消，未 detach counterpart 放大跨 cell 耦合，中后期 mode switch 后 sigmoid gate 饱和关闭；单组 ERE-g00 disposable 对照 250 steps 即 4/4，排除任务不可学或基础 XOR 表达能力不足。C1T 终态仍为 `authorizes=nothing`。fresh C1U Publicly-Grounded Gate-Free Workspace 已按 Gate 顺序完成唯一 S0 与 fresh S1：S0 P001–P006/U001–U010 全 PASS，public semantic bridge、gate-free transition 与 operation-2 梯度均由真实逐卡 Qwen cache 资格化；唯一 S1 preflight P101–P105 PASS 后，唯一正式 Overfit32 完成固定 4,000 updates 并在 R101–R107 全 PASS。终局为 answer `32/32`、factorial `8/8`、CPS/ERE no-core `4/16` 与 `2/16`、两族 two-contributor 均 `16/16`；result/seal/endpoint 为 `1FCD2D…6090C`/`A5211D…5350`/`88F538…159F`，44/44 replay。随后只做多 bank、不做多 seed 的 C1U S2 完成唯一 preflight 与 formal：24,000/24,000 updates、6/6 fixed endpoints 和 76/76 seal replay 完整，但 R203/R204/R205 sealed FAIL。K8/K1 heldout 仅 `52/192`/`50/192`，overall K8 gain `+1.04pp` 且区间跨零；post-stop Choices-only 与 opaque alpha-renaming 反事实分别证明 fixed A–I head 不具 label permutation equivariance、whole-card payload 不具 opaque-renaming invariance。S2 终态 `authorizes=nothing`，S3、multi-seed、C2、V2-A PASS、V2-B 与 V2-C 均未运行。

当前主线已经从旧 Stage A—AV-J-C 直接切换为：

1. V2-A：先在成熟文本基座上比较显式文本 CoT、单向量 latent recurrence 和多向量 latent recurrence。
2. V2-B：只有 V2-A 通过后，才验证固定输入/输出下的文本/视觉/动作表示、Boundary-MoE、FFN-MoE 与 Attention Pump。
3. V2-C：只有 V2-B 通过后，才把主动 Boundary、真实工具、工作树/记忆、人类目标连续性和隔离专家演化聚合为多层、多线程、树图混合全双工智能体实验。

旧实验不是当前实现路线。原有 Stage A—AV-J-C 阶段报告、代理脚本、测试、源码、artifact、旧 README 和思考稿保留作历史证据，不建立兼容入口；其原相对树 `archive/legacy-proxy-route-2026-07-11/` 已于 2026-08-24 与 V2-R1R/A1.20D 冷证据统一移出活动工作树，当前暂存包与预定 D 盘位置见[目录索引](docs/DIRECTORY_REFERENCE.md)。

## 当前入口

- [V2 架构白皮书](docs/Project-Yggdrasil%20%E5%A4%9A%E6%A8%A1%E6%80%81%E6%BD%9C%E5%8F%98%E9%87%8F%E6%8E%A8%E7%90%86%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6%20V2.md)：唯一目标架构规范。
- [V2 总路线图](docs/Project-Yggdrasil%20V2%20%E4%BB%8E%E6%9E%B6%E6%9E%84%E9%AA%8C%E8%AF%81%E5%88%B0%E5%95%86%E7%94%A8%E8%B7%AF%E7%BA%BF%E5%9B%BE.md)：从 R0 到架构完整版和商用版的唯一高层路线。
- [下一阶段测试计划](docs/next-stage-test-plan.md)：当前执行真源，定义 V2-A/V2-B/V2-C 的顺序、证据等级、Gate 和旧路线收口。
- [V2-A Closure C0 资格合同](docs/v2-a-closure-c0-task-baseline-qualification.md)：冻结 ERE/CPS 任务审计、四臂 public input、matched supervision、双成本切片、历史复用与 single-use C0。
- [V2-A Closure C0 结果复盘](docs/v2-a-closure-c0-result-review.md)：记录 C004/C006 两个 blocker、六个通过 Gate、旧 V2-A 复用矩阵、封存 hash 与下一轮 C0 修复边界。
- [V2-A Closure C0R 修复合同](docs/v2-a-closure-c0r-data-trace-qualification.md)：冻结新 relation-balanced data identity、strict compact trace、D001–D012、C001–C008 与两阶段 single-use 边界。
- [V2-A Closure C0R 结果复盘](docs/v2-a-closure-c0r-result-review.md)：记录两个正式 PASS、全 bank replay/fault-kill、封存 hash、架构证据边界与 C1 唯一授权。
- [V2-A Closure C1 结果复盘](docs/v2-a-closure-c1-result-review.md)：记录旧 joint answer/trace formal 的 G007 fail-stop、trace-credit 失败与未运行 Gate。
- [V2-A Closure C1 失败归因](docs/v2-a-closure-c1-failure-attribution.md)：记录旧 C1 的 exposure、shared-gradient、trace readout 与 post-stop architecture 只读拆分。
- [V2-A Closure C1R 合同](docs/v2-a-closure-c1r-staged-credit.md)：已消费的 staged-credit 合同；Stage A 先答题，只有全部前置行为/因果门通过才允许冻结 graph 后训练 trace probe。
- [V2-A Closure C1R 结果复盘](docs/v2-a-closure-c1r-result-review.md)：记录 G004 fail-stop、train/validation E/I collapse、标签闭环、H0–H10 slot consensus 与 answer-path effective K≈1 的边界。
- [V2-A Closure C1S 架构提案](docs/v2-a-closure-c1s-addressed-workspace-proposal.md)：已由冻结合同接替的设计来路；自身不授权训练或 formal。
- [V2-A Closure C1S 机制资格合同](docs/v2-a-closure-c1s-addressed-workspace-qualification.md)：冻结 addressed workspace、matched K=1、S0–S2 边界与 single-use Gate；S0 已 PASS。
- [V2-A Closure C1S S0 结果复盘](docs/v2-a-closure-c1s-s0-result-review.md)：记录零训练结构/量具/GPU 资格、封存 hash、证据边界与 S1 唯一授权。
- [V2-A Closure C1S SRW S1 结果复盘](docs/v2-a-closure-c1s-s1-result-review.md)：记录答案已拟合但 recurrence necessity、functional-K 与动态状态资格失败，以及 S2/formal 未授权的边界。
- [V2-A Closure C1S S1 失败归因结果复盘](docs/v2-a-closure-c1s-s1-failure-attribution-result-review.md)：记录旧诊断 D001–D003 的封存部分证据及 D004 inner-fold support 崩溃，明确旧身份禁止重跑。
- [V2-A Closure C1S S1 时序归因修复合同](docs/v2-a-closure-c1s-s1-temporal-attribution-repair.md) / [preflight 复盘](docs/v2-a-closure-c1s-s1-temporal-attribution-repair-preflight-review.md) / [结果复盘](docs/v2-a-closure-c1s-s1-temporal-attribution-repair-result-review.md)：D004R/D005R-only 身份已唯一完成；时序 target、latent motion、cross-fit readout 与 frozen decoder 均有正证据，两族 Axis C 因没有命中注册故障而保守为 `INCONCLUSIVE`。不授权训练或 S2/S3。
- [V2-A Closure C1T 因果分区工作区合同](docs/v2-a-closure-c1t-causally-partitioned-workspace.md) / [S0 执行合同](docs/v2-a-closure-c1t-s0-execution.md) / [S0 结果复盘](docs/v2-a-closure-c1t-s0-result-review.md) / [S1 执行合同](docs/v2-a-closure-c1t-s1-execution.md) / [S1 结果复盘](docs/v2-a-closure-c1t-s1-result-review.md) / [S1 失败诊断与归因](docs/v2-a-closure-c1t-s1-failure-attribution.md)：S0 sealed PASS 后唯一 S1 在 R103/R105 sealed FAIL；归因确认多组共享优化中的 operation-2 dead-gate、factorial hinge 梯度抵消与 counterpart 梯度耦合，单组对照排除基础表达能力不足。`authorizes=nothing`，禁止重跑或进入 S2/formal。
- [V2-A Closure C1U Publicly-Grounded Gate-Free Workspace](docs/v2-a-closure-c1u-publicly-grounded-gate-free-workspace.md) / [S0 执行合同](docs/v2-a-closure-c1u-s0-execution.md) / [S0 结果复核](docs/v2-a-closure-c1u-s0-result-review.md) / [S1 执行合同](docs/v2-a-closure-c1u-s1-execution.md) / [S1 结果复核](docs/v2-a-closure-c1u-s1-result-review.md)：唯一 S0 与唯一 fresh S1 均 sealed PASS；S1 R101–R107 全真，其 S2 设计授权已由用户的执行授权接续。
- [V2-A Closure C1U S2 多 Bank、单 Seed、Matched K1/K8 执行合同](docs/v2-a-closure-c1u-s2-multibank-matched-k1-k8-execution.md) / [结果复盘](docs/v2-a-closure-c1u-s2-result-review.md) / [失败归因与修复方向](docs/v2-a-closure-c1u-s2-failure-attribution.md)：唯一 formal 完成六端点后在 R203/R204/R205 sealed FAIL；post-stop 确认 choice-label 与 opaque-symbol 两类结构不变性都缺失，`authorizes=nothing`。
- [V2-C 全双工智能体实验合同](docs/v2-c-hierarchical-full-duplex-agent-experiment.md)：聚合多层、多线程、树图混合 Boundary-MoE、工具、目标连续性和专家演化；当前仅设计完成。
- [`thinking` 草稿综合分析](docs/thinking-draft-synthesis-2026-08-01.md)：保留原稿不动，区分五篇近期草稿与项目初期 V1 参照，并说明它们如何统一进入 V2-C。
- [V2-A 推理介质实验记录](docs/v2-a-reasoning-medium-experiment.md)：当前实现、命令、smoke 结果和失败边界。
- [V2-A1.5 潜空间基础实验记录](docs/v2-a1.5-latent-foundation.md)：独立数据合同、P0/P1/P2 结果、反事实干预和停止门禁。
- [V2-A1.6 连续核心失败记录](docs/v2-a1.6-core.md)：relation Gate 失败、只读 closure 诊断和无效旧指标修正。
- [V2-A1.7 连续核心实验记录](docs/v2-a1.7-core.md)：受控 relation holdout、三 seed formal/causal、2×2 消融、长程压力和分层成功概率。
- [V2-A1.8 长程递归实验记录](docs/v2-a1.8-long-horizon.md)：T1–16 随机深度训练、三组独立数据/model seed、T20/T24 Gate、T32 诊断、稳定性与成本。
- [V2-A1.9 冻结 Qwen hidden 边界记录](docs/v2-a1.9-qwen-boundary.md)：三组真实 Qwen role cache、adapter-only formal、hidden 反事实、成本与严格证据边界。
- [V2-A1.10 完整文本匿名工作区记录](docs/v2-a1.10-anonymous-workspace.md)：full-token cache、匿名 K-slot、通用 recurrent reasoner、方法修正、三 seed 正式失败与成本边界。
- [V2-A1.11 正交故障定位记录](docs/v2-a1.11-fault-localization.md)：Boundary strict-overfit 诊断、exact-symbolic Reasoner 三 seed formal、2×2 归因和证据边界。
- [V2-A1.12 Reasoner 根因拆分](docs/v2-a1.12-reasoner-root-cause.md)：binding × cursor 三臂合同、formal `0/3` 和排除结论。
- [V2-A1.13 Transition × Closure](docs/v2-a1.13-transition-closure-root-cause.md)：relation transition、soft closure、state/full 分离和因果结果。
- [V2-A1.13F Fixed-budget 审计](docs/v2-a1.13f-fixed-budget-method-audit.md)：固定 4000-step 排除 early-stop 主因。
- [V2-A1.15 Query-coupled core](docs/v2-a1.15-closed-coupled-core.md)：query-coupled + answer CE 的 fresh-seed 失败。
- [V2-A1.16 冗余答案损失](docs/v2-a1.16-redundant-answer-loss.md)：删除重复 answer CE 后仍为 state/full `0/3`。
- [V2-A1.17 配对目标审计](docs/v2-a1.17-paired-objective-initialization-audit.md)：同 seed、共享初始化位相等的 objective × initialization 根因定位。
- [V2-A1.18/A1.18B 训练脚手架](docs/v2-a1.18-training-scaffold.md)：FINAL-SAUX `2/3`、逐步全局 TSAUX 六 seed formal/causal、部署剥离和机制结论。
- [V2-A1.19H 可泛化混合 Core](docs/v2-a1.19h-hybrid-core.md)：opaque handle、可变实体数量、heldout N5、三 seed formal/causal 与参数级等价训练优化。
- [V2-A1.20B 完整文本 Boundary](docs/v2-a1.20b-full-text-boundary.md)：full-token Qwen cache、无 oracle Boundary、训练吞吐审计、run-1 eligibility 失败、oracle/梯度归因与停线结论。
- [V2-A1.20C 分层编译与信用修复](docs/v2-a1.20c-boundary-repair.md)：token anchor、entity-table-first compiler、hard-forward straight-through bridge、overfit32 失败、梯度冲突与停线结论。
- [V2-A1.20D full-text 机制修复](docs/v2-a1.20d-full-text-mechanism-repair.md)：双向 section 因果闭环、N5 修复、完整矩阵与 hidden causal 诊断。
- [V2-A1.21P matched Pareto](docs/v2-a1.21p-pareto.md)：K=1 负基线、在线 smoke、正式合同缺口与 A1.22A 停机判定。
- [V2-R1R 高层设计](docs/v2-r1-revalidation-task-design.md)：保留 ERE/CPS、共享 core、公平基线与 Pareto 高层合同；历史 P0-D 实现细节由逐版冻结合同和主审覆盖。
- [V2-R1R 阶段结果](docs/v2-r1r-p0-result.md)：记录 v1–v17 P0-D、P0-M v1–v5，以及 P1 v1–v8L 的 data/cache/训练/机制判决与当前停线边界。
- [V2-R1R v13 generator smoke 主审](docs/v2-r1r-r1g-v13-generator-smoke-main-review.md)：复算 62 文件 seal、正式 G01–G11、400 个新 seed 结构探针与三组非 formal shortcut 稳健性诊断，冻结有限接受边界。
- [V2-R1R v14 full production 冻结设计](docs/v2-r1r-p0d-v14-full-production-design.md)：冻结 4096/512、root provenance、扩展 ERE 状态域、simultaneous shortcut Gate 与唯一 formal。
- [V2-R1R v14 full production 主审](docs/v2-r1r-p0d-v14-full-production-main-review.md)：记录 14,336 records、G09 两格失败、其余 Gate/replay/seal 证据、统计 power 根因和后继边界。
- [V2-R1R v15 G09 qualification 冻结设计](docs/v2-r1r-p0d-v15-g09-qualified-production-design.md)：冻结 null/local/diffuse power、4,096/1,536、exact-equivalent batch scorer、Q-before-F 与 single-use 停线。
- [V2-R1R v15 G09 qualification 主审](docs/v2-r1r-p0d-v15-g09-qualified-production-main-review.md)：记录 Q01–Q07/Q09 通过、Q08 speedup 失败、sealed artifact 与 fresh formal 未运行边界。
- [V2-R1R v16 runtime-qualified production 冻结设计](docs/v2-r1r-p0d-v16-runtime-qualified-production-design.md)：拆分 correctness/runtime Gate，冻结 15 轮 paired、完整旧规模预算、18-fault 与新 seed production。
- [V2-R1R v16 执行合同](docs/v2-r1r-p0d-v16-runtime-qualified-production-execution-command.md)：Q→F single-use 历史命令；两个 fixed roots 均已执行并封存，禁止重跑。
- [V2-R1R v16 主审](docs/v2-r1r-p0d-v16-runtime-qualified-production-main-review.md)：接受 runtime-Q 窄组件，记录 fresh F 的 G05/G07 失败、307 条 ERE visible-domain 缺口与 alpha-renamer 路径误报。
- [V2-R1R v17 repaired production 冻结设计](docs/v2-r1r-p0d-v17-repaired-production-design.md)：冻结 visible-domain/alpha 两项机制修复资格与新 seed 完整 production conjunction。
- [V2-R1R v17 执行合同](docs/v2-r1r-p0d-v17-repaired-production-execution-command.md)：repair qualification → fresh production 的 single-use 历史命令；两个 roots 均已封存。
- [V2-R1R v17 主审](docs/v2-r1r-p0d-v17-repaired-production-main-review.md)：接受 R01–R07 与 G01–G11，正式关闭 P0-D 并只授权 P0-M。
- [V2-R1R P0-M v5 训练合同](docs/v2-r1r-p0m-v5-design.md)：冻结 cache、共享 K=8、可剥离 claim 目标、公平 LoRA baselines、throughput 与 M01–M08。
- [V2-R1R P0-M v5 执行合同](docs/v2-r1r-p0m-v5-execution-command.md)：固定顺序、roots、Gate 和 P1-before stop；已执行结束。
- [V2-R1R P0-M v5 主审](docs/v2-r1r-p0m-v5-main-review.md)：接受有限 training-path smoke，解释 v1–v4 失败—修复闭环，并限定 P1 必须验证的新问题。
- [V2-R1R P1 v1 冻结设计](docs/v2-r1r-p1-v1-single-seed-falsification-design.md)：冻结 fresh 8192/1024、K=8-first、因果干预、K=1 与 matched baselines 的 single-seed falsification 合同。
- [V2-R1R P1 v1 执行合同](docs/v2-r1r-p1-v1-execution-command.md)：固定 data→cache→K=8→对照顺序与任一 Gate 失败即停；唯一 data root 已消耗。
- [V2-R1R P1 v1 data 失败复核](docs/v2-r1r-p1-v1-data-failure-review.md)：记录 G09 唯一失败 cell、全量 replay、规模学习曲线、generator surface/label coupling 归因与 P1 v2 修复边界。
- [V2-R1R P1 v2 powered 设计](docs/v2-r1r-p1-v2-powered-single-seed-design.md)：以 4,096-heldout 解耦统计功效与 1,024-model subset 成本，冻结 Q-before-D、K=8-first 与 P2-before stop。
- [V2-R1R P1 v2 执行合同](docs/v2-r1r-p1-v2-execution-command.md)：唯一 preflight/power/data/cache 顺序与失败即停；已在 cache infrastructure FAIL 后结束。
- [V2-R1R P1 v2 cache 失败复核](docs/v2-r1r-p1-v2-cache-infrastructure-failure-review.md)：记录 4096 data PASS、完整 cache banks、sealed `EINVAL`、post-stop 内容 PASS 与 P1 v3 recovery 边界。
- [V2-R1R P1 v3 recovery 设计](docs/v2-r1r-p1-v3-cache-recovery-design.md)：冻结 telemetry decision power、immutable cache 双重审计、K=8-first 与失败即停。
- [V2-R1R P1 v3 执行合同](docs/v2-r1r-p1-v3-cache-recovery-execution-command.md)：唯一 preflight→telemetry→recovery→K=8 顺序；已在 K=8 FAIL 后结束。
- [V2-R1R P1 v3 K=8 失败复核](docs/v2-r1r-p1-v3-k8-failure-review.md)：接受 recovery，记录近随机 K=8、2.34× episode exposure、claim bootstrap 未启动、K08 batch-size 审计误报与 P1-LQ 后继边界。
- [V2-R1R P1 v4-LQ 设计](docs/v2-r1r-p1-v4-learning-qualification-design.md)：冻结四段嵌套规模课程、学习 Gate、continuation 链、K08 测量修复与 full K01–K09 停止规则。
- [V2-R1R P1 v4-LQ 执行合同](docs/v2-r1r-p1-v4-learning-qualification-execution-command.md)：唯一 `run-p1-lq` 顺序、六个 single-use roots 与结束回传格式。
- [V2-R1R P1 v4-LQ 失败复盘](docs/v2-r1r-p1-v4-lq-failure-review.md)：记录 S512 训练记忆、未见 episode 反事实诊断、owner-contrast 根因与 v5 约束。
- [V2-R1R P1 v5 语义迁移设计](docs/v2-r1r-p1-v5-semantic-transfer-design.md)：冻结 7,168/1,024 分层 train-audit、合法 paired claim 目标、batch 32、transfer Gate 与原 K01–K09。
- [V2-R1R P1 v5 执行合同](docs/v2-r1r-p1-v5-semantic-transfer-execution-command.md)：唯一 `run-p1-transfer` 顺序、四个 single-use roots 与结束唤醒格式。
- [V2-R1R P1 v5 失败复盘](docs/v2-r1r-p1-v5-semantic-transfer-failure-review.md)：记录 Q7168 的 answer/claim 脱钩、梯度与小规模探针、规模化信用对称根因。
- [V2-R1R P1 v6 CTW 机制设计](docs/v2-r1r-p1-v6-causal-temporal-witness-design.md)：冻结 static/temporal 等计算比较、T+1 状态定义、fresh formal seeds、隔离 audit 与因果 Gate。
- [V2-R1R P1 v6 执行合同](docs/v2-r1r-p1-v6-causal-temporal-witness-execution-command.md)：唯一 `run-p1-ctw` 顺序、五个 single-use roots 与结束唤醒格式。
- [V2-R1R P1 v6 assessment 失败复核](docs/v2-r1r-p1-v6-ctw-assessment-failure-review.md)：保留原 formal FAIL，定位 raw Python mapping 与 canonical JSON 等价关系不一致。
- [V2-R1R P1 v6R recovery 设计](docs/v2-r1r-p1-v6r-selection-recovery-design.md)：冻结五个旧 seal、canonical/hash replay、三类负控与 R601–R610。
- [V2-R1R P1 v6R 主审](docs/v2-r1r-p1-v6r-selection-recovery-main-review.md)：记录两个新 seal、全量 query 重审、机制资格恢复与 integrated P1 边界。
- [V2-R1R P1 v7 integrated K=8 设计](docs/v2-r1r-p1-v7-integrated-k8-design.md)：冻结 full-data fresh-seed 联训、T+1 状态语义、86016-query cache、K01–K10 与 fail-stop。
- [V2-R1R P1 v7 integrated K=8 执行合同](docs/v2-r1r-p1-v7-integrated-k8-execution-command.md)：固定 preflight/query-cache/K8 三根与唯一 `run-p1-integrated` 命令；已执行结束，禁止重跑。
- [V2-R1R P1 v7 失败复核](docs/v2-r1r-p1-v7-integrated-k8-failure-review.md)：分离真实 CPS causal/OOD 失败、pair-role evaluator 缺陷，以及不适合作必要 Gate 的 zero-middle/T1。
- [V2-R1R P1 v8 causal-bridge 设计](docs/v2-r1r-p1-v8-causal-bridge-design.md)：冻结 ordinary/causal-unpaired/causal-paired 三臂、等 exposure、sealed causal audit、最简机制优先级与 B01–B07。
- [V2-R1R P1 v8 执行合同](docs/v2-r1r-p1-v8-causal-bridge-execution-command.md)：三个 roots 已运行并封存；qualification 正式 FAIL，禁止重跑。
- [V2-R1R P1 v8 失败审阅](docs/v2-r1r-p1-v8-causal-bridge-failure-review.md)：区分 scratch competence collapse、训练内 pair 优化成功与 unseen pair 迁移失败。
- [V2-R1R P1 v8R causal curriculum 设计](docs/v2-r1r-p1-v8r-causal-curriculum-design.md)：冻结 shared v7 checkpoint、完整 ordinary rehearsal、同 batch CE/pair 两臂和 R01–R06。
- [V2-R1R P1 v8R 执行合同](docs/v2-r1r-p1-v8r-causal-curriculum-execution-command.md)：固定两根与已消耗的唯一 `run-p1-causal-curriculum` 命令；qualification sealed FAIL，禁止重跑。
- [V2-R1R P1 v8R 失败复核](docs/v2-r1r-p1-v8r-causal-curriculum-failure-review.md)：排除 scratch、mutation-family 漂移和纯 readout 容量，定位 final-decision state closure，并记录 BF16 gradient 假阴性。
- [V2-R1R P1 v8D causal-decision witness 设计](docs/v2-r1r-p1-v8d-causal-decision-witness-design.md)：以同-query base/flip 最终决策 witness 直接监督 `H_T`，严格区分 state formation 与 answer consumption。
- [V2-R1R P1 v8D 执行合同](docs/v2-r1r-p1-v8d-causal-decision-witness-execution-command.md)：已消耗的三根与唯一 `run-p1-decision-witness`；qualification sealed FAIL，禁止重跑。
- [V2-R1R P1 v8D 失败复核](docs/v2-r1r-p1-v8d-causal-decision-witness-failure-review.md)：排除 source/Boundary blindness 与纯 readout 缺口，定位 perturbation 已放大但 semantic reduction 未形成。
- [V2-R1R P1 v8L causal-state ladder 设计](docs/v2-r1r-p1-v8l-causal-state-ladder-design.md)：以无可学习 probe 的 lexical semantic contrasts，在 mutation-aligned `H_t` 建立因果状态梯子和最后 K=8 停止判据。
- [V2-R1R P1 v8L 执行合同](docs/v2-r1r-p1-v8l-causal-state-ladder-execution-command.md)：三根与唯一 `run-p1-causal-state-ladder` 已消耗，禁止重跑。
- [V2-R1R P1 v8L 失败复核](docs/v2-r1r-p1-v8l-causal-state-ladder-failure-review.md)：记录 ERE/CPS bootstrap 分裂、三个 seals、CPS lexical-anchor measurement 漏检与当前 K=8 主线停止边界。
- [V2-R1R P1-NR1 数值/关系测量设计](docs/v2-r1r-p1-nr1-numeric-relation-measurement-design.md)：独立 oracle、typed scalar/relation state、范围外 holdout、metamorphic 与 fault matrix；不是 V8 修补。
- [V2-R1R P1-NR1 执行合同](docs/v2-r1r-p1-nr1-execution-command.md)：已消耗的 seed、两根 single-use roots、唯一 `run-p1-nr1` 与 NR1-before-H1 停止规则；禁止重跑。
- [V2-R1R P1-NR1 正式主审](docs/v2-r1r-p1-nr1-main-review.md)：记录唯一 formal PASS、两根 seals、N01–N07、measurement/fault/topology 结果和只授权 H1 的边界。
- [V2-R1R P1-H1 开发对照设计](docs/v2-r1r-p1-h1-mixed-core-design.md)：定义公共 FFN、共享 `H=384` nonlinear features、generic/routed final projection、历史身份隔离、matched active compute 与含正常路径必要性的 H01–H08；当前 factorized screen 已失败停线。
- [V2-R1R P1-H1 唯一执行合同](docs/v2-r1r-p1-h1-execution-command.md)：正式入口、两根 fixed roots、before-mutation readiness 和 fail-stop；screen 未授权 calibration，阈值/hash 无法冻结，正式命令保持拒绝。
- V2-R1R P1-H1 factorized screen 复核（冷归档相对路径 `artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/SCREEN_REVIEW.md`）：记录 full-budget 正常完成、H05/H06 三门失败、机制归因与 calibration/formal absence。
- [V2-R1R P1-H1-WD 非正式设计](docs/v2-r1r-p1-h1-wd-overlap-residual-design.md)：冻结正向重合写入、公共残差化、W/D/J 预算、机制 Gate 与 matched-control Gate。
- [V2-R1R P1-H1-WD 已消耗命令](docs/v2-r1r-p1-h1-wd-execution-command.md)：唯一 `run-h1-wd` 已正常结束；固定 root 禁止删除后重跑。
- [V2-R1R P1-H1-WD 失败复盘](docs/v2-r1r-p1-h1-wd-failure-review.md)：接受输出空间重参数化，拒绝路径必要性、route 因果性与架构收益。
- [V2-R1R P1-H1-WD decision-causal 设计](docs/v2-r1r-p1-h1-wd-decision-causal-design.md)：以真实 common-off answer-margin drop 定义请求量、以 projection-output VJP/Fisher 定义方向，并冻结负 VJP 方向对照。
- [V2-R1R P1-H1-WD decision-causal 命令](docs/v2-r1r-p1-h1-wd-decision-causal-execution-command.md)：冻结隔离 root、无 family/teacher/route supervision 的 W/D/J 数据流、增量 fit Gate 与唯一命令。
- [V2-R1R P1-H1-WD decision-causal 失败复盘](docs/v2-r1r-p1-h1-wd-decision-causal-failure-review.md)：记录 target Gate PASS、W 增量泛化失败、完整输出 nMSE 的零学习假象，以及 parameter-reachable transfer 的后继研究条件。
- [V2-R1R P1-H1-WD direction-geometry 设计](docs/v2-r1r-h1-wd-direction-geometry-screen-design.md)：以 R0–R4 分解检查全局、route、target-before 可预测、共享参数可达和 registered-null residual，而非再次训练旧 WD。
- [V2-R1R P1-H1-WD direction-geometry 已消耗命令](docs/v2-r1r-h1-wd-direction-geometry-screen-execution-command.md)：记录唯一 root、全量 replay Gate、null family 与 fail-stop 边界；同 identity 禁止重跑。
- [V2-R1R P1-H1-WD direction-geometry 失败复盘](docs/v2-r1r-h1-wd-direction-geometry-screen-failure-review.md)：唯一运行因 microbatch `4` 与 replay batch `128` 的 CUDA 数值漂移在 identity Gate crash-stop；R2–R4 未运行，没有方向结论。
- [V2-R1R P1-H1-WD direction-geometry v2 设计](docs/v2-r1r-h1-wd-direction-geometry-v2-design.md)：新 successor 以 pre-root 全量 batch-4 replay 修复执行身份，并补齐 site-macro、feature permutation、双 sketch、regularization 与 negative-sign 控制。
- [V2-R1R P1-H1-WD direction-geometry v2 命令](docs/v2-r1r-h1-wd-direction-geometry-v2-execution-command.md)：唯一 `run-h1-wd-direction-geometry-v2`、科学结果/执行 crash 双层终态和最多三个基础设施 successor identity 的边界。
- [V2-R1R P1-H1-WD direction-geometry v2 结果复盘](docs/v2-r1r-h1-wd-direction-geometry-v2-result-review.md)：完整 R1–R4、target prevalence shift、R2/R3 不资格化、structured residual 与 null 分辨率边界。
- [V2-R1R P0-D v1 主设计层复核](docs/v2-r1r-p0-main-review.md)：记录 ERE/CPS 满分捷径、claim/split/token/certificate 缺口及 v2 修订依据。
- [V2-R1R P0-D v2 主设计层验收](docs/v2-r1r-p0-v2-main-review.md)：接受失败证据和停机纪律，拒绝当前 v2 合同完成度，并记录机器 Gate 未捕获的任务与审计缺口。
- [V2-R1R P0-D v3 主设计层验收](docs/v2-r1r-p0-v3-main-review.md)：接受单次 D2 失败与停机纪律，拒绝 v3 合同实现；区分 D0/审计器错误、CPS 真实生成缺陷、ERE 有效部分和下一次直接切换条件。
- [V2-R1R P0-D v4 主设计层验收](docs/v2-r1r-p0-v4-main-review.md)：接受 `FAIL_R0` 停机纪律，拒绝自算 expected values、浅层预测试及未实现的 G07/G08 正向结果。
- [V2-R1R P0-D v5 R0A 主设计层验收](docs/v2-r1r-p0-v5-r0a-main-review.md)：接受冻结样例与 artifact 完整性，因六项 fail-closed API 探针失败和一次错误-hash 预调用而拒绝阶段通过；R0B 未授权。
- [V2-R1R P0-D v6 R0A-strict 冻结设计](docs/v2-r1r-p0d-v6-r0a-strict-design.md)：冻结完整 AST schema、245 项非法输入矩阵、精确错误合同、原子 formal attempt 与 R0A-strict Gate。
- [V2-R1R P0-D v6 R0A-strict 执行命令](docs/v2-r1r-p0d-v6-r0a-strict-execution-command.md)：已执行并结束；只作唯一 formal attempt 的历史追溯，禁止修补或重跑。
- [V2-R1R P0-D v6 R0A-strict 主设计层验收](docs/v2-r1r-p0-v6-r0a-strict-main-review.md)：接受 14/4/12/245 与 sealed protocol 的窄证据，因 query placeholder 十项新鲜反例拒绝阶段通过；R0B 未授权。
- [V2-R1R P0-D v7 R0A-lattice 冻结设计](docs/v2-r1r-p0d-v7-r0a-lattice-design.md)：以 slot × scope × token 的 858-cell 覆盖格替代粗标签，并冻结 coverage auditor、正负控制与单次 formal。
- [V2-R1R P0-D v7 R0A-lattice 执行命令](docs/v2-r1r-p0d-v7-r0a-lattice-execution-command.md)：已执行结束；只作唯一 formal 追溯，禁止覆盖或重跑。
- [V2-R1R P0-D v7 R0A-lattice 主设计层验收](docs/v2-r1r-p0-v7-r0a-lattice-main-review.md)：复算七文件 artifact、858-cell ledger 与 27 项新鲜黑盒探针，最终接受 R0A-lattice。
- [V2-R1R P0-D v8 R0B-structural 冻结设计](docs/v2-r1r-p0d-v8-r0b-structural-design.md)：用小型 qualification bundle、41 faults、4 metamorphic 和 37 raw metrics 资格化 G02–G06，不生成生产数据。
- [V2-R1R P0-D v8 R0B-structural 执行命令](docs/v2-r1r-p0d-v8-r0b-structural-execution-command.md)：已执行结束；accepted simulator 只读，fixed-root formal 禁止覆盖或重跑。
- [V2-R1R P0-D v8 R0B-structural 主设计层验收](docs/v2-r1r-p0-v8-r0b-structural-main-review.md)：接受机器矩阵与封存事实，因五项 registry 外 false negative 拒绝 R0B 资格。
- [V2-R1R P0-D v9 R0B-invariant 冻结设计](docs/v2-r1r-p0d-v9-r0b-invariant-design.md)：以固定 profile、exact schema、可逆 grammar、typed ERE provenance 与 derived CPS composition 返工 G02–G06。
- [V2-R1R P0-D v9 R0B-invariant 执行合同](docs/v2-r1r-p0d-v9-r0b-invariant-execution-command.md)：唯一 formal 已结束；fixed root 禁止覆盖或重跑。
- [V2-R1R P0-D v9 R0B-invariant 主设计层验收](docs/v2-r1r-p0-v9-r0b-invariant-main-review.md)：复算 45 文件 seal，并以六项 registry 外反例接受有限 R0B measurement system。
- [V2-R1R P0-D v10 R0C-statistical 冻结设计](docs/v2-r1r-p0d-v10-r0c-statistical-design.md)：资格化 source-only parser、条件多数、NB、grouped fold、CV/heldout、定向反例与固定语义摘要。
- [V2-R1R P0-D v10 R0C-statistical 执行合同](docs/v2-r1r-p0d-v10-r0c-statistical-execution-command.md)：唯一 formal 已执行结束；fixed root 禁止覆盖或重跑。
- [V2-R1R P0-D v10 R0C-statistical 主设计层验收](docs/v2-r1r-p0-v10-r0c-statistical-main-review.md)：复算 52 文件根 seal、bundle 双层 seal 与六组 registry 外探针，最终接受有限 R0C measurement component。
- [V2-R1R P0-D v11 R0D-integrated 冻结设计](docs/v2-r1r-p0d-v11-r0d-integrated-design.md)：组合 accepted simulator/invariant/learner，冻结 12-case、19 metrics、20 faults、4 metamorphic、import 与 snapshot replay。
- [V2-R1R P0-D v11 R0D-integrated 执行合同](docs/v2-r1r-p0d-v11-r0d-integrated-execution-command.md)：唯一 formal 已执行结束；fixed root 禁止覆盖或重跑。
- [V2-R1R P0-D v11 R0D-integrated 主设计层验收](docs/v2-r1r-p0-v11-r0d-integrated-main-review.md)：复算 144 文件直接/传递 seal、全部 ledger 与三项 registry 外探针，最终接受有限 integrated measurement system。
- [V2-R1R P0-D v4 R0 历史执行命令](docs/v2-r1r-p0d-v4-r0-execution-command.md)：已执行并失败，只作 artifact 追溯，不可重跑。
- [V2-R1R P0-D v5 R0A 历史执行命令](docs/v2-r1r-p0d-v5-r0a-execution-command.md)：已执行并结束，只作 artifact 追溯；禁止修补、覆盖或重跑。
- [V2-A 实验脚本](experiments/v2_a_reasoning_medium.py)：数据生成、文本基线和 latent reasoner 入口。
- [V2-R1R 当前 CLI](experiments/v2_r1_revalidation.py)：物理入口仍只有 `run-p1-h1`；factorized screen FAIL 后 calibration/threshold/hash freeze 未获授权，launcher 只允许 before-mutation 拒绝，旧 V8L、NR1 与 P2 入口不存在。
- [V2-A1.5 实验脚本](experiments/v2_a1_5_latent_foundation.py)：A1.5 数据、P0、P1 cache/train/eval、P2 train/intervention 和 matched text baseline 入口。
- [V2-A1.7 实验脚本](experiments/v2_a1_7_core.py)：A1.6 closure diagnostic，以及 A1.7 data/audit/train/evaluate/intervene/stress/assessment 入口。
- [V2-A1.8 实验脚本](experiments/v2_a1_8_core.py)：A1.8 data/audit/random-depth train/evaluate/intervene/cost/assessment 入口。
- [V2-A1.9 实验脚本](experiments/v2_a1_9_boundary.py)：A1.9 cache/audit/train/evaluate/intervene/cost/assessment 入口。
- [V2-A1.10 实验脚本](experiments/v2_a1_10_anonymous_workspace.py)：full-token cache/audit、匿名 reasoner train/evaluate、formal 后干预、cost/assessment 入口。
- [V2-A1.11 实验脚本](experiments/v2_a1_11_localization.py)：Boundary/Reasoner 两臂 train/evaluate、Gate 后干预和 2×2 assessment 入口。
- [V2-A1.12 实验脚本](experiments/v2_a1_12_root_cause.py)：binding/cursor train/evaluate/intervene/assess 入口。
- [V2-A1.13 实验脚本](experiments/v2_a1_13_transition_closure.py)：transition/closure train/evaluate/intervene/assess 与 fixed-budget audit 入口。
- [V2-A1.15/A1.16/A1.17 实验入口](experiments/v2_a1_17_paired_objective_initialization.py)：query-coupled 两种 objective 的训练入口分别见相邻 A1.15/A1.16 脚本，本入口负责 paired 总判定。
- [V2-A1.18 训练入口](experiments/v2_a1_18_training_scaffold.py)：QAUX、FINAL-SAUX、TSAUX 的 train/evaluate/intervene 与辅助剥离部署入口。
- [V2-A1.18B 总判定](experiments/v2_a1_18b_trajectory_state_scaffold.py)：FINAL-SAUX、TSAUX paired/fresh 和 causal 的机器聚合入口。
- [V2-A1.19H 实验入口](experiments/v2_a1_19h_hybrid_core.py)：H1/H2 data/audit/train/benchmark/evaluate/intervene/assess 统一入口。
- [V2-A1.20B 实验入口](experiments/v2_a1_20b_full_text_boundary.py)：cache/audit/train/evaluate、失败 oracle diagnostic、gradient-credit audit 与 assessment 入口。
- [V2-A1.20C 实验入口](experiments/v2_a1_20c_boundary_repair.py)：compiler supervision、flat/hierarchical × hard/ST 训练、严格 Gate 与实际 checkpoint 梯度冲突诊断入口。
- [V2-A1.21P 实验入口](experiments/v2_a1_21p_pareto.py)：跨域 probe、K=1、在线 matched preflight 与机器 assessment 入口。
- [架构审阅记录](docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md)：V2 决策形成依据，不与白皮书并行定义规范。
- [目录索引](docs/DIRECTORY_REFERENCE.md)：当前保留项和归档边界。

## 当前状态与未完成项

P1 v7、v8、v8R、v8D、v8L 正式 roots 均已封存，五个失败不改判。v8D 证明 source perturbation 可进入并显著改变 state，但可学习 final-state probe 没有产生 CPS semantic reduction；v8L 用无可学习 fixed anchors 获得 ERE bootstrap transfer，却在 CPS optimization/audit `0.5682/0.5281` 处停止。formal 后分层复核又确认 global geometry Gate 漏掉 CPS 数值 measurement 资格，因此当前结论只关闭匿名 K=8 + lexical-anchor 组合。v8L 活动 CLI、活动测试和 live `r1_revalidation/p1/*.py` 已由 NR1 直接替换；历史实现只由 sealed `source_snapshot/` 与文档追溯，三个 fixed roots 禁止重跑。

P1-NR1 已以 `PASS_P1_NR1_MEASUREMENT_QUALIFICATION` 封存，两根 seal 为 `1825282B…95DAF` 与 `DADBDDBF…F6FB3`；fixed roots 与 transport 均已消耗，禁止重跑。H1 等分 shared+routed residual 的独立 calibration 先以 `authorizes=nothing` 关闭；机制复核据此把活动代码直接切换为共享 feature trunk + routed `H=384 -> D=256` projection，并预登记正常路径必要性 Gate。该新方向的 `2026081761/2026081762` full-budget screen 现已正常跑完，但 mixed 没有形成净收益或路由因果性：overall gain `-0.01074`，wrong-route 与 conditional-write 最大效应仅 `0.00281/0.00781`。因此 H05/H06 FAIL，`2026081763/2026081764` calibration、H1 formal roots 与 transport 均保持不存在。不得重跑旧或当前 identity、降低 Gate、冻结无效 threshold，或启动 formal。这条 NR1→H1→F1 路线现已整体降级为历史机制证据；当前授权链由 Closure C0R→C1→C2→C3 取代。C0R 只关闭任务/trace/公平资格，不构成跨 seed、matched Pareto 或架构通过结论；不得直接进入 P2、A1.22A 或 V2-B。

2026-08-21 另立并单次执行的 `H1-WD` overlap-write/common-residual 非正式 screen 已以 `FAIL_H1_WD_NONFORMAL_MECHANISM` 结束。W/D 成功把约 `25.07%` 的 common 输出能量对应分量写入 projection 并保持函数：W heldout nMSE `0.00196`，D common nMSE `0.01393`，D free-rollout prediction agreement `0.99121`。但最终 common-off 与 projection-off 都只下降 `0.01172`，projection effect 相对 predecessor 只增加 `0.00391`；WD 相对同预算 control 仅 `+0.00488`，95% CI `[-0.00195,0.01172]`。因此“先写后删”只实现了输出空间重参数化，没有形成两条路径各自必要的能力分工或架构收益。训练全程未读取 family route target，原 checkpoint hash 不变；结果 `authorizes=nothing`，不得重跑该 identity 或启动 H1 formal/F1/P2。完整复盘见 `docs/v2-r1r-p1-h1-wd-failure-review.md`。

同日另立的 decision-causal screen 不复用上述失败 checkpoint，而从原 immutable mixed deployment 重新构造 train/heldout 因果 target：请求量取真实 common-off answer-margin drop 的 `0.5`，方向取固定 route 下 selected projection output 的 scalar VJP/Fisher，并以逐 site `0.25×||common||` 封顶。target 以 microbatch `4` 物化为 CPU 数据集；W/D Gate 直接检查转移增量，J 只用 ordinary answer CE 与 two-path causal-target allocation lock，不使用 teacher logits、route CE 或 family target。负 VJP 臂只作方向性对照，不是 shared-only 架构基线；本屏无论结果仍 `authorizes=nothing`。

该 decision-causal 唯一运行现已在 W Gate 以 `FAIL_WD_DECISION_CAUSAL_WRITE_FIT` 正常停止。target 本身全部 finite，train/heldout realized-request 为 `0.856/0.931`，CUDA peak `0.323 GB`；但 causal/control heldout transfer-nMSE 为 `1.00450/0.99247`，几乎等于 projection 完全保持 `P0` 的零学习基线。两臂 800 updates 均有非零梯度且覆盖每条 train record `6–7` 次，因此当前证据指向 raw per-record output VJP 没有先被约束到共享 projection 参数可达空间，而不是 J 后期奖励不足。合同未进入 D/J、未保存 checkpoint；root 已消耗，仍不授权 H1 formal、F1 或 P2。

2026-08-23 单次启动的 direction-geometry 诊断原拟以 R0–R4 进一步区分稳定条件方向、输入可预测性、共享 projection 参数可达性与 residual null。它在重放全部 train 4096/heldout 1024 records 后先被 replay identity Gate 拒绝：common/projection 最大绝对误差 `7.96914e-05/9.50396e-05`，超过冻结的 `2.5e-05`。根因是旧 bank 的 microbatch `4` 与新 capture batch `128` 使用不同 CUDA batch geometry，而启动前 spot check 未覆盖尾部最坏记录。机器终态为 `CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY`，没有 `result.json`，R2–R4 未执行；这既不支持也不反驳 parameter-Jacobian/Fisher 猜想。root 已消耗，禁止重跑该 identity，仍为 `authorizes=nothing`。

用户随后授权另立 v2 successor，目标是取得完整可信的 R1–R4 测量而非强迫 signal。唯一运行现已以 `COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2` 完成，root 前 batch-4 全量 replay 的 common/projection 误差均为 `0.0`，科学终态为 `NO_QUALIFIED_R2_R3_COMPONENT`。正确 route 显著优于 route permutation（`p=9.999e-05`），但 heldout route gain 仍为 `-0.586%`；primary projection trunk 只有不稳定的 `+3.189%`，exact shared head 为 `-0.334%`，sampled local-J 为 `-5.922%`。双 sketch residual 均拒绝 current-null compatibility。只读分层进一步发现 nonzero target prevalence 从 train `97.90%` 降到 heldout `57.71%`，静态写入在 433 条 heldout 零 target 上系统性误写；这是当前最直接的失败原因。结果始终 `authorizes=nothing`。

同日返回整体 V2-A 后，Closure C0 已以固定 root 单次封存为 `FAIL_V2_A_CLOSURE_C0_READINESS`。P0-D/P0-M seals、26,624 records、ERE/CPS 双代数、fingerprint、历史七项证据和 H1 排除均通过；但 ERE ordinary relation-query 为全 TRUE，严格只读 `source_text` 的 visible-legend oracle 在 train/validation/language OOD 条件准确率均为 `1.0`，约污染 ordinary split 4.17 个百分点。另一个失败是 compact trace 只有 formatter/final-answer parser，没有可冻结的 trace semantic verifier。result/evidence-seal SHA-256 分别为 `3FC871…43B1`/`462317…9972`，`authorizes=nothing`。历史 A1.8/A1.18B/A1.19H 只复用训练/core 方法，A1.20B/C 复用负向诊断，A1.21P 复用 K=1/成本审计方法；旧权重与结果均不进入新 C1。

2026-08-24，C0R 以两个独立 single-use 正式阶段关闭上述缺口。新 generator identity 在四个 relation cell 上均为 TRUE/FALSE 严格配平，source-only oracle accuracy `0.5`；26,624 条 compact trace 全部 roundtrip 与 semantic replay，`394,278/394,278` 定向故障被拒绝，token max `389 < 512`。data/trace D001–D012 与 readiness C001–C008 全 PASS；data result/seal 为 `B2502F…FFF9`/`5AFB37…B570`，readiness result/seal 为 `391C84…8D5E`/`161FBB…0BA4`。两阶段训练、optimizer step、模型写入均为零，`four_arm_results_present=false`、`v2a_passed=false`；当前只授权 fresh C1 单 seed 实现与资格测试。

旧 C1 与 C1R 随后分别在 G007 与 G004 fail-stop，旧 dense anonymous workspace 不再修补。2026-08-28，C1S 以全新的 Addressed Content Workspace 状态代数完成唯一一次 S0：K=8/K=1 trainable parameters 均为 `34,125,197`，成对置换 logits/trajectory delta、未选槽更新、inactive 更新和 auxiliary-strip logits delta 均为 `0`；注册 positive/null controls 与 RTX 4070 BF16 forward/backward 全部通过。其后 SRW S1 固定 4,000 updates 完整训练并把 ERE/CPS 各 16 条答案全部拟合，但 no-core 后仍答对 `28/32`，functional multi-address 只有 `7/32` 记录达到至少两个因果贡献者，多个动态 feature 也低于 S1 的 `0.95` 门槛。S1 result/seal 为 `9D0B06…8892`/`FAFF30…A9D2`，63/63 replay，机器终态 `FAIL_V2_A_C1S_S1_QUALIFICATION`；S2 与 formal 未运行，当前 `authorizes=nothing`。

2026-08-31，S1 失败归因的唯一旧诊断在 D004 fail-closed：`CPS.running_best` 仅 12/16 条记录同时含正负 target，普通哈希 inner fold 产生 0 条 valid record 的评分 cell。旧 root 已 sealed CRASH，不能修补或重跑；D001–D003 只保留为部分证据。新的 temporal-attribution-repair 身份随后唯一完成 D004R/D005R：两族 active motion、固定通道 AUC、nested cross-fit readout 与 frozen decoder 均满足归因阈值，所有 score-null 为 `p=1/10001`，所以 Axis C 没有命中 target/latent/readout 故障类别而保守返回 `INCONCLUSIVE`。正式 result/seal 为 `D1603C…6A496`/`E8A61C…562B0`，103/103 replay，模型状态未变；这排除了“时序状态不存在”作为主因，但 no-core/H0 shortcut 与 multi-address causal necessity 仍未解决，整体 `authorizes=nothing`。

## 文件治理原则

归档代表降级为历史证据，不代表删除。后续实现 V2 时应直接创建新的 V2-A 代码和测试，删除或重写旧契约，不在旧代理代码上添加兼容 wrapper。
