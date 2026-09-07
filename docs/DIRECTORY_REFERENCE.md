# Directory Reference

本仓库是 Project-Yggdrasil V2 的独立研究工作区。当前主线已经从旧 Stage A—AV-J-C 直接切换为 V2-A 推理介质、V2-B 静态多模态模型核、V2-C 多层多线程树图全双工智能体；严格按 A→B→C Gate 推进。关联项目 `C:\skzy\QuickFileTransport\世界树计划` 的设计哲学是上位概念真源，本仓库不修改其源码。

## 当前真源

| 路径 | 地位与用途 |
| --- | --- |
| `docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md` | 当前唯一目标架构规范；定义连续语义 recurrence、离散地址/控制 + 连续 payload 的合规混合 core、静态 Boundary-MoE/FFN-MoE，以及 V2-C 系统级全双工 Boundary-MoE 与人类目标能力边界。 |
| `docs/Project-Yggdrasil V2 从架构验证到商用路线图.md` | 当前唯一高层路线；按 Gate 推进 R0/R1、R2/R3 静态模型核、V2-C 架构完整版和商用路线；旧 A1-A4 后续路线已被直接替换。 |
| `docs/next-stage-test-plan.md` | 最近阶段状态汇总；记录 V2-A/A1.5–A1.21P、V2-R1R 闭环、Closure C1/C1R/C1S 终局、C1T sealed FAIL/归因、C1U S0/S1 sealed PASS，以及 C1U S2 多-bank单-seed sealed FAIL 与 post-stop 结构归因。未来 V2-A Closure C2 仍仅是规划；当前 C1U 后继也只有 symbol-bound choice workspace 修复建议，没有合同、代码或 root/lease。 |
| `docs/v2-a-closure-c0-task-baseline-qualification.md` | V2-A Closure C0 冻结合同；定义任务结构/捷径、四臂 public input、matched compact-trace、双成本切片、历史复用和单次封存。 |
| `docs/v2-a-closure-c0-result-review.md` | C0 人类可读终局；C004 relation-query 捷径与 C006 trace-verifier 缺失失败、六门通过、旧 V2-A 复用矩阵及下一修复边界。 |
| `docs/v2-a-closure-c0r-data-trace-qualification.md` | C0R 直接 successor 冻结合同；定义 relation-balanced 新 data identity、strict compact trace、D001–D012、C001–C008 与两阶段 single-use。 |
| `docs/v2-a-closure-c0r-result-review.md` | C0R 人类可读终局；记录两个正式 PASS、全 bank replay/fault-kill、封存 hash、证据边界与 C1 唯一授权。 |
| `docs/v2-a-closure-c1-single-seed-eligibility.md` / `docs/v2-a-closure-c1-result-review.md` | C1 历史冻结合同与当前结果真源；唯一 single-seed formal 在 G007 trace credit FAIL-stop，后续 behavior/OOD/causal/intervention 未运行，`authorizes=nothing`。 |
| `docs/v2-a-closure-c1-failure-attribution.md` | C1 post-stop 只读归因与完成证据真源；区分 exposure hygiene、shared-gradient conflict、trace readout 与整体架构失败，不回填旧 Gate。 |
| `docs/v2-a-closure-c1r-staged-credit.md` | 已消费的 C1R 冻结合同；唯一 formal 在 Stage A 的 G004 FAIL-stop，Stage B 与后续 Gate 均未运行，合同不授权重跑或后继正式实验。 |
| `docs/v2-a-closure-c1r-result-review.md` | C1R 人类可读终局；记录 train/validation 同步欠拟合、H1 起答案类别塌缩、最终答案通路 effective K≈1 的直接证据，以及“不能外推为整个 latent workspace K=1”的边界。 |
| `docs/v2-a-closure-c1s-addressed-workspace-proposal.md` | C1R 后的地址化内容工作区设计来路；已由 C1S 冻结资格合同接替，自身不授权训练或 formal。 |
| `docs/v2-a-closure-c1s-addressed-workspace-qualification.md` / `docs/v2-a-closure-c1s-s0-result-review.md` | C1S 冻结机制资格合同与 S0 结构证据真源；S0 zero-update PASS 曾只授权独立 S1 Overfit32，该授权现已消费。 |
| `docs/v2-a-closure-c1s-s1-s2-s3-execution.md` / `docs/v2-a-closure-c1s-s1-result-review.md` | Semantic-Routed Workspace（SRW）执行合同与当前终局；S1 answer `32/32`，但 no-core `28/32`、至少两个因果贡献者 `7/32` 且动态状态未全过门，故 sealed FAIL。S2/formal 未运行，`authorizes=nothing`。 |
| `docs/v2-a-closure-c1s-s1-failure-attribution.md` / `docs/v2-a-closure-c1s-s1-failure-attribution-result-review.md` | S1 FAIL 后已消费的只读归因合同与终局；旧 diagnosis 只完成 D001–D003，随后在 D004 inner-fold support 处 sealed CRASH。 | 旧身份禁止修补或重跑；没有 D004/D005 结论，`authorizes=nothing`。 |
| `docs/v2-a-closure-c1s-s1-temporal-attribution-repair.md` / `docs/v2-a-closure-c1s-s1-temporal-attribution-repair-preflight-review.md` / `docs/v2-a-closure-c1s-s1-temporal-attribution-repair-result-review.md` | D004R/D005R-only 修复诊断合同、唯一 preflight 与正式结果复盘；以 target-only eligible-record support 生成 feature-specific nested folds，再测 temporal latent/readout。 | 正式诊断已完成；两族 target/motion/readout/frozen decoder 均有正证据，Axis C 因无注册故障可归入而保守 `INCONCLUSIVE`。永远不授权训练、S2/S3 或 V2-A PASS。 |
| `docs/v2-a-closure-c1t-causally-partitioned-workspace.md` / `docs/v2-a-closure-c1t-s0-execution.md` / `docs/v2-a-closure-c1t-s0-result-review.md` / `docs/v2-a-closure-c1t-s1-execution.md` / `docs/v2-a-closure-c1t-s1-result-review.md` / `docs/v2-a-closure-c1t-s1-failure-attribution.md` | C1T 直接 successor、S0/S1 单用合同、结果与 post-stop 归因真源。S0 sealed PASS 后唯一 S1 在 R103/R105 sealed FAIL；归因定位 operation-2 dead-gate、hinge 梯度抵消和 counterpart 跨 cell 耦合，并以单组 screen 排除基础 XOR 不可表达。`authorizes=nothing`。 |
| `docs/v2-a-closure-c1u-publicly-grounded-gate-free-workspace.md` / `docs/v2-a-closure-c1u-s0-execution.md` / `docs/v2-a-closure-c1u-s0-result-review.md` / `docs/v2-a-closure-c1u-s1-execution.md` / `docs/v2-a-closure-c1u-s1-result-review.md` | C1U fresh successor、S0 机制资格、S0/S1 single-use 合同与人类可读终局；删除 learned write gate，补齐 public semantic bridge，并保留旧 causal objective/optimizer/行为 Gate。 | S0/S1 均 sealed PASS；其 S2 合同设计授权已由新的用户执行授权接续。扩展 C2 仍是未来计划，不构成后继授权。 |
| `docs/v2-a-closure-c1u-s2-multibank-matched-k1-k8-execution.md` | C1U S2 多-bank、单 scientific seed、matched learned K1/K8 冻结执行合同；定义六个 fresh banks、三个 heldout folds、同参数/初始化/优化的 K1/K8、六端点固定会计、fresh-bank/paired/causal/functional-K Gate 与 single-use fail-stop。 | 已由唯一 preflight/formal 消费；正式终态以 S2 结果复盘为准。 |
| `docs/v2-a-closure-c1u-s2-result-review.md` / `docs/v2-a-closure-c1u-s2-failure-attribution.md` | S2 人类可读终局及 post-stop 归因；记录 R203/R204/R205 FAIL、K8/K1 heldout、paired/causal/functional-K 边界、train–heldout 重放、Choices-only permutation、opaque alpha-renaming 与建议的直接结构修复。 | 当前 C1U 真源：fixed-head label binding 与 whole-card opaque binding 都不具所需不变性；formal `authorizes=nothing`，修复建议未授权实施。 |
| `docs/v2-c-hierarchical-full-duplex-agent-experiment.md` | V2-C 统一系统实验合同；定义多层、多线程、责任树 + 受控图边、全双工消息、主动 Boundary、必要工具、人类目标连续性、Flat/同步基线、C0-C5 Gate 和停止规则；当前设计完成但未实施。 |
| `docs/v2-a-route-reassessment-2026-07-17.md` | A1.18B 后路线重审；定义合规混合 core，以及 A1.19H generalized hybrid core、A1.20B/A1.20C full-text boundary、A1.21P Pareto、A1.22A audit 的顺序与停机门；2026-08-01 已同步 B→C 高层路线覆盖决定。 |
| `docs/v2-a1.20b-full-text-boundary.md` | A1.20B 无 oracle full-text Boundary 合同、启动前吞吐优化、run-1 eligibility 失败、联合 mapping、oracle 替换、梯度信用审计与停线结论。 |
| `docs/v2-a1.20c-boundary-repair.md` | A1.20C 分层 compiler × execution-credit 合同、token anchor、ST hard-forward、overfit32 失败、真实梯度冲突与停线结论。 |
| `docs/v2-a1.20d-full-text-mechanism-repair.md` | A1.20C 后的 post-stop 双向 section 推断修复；记录 N5 截断根因、三遍 decode、完整 split/hidden causal/routing schedule 诊断与非 formal 边界。 |
| `docs/v2-a1.21p-pareto.md` | K=1 容量负基线、在线 canonical/routing Pareto smoke、路径稳定性、六项正式缺口和 A1.22A 停机判定。 |
| `docs/v2-r1-revalidation-task-design.md` | A1.21P 后的 R1 历史高层任务/模型/公平基线；逐版推进到 v8L、NR1、H1/WD 后已由 Closure C0 路线覆盖。 |
| `docs/v2-r1r-p0-result.md` | V2-R1R 阶段汇总真源；记录 v1–v17 P0-D、P0-M v1–v5、P1 v1–v8L 判决、NR1 formal PASS 与 H1 授权边界。 |
| `docs/v2-r1r-p1-v1-data-failure-review.md` | P1 v1 data formal 的唯一失败 cell、可复现性、只读归因、架构边界与 generator-only P1 v2 条件。 |
| `docs/v2-r1r-p1-v2-cache-infrastructure-failure-review.md` | P1 v2 4096-powered data PASS、完整 cache banks、sealed telemetry `EINVAL`、post-stop full-audit PASS 与 P1 v3 recovery 边界。 |
| `docs/v2-r1r-p1-v3-k8-failure-review.md` | P1 v3 telemetry/cache recovery、K=8 K01–K09、暴露预算、claim bootstrap、K08 审计误报、架构证据边界与 P1-LQ 后继判决。 |
| `docs/v2-r1r-p1-v4-learning-qualification-design.md` | P1 v4-LQ 历史设计：64→512→2048→8192 嵌套规模、机制启动/扩展 Gate、continuation hash 链、K08 修复与 full K01–K09。 |
| `docs/v2-r1r-p1-v4-learning-qualification-execution-command.md` | P1 v4-LQ 历史 `run-p1-lq` 命令；已在 S512 FAIL 后结束，禁止重跑。 |
| `docs/v2-r1r-p1-v4-lq-failure-review.md` | P1 v4 B128 PASS、S512 训练记忆、未见 episode 反事实、owner-contrast 根因与 v5 约束。 |
| `docs/v2-r1r-p1-v5-semantic-transfer-design.md` | 历史 P1 v5 合同；冻结 7,168/1,024 optimization-audit、合法 paired claim、batch32、transfer Gate 与原 K01–K09。 |
| `docs/v2-r1r-p1-v5-semantic-transfer-execution-command.md` | 已消耗的 P1 v5 `run-p1-transfer` 命令；Q7168 FAIL 后禁止重跑。 |
| `docs/v2-r1r-p1-v5-semantic-transfer-failure-review.md` | v5 Q7168 answer/claim 脱钩、梯度与小规模诊断、静态信用对称根因和 v6 约束。 |
| `docs/v2-r1r-p1-v6-causal-temporal-witness-design.md` | 已消耗 P1 v6 合同；冻结同-query 相邻状态 witness、static 等计算对照、fresh formal seeds、隔离 audit 与因果 Gate。 |
| `docs/v2-r1r-p1-v6-causal-temporal-witness-execution-command.md` | 已消耗 P1 v6 `run-p1-ctw` 命令、五个 single-use roots、失败即停与 wake 回传合同；禁止重跑。 |
| `docs/v2-r1r-p1-v6-ctw-assessment-failure-review.md` | 原 v6 assessment 唯一 W604 false 的精确复核；区分 raw Python key type 与 canonical JSON 持久身份。 |
| `docs/v2-r1r-p1-v6r-selection-recovery-design.md` / `docs/v2-r1r-p1-v6r-selection-recovery-execution-command.md` | v6R 冻结恢复合同与已消耗命令；固定旧 seals、canonical/hash replay、三负控与 R601–R610。 |
| `docs/v2-r1r-p1-v6r-selection-recovery-main-review.md` | v6R 两个新 roots、seal、18,899-query 重审、机制资格恢复与 integrated P1 边界的正式主审。 |
| `docs/v2-r1r-p1-v7-integrated-k8-design.md` | 已消耗历史合同；定义 fresh seeds、T+1 state mapping、full 8192/族、86,016-query cache、40-epoch 长度分桶、temporal→joint 路径与 K01–K10。 |
| `docs/v2-r1r-p1-v7-integrated-k8-execution-command.md` | 已消耗 single-use 合同；固定三个 roots 与唯一 `run-p1-integrated` 命令，禁止重跑。 |
| `docs/v2-r1r-p1-v7-integrated-k8-failure-review.md` | v7 K10/state dependence 正证据、CPS causal/OOD 失败、pair-role evaluator bug、zero-middle/T1 Gate 语义与后继边界。 |
| `docs/v2-r1r-p1-v8-causal-bridge-design.md` / `docs/v2-r1r-p1-v8-causal-bridge-execution-command.md` | 已消耗 v8 三臂 scratch 合同与 single-use 命令；三个 roots 已封存。 |
| `docs/v2-r1r-p1-v8-causal-bridge-failure-review.md` | v8 sealed FAIL、训练内 pair 增益、scratch competence collapse、缺少 rehearsal 与架构证据边界。 |
| `docs/v2-r1r-p1-v8r-causal-curriculum-design.md` / `docs/v2-r1r-p1-v8r-causal-curriculum-execution-command.md` | 已消耗 recovery 合同；定义 shared v7 checkpoint、同 mixed batch CE/pair 两臂、R01–R06 与唯一 `run-p1-causal-curriculum`。 |
| `docs/v2-r1r-p1-v8r-causal-curriculum-failure-review.md` | v8R sealed FAIL、CPS 训练记忆、冻结表示/readout 排除、final-decision closure 根因与 BF16 gradient 假阴性复核。 |
| `docs/v2-r1r-p1-v8d-causal-decision-witness-design.md` / `docs/v2-r1r-p1-v8d-causal-decision-witness-execution-command.md` / `docs/v2-r1r-p1-v8d-causal-decision-witness-failure-review.md` | 已消耗机制资格、命令与失败复核；记录 D03/D04/D06 失败，以及 perturbation 被放大但 semantic reduction 未形成。 |
| `docs/v2-r1r-p1-nr1-numeric-relation-measurement-design.md` / `docs/v2-r1r-p1-nr1-execution-command.md` | V8L 后直接切换且已消耗的新路线入口；冻结手工外部预期、typed measurement、独立 oracle、非链式拓扑、metric-to-fault ledger、单一 CLI、两根 roots 与 fixed transport。 | 历史合同，禁止修改或重跑 |
| `docs/v2-r1r-p1-nr1-main-review.md` | NR1 唯一正式 PASS 的父任务主审；复算两根 seal、N01–N07、source/Git/process/transport、measurement/fault/topology 与 H1-only 授权边界。 | 当前 NR1 结果真源；不构成模型或 P1 通过 |
| `docs/v2-r1r-p1-h1-mixed-core-design.md` / `docs/v2-r1r-p1-h1-execution-command.md` | NR1 后的 H1 shared-vs-mixed 开发对照；定义 full supervised-semantic fingerprint、历史旧身份/current screen/calibration/formal/smoke 七域隔离、23,456-record 正式预检 cache、公共 FFN + 共享 `D->H` nonlinear features + generic/routed `H->D` state-write projection、matched active compute、三 seed、H01–H08 与唯一 single-use 命令。 | factorized full-budget screen 已 H05/H06 FAIL；calibration、阈值/hash freeze 与 formal 均未授权 |
| `docs/v2-r1r-p1-h1-wd-overlap-residual-design.md` / `docs/v2-r1r-p1-h1-wd-execution-command.md` / `docs/v2-r1r-p1-h1-wd-failure-review.md` | H1-WD 正向重合写入、公共残差化、matched continuation、冻结 Gate、唯一命令与失败归因。 | 唯一 root 已 `FAIL_H1_WD_NONFORMAL_MECHANISM`；只接受重参数化可行，拒绝路径必要性与架构收益 |
| `docs/v2-r1r-p1-v8l-causal-state-ladder-design.md` / `docs/v2-r1r-p1-v8l-causal-state-ladder-execution-command.md` / `docs/v2-r1r-p1-v8l-causal-state-ladder-failure-review.md` | 已消耗机制资格、single-use 命令与失败复核；记录 bootstrap ERE/CPS 分裂、lexical numeric measurement 漏检和当前 K=8 主线停线。 |
| `docs/v2-r1r-p0-main-review.md` | P0-D v1 主设计层独立复核；记录 ERE/CPS surface 满分捷径、claim/split/token/certificate 缺口及 v2 修订理由；仅为历史判决依据。 |
| `docs/v2-r1r-p0-v2-main-review.md` | P0-D v2 主设计层独立验收；接受失败 artifact 与停机纪律，拒绝 v2 合同完成度，并记录 horizon 条件捷径、ERE provenance、CPS 深度和 audit/provenance 盲区。 |
| `docs/v2-r1r-p0-v3-main-review.md` | P0-D v3 主设计层独立验收；接受单次 D2 失败与停机纪律，拒绝 D0/审计器/CPS/provenance 合同实现，并给出 v4 直接切换依据。 |
| `docs/v2-r1r-p0-v4-main-review.md` | P0-D v4 主设计层独立验收；接受 `FAIL_R0` 和停机纪律，拒绝 reference 自算 expected、浅层预测试、伪 G07/G08 和正向 Gate 地位。 |
| `docs/v2-r1r-p0-v5-r0a-main-review.md` | P0-D v5 R0A 主设计层独立验收；接受 14/4/12 与 artifact hash 的窄证据，因六项 fail-closed AST 探针失败和一次错误-hash 预调用而拒绝阶段通过。 |
| `docs/v2-r1r-p0d-v6-r0a-strict-design.md` | v6 历史 R0A-strict 冻结设计；定义完整 ERE/CPS AST schema、245 项非法输入矩阵、精确错误、输入不变、原子 formal attempt 和严格停止规则。 |
| `docs/v2-r1r-p0d-v6-r0a-strict-execution-command.md` | v6 历史执行合同；四文件直接切换、v5 tests 删除与唯一 fixed-root `seal-strict` 已执行结束，禁止重跑。 |
| `docs/v2-r1r-p0-v6-r0a-strict-main-review.md` | v6 sealed artifact、当前实现与新鲜 query-placeholder 探针的独立验收。接受 14/4/12/245 和 protocol 窄证据，因十项作用域反例拒绝阶段通过。 |
| `docs/v2-r1r-p0d-v7-r0a-lattice-design.md` | v7 历史冻结设计；以 23 rule slot、5 query slot、6 scope、6 token 形成 858-cell operand lattice。 |
| `docs/v2-r1r-p0d-v7-r0a-lattice-execution-command.md` | v7 历史执行合同；唯一 formal 已执行结束，禁止覆盖或重跑。 |
| `docs/v2-r1r-p0-v7-r0a-lattice-main-review.md` | v7 七文件 artifact、858-cell coverage 与 27 项新鲜公共 API 探针的独立验收；最终判定 `R0A-lattice accepted`。 |
| `docs/v2-r1r-p0d-v8-r0b-structural-design.md` | v8 历史冻结设计；用 11-record qualification bundle、41 faults、4 metamorphic 与 37 raw metrics 资格化 G02–G06。 |
| `docs/v2-r1r-p0d-v8-r0b-structural-execution-command.md` | v8 历史执行合同；唯一 fixed-root formal 已执行结束，禁止覆盖或重跑。 |
| `docs/v2-r1r-p0-v8-r0b-structural-main-review.md` | v8 artifact、机器矩阵和 registry 外反例的独立验收；最终判定 machine-pass、main-review rejected。 |
| `docs/v2-r1r-p0d-v9-r0b-invariant-design.md` | v9 历史 accepted R0B 冻结设计；以固定 qualification profile、exact schema、可逆 grammar、typed ERE provenance 与 derived CPS composition 资格化 G02–G06。 |
| `docs/v2-r1r-p0d-v9-r0b-invariant-execution-command.md` | v9 历史执行合同；唯一 fixed-root formal 已执行结束，禁止覆盖或重跑。 |
| `docs/v2-r1r-p0-v9-r0b-invariant-main-review.md` | v9 formal artifact、45 文件 seal 与六项 registry 外公共 API 探针的独立验收；最终判定有限 `R0B-invariant accepted`。 |
| `docs/v2-r1r-p0d-v10-r0c-statistical-design.md` | v10 历史 accepted 冻结设计；定义 source-only parser、条件多数、word/character NB、grouped fold、CV/heldout、34 raw metrics、39 adversary、5 metamorphic 与固定语义摘要。 |
| `docs/v2-r1r-p0d-v10-r0c-statistical-execution-command.md` | v10 历史执行合同；42 文件 frozen guard、prior seal/runtime identity、preflight 与唯一 formal 已执行结束，禁止重跑。 |
| `docs/v2-r1r-p0-v10-r0c-statistical-main-review.md` | v10 formal、52 文件根 seal、bundle 双层 seal、自证攻击与六组 registry 外公共函数探针的独立验收；最终判定有限 `R0C-statistical accepted`。 |
| `docs/v2-r1r-p0d-v11-r0d-integrated-design.md` | v11 accepted R0D 冻结设计；显式区分 semantic projection 与 qualification surface，定义 12 cases、19 metrics、20 faults、4 metamorphic、import 与 replay。 |
| `docs/v2-r1r-p0d-v11-r0d-integrated-execution-command.md` | v11 历史执行合同；29 文件 frozen guard、33 项 pytest、完整 preflight 与唯一 formal 已执行结束，禁止重跑。 |
| `docs/v2-r1r-p0-v11-r0d-integrated-main-review.md` | v11 formal、144 文件直接/传递 seal、跨 Gate fault、三项 registry 外探针与 `<=400` 长度边界的独立验收；最终接受有限 integrated measurement system。 |
| `docs/v2-r1r-r1e-v12-entry-qualification-design.md` | 历史 accepted R1E v12 冻结设计；直接新建 production renderer/scalable learner 层，以可逆完整 AST、source-only 边界、真实长度和 compact exact scoring 关闭 generator 的两个入口条件。 |
| `docs/v2-r1r-r1e-v12-entry-qualification-execution-command.md` | 历史 v12 执行合同；唯一 fixed-root formal 已结束，活动 CLI/tests 已由 v13 直接替换。 |
| `docs/v2-r1r-r1e-v12-entry-qualification-main-review.md` | v12 fixed root、57 文件 seal、480 个随机 learner 对照和三项深层嵌套 renderer 外部探针的独立验收；接受有限 production entry，并只授权另立 generator smoke。 |
| `docs/v2-r1r-r1g-v13-generator-smoke-design.md` | v13 accepted 冻结设计；每族 720/总 1440 records，定义 production ERE/CPS 构造、alpha-invariant fingerprint、因果 pair、teacher/claim、shortcut、G01–G11 和硬停线。 |
| `docs/v2-r1r-r1g-v13-generator-smoke-execution-command.md` | v13 历史执行合同；两次确定性 preflight、直接切换和唯一 fixed-root formal 已结束，禁止覆盖或重跑。 |
| `docs/v2-r1r-r1g-v13-generator-smoke-main-review.md` | v13 formal、62 文件 seal、400 个新 seed 结构探针与三组非 formal shortcut seed 的独立验收；接受 fixed-seed generator smoke，并冻结完整 P0-D 前的统计/provenance 条件。 |
| `docs/v2-r1r-p0d-v14-full-production-design.md` | v14 冻结设计；定义 4096/512、root provenance、3-attribute/6-value ERE 域、98/14-cell simultaneous shortcut Gate 与 single-use formal。 |
| `docs/v2-r1r-p0d-v14-full-production-execution-command.md` | v14 历史执行合同；8,192-record 双运行 preflight 与唯一 14,336-record formal 已结束，禁止覆盖或重跑。 |
| `docs/v2-r1r-p0d-v14-full-production-main-review.md` | v14 正式 G09 两格失败、其余 Gate、235 MB sealed artifact、统计 power 与后继边界的独立验收；最终 rejected。 |
| `docs/v2-r1r-p0d-v15-g09-qualified-production-design.md` | v15 冻结设计；定义 98/14-cell null/local/diffuse power、4,096/1,536、exact-equivalent batch scorer、Q-before-F 与 single-use 停线。 |
| `docs/v2-r1r-p0d-v15-g09-qualified-production-execution-command.md` | v15 历史执行合同；唯一 Q 已结束为 FAIL，固定 F 未运行，两个命令均禁止重跑。 |
| `docs/v2-r1r-p0d-v15-g09-qualified-production-main-review.md` | v15 Q01–Q07/Q09、Q08 4.7617x 性能失败、根 seal 与 fresh formal 未运行的独立验收；最终 rejected。 |
| `docs/v2-r1r-p0d-v16-runtime-qualified-production-design.md` | v16 历史冻结设计；把 correctness 与 runtime 拆 Gate，定义 15 轮 paired、v14 完整 G09/G10 投影/绝对预算、18-fault 与 seed 2026081602 fresh F。 |
| `docs/v2-r1r-p0d-v16-runtime-qualified-production-execution-command.md` | v16 已完成的 single-use 执行合同；Q accepted、F rejected，固定 Q/F 命令均禁止重跑。 |
| `docs/v2-r1r-p0d-v16-runtime-qualified-production-main-review.md` | v16 Q/F seal、运行指标、G05 307 条 visible-domain 失败、G07 path-aware alpha 误报与后继边界的独立验收。 |
| `docs/v2-r1r-p0d-v17-repaired-production-design.md` | v17 冻结设计；定义 ERE visible-domain/alpha 两项 repair qualification、fresh seed 2026081702 与 G01–G11 conjunction。 |
| `docs/v2-r1r-p0d-v17-repaired-production-execution-command.md` | v17 历史 single-use 执行合同；repair qualification 与 fresh production 均已通过并封存，禁止覆盖或重跑。 |
| `docs/v2-r1r-p0d-v17-repaired-production-main-review.md` | v17 两个 roots、R01–R07、G01–G11、seal、根因关闭与 P0-M-only 授权的独立验收；P0-D 最终 accepted 真源。 |
| `docs/v2-r1r-p0m-v1-design.md` 至 `docs/v2-r1r-p0m-v4-failure-review.md` | P0-M v1–v4 的冻结合同和失败/预测试主审；记录 cache 判据、BF16 mask、baseline 效率、claim compatibility、shortcut 与监督时序闭环。 |
| `docs/v2-r1r-p0m-v5-design.md` | P0-M accepted 冻结合同；定义 K=8 shared core、interaction claim probe、同族 owner contrast、公平 baselines、throughput 与 M01–M08。 |
| `docs/v2-r1r-p0m-v5-execution-command.md` | P0-M v5 single-use 执行顺序和 P1-before stop；所有 fixed roots 已运行并封存。 |
| `docs/v2-r1r-p0m-v5-main-review.md` | P0-M v5 M01–M08、八个 seal、失败—修复机制、架构有限含义与 P1 边界的独立验收；最终接受 training-path smoke。 |
| `docs/v2-r1r-p1-v1-single-seed-falsification-design.md` | P1 v1 冻结合同：fresh 8192/1024 ERE/CPS、packed mmap cache、共享 K=8/K=1、机制监督全程保留、因果干预、matched-training baselines 与 K01–K09。 |
| `docs/v2-r1r-p1-v1-execution-command.md` | P1 v1 single-use 阶段顺序、正式 roots、失败即停与 P2-before stop。 |
| `docs/v2-r1r-p1-v1-data-failure-review.md` | P1 v1 唯一 fresh-data formal 的 G09 失败、全量 replay、规模/消融诊断、架构证据边界与 generator-only P1 v2 后继条件。 |
| `docs/v2-r1r-p1-v2-powered-single-seed-design.md` | P1 v2 冻结合同：8192/4096 measurement power、1024 model subset、Q-before-D、K=8-first 与 single-use stop。 |
| `docs/v2-r1r-p1-v2-execution-command.md` | P1 v2 唯一正式顺序；preflight/power/data PASS，cache infrastructure FAIL 后已结束。 |
| `docs/v2-r1r-p1-v2-cache-infrastructure-failure-review.md` | P1 v2 cache 失败的正式/诊断证据分层、根因、架构边界与 P1 v3 cache-recovery 建议。 |
| `docs/v2-r1r-p1-v3-cache-recovery-design.md` | P1 v3 frozen 合同：telemetry decision power、immutable cache 双重 full audit、K=8-first 与失败即停。 |
| `docs/v2-r1r-p1-v3-cache-recovery-execution-command.md` | P1 v3 single-use 顺序；preflight/telemetry/recovery PASS，K=8 FAIL 后已结束。 |
| `docs/v2-r1r-p1-v3-k8-failure-review.md` | P1 v3 recovery/K=8 artifact、训练暴露、目标 bootstrap、K08 误报与 P1-LQ 路线的父任务复核。 |
| `docs/v2-r1r-p1-v4-learning-qualification-design.md` | 已执行结束的 P1-LQ 历史合同；B128 PASS、S512 FAIL 后不得续跑。 |
| `docs/v2-r1r-p1-v4-learning-qualification-execution-command.md` | 已消耗的历史 single-use 执行顺序和 root 边界。 |
| `docs/v2-r1r-p1-v4-lq-failure-review.md` | v4 当前失败判决真源；区分正式 Gate 与父任务 post-hoc 诊断。 |
| `docs/v2-r1r-p1-v6-causal-temporal-witness-design.md` / `docs/v2-r1r-p1-v6-causal-temporal-witness-execution-command.md` | 已消耗训练机制资格合同；原 assessment FAIL 永久保留，后继资格由 v6R 独立恢复。 |
| `docs/v2-r1r-p0d-v4-r0-execution-command.md` | v4 历史执行命令；已执行并失败，只作 artifact 追溯，不可修复或重跑。 |
| `docs/v2-r1r-p0d-v5-r0a-execution-command.md` | v5 R0A 历史执行命令；已经执行并结束，只作 artifact 追溯，禁止修补、覆盖或重跑。 |
| `docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md` | V2 决策形成记录；只保存推导和审阅依据，不与白皮书并行定义规范。 |
| `docs/moe-model-assembly-comparative-review-2026-07-14.md` | V2 的 Boundary-MoE/FFN-MoE 与公开模型路线的中文对照；区分已被其他模型验证的局部思想、完整架构未验证边界和 V2-B 未启动状态。 |
| `docs/DIRECTORY_REFERENCE.md` | 本索引；新代码、测试、文档和归档必须同步这里。 |
| `README.md` | 面向仓库使用者的当前状态说明；展示 V2-R1R/H1-WD 历史闭环、Closure C0/C0R/C1/C1R/C1S/C1T 结果、C1U S0/S1 PASS、C1U S2 sealed FAIL/结构归因，以及 V2-C 仅设计状态。 |
| `docs/v2-a-reasoning-medium-experiment.md` | V2-A 当前实现合同、Qwen3.5 基座、数据 schema、A0/A1/A2 结果、Gate 判定和失败边界。 |
| `docs/v2-a1.5-latent-foundation.md` | A1.5 独立 schema、P0 结构化正控制、P1 Qwen hidden 接口、P2 learned-slot formal、因果干预和停止门禁。 |
| `docs/v2-a1.9-qwen-boundary.md` | A1.9 冻结 Qwen hidden → 冻结 A1.8 structured core 的预注册合同、三 run formal/causal 结果、成本、捷径干预和证据边界。 |
| `docs/v2-a1.10-anonymous-workspace.md` | A1.10 full-token frozen-Qwen cache、匿名 K-slot workspace、通用 recurrent reasoner、方法修正、三 run formal 失败、成本和证据边界。 |
| `docs/v2-a1.11-fault-localization.md` | A1.11 Boundary × Reasoner 2×2 合同、方法纠正、严格 overfit/formal 结果、一级故障归因和 A1.12 边界。 |
| `docs/v2-a1.12-reasoner-root-cause.md` | A1.12 binding × cursor 正交拆分、三臂 formal 失败和排除结论。 |
| `docs/v2-a1.13-transition-closure-root-cause.md` | A1.13 transition identity × latent closure 正交矩阵、三 seed formal/causal 结果和 state/answer 分离判定。 |
| `docs/v2-a1.13f-fixed-budget-method-audit.md` | A1.13F 固定 4000-step 审计；排除 early-stop 作为单因素臂不稳定的主要解释。 |
| `docs/v2-a1.15-closed-coupled-core.md` | A1.15 query-coupled readout 合同、overfit 正控制和 fresh-seed formal `0/3` 结果。 |
| `docs/v2-a1.16-redundant-answer-loss.md` | A1.16 删除重复 answer CE 的唯一变量实验；fresh-seed formal `0/3`。 |
| `docs/v2-a1.17-paired-objective-initialization-audit.md` | A1.17 同 seed/同共享初始化的 objective × initialization 配对审计合同与最终归因。 |
| `docs/v2-a1.18-training-scaffold.md` | A1.18 FINAL-SAUX `2/3`、A1.18B TSAUX paired/fresh 六 seed formal/causal、部署剥离、机制结论和 state-target 来源边界。 |
| `docs/v2-a1.6-core.md` | A1.6 relation-addressed continuous state core 的数据合同、结构完整性、C0 停止点和证据边界。 |
| `docs/v2-a1.7-core.md` | A1.6 closure 归因、A1.7 受控 relation 数据、三 seed formal/causal、2×2 消融、8/12/16 步压力与成功概率真源。 |
| `docs/v2-a1.8-long-horizon.md` | A1.8 T1–16 均衡随机深度合同、三组独立 data/model seed、T20/T24 Gate、T32 诊断、稳定性、成本和归因真源。 |

## 研究辅助入口

| 路径 | 用途与边界 |
| --- | --- |
| `docs/research-agent-literature-commands.md` | 面向 Google Gemini Deep Research 的历史研究冲刺命令；其中旧 A1-A4 仅保留为专题分组，当前正式路线已将 I/O、工作树/记忆、专家晋升和整机评测聚合为 V2-C。研究报告只作为决策输入，不自动修改架构真源或 Gate。 |
| `docs/thinking-draft-synthesis-2026-08-01.md` | 对 `thinking/` 五篇近期草稿的主题重建、逐命题判断和 V2-C 聚合映射，并把项目初期 V1 白皮书单列为历史参照；明确排除 `thinking/research/`，不修改原稿，也不自动升级为实验通过证据。 |

## 当前实现状态

V2-A0 数据/基座链路、Qwen3.5-2B/0.8B text-CoT probe 和当前结构的 V2-A1 mechanism smoke 已实现；旧 A2 的 K/T、prompt、transition、读出、full-data、teacher/masked state supervision、token-wise source adapter、step-level verifier RL、latent-attention probe 与 0.8B/2B 对照仍未达到强 text-CoT 基线。source-layer bank、query init、token mixer、latent-attention 等失败入口已从当前代码删除。新增的 A1.5 独立 schema 与 P0/P1/P2 代码已完成：P0 32-example overfit final/state full exact `1.0/1.0`，4096-example best ordinary test `1.0/1.0`，composition-heldout `0.2734/0`，length-heldout final/state full exact `1.0/0.2266`；真实 Qwen3.5-2B FP16 hidden cache 已分片生成，P1 4096-cache formal ordinary validation final/state `1.0/1.0`，小 probe 仅作为 underfit 诊断；P2 K=8 formal ordinary test `1.0/1.0`、composition `0.2773/0`、length `1.0/0.6484`，same-answer composition shuffle 失败；A1.5 0.8B no-cap zero-shot 128 条 test/composition/length formal 分别为 `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`，2-shot test/composition/length final/state 也分别为 `0.1328/0.0156`、`0.1953/0`、`0.0938/0`（格式解析率高但 state 失败）。A1.5 未通过，V2-A3/V2-A4/V2-B 均未启动。A1.5 真源见 `docs/v2-a1.5-latent-foundation.md` 与 `tmp/V2-A1.5 result.md`；旧 A2 结果仍只作历史 probe，不得与 A1.5 混写。

A1.6 已作为失败证据保留：data audit 与 overfit32 通过，正式 C0 ordinary/length trajectory full 为 `1.0/1.0`，relation-heldout 为 `0.765625`，按 Gate 停止。只读 checkpoint 诊断显示 oracle-reset one-step 与 predicted hard re-embed diagnostic 在 test/length/relation/causal 均为 `1.0`，失败定位为 continuous latent closure；旧 final-vs-trajectory Gate 无效，旧 relation split 也混入多变量。因果干预与 C1 未启动。

A1.7 是当前结构化 core 证据：独立受控数据合同把唯一 holdout 冻结为 `COPY amber→jade`。三个初始化 seed 的 5–6 步 formal C0 与 causal intervention 均通过；目标 seed 的 test/length/relation trajectory full 为 `1.0/0.996094/0.998047`。但 2×2 消融表明短程通过不能只归因于地址/内容分离或 closure；closure 的明确作用是减缓长程漂移。三个目标 checkpoint 的 8/12/16 步严格压力 Gate 为 `0/3`，T16 supported 为 `0.894531/0.941406/0.894531`。当前对 V2-A 形成 matched text-CoT Pareto 的工程判断为 `40%–55%`、中心约 `48%`。该结论只覆盖三寄存器结构化合成 core；Qwen/C1、匿名 workspace、完整 V2-A、V2-A3/V2-A4/V2-B 均未启动。真源见 `docs/v2-a1.7-core.md`、`tmp/V2-A1.7 result.md` 与 `artifacts/v2-a/a1_7/core-assessment-summary.json`。

A1.8 是当前最新 core 证据：保持 A1.7 架构和 closure 不变，把训练切换为 T1–16 batch 内均衡随机深度。三组独立 data/model seed 的 short、T8/12/16、OOD T20/T24、relation 和 causal Gate 全部通过；T16 supported/relation 最低为 `0.996094/1.0`，T24 为 `1.0/0.996094`，诊断性 T32 为 `0.988281/1.0`。跨 run 及相对 A1.7 fingerprints overlap 为 `0`。证据否定 T16 必然内在发散并支持 horizon mismatch，但没有 compute-matched 地拆分长度覆盖和 transition exposure；仍不覆盖 Qwen hidden、匿名 workspace 或 matched text-CoT Pareto。A1.8 完成时的工程概率为 V2-A Pareto `45%–60%`、中心约 `53%`。真源见 `docs/v2-a1.8-long-horizon.md`、`tmp/V2-A1.8 result.md` 与 `artifacts/v2-a/a1_8/assessment-summary.json`。

A1.9 是当前最新 boundary 证据：冻结 Qwen3.5-2B 与三组已经通过的 A1.8 core，只训练 value、family 和共享 register adapter。三个独立 data/core/adapter seed 的 cache audit、formal 与 hidden causal Gate 全部通过；T16 supported/relation 最低 `1.0/0.996094`，T24 为 `1.0/1.0`，诊断性 T32 为 `0.996094/1.0`，mapping 最低 `1.0`。query swap 和 same-answer/different-trajectory 跟随新 oracle 为 `1.0`，no-hidden 与 independent role shuffle trajectory 为 `0`；core hash 未改变、跨 run fingerprint overlap 为 `0`。证据只覆盖 oracle-role-segmented Qwen hidden，不覆盖 learned full-text role extraction、匿名 workspace 或 matched text-CoT Pareto。当前 V2-A Pareto 工程判断为 `48%–63%`、中心约 `55%`。真源见 `docs/v2-a1.9-qwen-boundary.md`、`tmp/V2-A1.9 result.md` 与 `artifacts/v2-a/a1_9/assessment-summary.json`。

A1.10 是联合目标配置的失败证据：在同一轮删除 oracle span mask 和显式三寄存器 scaffold，使用 frozen Qwen3.5-2B 的完整 last-hidden + attention mask、匿名 `K=8` learned slots，以及两层参数共享的通用 recurrent Transformer。修正版 overfit32 通过，但三个独立 data/model seed 正式 Gate 为 `0/3`，hidden interventions 按停止规则未运行。该阶段自身不能单独归因，后续 A1.11 已补齐正交诊断。真源见 `docs/v2-a1.10-anonymous-workspace.md`、`tmp/V2-A1.10 result.md` 与 `artifacts/v2-a/a1_10/assessment-summary.json`。

A1.11 是一级故障定位证据。Boundary 臂在无 oracle span、保留 frozen A1.8 core 的修正合同下，overfit32 trajectory/answer/mapping 全为 `1.0`，但 source/target pointer 最低均为 `0.9642857143`，严格 Gate failed；只读 hard re-embedding 全部恢复 `1.0`，所以存在连续 latent → frozen address geometry 缺陷，但 task-level formal 未运行。Reasoner 臂直接使用 exact symbolic typed roles，不加载 Qwen/cache/adapter/core；overfit32 全通过，三组 formal 稳定 `0/3`，全部主要 trajectory 为 `0–0.003906`，训练集均衡诊断 trajectory 也只有 `0–0.003906`。这足以否定纯组合故障并把主要独立失败源定位到 anonymous binding／generic transition／当前 readout objective 这一整臂，但尚未拆开三者。当前 V2-A matched Pareto 工程判断为 `22%–35%`、中心约 `28%`。真源见 `docs/v2-a1.11-fault-localization.md`、`tmp/V2-A1.11 result.md` 与 `artifacts/v2-a/a1_11/assessment-summary.json`。

A1.12–A1.17 是当前最新二级根因证据。A1.12 BIND/CURSOR/BOTH 均为 formal `0/3`；A1.13 GENERIC-CLOSURE state `0/3`、STRUCTURED-CE `1/3`、STRUCTURED-CLOSURE state `3/3`/full `1/3`；A1.13F fixed 4000-step 排除 early-stop 主因。A1.15/A1.16 的 query-coupled + CE/noCE fresh seed 均 state/full `0/3`。A1.17 在 A1.13 三个成功 model/data seed 上验证全部共享初始 tensor 位相等后重跑，两条 coupled 臂仍均 state/full `0/3`。机器分类 `independent_answer_auxiliary_gradient_required`：当前 state 学习依赖独立 pooled-answer objective 的全局辅助梯度，但该旁路不能形成可靠因果答案。根因在 exact-symbolic 三寄存器合同内已定位，架构未通过；下一阶段只允许 training-only QAUX/SAUX 目标重设。真源见 `docs/v2-a1.12-reasoner-root-cause.md` 至 `docs/v2-a1.17-paired-objective-initialization-audit.md`、`tmp/V2-A1.12-A1.17 root-cause result.md` 与各阶段 assessment。

A1.18/A1.18B 是当前最新训练机制证据。QAUX/FINAL-SAUX overfit32 通过；FINAL-SAUX paired formal/causal 为 `2/3`，证明 final-only 全局完整状态梯度方向正确但 seed 不稳定。TSAUX 用一组跨步共享训练头在每个递归步从 global workspace mean 预测完整 state；三个 paired seed 与三个 fresh model seed 的 formal/causal 均为 `3/3`。六个通过部署模型的 causal trajectory/answer 都为 `1.0`，全部反事实 Gate 通过，disable-recurrence 与 wrong-start trajectory 都为 `0`；formal 前辅助参数已物理删除。机器分类 `per_step_global_state_credit_assignment_confirmed`、`mechanism_solved=true`。结论只覆盖 exact-symbolic 三寄存器 core；逐步 oracle state target 的开放任务来源仍未解决。真源见 `docs/v2-a1.18-training-scaffold.md`、`tmp/V2-A1.18 result.md` 与 `artifacts/v2-a/a1_18b/assessment-summary.json`。

2026-07-17 白皮书与路线重审已接受 `S_t=(A_t,H_t)` 合规混合 core；A1.19H formal/causal `3/3`，A1.20D 形成 full-text mechanism 强诊断，A1.21P 因任务同构、baseline 不公平和 formal 缺口停止。V2-R1R 以 ERE/CPS 检验同一 Boundary/core。v17 的 26,624-record formal G01–G11 全 true，P0-D accepted；P0-M v5 的 M01–M08 全 true，只接受 training-path smoke。P1 v6/v6R 正式资格化 causal-temporal witness；v7 保留 state mechanism 但 CPS causal/OOD 失败；v8R 排除 scratch/coverage，v8D 证明 CPS perturbation 可进入并放大但未归约为可迁移决策代数；v8L 在 ERE bootstrap 通过、CPS bootstrap 失败处正式停线，并暴露 lexical numeric teacher 未被单独资格化。匿名 K=8 + lexical-anchor 主线保持关闭；P1-NR1 已 sealed PASS 并只授权 H1。H1 的完整-transition、互斥完整 FFN、2:1 residual、等分 residual 与 factorized routed projection 五个非正式方向均已否决；factorized full-budget screen 的 overall gain 为 `-0.01074`，wrong-route/conditional-write 最大效应仅 `0.00281/0.00781`。后续 H1-WD 又在无 family target 下成功搬运约 `25.07%` common 输出能量，却只把 projection-off effect 提高 `0.00391`，common-off/projection-off 都仅 `0.01172`，matched-control gain `+0.00488` 且 CI 跨零，故同样 FAIL。calibration、阈值/hash freeze 与 formal 均未授权。仍无完整组合泛化、K 容量、跨 seed 或 Pareto 通过结论，V2-C 继续等待 V2-A/V2-B。

2026-08-21 又冻结并唯一执行了一个不复用失败 WD checkpoint 的 decision-causal 非正式 screen：从原 mixed deployment 物化 common-off answer-margin drop 与 projection-output VJP/Fisher target。target Gate PASS，但 W causal/control heldout transfer-nMSE 为 `1.00450/0.99247`，故以 `FAIL_WD_DECISION_CAUSAL_WRITE_FIT` 在 D/J 前停止。它不恢复 H1 路线，不构成 shared-only 架构对照，机器边界始终为 `authorizes=nothing`。

2026-08-23 又冻结并唯一启动了 R0–R4 direction-geometry 诊断，用于区分 global、route-conditioned、target-before input-predictable、projection-parameter-reachable 与 residual/null 分量。唯一运行在全量 replay identity Gate 以 `CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY` 停止：common/projection 最大绝对误差为 `7.96914e-05/9.50396e-05`，超过冻结容差 `2.5e-05`。旧 target bank 用 microbatch `4` 物化，新 replay 用 batch `128`；不同 CUDA batch geometry 造成的尾部有限精度漂移未被首批 spot check 捕获。R2–R4 均未运行且无 `result.json`，因此没有方向几何结论；root 已消耗，`authorizes=nothing`，禁止同 identity 重跑。

同日 v2 successor 以 exclusive sibling lease 修复单次身份，并在 root 前按原生 microbatch `4` 全量重放，common/projection 误差均为 `0.0`。唯一运行完整产出 R1–R4，机器终态 `COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`、科学终态 `NO_QUALIFIED_R2_R3_COMPONENT`：正确 route pairing 显著但 heldout energy gain 为负，projection trunk 的 `+3.189%` 不稳定，exact shared head 与 sampled local-J 分别为 `-0.334%/-5.922%`，双 sketch residual 不与 current null 相容。只读分层确认 nonzero target prevalence 从 train `4010/4096` 降至 heldout `591/1024`，当前首要问题是 target-before write gate 与 ordered split shift，而不是方向纯随机散步。root/lease 均已消费，仍为 `authorizes=nothing`。

同日项目回到整体 V2-A 并单次完成 Closure C0。机器终态 `FAIL_V2_A_CLOSURE_C0_READINESS`：C001/C002/C003/C005/C007/C008 通过；C004 因 ordinary ERE relation-query 全 TRUE 和 source-only visible-legend oracle 条件准确率 `1.0` 失败；C006 因 compact trace 缺 parser/semantic verifier/full-bank roundtrip/fault-kill 失败。root 的 13 个 sealed files 全量复验通过，result/evidence-seal SHA-256 为 `3FC871…43B1`/`462317…9972`，训练、optimizer step 与模型写入均为零。A1.8/A1.18B/A1.19H 只复用 core/训练方法，A1.20B/C 只复用负向诊断，A1.21P 只复用 K=1/成本审计方法；所有旧 checkpoint 与 H1/WD route target 均排除。该时点只允许新 data identity 与 trace verifier 后重做 C0，不授权 C1。

2026-08-24 的直接 successor C0R 不覆盖上述失败。新 relation-balanced generator identity 在四个 relation cell 上把 source-only oracle 降回 `0.5`；strict compact trace 对 26,624 条记录全部 roundtrip/replay，并 kill `394,278/394,278` 个定向故障。data/trace D001–D012 与 readiness C001–C008 全 PASS，正式 result/seal 分别为 `B2502F…FFF9`/`5AFB37…B570` 与 `391C84…8D5E`/`161FBB…0BA4`；两根 seal 复验为 `42/42`、`25/25`。C0R 仍未训练模型，`four_arm_results_present=false`、`v2a_passed=false`，只授权 C1 single-seed implementation/eligibility。

## V2-R1R P0-D / P0-M / P1 代码、测试与产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `src/yggdrasil_v2/r1_revalidation/` | 保留 accepted `common/learner/integration/production` surface 与已资格化 `nr1/` measurement；`h1/` 实现 factorized routed-projection workflow，`h1_wd/` 隔离保留已消耗 overlap-write/common-residual screen，`h1_wd_causal/` 保留已消耗 decision-causal target/W/D/J runner；旧 live `p1/*.py` 已删除。 | NR1 只允许离线 target/历史复算；两个 WD screen 都已 FAIL、非 formal 且 `authorizes=nothing`；F1/P2 未实现 |
| `experiments/v2_r1_revalidation.py` | V2-R1R 历史物理入口，仍只暴露 `run-p1-h1` 并在写 transport/root 前复算 readiness。 | factorized screen FAIL 后只能 before-mutation 拒绝；不再是活动主线，Closure C0 使用独立入口 |
| `tests/v2_r1r_v4/` | 已在 v5 R0A 直接切换时删除。 | 废旧测试已删除；v4 artifact/文档保留追溯 |
| `tests/v2_r1r_v5/` | v5 R0A 活跃旧测试目录已在 v6 直接切换时整目录删除。 | 废旧测试已删除；v5 artifact/主审文档保留追溯 |
| `tests/v2_r1r_v6/` | 独立 oracle、245 项 invalid-AST matrix、mechanical validator/runner、CLI template、contract guard、15 文件 SHA manifest 与六组测试。 | 冻结历史合同；pytest `18 passed`，但主审证明 query × placeholder 组合覆盖不完整 |
| `tests/v2_r1r_v7/` | 已在 v8 R0B 直接切换时删除。 | 废旧测试已删除；v7 artifact/主审保留追溯 |
| `tests/v2_r1r_v8/` | v8 active Python/JSON 测试源已在 v9 直接切换时删除；历史合同只由文档与 sealed artifact 保留。 | 非现行测试入口；残留缓存或空目录不构成可执行真源 |
| `tests/v2_r1r_v9/` | v9 active Python/JSON 测试源已在 v10 直接切换时删除；历史合同由 v9 source snapshot、文档与 sealed artifact 保留。 | 非现行测试入口；残留缓存或空目录不构成可执行真源 |
| `tests/v2_r1r_v10/` | v10 active Python/JSON 测试源已在 v11 直接切换时删除；历史合同由 v10 source snapshot、文档与 sealed artifact 保留。 | 非现行测试入口；残留缓存或空目录不构成可执行真源 |
| `tests/v2_r1r_v12/` | v12 活动 Python 测试源已在 v13 首次完整 preflight 通过后删除；51 项历史测试由 v12 source snapshot 保留。 | 非现行测试入口；不得恢复兼容 alias |
| `tests/v2_r1r_v13/` | v13 Python 测试源已在 v14 preflight 通过后删除；历史 26 项由 v13 source snapshot 保留。 | 非现行测试入口；残留缓存或空目录不构成可执行真源 |
| `tests/v2_r1r_v14/` | v14 Python 测试源已在 v15 活动入口清理时删除；历史 19 项由 v14 source snapshot 保留。 | 非现行测试入口；残留缓存或空目录不构成可执行真源 |
| `tests/v2_r1r_v15/` | v15 Python 测试源已在 v16 preflight 通过后删除；历史 19 项由 v15 source snapshot 保留。 | 非现行测试入口；残留缓存不构成可执行真源 |
| `tests/v2_r1r_v16/` / `tests/v2_r1r_v17/` | 活动 Python 测试源已在 P0-M 直接切换时删除；历史测试由对应 source snapshot 保留。 | 非现行测试入口；残留缓存不构成可执行真源 |
| `tests/v2_r1r_p0m_v1/` 至 `tests/v2_r1r_p0m_v4/` | 旧活动 Python 测试源均在下一版本直接切换后删除。 | 非现行测试入口；历史 source snapshot 与失败 artifact 保留 |
| `artifacts/v2-r1r/p0m-v5-*/source_snapshot/` | 已删除的 P0-M v5 活动实现与 9 项历史预测试源码。 | 只读复现证据；不得恢复成并行活动入口 |
| `tests/v2_r1r_p1_v2/` | P1 v2 powered contract、label-blind/pair-complete subset、cache/loader、模型干预、Gate、baseline 与 CLI 预测试。 | 13 项目标测试与 107 项全仓测试通过；不覆盖正式 cache telemetry FAIL |
| `artifacts/v2-r1r/p1-v2-{preflight,g09-power,data,cache}-20260810-1/` | P1 v2 唯一正式 roots；前三者 sealed PASS，cache 为完整 banks + sealed `EINVAL` FAIL。 | fixed roots 已消耗，禁止修改、覆盖或重跑；K=8 等后序 roots 不存在 |
| `tests/v2_r1r_p1_v5_transfer/` | v5 活动 Python 测试已在 v6 直接切换时删除。 | 历史测试仅保留在 v5 sealed source snapshot |
| `tests/v2_r1r_p1_v6_ctw/` | v6 活动 Python 测试已在 v6R 直接切换时删除。 | 历史测试仅保留在 v6 sealed source snapshot |
| `tests/v2_r1r_p1_v6r_selection_recovery/` | 已在 v7 直接切换时删除；历史 selection recovery 测试由 v6R sealed source snapshot 保留。 | 非现行测试入口，不得恢复 compatibility surface |
| `tests/v2_r1r_p1_v7_integrated_k8/` | 已在 v8 直接切换时删除；历史 9 项测试由 v7 sealed source snapshot 保留。 | 非现行入口，不得恢复 compatibility surface |
| `tests/v2_r1r_p1_v8_causal_bridge/` | v8 活动测试已在 v8R 直接切换时删除；历史 14 项由 v8 sealed source snapshot 保留。 | 非现行入口，不得恢复 compatibility surface |
| `tests/v2_r1r_p1_v8r_causal_curriculum/` | v8R 活动测试已在 v8D 直接切换时删除；历史测试由 v8R sealed source snapshot 保留。 | 非现行入口，不得恢复 compatibility surface |
| `tests/v2_r1r_p1_v8d_decision_witness/` | v8D 活动测试已在 v8L direct switch 时删除；历史测试由 v8D sealed source snapshot 保留。 | 非现行入口，不得恢复 compatibility surface |
| `tests/v2_r1r_p1_v8l_causal_state_ladder/` | 活动 Python 测试已在 NR1 direct switch 时删除；历史 12 项由 V8L sealed source snapshot 保留。 | 非现行入口；残留 `__pycache__` 不构成可执行真源 |
| `tests/v2_r1r_p1_nr1/` | NR1 严格 schema、手工 fixture、独立 numeric/relation 实现、范围外 holdout、复杂 DAG、metamorphic、11 fault ledger 与 runner 单元测试。 | accepted measurement 的自包含回归；依赖已冷归档 V8L/root/CLI 现场状态的旧测试已删除，NR1 formal 不得重跑 |
| `src/yggdrasil_v2/r1_revalidation/h1/` / `tests/v2_r1r_p1_h1/` | H1 fresh/full-semantic data、七域全局确定性去重、contextual-token cache、公共 recurrent attention/FFN + 共享 feature trunk + matched generic/routed final projection、训练/metrics、含 conditional-write necessity 的 screen/calibration Gate、architecture/data/cache audits、H01–H08、artifacts、runner/workflow 与 fail-stop 测试。 | 61 项自包含 H1 测试；依赖冷归档 history replay 的旧断言已删除；factorized screen 已正常运行但三项方向 Gate FAIL，当前实现不得 calibration/formal |
| `artifacts/v2-r1r/p1-v3-{preflight,telemetry-qualification,cache-recovery,k8}-20260811-1/` | P1 v3 唯一 roots；前三者 sealed PASS，K=8 为 `FAIL_P1_K8`，四个 evidence seal 均可复算。 | fixed roots 已消耗，禁止修改、覆盖或重跑；K=1/direct/text-CoT/assessment roots 不存在 |
| `artifacts/v2-r1r/p1-v4-lq-{preflight,bootstrap128,scale512}-20260811-1/` | P1 v4-LQ 已创建 roots；前两者 sealed PASS，S512 sealed `FAIL_P1_LQ_SCALE512`。 | fixed roots 已消耗，禁止修改、覆盖或重跑；S2048/F8192/assessment 不存在 |
| `artifacts/v2-r1r/p1-v5-transfer-preflight-20260811-1/` | P1 v5 preflight：旧来源 hash/seal、合同 hash、12 项预测试、8 GiB GPU、batch32 真实反传与 root absence 全通过。 | sealed PASS；evidence-seal 文件 SHA-256 `C50DD3…85A1B`，fixed root 已消耗 |
| `artifacts/v2-r1r/p1-v5-transfer-qualification7168-20260811-1/` | P1 v5 唯一 Q7168：train answer `1.0/1.0`，audit claim `0.5/0.5`、state drop 近零。 | sealed `FAIL_P1_V5_QUALIFICATION7168`；seal 文件 SHA-256 `599E1A…C591A`，禁止修改或重跑 |
| `artifacts/v2-r1r/p1-v5-transfer-{full8192,assessment}-20260811-1/` | P1 v5 失败后的后序 roots。 | 按合同不存在，禁止补跑 |
| `artifacts/v2-r1r/p1-v6-ctw-{preflight,witness-audit,query-cache,mechanism-compare,assessment}-20260811-1/` | P1 v6 五个 formal roots；前四 PASS，assessment 仅 W604 false。 | 全部 sealed 且不可修改/重跑；temporal mechanism 证据有效，形式资格由 v6R 恢复 |
| `artifacts/v2-r1r/p1-v6r-selection-recovery-{preflight,assessment}-20260811-1/` | v6R 两个 formal roots；R601–R610、query 18,899 content reaudit 与三 mutation 负控。 | sealed PASS；assessment seal SHA-256 `B191B4…E5F98`，只授权另立 integrated P1 |
| `artifacts/v2-r1r/p1-v7-integrated-{preflight,query-cache,k8}-20260811-1/` | v7 三个 formal roots；preflight/query-cache PASS，K8 为 `FAIL_P1_V7_INTEGRATED_K8`。 | seals 均已复算；fixed roots 已消耗且禁止修改/重跑，K=1/baselines/P2 roots 不存在 |
| `artifacts/v2-r1r/p1-v8-causal-bridge-{preflight,query-cache,qualification}-20260811-1/` | v8 三个 fixed roots；前两根 PASS，qualification 为 `FAIL_P1_V8_CAUSAL_BRIDGE`。 | seal 均已复算；禁止修改、覆盖或重跑，未授权 v9/P2 |
| `artifacts/v2-r1r/p1-v8r-causal-curriculum-{preflight,qualification}-20260811-1/` | v8R 两个 fixed roots；preflight PASS，qualification 为 `FAIL_P1_V8R_CAUSAL_CURRICULUM`。 | seal 分别 `6891D7…6690D`、`806085…E0DE4`，禁止修改、覆盖或重跑 |
| `artifacts/v2-r1r/p1-v8d-decision-witness-{preflight,query-cache,qualification}-20260811-1/` | v8D 三个 formal roots；前两根 PASS，qualification 为 `FAIL_P1_V8D_CAUSAL_DECISION_WITNESS`。 | seals 分别 `2A2009…18E3`、`EFA106…EFFE`、`674AC6…E047`，禁止修改、覆盖或重跑 |
| `artifacts/v2-r1r/p1-v8l-causal-state-ladder-{preflight,anchor-cache,qualification}-20260812-1/` | v8L 三个 formal roots；preflight/anchor-cache PASS，qualification 为 `FAIL_P1_V8L_BOOTSTRAP`。 | seals 为 `06BA4E…5115`、`5F1AAB…3C14`、`3332CD…FE75`；禁止修改、覆盖或重跑 |
| `artifacts/v2-r1r/p1-nr1-{preflight,qualification}-20260817-1/` / `tmp/p1-nr1-transport-20260817-1/` | NR1 已消耗的两根 fixed roots 与 launcher transport；preflight/qualification 均 PASS。 | seals 为 `1825282B…95DAF`、`DADBDDBF…F6FB3`，禁止修改、覆盖或重跑；只授权 H1 设计 |
| `artifacts/v2-r1r/p1-h1-{preflight,development}-20260817-1/` / `tmp/p1-h1-transport-20260817-1/` | H1 预注册 fixed paths；正式时依次保存 23,456-record fresh data/cache preflight、800-update smoke 与三 seed 双臂 development comparison。 | 当前已复核不存在；factorized screen FAIL 后 calibration/合同 freeze 未授权，launcher 必须继续禁止创建 |
| `artifacts/v2-r1r/p1-h1-nonformal-calibration-20260817-{1,2}/` | H1 完整-transition experts 的开发校准；`-2` 完整结果总增益仅 `+0.00684`，结构已否决。 | 非 formal，不消耗 fixed roots，不得提升为 H1 证据或用于冻结当前结构阈值 |
| `artifacts/v2-r1r/p1-h1-nonformal-ffn-moe-screen-20260817-{1,2}/` | 互斥完整 FFN-MoE 的开发筛选；`-1` 因实现信号错误主动终止，`-2` 正常结束但总增益 `+0.03027`、relation `-0.06055`、primary Gate false。 | 非 formal；失败证据保留用于结构归因，不得重用为 shared+routed residual 的资格结果 |
| `artifacts/v2-r1r/p1-h1-nonformal-shared-routed-screen-20260817-{1,2}/` | 2:1 shared+routed residual 的开发筛选；`-1` 因旧 cache 与当前 fingerprint 真源差 1 条 validation record，在训练前 fail closed；`-2` 用 fresh cache 正常结束，H05 `+0.12695`/CI 下界 `+0.09375`，但 H06 最大 route effect `0.08887<0.10`。 | 非 formal；只证明公共 FFN 恢复净收益，方向仍失败，不授权新 seed 校准、H1/F1/P1/P2 |
| `artifacts/v2-r1r/p1-h1-nonformal-balanced-routed-screen-20260817-1/` | 等分 `H=384+384` shared+routed residual 的非正式 2,400-update screen；复用 current-fingerprint cache，heldout 增益 `+0.13867`、CI `[0.10352,0.17285]`，numeric/relation `+0.26172/+0.01563`，最大 route answer drop `0.13672`。 | 当时 H05/H06 方向 Gate 通过并只授权全新 seed/full-budget calibration；后者未复现，screen 不构成稳定机制或 formal 证据 |
| `artifacts/v2-r1r/p1-h1-nonformal-balanced-routed-calibration-20260817-1/` | 等分结构预登记 `2026081793/2026081794` 独立 4,000-update calibration；fresh 7,808-record cache，overall gain `-0.00098`、CI `[-0.03613,0.03516]`、relation `-0.03320`、最大 route effect `0.03845`；含 `CALIBRATION_REVIEW.md`。 | 正常完成但 H05/H06 FAIL，`authorizes=nothing`；threshold proposal 无效，禁止重跑、降门槛、冻结合同或启动 formal |
| `artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/` | 预登记 `2026081761/2026081762`、package identity `A52C5222…486DA4` 的 factorized projection full-budget screen；两臂各 4,000 updates，overall gain `-0.01074`，wrong-route/conditional-write 最大效应 `0.00281/0.00781`；含 `SCREEN_REVIEW.md`，`probe-result.json` SHA-256 `12D609…B1544`。 | 正常完成但 H05/H06 三门 FAIL，`passed=false`、`authorizes=nothing`；不得重跑、换 seed、降低 Gate、启动 calibration/formal |
| `artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-calibration-20260817-1/` | 预登记 `2026081763/2026081764`、package identity `21CBD922…E761E9` 的独立 full-budget calibration root。 | screen FAIL 后按合同保持不存在；禁止补跑 |
| `artifacts/v2-r1r/p1-h1-wd-decision-causal-screen-20260821-1/` | decision-causal 非正式 screen 的隔离 root；从原 mixed deployment 物化 margin-VJP/Fisher train/heldout target，正/负 VJP W 各完成 800 updates。 | 已消耗；`FAIL_WD_DECISION_CAUSAL_WRITE_FIT`、`authorizes=nothing`，无 checkpoint/D/J，禁止删除后重跑 |
| `docs/v2-r1r-p1-v3-cache-recovery-design.md` / `docs/v2-r1r-p1-v3-cache-recovery-execution-command.md` | P1 v3 frozen recovery/训练设计与 single-use 执行顺序。 | 已执行结束，只作合同追溯；不得重跑 |
| `docs/v2-r1r-p1-v3-k8-failure-review.md` | 父任务对 recovery/K=8 artifact、训练预算、学习曲线、K08 误报和后继路线的独立复核。 | 当前 P1 判决真源；P2 未授权 |
| `docs/v2-r1r-p1-v4-learning-qualification-design.md` / `docs/v2-r1r-p1-v4-learning-qualification-execution-command.md` | P1 v4-LQ frozen 设计与已消耗 single-use 命令。 | 历史合同；不得续跑或恢复 owner-contrast |
| `docs/v2-r1r-p1-v4-lq-failure-review.md` | v4 sealed 结果、未见 episode 诊断与目标根因。 | 当前 v4 判决真源 |
| `docs/v2-r1r-p1-v5-semantic-transfer-design.md` / `docs/v2-r1r-p1-v5-semantic-transfer-execution-command.md` / `docs/v2-r1r-p1-v5-semantic-transfer-failure-review.md` | P1 v5 frozen 设计、已消耗命令与失败复盘。 | 历史正式失败；禁止续跑 |
| `docs/v2-r1r-p1-v6-causal-temporal-witness-design.md` / `docs/v2-r1r-p1-v6-causal-temporal-witness-execution-command.md` | P1 v6 frozen 设计与已消耗 single-use 命令。 | 历史正式机制证据；原 assessment FAIL 不改写 |
| `docs/v2-r1r-p1-v6r-selection-recovery-design.md` / `docs/v2-r1r-p1-v6r-selection-recovery-main-review.md` | v6R 恢复合同与 accepted 判决。 | 历史机制资格真源；已由 v7 验证其规模迁移边界，P2 未授权 |
| `docs/v2-r1r-p1-v7-integrated-k8-design.md` / `docs/v2-r1r-p1-v7-integrated-k8-execution-command.md` / `docs/v2-r1r-p1-v7-integrated-k8-failure-review.md` | v7 已消耗合同、命令与失败复核。 | 历史正式 FAIL；接受 temporal/state 机制，拒绝 CPS causal/OOD answer，禁止重跑 |
| `docs/v2-r1r-p1-v8-causal-bridge-design.md` / `docs/v2-r1r-p1-v8-causal-bridge-execution-command.md` / `docs/v2-r1r-p1-v8-causal-bridge-failure-review.md` | v8 已消耗设计、命令与失败复核。 | 历史正式 FAIL；接受训练内 pair decision power，拒绝 scratch 迁移资格，禁止重跑 |
| `docs/v2-r1r-p1-v8r-causal-curriculum-design.md` / `docs/v2-r1r-p1-v8r-causal-curriculum-execution-command.md` / `docs/v2-r1r-p1-v8r-causal-curriculum-failure-review.md` | v8R 已消耗设计、命令与失败复核。 | 历史正式 FAIL；排除 scratch/coverage，拒绝可分解 answer-pair 目标，禁止重跑 |
| `docs/v2-r1r-p1-v8d-causal-decision-witness-design.md` / `docs/v2-r1r-p1-v8d-causal-decision-witness-execution-command.md` / `docs/v2-r1r-p1-v8d-causal-decision-witness-failure-review.md` | v8D 已消耗设计、命令与失败复核。 | 历史正式 FAIL；排除 source blindness，拒绝 learnable final-state probe 路径，禁止重跑 |
| `docs/v2-r1r-p1-v8l-causal-state-ladder-design.md` / `docs/v2-r1r-p1-v8l-causal-state-ladder-execution-command.md` / `docs/v2-r1r-p1-v8l-causal-state-ladder-failure-review.md` | v8L 已消耗设计、命令与失败复核。 | 历史正式 bootstrap FAIL；关闭匿名 K=8 + lexical-anchor 路径，禁止重跑 |
| `docs/v2-r1r-p1-nr1-numeric-relation-measurement-design.md` / `docs/v2-r1r-p1-nr1-execution-command.md` | NR1 新路线冻结设计与唯一 single-use 命令。 | 已消耗历史合同；PASS 只授权 H1 设计，不完成 P1/P2 |
| `docs/v2-r1r-p1-h1-mixed-core-design.md` / `docs/v2-r1r-p1-h1-execution-command.md` | H1 架构、完整监督语义去重数据、teacher、两臂、公平预算、smoke、H01–H08、before-mutation readiness 与 single-use 顺序。 | 当前 factorized 开发方向已 screen FAIL；calibration/hash freeze/formal 未授权 |
| `docs/v2-r1r-p0-result.md` | P0-D v1–v17、P0-M v1–v5、P1 v1–v8L、NR1 与 H1 非正式开发判决。 | 当前阶段总状态真源；P1/P2 未完成 |
| `docs/v2-r1r-p0-v2-main-review.md` | v2 formal artifact、当前代码和第 16 节的独立验收。 | 当前主设计层判决真源 |
| `docs/v2-r1r-p0-v3-main-review.md` | v3 artifact、当前代码、测试和 D2 snapshot 旧合同的独立验收。 | 当前主设计层判决真源 |
| `docs/v2-r1r-p0-v4-main-review.md` | v4 partial artifact、reference builder、G07/G08、预测试与 F401 的独立验收。 | v4 最终主设计层判决真源 |
| `docs/v2-r1r-p0-v5-r0a-main-review.md` | v5 sealed artifact、公共 simulator、六项 malformed-AST 黑盒探针和执行偏差的独立验收。 | v5 R0A 最终主设计层判决真源；R0B 未授权 |
| `docs/v2-r1r-p0d-v4-r0-execution-command.md` | v4 R0 历史执行合同。 | 已执行一次；结果 `FAIL_R0`，不可重跑 |
| `docs/v2-r1r-p0d-v5-r0a-execution-command.md` | v5 R0A 的 frozen hash、直接删除范围、三个 simulator API、预测试、唯一 sealed run、停机与父任务回传协议。 | 已执行并结束；只作历史追溯，禁止重跑 |
| `docs/v2-r1r-p0d-v6-r0a-strict-design.md` | v6 完整 AST schema、245 probe matrix、eager validation、精确错误、fixed root 与 PASS conjunction。 | 历史冻结设计；v6 判决追溯依据 |
| `docs/v2-r1r-p0d-v6-r0a-strict-execution-command.md` | v6 四文件权限、v5 tests 删除、preflight、一次 formal 与唤醒协议。 | 已执行结束；只作追溯，R0B 未授权 |
| `docs/v2-r1r-p0-v6-r0a-strict-main-review.md` | v6 formal artifact、实现、冻结矩阵与十项新鲜 query-placeholder 探针复核。 | 最终判定 machine-pass、main-review rejected |
| `artifacts/v2-r1r/p0d-v6-r0a-strict-20260801-1/` | v6 唯一 formal root，精确六个 sealed JSON；machine 14/4/12/245 与 evidence hash 全通过。 | 窄机器证据；主审 rejected，不可修改、覆盖或重跑 |
| `docs/v2-r1r-p0d-v7-r0a-lattice-design.md` | v7 normative token scope、858-cell lattice、coverage 正负控制、七文件 artifact 与 PASS conjunction。 | 历史 accepted R0A 设计依据；不可改写 |
| `docs/v2-r1r-p0d-v7-r0a-lattice-execution-command.md` | v7 精确权限、hash guard、四文件切换、preflight、一次 formal 与唤醒协议。 | 已执行结束；禁止重跑 |
| `docs/v2-r1r-p0-v7-r0a-lattice-main-review.md` | v7 machine、artifact 与 registry 外探针复核。 | R0A-lattice 最终 accepted 判决真源 |
| `artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1/` | v7 唯一 formal root，精确七文件；machine 与 evidence seal 全通过。 | accepted R0A 窄证据；不可修改、覆盖或重跑 |
| `docs/v2-r1r-p0d-v8-r0b-structural-design.md` | G02–G06 qualification bundle、raw metrics、fault/metamorphic 与 formal conjunction。 | v8 历史冻结设计；不可改写 |
| `docs/v2-r1r-p0d-v8-r0b-structural-execution-command.md` | v8 精确权限、实现分工、预测试、single formal 与唤醒协议。 | 已执行结束；禁止重跑 |
| `docs/v2-r1r-p0-v8-r0b-structural-main-review.md` | v8 formal artifact、公共 audit 与五项 registry 外反例的独立验收。 | 最终判定 machine-pass、main-review rejected；由 v9 直接替换 active implementation |
| `artifacts/v2-r1r/p0d-v8-r0b-structural-20260801-1/` | v8 唯一 formal root，精确八个顶层 entry；机器正控 5/5、fault 41/41、metamorphic 4/4、metric kill 37/37。 | 窄机器证据；主审 rejected，不可修改、覆盖或重跑 |
| `docs/v2-r1r-p0d-v9-r0b-invariant-design.md` | 固定 profile、exact schemas、typed derivation、四种可逆 language grammar、45 metrics、48 adversary 与 6 metamorphic 的 PASS conjunction。 | 当前 accepted R0B 设计依据；不可改写 |
| `docs/v2-r1r-p0d-v9-r0b-invariant-execution-command.md` | v9 精确权限、直接切换范围、预测试、单次 formal、停机与回传协议。 | 已执行结束；禁止重跑 |
| `docs/v2-r1r-p0-v9-r0b-invariant-main-review.md` | v9 formal、evidence seal 与六项 registry 外探针复核。 | 有限 R0B measurement-system qualification 最终 accepted 判决真源 |
| `artifacts/v2-r1r/p0d-v9-r0b-invariant-20260802-1/` | v9 唯一 formal root，精确九项顶层 entry；G02–G06 5/5、adversary 48/48、metric kill 45/45、metamorphic 6/6，45 个递归 evidence hash。 | accepted R0B 窄测量证据；seal SHA-256 `94B80FBD2BB38BC46E95F88E0747D9DB1D8CB279E689F68BB4A43AE7B76D8209`；不可修改、覆盖或重跑 |
| `docs/v2-r1r-p0d-v10-r0c-statistical-design.md` | source/numeric profile、算法公式、34 metrics、39 adversary、5 metamorphic、语义摘要和 formal conjunction。 | 当前 accepted R0C 设计依据；不可改写 |
| `docs/v2-r1r-p0d-v10-r0c-statistical-execution-command.md` | v10 精确权限、prior guard、直接切换、预测试、单次 formal 与停机协议。 | 已执行结束；禁止重跑 |
| `docs/v2-r1r-p0-v10-r0c-statistical-main-review.md` | v10 formal、双层 seal、外部探针、异常复核与证据边界。 | 有限 R0C measurement-component qualification 最终 accepted 判决真源 |
| `artifacts/v2-r1r/p0d-v10-r0c-statistical-20260802-1/` | v10 唯一 formal root，精确九项顶层 entry；G07/G08 2/2、adversary 39/39、metric kill 34/34、metamorphic 5/5，52 个递归 evidence hash。 | accepted R0C 窄测量证据；seal SHA-256 `3F003C8BED40F54F010F4C7A96207D703ABA59F1088F189C1265D0F9E1C79BA1`；不可修改、覆盖或重跑 |
| `docs/v2-r1r-p0d-v11-r0d-integrated-design.md` | 12-case semantic/surface interface、G01–G08、19 metrics、20 fault、4 metamorphic、import/replay 与 formal conjunction。 | 当前 accepted R0D 设计依据；不可改写 |
| `docs/v2-r1r-p0d-v11-r0d-integrated-execution-command.md` | v11 精确权限、直接切换、prior/frozen guard、预测试、唯一 formal 与停机协议。 | 已执行结束；禁止重跑 |
| `docs/v2-r1r-p0-v11-r0d-integrated-main-review.md` | v11 formal、传递 seal、跨 Gate fault、外部探针、长度边界和后继条件。 | 有限 integrated measurement-system qualification 最终 accepted 判决真源 |
| `artifacts/v2-r1r/p0d-v11-r0d-integrated-20260802-1/` | v11 唯一 formal root，精确十项；G01–G08 8/8、fault 20/20、metric kill 19/19、metamorphic 4/4、replay 1/1，144 文件直接或传递覆盖。 | accepted R0D 窄集成测量证据；seal SHA-256 `1C4AD436CECA3F96F4286A2E5A62D272782E7EF02DF5A52BBD5C6493A43408CF`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/r1e-v12-entry-qualification-20260809-1/` | v12 唯一 formal root，精确十二项；G01–G08、15 metrics、29 adversary、20 metamorphic 与 replay 全通过，57 个非 root-seal 文件全量封印。 | accepted production renderer/scalable learner 入口证据；seal SHA-256 `86B49779D4E5A7EF496EE6FC0C0C722071983E5B3C6FFB9F8D90D92FBB9E78CA`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/r1g-v13-generator-smoke-20260809-1/` | v13 唯一 formal root；14 split、1,440 records、72 causal pairs、16,848 claims，G01–G11 全 true，双生成/报告与 artifact replay 一致。 | accepted fixed-seed R1G smoke；62 个非 root-seal 文件，seal SHA-256 `A9CF968F15634E7F1F9471CDE5383736C6D0E17DCC8208E871B6C779062AE455`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0d-v14-full-production-20260809-1/` | v14 唯一 formal root；14,336 records、512 causal pairs、167,580 claims，G01–G08/G10/G11 true、G09 false，regeneration/replay 一致。 | rejected full-production diagnostic；seal SHA-256 `6FD9AB9CD3F7CB1793F959762BA07AF0A0931052E7A3D05E3DD2C35455D19F52`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0d-v15-g09-qualification-20260810-1/` | v15 唯一 Q root；100,000-trial decision power、14 fault、3,264 exact 对照、single-bank/progress 与 replay，Q01–Q07/Q09 true、Q08 false。 | rejected G09 qualification；seal SHA-256 `997166EECC490F7B6AF6E80C01E28E187539BEDA351764597F0D79DE42C1EE38`；fresh F root 未创建，不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0d-v16-runtime-qualification-20260810-1/` | v16 唯一 Q root；15 轮 paired、14,336-record G09/G10、18 fault、3.46 GB peak 与 Q01–Q09 全 true。 | accepted runtime measurement component；seal SHA-256 `EFBF833D5E55321012C98E6FF67C6B19E509687360E61588D4E34063B958513E`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0d-v16-full-production-20260810-1/` | v16 唯一 fresh F root；26,624 records、311,636 claims、G01–G04/G06/G08–G11 true，G05/G07 false，regeneration/replay 一致。 | rejected full-production diagnostic；seal SHA-256 `2CE3CBCDB720FA777C1E357026209108D3B57C444E751FD2689ED9C6B6AB74E7`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0d-v17-repair-qualification-20260810-1/` | v17 唯一 repair root；50,000-seed ERE sweep、13,312-fingerprint capacity、307 条回归、58 collision/fault-kill 与 R01–R07 全 true。 | accepted repair qualification；seal SHA-256 `6A31B0218F2B46979714A1D8F296BBE085E8FE9F0FF6E9B10476904C3885D103`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0d-v17-full-production-20260810-1/` | v17 唯一 fresh production root；seed 2026081702、26,624 records、1,536 causal pairs、G01–G11 true、regeneration/replay 一致。 | accepted P0-D production evidence；seal SHA-256 `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0m-v1-*` 至 `artifacts/v2-r1r/p0m-v4-*` | cache 判据、BF16/baseline preflight、claim compatibility 与 owner/监督时序的逐版失败或窄 PASS roots。 | P0-M 失败—修复诊断；各 root 已封存，不得覆盖或提升为 v5/P1 证据 |
| `artifacts/v2-r1r/p0m-v5-cache-qualification-20260810-1/` | exact Qwen/cache 内容、禁字段/截断、batch8 吞吐与 ≤6 GiB qualification。 | M01 accepted；seal SHA-256 `F5FEF16EE4D1AFC3BDBB7779E681A806B289DF9B31781B995C6712567134FF9D` |
| `artifacts/v2-r1r/p0m-v5-ere-k8-overfit64-20260810-1/` / `p0m-v5-cps-k8-overfit64-20260810-1/` | 单任务 K=8 update1200；answers 各 1.0，claims `0.99740/0.99861`，owner-shuffle drop `0.36589/0.34583`，剥离复载相同。 | M02/M03 accepted；seals `DAB1…4E35` / `5EC4…390F` |
| `artifacts/v2-r1r/p0m-v5-joint-k8-overfit128-20260810-1/` | 唯一 shared K=8 joint；ERE/CPS answer 1.0/1.0、claim 0.90659、owner-shuffle drop 0.37231、剥离复载相同。 | M04 accepted；seal SHA-256 `91921CD18A245D34CB1DF3CBEE94706FA11C01E3B1735513E788AD7A2AEB341B` |
| `artifacts/v2-r1r/p0m-v5-direct-overfit64-20260810-1/` / `p0m-v5-text-cot-overfit64-20260810-1/` | 同 64 episodes、公平 top-4 LoRA；两者 greedy answer 都为 1.0，64/64 可解析。 | M05/M06 accepted；seals `E357…1AAE` / `2F25…D618` |
| `artifacts/v2-r1r/p0m-v5-throughput-20260810-1/` | stripped joint checkpoint 连续 100 optimizer steps；71.706 examples/s、finite、无 OOM/fallback。 | M07 accepted；seal SHA-256 `30BD34FF6C11069E3FF854E7F7D5E9ED4D3A350817A5E7A731C741162E897684` |
| `artifacts/v2-r1r/p0m-v5-assessment-20260810-1/` | 八 root conjunction；M01–M08 true，`PASS_P0M`、`p1_eligible=true`、`p1_started=false`。 | P0-M accepted smoke；seal SHA-256 `17F45EAC8A3DF129B236688D3D1E2B22BD9B634D4098F903142CA42E7FEEF09E` |
| `artifacts/v2-r1r/p1-v1-preflight-20260810-1/` | P1 v1 项目 Python、CUDA/GPU、磁盘、前提 hash 与 fixed-root absence 检查。 | preflight PASS；seal SHA-256 `941A985A475F1E08DD793C05F02C37B99E4625C2895B5204FE0E0B553156263D` |
| `artifacts/v2-r1r/p1-v1-data-20260810-1/` | seed 2026081901 的 28,672-record fresh data；G01–G08/G10/G11 true、G09 false，全量 regeneration/replay 相等。 | `FAIL_P1_DATA`；seal SHA-256 `51D57EB3083A5BDF25C42C87A40B3C9BE642B57DEE95B1D2C0D159DA0478B94B`；不可修改、覆盖或重跑，cache 未授权 |
| `artifacts/v2-r1r/p0d-v5-r0a-20260801-1/` | v5 R0A 五文件 sealed artifact；机器 14/14 canonical、4/4 prefix、12/12 negative，evidence hash 可复算。 | machine-pass、main-review rejected diagnostic；不可修改、覆盖或重跑 |
| `artifacts/v2-r1r/p0d-v4-r0-20260801-1/` | v4 唯一 partial sealed root：reference data、manifest、input-seal、source snapshot 和通过的 audit-core；fault-matrix/assessment/evidence-seal/replay 未完成。 | `FAIL_R0`；保留失败证据，禁止修复/覆盖/重跑 |
| `artifacts/v2-r1r/p0-preflight-v3-20260801-1/source_snapshot/docs/v2-r1r-p0d-v3-execution-command.md` | v3 D0–D2 历史命令的运行时精确副本。 | rejected artifact 内历史证据；禁止再次执行 |
| `docs/v2-r1r-p0-main-review.md` | v1 主设计层复核和 v2 修订依据。 | 当前复核结论 |
| `artifacts/v2-r1r/p0-v1/` | v1 formal ERE/CPS JSONL、manifest、audit、heuristics 和 assessment。 | 旧机器 10/10；主设计层 rejected |

## V2-R1R P0-D 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-r1r/p0-v1/data/ere/` | v1 ERE train、validation、OOD 与 causal pairs JSONL。 | rejected diagnostic；含 seed-literal 满分捷径 |
| `artifacts/v2-r1r/p0-v1/data/cps/` | v1 CPS train、validation、OOD 与 causal pairs JSONL。 | rejected diagnostic；含 candidate 位置满分捷径 |
| `artifacts/v2-r1r/p0-v1/manifest.json` | v1 generator/version/seed、旧设计 hash、环境、命令、时间和文件 SHA-256。 | v1 provenance 记录 |
| `artifacts/v2-r1r/p0-v1/audit.json` | 旧十项 P0-D 机器审计明细。 | 自判通过但规格覆盖不足 |
| `artifacts/v2-r1r/p0-v1/heuristics.json` | 旧 unigram、claim 数量和标签/选择平衡统计。 | 漏检结构化捷径与 claim 真值 |
| `artifacts/v2-r1r/p0-v1/p0-assessment.json` | 旧 `passed` conjunction。 | 文件值 `true`，路线判定 rejected |
| `artifacts/v2-r1r/p0-smoke-v1/` | 每族 64 train、32 validation/OOD 的 smoke 数据与审计。 | smoke，不是 formal |
| `artifacts/v2-r1r/p0-v1-failed-*/` | 三次被保留的 formal 失败重试及其修复前证据。 | 诊断归档，不是最终结果 |
| `artifacts/v2-r1r/p0-smoke-v2/` | 按 generator v2 生成的 ERE/CPS smoke 数据、manifest、heuristics 和 audit。 | smoke；13/13 true，非 formal |
| `artifacts/v2-r1r/p0-smoke-v2-audit-20260801-5/` | smoke 的历史审计输出副本。 | 失败/中间诊断，保留追溯 |
| `artifacts/v2-r1r/p0-smoke-v2-failed-audit-20260801*/` | generator v2 多次 smoke 失败审计尝试。 | 失败诊断；不作为 formal 证据 |
| `artifacts/v2-r1r/p0-v2-failed-generation-20260801-1/` | formal 第一次生成在 ERE train record 96 因唯一性耗尽停止的证据。 | failed generation diagnostic |
| `artifacts/v2-r1r/p0-v2-failed-generation-20260801-2/` | formal 第二次生成在 ERE validation record 0 因跨 split semantic overlap 停止的证据。 | failed generation diagnostic |
| `artifacts/v2-r1r/p0-v2-failed-audit-20260801-1/` | formal 完整 14,336 records、manifest、audit、heuristics 和 assessment；13 项仅 10 项通过，主设计层另发现未捕获的合同失败。 | v2 machine failure diagnostic；禁止 P0-M/训练 |
| `artifacts/v2-r1r/p0-smoke-v3-20260801-2/` 至 `-9/` | v3 D1 smoke 的失败/中间证据；`-1` 未形成目录，每次实际重跑均未覆盖旧 artifact。 | 失败诊断；不作 formal |
| `artifacts/v2-r1r/p0-smoke-v3-20260801-10/` | v3 最后一次 D1 smoke；两族 train 128、validation/OOD 64、causal 64，`smoke_passed=true`。 | D1 functional smoke；distribution 不作 formal |
| `artifacts/v2-r1r/p0-preflight-v3-20260801-1/` | v3 唯一 D2 preflight；两族 train 1024、validation/OOD 256、causal 256，含 manifest、source snapshot、audit 和 assessment。 | `FAIL_D2` 且主设计层 rejected；禁止 D3/formal/P0-M |

## 2026-08-01 artifacts 清理与外部归档

2026-08-01 当时只保留仍有明确运行职责的 `artifacts/v2-a/a1_20d/`（V2-R1R 机制正控制）和完整的 `artifacts/v2-r1r/`（包含 v1 rejected diagnostic 与 v2 smoke 失败证据）。其余 V2-A 旧路线、失败路径和中间缓存没有被当作当前运行输入，统一镜像移动到 `D:\归档\onmi-yggdrasil-test-artifacts-2026-08-01\artifacts\v2-a\`；该目录保留原有相对结构，可按原阶段名追溯，不再是本仓库的可运行入口。

归档的 V2-A 阶段包括 `a0`、`data`、`a1`、`a2`、`a1_5`、`a1_6`、`a1_7`、`a1_8`、`a1_9`、`a1_10`、`a1_11`、`a1_12`、`a1_13`、`a1_13f`、`a1_15`、`a1_16`、`a1_17`、`a1_18`、`a1_18b`、`a1_19h`、`a1_20b`、`a1_20c` 和 `a1_21p`。五个已有最终 JSON 的 `.partial.json` 进度快照已从归档中删除；正式 JSON、checkpoint、缓存和数据集均保留。旧文档中的这些相对路径仍表示历史证据，不表示文件仍位于仓库内。

## 2026-08-24 活动工作树收口

Closure C0R 已 PASS 并只授权 fresh C1；C1 明确禁止读取旧 checkpoint，V2-R1R/H1/WD 也已被 Closure 覆盖且全部后继 `authorizes=nothing`。因此本轮没有拆散或删除 sealed roots，而是把 `artifacts/v2-r1r/`、`artifacts/v2-a/a1_20d/`、`archive/legacy-proxy-route-2026-07-11/`、旧 `tmp/p1-*-transport*/`、旧 `tmp/V2-A*result.md` 与根 `tmp.md` 按原相对树统一收进：

- 当前人工转移暂存包：`C:\Users\24408\Documents\onmi-yggdrasil-test-cleanup-2026-08-24\`
- 用户整包移动后的预定位置：`D:\归档\onmi-yggdrasil-test-cleanup-2026-08-24\`
- 包内说明：`MOVE_TO_D_MANIFEST.md`
- 归档数据（说明文件加入前）：5,386 files，155,605,463,533 bytes（约 144.9189 GiB），1,363 directories

本索引后文的 `artifacts/v2-r1r/...`、`artifacts/v2-a/a1_20d/...`、`archive/legacy-proxy-route-2026-07-11/...` 与上述旧 `tmp/...` 路径继续表示包内原相对路径，不表示文件仍位于活动仓库；唯一例外是 `artifacts/v2-r1r/p0d-v17-full-production-20260810-1/`，清理后回归确认 C0R compact-trace 测试仍直接读取其 dataset，因此已移回仓库作为当前测试夹具。当前工作树还保留 Closure C0/C0R 三个固定 artifact roots 及其 leases，以及最近的 C0R preflight sealed FAIL；这些是 C1 的直接资格与审计前提。

明确可再生或为空的 `.tmp-r0-*`、`.tmp-r1g-v13-debug*`、`.pytest_cache/`、`.ruff_cache/` 和工作树 Python `__pycache__/` 已集中到 `C:\Users\24408\Documents\onmi-yggdrasil-test-待删除-2026-08-24\`；连同回归验证重新生成后再次收走的缓存，共 1,027 files、242,590,104 bytes（约 0.2259 GiB）。Codex 未删除它们；用户确认后可整包删除。`.venv/` 因近期 C1 仍需运行而保留。

## 当前 V2-A 代码与测试

| 路径 | 用途 | 当前证据地位 |
| --- | --- | --- |
| `pyproject.toml` | 当前 V2-A Python 依赖、包布局和 pytest 配置；Torch CUDA 运行时由本机环境选择。 | 当前工程合同 |
| `uv.lock` | 当前 Python 依赖解析锁文件；不等同于 GPU 运行时证据。 | 环境复现辅助 |
| `experiments/v2_a_reasoning_medium.py` | 统一 CLI：数据生成、direct/answer-only/text-CoT baseline、latent training/resume。 | 当前 V2-A 运行入口 |
| `src/yggdrasil_v2/reasoning_medium/data.py` | `symbolic-state-machine.v4` 数据生成器、heldout split、依赖和泄漏 manifest。 | A0 数据合同 |
| `src/yggdrasil_v2/reasoning_medium/prompts.py` | prompt contract v3、三路/encoder prompt、可选 demonstrations、答案解析和语义终止。 | A0/A1 计量组件 |
| `src/yggdrasil_v2/reasoning_medium/model.py` | 冻结 Qwen 文本塔提取、latent queries、copied/MLP transitions、token-wise source adapter、source 初始化/逐步 reread、mean/flatten readout 和 no-bypass 检查。 | A1/A2 mechanism 与诊断实现 |
| `src/yggdrasil_v2/reasoning_medium/train.py` | hidden-cache v5（last hidden + 有效程序步骤 mask）、中间 state/query-state labels、可选 teacher/state 训练辅助、step-level self-critical verifier RL、fit/评估、干预、checkpoint/resume 和结果 JSON。 | A1/A2 probe 实现；verifier-RL 当前只形成失败 probe，不是 Gate 证据 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_data.py` | A1.5 v1 必要步骤数据、长度/组合 split、operation spans/masks、反事实检查和弱基线。 | A1.5 contract/mechanism |
| `src/yggdrasil_v2/reasoning_medium/a1_5_model.py` | P0 结构化显式 register-slot recurrent core 与 shared self/cross-attention + FFN transition。 | A1.5 P0 surrogate positive control |
| `src/yggdrasil_v2/reasoning_medium/a1_5_train.py` | P0 overfit/full training、逐步 state CE、checkpoint、弱基线和 no-op/delete/shuffle/truncate/T0/T1/disable-delta 干预。 | A1.5 P0 probe；composition 因果未通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_data.py` | A1.6 relation-state-machine 数据、字符 span、causal_core 必要性和 audit Gate。 | A1.6 数据合同；audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_core.py` | 独立 RelationAddressedCore/Transition；三路 role pointer、共享 operator 和 shared state head。 | A1.6 C0 实现；formal relation Gate 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_train.py` | C0 固定 loss、checkpoint、评测和 prefix/counterfactual 代码。 | 已实现；formal relation Gate 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_qwen.py` | 冻结 Qwen hidden span cache 与复用 C0 core 的边界 adapter。 | 已实现；C1 未启动 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_qwen_train.py` | C1 boundary/joint 训练入口和缓存 batch 处理。 | 已实现；C1 未启动 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_closure_diagnostic.py` | A1.6 只读 checkpoint 的 free/oracle-reset/hard-reembed/first-error 诊断。 | A1.6 closure 失败归因证据；hard re-embed 仅诊断 |
| `experiments/v2_a1_6_core.py` | A1.6 prepare/audit、C0 train/evaluate/intervene、C1 cache/train/evaluate CLI。 | A1.6 历史入口；停止在 formal relation Gate |
| `src/yggdrasil_v2/reasoning_medium/a1_7_data.py` | A1.7 12-combination universe、唯一 relation holdout、分层随机 sequence 与 causal necessary audit。 | A1.7 当前数据合同；audit Gate 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_core.py` | 默认 content-only slots、address-only register keys、唯一 shared transition、last-active shared state readout，以及显式 address-mixed 消融开关。 | A1.7 目标 core 与受控结构消融 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_train.py` | 可配置 stop-gradient prototype closure、trajectory/final/query metrics、identity/non-collapse、训练与 intervention。 | A1.7 formal、causal 与 closure 真消融实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_stress.py` | 与正式数据零 fingerprint 重叠的 8/12/16 步 supported/relation 压力生成、审计与逐长度 Gate。 | A1.7 长程漂移评测实现；目标 0/3 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_assessment.py` | 多 seed、2×2 消融和压力结果的机器可读聚合与判断。 | A1.7 核心证据汇总实现 |
| `experiments/v2_a1_7_core.py` | A1.6 diagnostic、A1.7 prepare/audit/train/evaluate/intervene/stress/assessment 统一 CLI。 | 当前 A1.7 运行入口；不含 C1 |
| `src/yggdrasil_v2/reasoning_medium/a1_8_data.py` | T1–16 training、short、T8/12/16、T20/24/32、relation、causal 数据生成，外部 fingerprint 排除和十二项 audit。 | A1.8 独立数据合同；三 run audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_8_train.py` | batch 内长度均衡训练、最差长度 checkpoint scoring、formal Gate、逐步稳定性、因果与 CUDA 成本 benchmark。 | A1.8 正式训练/评测实现；不更改 A1.7 core |
| `src/yggdrasil_v2/reasoning_medium/a1_8_assessment.py` | 三 run、cross-run overlap、A1.7 对照、成本和总 Gate 的机器可读聚合。 | A1.8 总判定实现 |
| `experiments/v2_a1_8_core.py` | A1.8 prepare/audit/train/evaluate/intervene/benchmark/summarize CLI。 | 当前结构化 core 最新入口；不含 Qwen boundary |
| `src/yggdrasil_v2/reasoning_medium/a1_9_cache.py` | 固定 revision Qwen3.5-2B 的 oracle role span pooled hidden 分片 cache 与严格 cache audit。 | A1.9 真实 Qwen boundary 输入合同；三 run audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_model.py` | 冻结 A1.8 core 与 value/family/shared-register 三类窄 adapter；无 full-source、答案头或 query-state 写入。 | A1.9 adapter-only 结构合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_train.py` | T1–16 均衡 boundary 训练、最差长度 checkpoint、formal Gate、core hash 与 cached CUDA 成本。 | A1.9 三 run formal 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_interventions.py` | prefix、operation replacement/deletion/shuffle、query swap、same-answer、no-hidden 与 role shuffle。 | A1.9 hidden causal Gate 三 run 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_assessment.py` | cache/formal/causal/core hash/cross-run/cost 的机器可读总判定与证据边界。 | A1.9 总 Gate `3/3` passed |
| `experiments/v2_a1_9_boundary.py` | A1.9 cache/audit/train/evaluate/intervene/benchmark/assess 统一 CLI。 | 当前 frozen-Qwen boundary 入口；不含 learned full-text reader |
| `src/yggdrasil_v2/reasoning_medium/a1_10_cache.py` | 固定 revision Qwen3.5-2B 的完整 last-hidden + attention mask 分片 cache；不保存 spans、operation mask、role tensor 或 input IDs。 | A1.10 full-text 输入合同；三 run cache audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_10_model.py` | 匿名 `K=8` learned workspace、两层共享通用 recurrent Transformer、workspace-only state/answer heads。 | A1.10 联合架构实现；无显式寄存器 scaffold |
| `src/yggdrasil_v2/reasoning_medium/a1_10_train.py` | 全局 16-step T1–16 训练、formal Gate、训练子集诊断和 cached CUDA 成本；短程序 final state label 在剩余步骤重复。 | A1.10 三 run formal `0/3`；联合合同失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_10_interventions.py` | prefix、替换/删除/打乱、query swap、same-answer、no-source/source shuffle、disable recurrence 与 slot permutation。 | 已实现；因 ordinary formal 失败而未执行正式干预 |
| `src/yggdrasil_v2/reasoning_medium/a1_10_assessment.py` | overfit、cache/formal/干预先后、cross-run、成本和证据边界的机器可读总判定。 | A1.10 总 Gate `0/3` failed |
| `experiments/v2_a1_10_anonymous_workspace.py` | A1.10 full-token cache/audit/train/evaluate/intervene/benchmark/assess 统一 CLI。 | 当前 A1.10 历史入口；不得绕过 formal 停止规则 |
| `src/yggdrasil_v2/reasoning_medium/a1_11_models.py` | learned full-text typed reader → frozen core，以及 exact-symbolic typed roles → anonymous reasoner 两个正交模型。 | A1.11 模型合同；不保留旧 continuous-adapter Reasoner 入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_11_train.py` | 两臂 overfit/formal、strict Gate、hard re-embedding 只读诊断、checkpoint 与数据/核心完整性。 | Boundary strict overfit failed；Reasoner formal `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_11_interventions.py` | 两臂 ordinary-formal 后 counterfactual、disable recurrence 与 slot permutation。 | 已实现；本轮无 run 通过 ordinary formal，故未执行 |
| `src/yggdrasil_v2/reasoning_medium/a1_11_assessment.py` | A1.9/A1.10 参考格、两臂 overfit/formal、停止顺序与 2×2 一级归因。 | A1.11 机器总判定；纯组合解释 rejected |
| `experiments/v2_a1_11_localization.py` | Boundary/Reasoner train/evaluate、Gate 后 intervention 和 assess 统一 CLI。 | 当前 A1.11 历史入口；下一阶段应新建 A1.12 合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_12_models.py` | exact-symbolic A1.11 Reasoner 的 entity binding 与 aligned cursor 两个正交开关。 | A1.12 诊断模型；三臂均不足 |
| `src/yggdrasil_v2/reasoning_medium/a1_12_train.py` | A1.12 overfit/formal、统一 Gate、checkpoint 与评估。 | A1.12 三臂 formal `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_12_interventions.py` | ordinary formal 后的结构反事实、recurrence 与 start-state 干预。 | 因 formal 全失败未执行 |
| `src/yggdrasil_v2/reasoning_medium/a1_12_assessment.py` | binding × cursor 三臂机器汇总与严格分类。 | `binding_and_cursor_insufficient` |
| `experiments/v2_a1_12_root_cause.py` | A1.12 train/evaluate/intervene/assess CLI。 | A1.12 可复现实验入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_13_models.py` | relation-addressed transition、soft prototype closure、query-coupled answer 与训练期 QAUX/SAUX/TSAUX 受控头。 | A1.13–A1.18B 诊断核心；不是完整 V2-A |
| `src/yggdrasil_v2/reasoning_medium/a1_13_train.py` | A1.13–A1.17 训练、formal Gate、可选固定预算与 answer-loss 唯一变量。 | transition/closure/objective 归因实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_13_interventions.py` | relation/OOD/causal ordinary pass 后的反事实与 recurrence 干预。 | A1.13 joint run-2 causal 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_13_assessment.py` | transition × closure 2×2 聚合、state/full 分离和自适应分类。 | A1.13 joint state `3/3`、full `1/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_13f_assessment.py` | 固定预算三臂审计与 early-stop 判定。 | early-stop primary=false |
| `src/yggdrasil_v2/reasoning_medium/a1_15_assessment.py` | query-coupled 三 seed formal/causal 聚合。 | A1.15 state/full `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_16_assessment.py` | no-answer-CE 三 seed formal/causal 与触发条件聚合。 | A1.16 state/full `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_17_assessment.py` | 配对 seed、共享初始化逐 tensor 等同性、两种 coupled objective 与 A1.13 reference 的机器归因。 | A1.17 最终根因判定实现 |
| `experiments/v2_a1_13_transition_closure.py` | A1.13 train/evaluate/intervene/assess 与 A1.13F audit CLI。 | transition × closure 主入口 |
| `experiments/v2_a1_15_closed_coupled_core.py` | A1.15 train/evaluate/intervene/assess CLI。 | query-coupled + answer CE 入口 |
| `experiments/v2_a1_16_redundant_answer_loss.py` | A1.16 train/evaluate/intervene/assess CLI。 | query-coupled + no answer CE 入口 |
| `experiments/v2_a1_17_paired_objective_initialization.py` | 聚合 A1.17 paired reference、coupled-CE 与 coupled-noCE。 | A1.17 assessment 入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_18_train.py` | QAUX/FINAL-SAUX/TSAUX 训练、state-only checkpoint、固定 formal budget、辅助剥离部署与 formal loader。 | A1.18/A1.18B 长期训练合同实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_18_interventions.py` | 只接受辅助剥离部署模型的 structural/query/same-answer/recurrence/start-state 因果干预。 | TSAUX paired/fresh causal 均 `3/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_18b_assessment.py` | FINAL-SAUX、TSAUX paired/fresh、seed/budget/target/deployment integrity 与机制分类。 | `per_step_global_state_credit_assignment_confirmed` |
| `experiments/v2_a1_18_training_scaffold.py` | QAUX/FINAL-SAUX/TSAUX train/evaluate/intervene CLI。 | A1.18/A1.18B 统一执行入口 |
| `experiments/v2_a1_18b_trajectory_state_scaffold.py` | A1.18B 过拟合、FINAL-SAUX、TSAUX paired/fresh 的机器总判定入口。 | 当前机制 assessment 入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_data.py` | opaque-handle 编码、随机语义到物理 slot 映射、批量索引/GPU 搬运与缓存期地址完整性校验。 | A1.19H 共享数据边界；热路径不再执行 CUDA 标量校验 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_model.py` | equality-only opaque address sidecar、exchangeable continuous entity payload、共享 transition/state head 与 query-coupled answer。 | A1.19H generalized hybrid core 实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_train.py` | H1 fixed-cardinality 训练、辅助剥离部署、formal 与状态评测。 | H1 formal/causal `3/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_h2_data.py` | N2/N3/N4 训练、N5 heldout、relation heldout、独立数据 seed 与 overlap audit。 | H2 variable-cardinality 数据合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_h2_train.py` | H2 固定预算训练、一次性 CPU 编码/GPU tensor cache、预生成采样、无逐步 CUDA 同步的 loss、cached formal 与吞吐等价基准。 | run-3 fresh 重跑入口；60-step 参数级等价基准为 `10.74×` |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_{interventions,assessment}.py` | H1 structural/address/recurrence 干预与三 seed 汇总。 | H1 机器 Gate 已通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_h2_{interventions,assessment}.py` | H2 N5/address/recurrence 干预与 H1+H2 总判定。 | H2 formal/causal `3/3`；A1.19H 已关闭通过 |
| `experiments/v2_a1_19h_hybrid_core.py` | H1/H2 prepare、audit、train、benchmark、evaluate、intervene、assess 统一 CLI。 | A1.19H 可复现历史入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_cache.py` | H2 数据上的完整 Qwen last-hidden 分片 cache、strict audit 与 mmap dataset；禁止保存 span/role/entity mask/operation mask/input IDs。 | A1.20B full-text 输入合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_model.py` | 共享 entity/operation typed queries，自主预测 entity/operation presence、value、family、source/target/query pointer，并把连续 value payload 送入冻结 A1.19H core。 | A1.20B Boundary 实现；无 oracle mask 输入 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_train.py` | Boundary factorized mapping/state loss、冻结 core hash、N/T 均衡采样、overfit/formal evaluator、同步 mmap training 与吞吐对照。 | run-1 fixed-5000 已完成；formal eligibility 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_diagnostics.py` | 选择性 oracle 替换、逐样本联合 mapping exact 与 state CE→离散控制 logits 梯度信用审计。 | A1.20B 失败归因；oracle 只作诊断 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_assessment.py` | overfit/cache/training/oracle/gradient evidence 的机器总判定与停线列表。 | `full_text_entity_binding_and_program_extraction_failure` |
| `experiments/v2_a1_20b_full_text_boundary.py` | cache/audit/train/evaluate、diagnose、gradient-audit、assess-failure CLI。 | A1.20B 可复现失败入口；不得继续 run-2/run-3 |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_supervision.py` | 从 tokenizer offset 生成 entity name/value、operation family/source/target、query 的训练专用 token anchor target，并审计 target 不进入 forward。 | A1.20C compiler supervision；overfit/run-1 audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_model.py` | flat/hierarchical/factorized compiler、shared entity/operation Viterbi、三遍双向 section decode 与 frozen-core bridge。 | A1.20C 历史失败实现 + A1.20D post-stop 机制修复 |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_train.py` | anchor + mapping + state loss、分离 reader/schedule seed、优化 scope、formal eligibility、hard-forward equivalence、core hash 与 checkpoint。 | A1.20D 路径稳定性可复现训练基础；当前非 fresh formal |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_diagnostics.py` | 对实际 checkpoint 计算 state-vs-local 梯度夹角、失败 cell、tail loss 与 section 诊断。 | A1.20C 失败归因和 A1.20D 根因搜索辅助 |
| `experiments/v2_a1_20c_boundary_repair.py` | prepare/audit、训练、schedule seed、优化 scope 与诊断 CLI。 | A1.20C/A1.20D 可复现入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_21p_k1.py` | 单 slot anonymous recurrent reasoner 的训练、formal evaluator 与完整性报告。 | A1.21P 容量负基线；validation trajectory `0.106934` |
| `src/yggdrasil_v2/reasoning_medium/a1_21p_pareto.py` | routing surface data、section/capacity 诊断、跨域 probe、official-chat 在线 direct/text-CoT/hybrid preflight 与正式 assessment。 | A1.20D/A1.21P 机器评估入口；`a121p_passed=false` |
| `experiments/v2_a1_21p_pareto.py` | prepare/probe/K1/preflight/assess 统一 CLI。 | 当前 A1.21P 可复现入口；A1.22A 未授权 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p1.py` | Qwen3.5-2B FP16 hidden cache、operation span token mask、P1 hidden-to-latent interface。 | A1.5 P1 mechanism/surrogate probe |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p1_train.py` | P1 hidden cache training、start/query/operation warm-up、state/final CE、best reload 和多 split 诊断。 | A1.5 P1 surrogate formal；ordinary validation 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p2.py` | K=8 learned multi-slot workspace、共享 transition、无答案旁路和 permutation probe。 | A1.5 P2 formal core；composition 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p2_train.py` | P2 frozen-hidden training、state/final loss、same-answer shuffle、operation counterfactual 和轨迹干预。 | A1.5 P2 formal training/evidence |
| `src/yggdrasil_v2/reasoning_medium/baseline.py` | 旧 V2-A0 文本基线与 Qwen3.5 文本塔加载。 | A0 smoke/probe 实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_text_baseline.py` | A1.5 matched visible text-CoT baseline、逐步 state parser、Qwen3.5-0.8B/2B 选择、无人工总输出 token cap 的语义终止生成。 | A1.5 smoke/probe；需与 latent 使用同样本数比较 |
| `tests/test_v2_a_data.py` | split 可复现、heldout/length、解析器和泄漏断言。 | 单元验证 |
| `tests/test_v2_a_model.py` | latent shape、trajectory、干预和 no-bypass 断言。 | 单元验证 |
| `tests/test_v2_a_train.py` | step-level verifier policy-gradient 的 mask、reward 统计和反传断言。 | 单元验证 |
| `tests/test_v2_a1_5.py` | A1.5 split/necessity/span contract、P0 shared transition、variable T 和 no-gate 断言。 | A1.5 单元验证 |
| `tests/test_v2_a1_5_p1_p2.py` | P1 span/warm-up shape、P2 learned-slot/permutation smoke 断言。 | A1.5 单元验证 |
| `tests/test_v2_a1_6_data.py` | A1.6 split、span、causal necessary 和 audit contract。 | A1.6 单元验证 |
| `tests/test_v2_a1_6_core.py` | operation-free initialization、prefix boundary、query isolation、shared transition/head 和 pointer contract。 | A1.6 单元验证 |
| `tests/test_v2_a1_6_qwen.py` | C1 core reuse、无 full-source path 和空 span 失败检查。 | A1.6 单元验证 |
| `tests/test_v2_a1_7_data.py` | 受控 11+1 relation support、位置分层、overlap 和 causal necessary contract。 | A1.7 数据完整性测试 |
| `tests/test_v2_a1_7_core.py` | 默认地址/内容分离、显式 address-mixed 消融、operation-free initial state、shared transition、T=0、query isolation、last-active identity、closure stop-gradient 与零权重真消融。 | A1.7 core 完整性测试 |
| `tests/test_v2_a1_7_stress.py` | 长度/位置平衡、唯一 holdout、base fingerprint 零重叠和 stress audit Gate。 | A1.7 长程压力合同测试 |
| `tests/test_v2_a1_8_data.py` | T1–32 split、relation holdout、外部数据排除和跨数据 seed fingerprint 零重叠。 | A1.8 数据合同测试 |
| `tests/test_v2_a1_8_train.py` | 16 长度 batch 均衡采样和 T32 连续扰动稳定性诊断。 | A1.8 训练/诊断合同测试 |
| `tests/test_v2_a1_9_boundary.py` | oracle role cache 审计、共享 register adapter、冻结 core、无 full-source path 和 mapping/trajectory 反传合同。 | A1.9 边界完整性单元测试 |
| `tests/test_v2_a1_10_anonymous_workspace.py` | full-token cache 禁止项、匿名 slot 对称性、通用共享 recurrence、无 scaffold/bypass、统一 recurrent budget 和干预合同。 | A1.10 架构与实验合同单元测试 |
| `tests/test_v2_a1_11_localization.py` | Boundary 无 span/frozen core、Reasoner exact-symbolic/no-Qwen/no-core、固定 recurrent budget 与 2×2 分类。 | A1.11 正交隔离合同单元测试 |
| `tests/test_v2_a1_12_root_cause.py` | binding/cursor 唯一变量、forward shape、训练合同和分类。 | A1.12 完整性单元测试 |
| `tests/test_v2_a1_13_transition_closure.py` | transition/closure/coupled/noCE 合同、loss 差分、配对初始化逐 tensor 等同和 A1.13F–A1.17 分类。 | A1.13–A1.17 完整性单元测试 |
| `tests/test_v2_a1_18_training_scaffold.py` | QAUX 初始化等同、SAUX/TSAUX shape/gradient、query-coupled answer、训练期限定、物理部署剥离和 A1.18B 分类。 | A1.18/A1.18B 完整性单元测试 |
| `tests/test_v2_a1_19h_hybrid_core.py` | opaque-handle/slot 等变、辅助剥离、H2 N5 合同、缓存编码等价、冻结 loss 与缓存期严格地址校验。 | A1.19H 完整性与训练优化回归测试 |
| `tests/test_v2_a1_20b_full_text_boundary.py` | continuous-payload core 等价、Boundary 无 oracle mask/role 输入、冻结 core 梯度隔离、N/T sampler、oracle 诊断选择与 hard-control state-credit 断裂。 | A1.20B 架构/训练/归因合同测试 |
| `tests/test_v2_a1_20c_boundary_repair.py` | token anchor、flat/hierarchical/factorized no-oracle forward、shared Viterbi、三遍 section decode、ST/state-credit 与训练 schedule 合同。 | A1.20C/A1.20D 回归测试通过 |
| `tests/test_v2_a1_21p_pareto.py` | trace parser、official prompt、routing 语义保持、Pareto dominance、K1 容量/梯度和 formal assessment 停机合同。 | A1.21P 8 项回归测试通过 |
| `tests/conftest.py` | 为当前 CPU torchvision wheel 预声明缺失的 NMS operator，保证 Transformers 测试收集可重复；不改变模型运行语义。 | 测试环境隔离 |

本地 `artifacts/v2-a/` 被 `.gitignore` 忽略；阶段结果路径、配置和证据等级必须以对应阶段文档与本目录索引为准，不把未索引的本地文件当作 repo truth。

## 公开路线对照入口

| 主题 | 路径 | 用途 |
| --- | --- | --- |
| MoE/模型组装对照 | `docs/moe-model-assembly-comparative-review-2026-07-14.md` | 对照 VLMo、Uni-MoE、MoME、Uni-Med、DeepSeek-VL2、MoE-LLaVA、DeepSeekMoE/DeepSeek-V3、Qwen3、SMoES；记录成果、差异和 V2-B 未完成项。 |

## V2-A Closure C0/C0R 代码与运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `src/yggdrasil_v2/v2_a/closure_c0/` | C0 frozen contract、strict JSON/seal、26,624-record stream audit、source-only visible-legend oracle、P0-M/历史/H1 审计、decision 与 single-use runner。 | 已消耗 C0 实现；C004/C006 FAIL，固定 identity 禁止重跑 |
| `experiments/v2_a_closure_c0.py` | 唯一 `run-v2-a-closure-c0` CLI；不接受 root/seed/阈值覆盖。 | 已消耗；再次调用必须在 mutation 前拒绝 |
| `tests/v2_a_closure_c0/` | 四臂/授权边界、PASS/FAIL/INCOMPLETE、seal/tree/hash、stream dataset 与 single-use runner 回归。 | 专项 `6 passed` |
| `artifacts/v2-a/closure-c0-20260823-1/` | contract/fairness/input audit、七项历史 reuse matrix、result/run-state、source snapshot 与 13-file evidence seal。 | `FAIL_V2_A_CLOSURE_C0_READINESS`；result/seal `3FC871…43B1`/`462317…9972`，`authorizes=nothing` |
| `artifacts/v2-a/closure-c0-20260823-1.preflight-lease.jsonl` | sibling exclusive single-use lease。 | 已消费；固定 identity 禁止删除后重跑 |
| `docs/v2-a-closure-c0-task-baseline-qualification.md` / `docs/v2-a-closure-c0-result-review.md` | 冻结 C0 合同与人类可读终局。 | 历史 FAIL 真源；C004/C006 不改判，已由 C0R 新身份修复 |
| `src/yggdrasil_v2/v2_a/closure_c0r/` | 新 data wrapper、D001–D011 audit、strict CT1 formatter/parser/replay/fault-kill、readiness 与 single-use runners。 | 已消耗 C0R 实现；当前只允许只读复验，不得删除 roots 后重跑 |
| `experiments/v2_a_closure_c0r.py` | C0R data preflight、正式 data/trace qualification 与 readiness 三个固定 CLI。 | preflight 与两个正式 identity 均已消费；再次调用必须在 mutation 前拒绝 |
| `tests/v2_a_closure_c0r/` | generator identity、D002/D011、strict trace、fault registry、readiness、seal 与 single-use runner 回归。 | 专项 `26 passed` |
| `tmp/v2-a-closure-c0r-data-trace-preflight-20260824-1/` | C0R 唯一 preflight；除 D002 manifest/record schema 审计器混淆外全部通过，42-file seal 完整。 | 历史 sealed FAIL；未覆盖、未重跑，不授权任何阶段 |
| `artifacts/v2-a/closure-c0r-data-trace-20260824-1/` | 26,624-record 新 bank、D001–D012、trace qualification、source/substrate snapshot 与 evidence seal。 | `PASS_V2_A_C0R_DATA_TRACE_QUALIFICATION`；result/seal `B2502F…FFF9`/`5AFB37…B570`，42/42 sealed |
| `artifacts/v2-a/closure-c0r-data-trace-20260824-1.preflight-lease.jsonl` | 正式 data/trace sibling exclusive lease。 | 已消费；固定 identity 禁止删除后重跑 |
| `artifacts/v2-a/closure-c0r-20260824-1/` | old/new seal audit、fairness contract、C001–C008、result/run-state 与 source/substrate snapshot。 | `PASS_V2_A_CLOSURE_C0R_READINESS`；result/seal `391C84…8D5E`/`161FBB…0BA4`，25/25 sealed；只授权 C1 单 seed |
| `artifacts/v2-a/closure-c0r-20260824-1.preflight-lease.jsonl` | C0R readiness sibling exclusive lease。 | 已消费；固定 identity 禁止删除后重跑 |
| `docs/v2-a-closure-c0r-data-trace-qualification.md` / `docs/v2-a-closure-c0r-result-review.md` | 冻结 C0R 合同与人类可读终局。 | 当前 C0R 设计/结果真源；下一步是独立 C1 合同，不是 C0R 后补丁 |
| `docs/v2-a-closure-c1-single-seed-eligibility.md` / `docs/v2-a-closure-c1-result-review.md` | C1 历史冻结合同与人类可读终局；Qwen hidden 只进入 learned Boundary 一次，`H0` 后 core source-closed，CT1 probe 仅训练期读取 latent trajectory。 | 唯一 formal 已在 G007 FAIL-stop；`authorizes=nothing`，不得改旧 root、重跑、换 seed 或创建未授权 successor |
| `src/yggdrasil_v2/v2_a/closure_c1/` | C1 fresh text-only Qwen cache、indexed mmap、offline answer/CT1 ledger、learned Boundary→K8→T10 shared core、training-only probe、物理剥离、batched behavior/causal/intervention evaluator、G001–G011 与 single-use runners。 | 历史实现；当前工作区仅将 target-bank 审计的词表映射改为一次物化以消除 O(tokens×vocab) 性能浪费，不改任何 sealed root、target、Gate 或旧结论，也不允许重跑旧 formal |
| `experiments/v2_a_closure_c1.py` | cache preflight/qualification 与 learner preflight/single-seed 四个固定 CLI；正式命令分别消费独立 root/lease。 | 四个固定 identity 均已消费；single-use guard 必须拒绝再次调用 |
| `tests/v2_a_closure_c1/` | source-only cache、local trace vocab/grammar mask、随机 mmap batch、模型等变/剥离、streaming trace credit、因果 bootstrap、双 hidden intervention、训练调度与 runner 回归。 | 当前专项实现回归；不能替代真实 GPU cache/overfit/formal |
| `tmp/v2-a-closure-c1-cache-preflight-20260825-1/` / `tmp/v2-a-closure-c1-single-seed-preflight-20260825-1/` | cache 最长样本真实 Qwen 检查与 learner trace chunk 64/512 forward-backward 检查。 | 两项 sealed PASS；result/seal 分别为 `D98B96…E2B4`/`BC774B…C35` 与 `2B2749…753`/`27A0DA…A6EA`；均已消费 |
| `artifacts/v2-a/closure-c1-cache-20260825-1/` | 26,624-record fresh Qwen full-token FP16 mmap cache、trace target bank、K001–K008 与 seal；16,596,184 source tokens、416 shards、67,998,434,400 bytes。 | `PASS_V2_A_CLOSURE_C1_CACHE_QUALIFICATION`；result/seal `A05868…D4A`/`68CA34…DC34`，867/867 replay；只曾授权唯一 C1 formal |
| `artifacts/v2-a/closure-c1-single-seed-20260825-1/` | Overfit32、6,144-update primary、trace-only selection 与 G007 trace-credit 证据；selected update 5,120。 | `FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY` at G007；result/seal `FE9F22…972C`/`BEB465…FA19`，55/55 replay；G003–G006/G008–G011 NOT_RUN，`authorizes=nothing` |
| `docs/v2-a-closure-c1-failure-attribution.md` | C1 G007 后的只读归因合同；冻结 identity/selection、alignment、exposure、selected/final trace、shared-gradient 与 post-stop architecture 六层测量。 | 已完成并封存；结论为 latent/training primary，exposure 是合同破坏但非直接主因，shared-gradient conflict 显著；只授权一个 fresh C1 repair design |
| `src/yggdrasil_v2/v2_a/closure_c1_diagnosis/` / `tests/v2_a_closure_c1_diagnosis/` / `experiments/v2_a_closure_c1_diagnosis.py` | 独立 post-stop 诊断实现、专项测试与 single-use CLI；包含 selection/pin、16-worker full alignment replay、逐 token exposure、selected/final top-k、共享梯度和完整 post-stop architecture evaluator。 | 已封存实现；不得重跑或改旧诊断 root，不能回填旧 G003–G011 |
| `artifacts/v2-a/closure-c1-failure-attribution-20260826-1/` | A001–A006 只读诊断、归因决策与 evidence seal。 | `COMPLETE_V2_A_C1_FAILURE_ATTRIBUTION`；result/seal `0726F4…AADA`/`D63231…2941`，32/32 replay；只授权一个 fresh C1 repair design |
| `docs/v2-a-closure-c1r-staged-credit.md` | 独立 C1R repair contract；Stage A fresh answer-only 6144 steps，全部 G004/G005/G006/G008/G009/G010 通过后才允许 Stage B frozen-probe positive control 与 1536-step complete-target training。 | 已消费；唯一 formal 在 G004 因 ERE/CPS accuracy `0.191406/0.156250` 失败，Stage B 未运行，`authorizes=nothing` |
| `docs/v2-a-closure-c1r-result-review.md` | C1R 终局复盘；整合封印、train/validation、label/evaluator 反证、H0–H10 轨迹、slot geometry、readout 干预与性能修复边界。 | 当前 C1R 结果解释真源；结论限定为 final answer path effective K≈1，不声称整个 latent workspace K=1 |
| `src/yggdrasil_v2/v2_a/closure_c1r/` / `tests/v2_a_closure_c1r/` / `experiments/v2_a_closure_c1r.py` | 独立合同、coverage、answer-only/trace-only trainer、fail-stop runner、C1R seal 与固定 CLI；设备门锁定 RTX 4070，coverage plan/训练 schedule 要求逐行与 SHA-256 双一致。 | 已封存；preflight 与 formal identity/lease 均已消费，不得重跑、换 seed 或补跑 Stage B |
| `tmp/v2-a-closure-c1r-staged-credit-preflight-20260827-1/` | 唯一 C1R zero-training preflight 与 sibling lease。 | sealed PASS；result/seal `E1C203…02A1`/`89387E…0C49`，13/13 replay；identity 已消费 |
| `artifacts/v2-a/closure-c1r-staged-credit-20260827-1/` | 唯一 C1R formal root 与 sibling lease；含 Overfit32、Stage A、G004 和 fail-stop seal。 | `FAIL_V2_A_C1R` at G004；result/seal `54F426…590D`/`28B4FD…6C7E`，20/20 replay；后续 Gate NOT_RUN，`authorizes=nothing` |
| `docs/v2-a-closure-c1s-addressed-workspace-proposal.md` | 地址/内容分离、输入条件 query、address-routed gated update、training-only state/closure 辅助与 matched K=1 控制的直接 successor 设计来路。 | 已由冻结 qualification 接替；提案自身不授权训练或 formal |
| `docs/v2-a-closure-c1s-addressed-workspace-qualification.md` | C1S 新状态代数、public-only forward、matched K=1、S0–S2 Gate、single-use 与停止边界。 | 历史冻结合同；S0 PASS 授权已由唯一 S1 消费，当前不再提供后继权限 |
| `src/yggdrasil_v2/v2_a/closure_c1s/` / `experiments/v2_a_closure_c1s.py` / `tests/v2_a_closure_c1s/` | 独立 addressed-workspace model、functional-K/ownership controls、source pin/seal、零训练 S0 runner、固定 CLI 与 synthetic/tmp-path 回归；不导入 C1R runner/checkpoint。 | 当前 C1S 实现真源；S0 源码 identity `1F6FA0…A759` 已封存，专项 `13 passed`，旧 C1/C1R 回归另有 `87 passed` |
| `tmp/v2-a-closure-c1s-s0-preflight-20260828-1/` / `tmp/v2-a-closure-c1s-s0-preflight-20260828-1.preflight-lease.jsonl` | C1S 唯一 zero-update S0 root 与 sibling lease；含 source snapshot、三组 predecessor pin replay、structure/controls、CUDA BF16 smoke、result 与 evidence seal。 | `PASS_V2_A_C1S_S0_QUALIFICATION`；result/seal `022DD0…698D`/`2117B1…6E3D`，17/17 replay，optimizer/model writes 均 0；identity 已消费 |
| `docs/v2-a-closure-c1s-s0-result-review.md` | S0 人类可读复盘；解释结构资格、关键数值、证据边界和当时的 S1 后继权限。 | 历史 S0 结果真源；S1 授权已消费，当前终局以 S1 复盘为准 |
| `docs/v2-a-closure-c1s-s1-s2-s3-execution.md` / `docs/v2-a-closure-c1s-s1-result-review.md` | C1S SRW 的 single-use 执行合同与 S1 人类可读终局。 | S1 sealed FAIL：答案 `32/32`，但 recurrence necessity、functional-K 与动态状态 Gate 失败；S2/formal 未运行，`authorizes=nothing` |
| `src/yggdrasil_v2/v2_a/closure_c1s_successor/` / `tests/v2_a_closure_c1s_successor/` / `experiments/v2_a_closure_c1s_successor.py` | SRW target/runtime/objective/evaluator、固定 endpoint trainer、preflight/seal/fail-stop runner、matched K1 和唯一 CLI；forward 仍只接收 `source_hidden/source_mask`。 | 已封存 S1 source identity `E3C5D7…B77F4`；专项 `83 passed`、source closure `54/54`。旧 S1 identity 不得重跑，S2/S3 CLI 因 predecessor FAIL 必须拒绝 |
| `tmp/v2-a-closure-c1s-srw-s1-preflight-20260829-1` / `artifacts/v2-a/closure-c1s-srw-s1-overfit32-20260829-1` | 已消费的 S1 preflight 与唯一 S1 stage root/lease。 | preflight PASS；S1 为 `FAIL_V2_A_C1S_S1_QUALIFICATION`，result/seal/endpoint `9D0B06…8892`/`FAFF30…A9D2`/`D6F44C…A2E9`，63/63 replay，`authorizes=nothing` |
| `tmp/v2-a-closure-c1s-srw-s2-preflight-20260829-1` → `artifacts/v2-a/closure-c1s-srw-s2-discovery-20260829-1` → S3 preflight/formal | S1 后预注册但必须由前一 sealed PASS 授权的固定路径。 | 全部不存在且保持 `NOT_RUN`；S1 FAIL 后禁止创建 |
| `src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/` / `experiments/v2_a_closure_c1s_s1_diagnosis.py` / `tests/v2_a_closure_c1s_s1_diagnosis/` | 独立只读 S1 归因实现：endpoint weights-only pin、objective credit、H0/core 路径、CPS semantic candidate→physical slot 双射、task certificate/answer-support/反事实影响三分、逐槽 leave-one-out content functional effect、answer-permutation control、source-AST temporal target replay、全链 record-macro temporal readout/null 与终态证据完整性审计。 | 已由唯一诊断消费；launch 时专项 `59 passed`、successor `83 passed`。旧源码与固定 identity 不得修后重跑；当前结论以 sealed root 与结果复盘为准。 |
| `docs/v2-a-closure-c1s-s1-failure-attribution-result-review.md` | 唯一诊断的封存终态、D001–D003 部分证据、D004 fold-support 崩溃归因和新身份修复原则。 | 当前人类可读真源；明确没有 D004/D005 完整结论，也不授权 successor。 |
| `tmp/v2-a-closure-c1s-srw-s1-failure-attribution-preflight-20260831-1` / `artifacts/v2-a/closure-c1s-srw-s1-failure-attribution-20260831-1` | 已消费的诊断 preflight 与单次只读归因输出。 | preflight PASS，result/seal `31242B…02C0`/`E065BB…CBD3`；diagnosis 在 CPS `running_best` inner-fold 0 valid-record 处 sealed `CRASH_V2_A_C1S_S1_FAILURE_ATTRIBUTION`，result/seal `58F45C…6163`/`AF45A8…4242`，86/86 replay，`authorizes=nothing`，禁止重跑。 |
| `src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/` / `experiments/v2_a_closure_c1s_s1_temporal_diagnosis.py` / `tests/v2_a_closure_c1s_s1_temporal_diagnosis/` | 全新 temporal-only 诊断包、single-use CLI 与回归；不修改旧 diagnosis。包含 target-only support ledger、4×3 受约束 nested folds、record/within-record-class-balanced ridge、共享 fold 的 time/mean/target fit null、10k score null、自然 rollout collector 与 D005R family-primary classifier。 | 已由唯一 preflight 与 diagnosis 消费；源码 identity `C4FAD9…883E5`，不得修改后重跑。当前结论以 sealed root 与结果复盘为准。 |
| `tmp/v2-a-closure-c1s-srw-s1-temporal-attribution-repair-preflight-20260831-1` / `artifacts/v2-a/closure-c1s-srw-s1-temporal-attribution-repair-20260831-1` | 新 temporal 身份的固定 preflight/diagnosis 路径。 | preflight result/seal `F7AA94…5049E`/`CCFAE2…DFB4A`，102/102 replay；diagnosis result/seal `D1603C…6A496`/`E8A61C…562B0`，103/103 replay，Axis C 两族 `INCONCLUSIVE`，`authorizes=nothing`。 |
| `docs/v2-a-closure-c1t-causally-partitioned-workspace.md` / `docs/v2-a-closure-c1t-s0-execution.md` / `docs/v2-a-closure-c1t-s0-result-review.md` | C1T CPW 直接 successor、S0 执行合同与人类可读终局；记录固定 source/硬件、八门 Gate、封存 hash 和严格后继边界。 | 正式 S0 sealed PASS；result/seal `831BC5…D6A8`/`E4D9B3…A0EE`，其 `C1T_S1_CONTRACT_DESIGN_ONLY` 授权现已消费。 |
| `docs/v2-a-closure-c1t-s1-execution.md` / `docs/v2-a-closure-c1t-s1-result-review.md` | fresh S1 Overfit32 的冻结 single-use 合同与人类可读终局；定义完整 group batch、固定 4,000-step schedule、同地址 counterfactual margin、no-core/two-contributor Gate 与失败停止。 | 唯一 S1 为 `FAIL_V2_A_C1T_S1_QUALIFICATION`；R103/R105 FAIL，S2/S3/formal 未运行，`authorizes=nothing`。 |
| `docs/v2-a-closure-c1t-s1-failure-attribution.md` | sealed S1 的 post-stop 失败诊断；汇总逐组拓扑、operation-2 gate、counterpart 等价、hinge/CE 梯度、stop-gradient 与 gate-floor ablation、训练 mode switch 及 fresh single-group bounded screen。 | 当前归因真源：多组共享优化中的 dead-gate + causal-objective 梯度抵消；排除 schedule/support-swap/card 缺失/基础 XOR 不可表达。只诊断，不授权修复或 successor。 |
| `src/yggdrasil_v2/v2_a/closure_c1t/` / `experiments/v2_a_closure_c1t.py` / `tests/v2_a_closure_c1t/` | C1T package、single-use S0/S1 CLI 与隔离回归；包含 2×2 generator、真实逐卡 Qwen cache/readback、PartitionedBoundary、target-only transition、固定 endpoint trainer、同地址 counterfactual evaluator、seal/fault-kill 与 fail-closed audit。 | 正式 S1 前专项 `33 passed`；当前封存 source identity `07E8707E…5900E`。不得修改已封存 source 后重跑 S0/S1。 |
| `tmp/v2-a-closure-c1t-cpw-s0-preflight-20260901-1` / `artifacts/v2-a/closure-c1t-cpw-s0-20260901-1` 及 sibling leases | 已消费的 C1T S0 preflight 与唯一 zero-training S0 root。 | preflight P001–P005 PASS，result/seal `BC1B18…F76C7`/`F32684…6DDCC`；正式 S001–S008 PASS，192 卡/9,772 tokens，result/seal `831BC5…D6A8`/`E4D9B3…A0EE`，optimizer/model writes `0/0`。 |
| `tmp/v2-a-closure-c1t-cpw-s1-overfit32-preflight-20260901-1` / `artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1` 及 sibling leases | 已消费的 C1T S1 preflight 与唯一正式 Overfit32 root。 | preflight P101–P105 PASS，result/seal `3E48FD…1C646`/`3E7849…BF62`；正式 R101/R102/R104/R106/R107 PASS、R103/R105 FAIL，result/seal/endpoint `9B3762…C1DFF`/`89C530…9BCF`/`E7E66F…96F9`，43/43 replay，`authorizes=nothing`。 |
| `src/yggdrasil_v2/v2_a/closure_c1u/` / `experiments/v2_a_closure_c1u.py` / `tests/v2_a_closure_c1u/` | C1U package、single-use S0/S1 CLI 与隔离回归；含 public-only replay、六项 bridge faults、gate-free target overwrite、双 operation-2 gradient faults、真实逐卡 cache、fresh fixed-endpoint trainer、counterfactual evaluator 与 fail-closed seal audit。 | C1U `36 passed`、C1T 独立回归 `33 passed`；封存 S1 33-file source identity `8BD618…94FD`，同一身份不得修改后重跑。 |
| `tmp/v2-a-closure-c1u-pgf-s0-preflight-20260901-1` / `artifacts/v2-a/closure-c1u-pgf-s0-20260901-1` 及 sibling leases | 已消费的 C1U S0 preflight 与唯一 zero-training S0 root。 | preflight P001–P006 PASS，result/seal `4D5755…4BBB2`/`63646A…E3516`；正式 U001–U010 PASS，result/seal `39070B…A714A`/`F224DB…9ADE`，38/38 replay，optimizer/model writes `0/0`。 |
| `tmp/v2-a-closure-c1u-pgf-s1-overfit32-preflight-20260901-1` / `artifacts/v2-a/closure-c1u-pgf-s1-overfit32-20260901-1` 及 sibling leases | 已消费的 C1U fresh S1 preflight 与唯一正式 Overfit32 root。 | preflight P101–P105 PASS，result/seal `E62E1C…E2034`/`484AF8…2D309`；formal R101–R107 PASS，result/seal/endpoint `1FCD2D…6090C`/`A5211D…5350`/`88F538…159F`，44/44 replay。 |
| `src/yggdrasil_v2/v2_a/closure_c1u_s2/` / `experiments/v2_a_closure_c1u_s2.py` / `tests/v2_a_closure_c1u_s2/` | 独立 C1U S2 package、single-use CLI 与隔离回归；含多-bank generator/audit、target-free cache、matched K1/K8、固定端点 trainer、heldout causal evaluator、bank-cluster paired bootstrap、seal 与 fail-closed runner。 | 启动前专项 `16 passed`、C1U 前驱独立 `36 passed`、CUDA BF16 微烟测通过；源码已由 formal 封存，不得改后重跑。 |
| `tmp/v2-a-closure-c1u-pgf-s2-multibank-matched-k1-k8-preflight-20260902-1` / `artifacts/v2-a/closure-c1u-pgf-s2-multibank-matched-k1-k8-20260902-1` 及 sibling leases | 已消费的 C1U S2 single-use preflight 与 formal；含 1,152-card target-free cache、六个 `fixed_4000` endpoints、heldout evaluation、qualification、result 与 seals。 | preflight PASS，result/seal `494A72…5C23B`/`EBB51D…02421`，60/60 replay；formal R203/R204/R205 FAIL，result/seal `F0EABD…6EAA4`/`00B782…25A4B`，76/76 replay，24,000 steps、六写入，`authorizes=nothing`。 |

## A1.6 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_6/data/` | seed `20260715` 的六 split relation-state-machine 数据。 | 数据合同证据 |
| `artifacts/v2-a/a1_6/data-audit.json` | 长度/family/bigram/COPY 位置、relation holdout、causal necessary 和 fingerprint overlap 审计。 | audit Gate 通过 |
| `artifacts/v2-a/a1_6/c0-overfit32/` | C0 5000-step overfit checkpoint、history、progress 和 results；fit Gate 通过，validation 仅作 few-shot diagnostic。 | C0 overfit sanity evidence |
| `artifacts/v2-a/a1_6/c0-formal/` | fresh formal C0 checkpoint、history、progress、results 和六 split `eval-all.json`；relation-heldout Gate 失败。 | C0 formal evidence；停止在正式 Gate |
| `artifacts/v2-a/a1_6/diagnostics/closure-diagnostic.json` | free recurrence、oracle-reset one-step、predicted hard re-embed 与首次错误 step。 | A1.6 只读失败归因；不改变旧 checkpoint |

## A1.7 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_7/data/` | seed `20260715`、11 个训练组合 + 唯一 `COPY amber→jade` holdout 的六 split 数据。 | A1.7 数据合同证据 |
| `artifacts/v2-a/a1_7/data-audit.json` | joint/position/sequence/marginal/overlap/causal necessary 审计。 | 十项 data Gate 全部通过 |
| `artifacts/v2-a/a1_7/c0-overfit32/` | 32-example fit-only checkpoint/results；step 200 五项指标全 `1.0`。 | A1.7 overfit sanity evidence |
| `artifacts/v2-a/a1_7/c0-formal/` | seed `20260715` 的 fresh formal checkpoint、history、results、`eval-all.json`、`interventions.json`。 | A1.7 目标 seed formal + causal Gate 通过 |
| `artifacts/v2-a/a1_7/multiseed/seed-{20260716,20260717}/` | 两个附加初始化 seed 的独立 checkpoint、formal 评测与因果干预。 | 与目标 seed 合计 3/3 短程 formal + causal 通过 |
| `artifacts/v2-a/a1_7/ablations/` | content-only/address-mixed × closure on/off 的另外三种受控训练、formal 和 intervention 结果。 | 四组合均过短程 Gate；短程必要性归因不成立 |
| `artifacts/v2-a/a1_7/stress-data/` | seed `20260718`、supported/relation 各 768 条、长度 8/12/16 的零重叠压力数据。 | A1.7 长程评测合同 |
| `artifacts/v2-a/a1_7/stress-data-audit.json` | 长度/位置/holdout/support/fingerprint 审计。 | stress data Gate 全部通过 |
| `artifacts/v2-a/a1_7/stress/` | 三个目标 seed 与三种消融 checkpoint 的逐长度压力结果。 | 目标严格 Gate 0/3；closure 显著缓解 T16 漂移 |
| `artifacts/v2-a/a1_7/core-assessment-summary.json` | formal、causal、2×2 消融、stress 的机器可读聚合和边界判断。 | A1.7 当前核心评估总表 |

## A1.8 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_8/overfit32/` | 每个 T1–16 两条记录的 length-balanced overfit checkpoint/history/results。 | step 500 停止，16 长度 trajectory/pointers 全 `1.0` |
| `artifacts/v2-a/a1_8/runs/run-{1,2,3}/data/` | 三个独立 data seed，各 22,784 条 T1–32 数据与 manifest。 | A1.8 formal 数据；彼此及 A1.7 overlap `0` |
| `artifacts/v2-a/a1_8/runs/run-{1,2,3}/data-audit.json` | 长度、support、relation 位置、diversity、causal necessity、内部/外部 overlap audit。 | 三组全部通过 |
| `artifacts/v2-a/a1_8/runs/run-{1,2,3}/formal/` | fresh random-depth checkpoints、history、results、formal eval/stability 和 interventions。 | 三组 formal + causal 全部通过 |
| `artifacts/v2-a/a1_8/cost-benchmark.json` | RTX 4070 Laptop GPU 上的训练/推理延迟、peak memory、参数、KV 与 FLOPs proxy。 | A1.8 structured core 成本证据；非 text-CoT Pareto |
| `artifacts/v2-a/a1_8/assessment-summary.json` | 三 run metrics、cross-run overlap、A1.7 delta、成本与总 Gate。 | A1.8 机器可读最终真源；passed |

## A1.9 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_9/overfit32/` | 真实 Qwen oracle-role cache/audit 与 T1–16 length-balanced fit-only checkpoint/results。 | mapping、pointer、trajectory、answer 全 `1.0`；overfit Gate 通过 |
| `artifacts/v2-a/a1_9/runs/run-{1,2,3}/cache/` | 三组独立 data seed，各 `22,784` 条 role-pooled Qwen hidden 分片与 manifest。 | A1.9 formal 输入；三组 fingerprint overlap `0` |
| `artifacts/v2-a/a1_9/runs/run-{1,2,3}/cache-audit.json` | fingerprint、shape、finite、operation count、split overlap、full-source 禁止项审计。 | 三组 cache audit 全部通过 |
| `artifacts/v2-a/a1_9/runs/run-{1,2,3}/formal/` | adapter-only checkpoints、history、results、formal eval 与 formal 后 hidden interventions。 | 三组 formal + hidden causal 全部通过 |
| `artifacts/v2-a/a1_9/cost-benchmark.json` | RTX 4070 Laptop GPU 上的 cache manifest 编码成本、cached train/inference latency、memory、参数和 KV 口径。 | A1.9 成本证据；非在线端到端、非 text-CoT Pareto |
| `artifacts/v2-a/a1_9/assessment-summary.json` | overfit、三 run、cross-run overlap、成本、证据边界和总 Gate。 | A1.9 机器可读最终真源；passed |
| `tmp/V2-A1.9 result.md` | A1.9 执行结果、异常、成本、完成/未完成项和下一阶段建议。 | 人类可读结果交接 |

## A1.10 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_10/overfit32/` | 64 条 T1–16 full-token cache、audit、fresh checkpoint、history 和逐长度评测。 | 修正 recurrent budget 后 trajectory/final/answer 全 `1.0`；fit-only sanity 通过 |
| `artifacts/v2-a/a1_10/runs/run-{1,2,3}/cache/` | 三组独立 data seed，各 `22,784` 条完整 Qwen last-hidden 分片与 attention mask。 | A1.10 formal 输入；合计 `9,046,290` source tokens、`38.36` GB，fingerprint overlap `0` |
| `artifacts/v2-a/a1_10/runs/run-{1,2,3}/cache-audit.json` | fingerprint、shape、finite、attention mask、split overlap 和 oracle/scaffold 禁止项审计。 | 三组 cache audit 全部通过 |
| `artifacts/v2-a/a1_10/runs/run-{1,2,3}/formal/` | fresh anonymous-reasoner checkpoints、history、formal eval 与训练子集 fit 诊断。 | 三组 formal `0/3`；trajectory 近零，未进入正式干预 |
| `artifacts/v2-a/a1_10/cost-benchmark.json` | RTX 4070 Laptop GPU 上的 cache manifest 编码成本、cached reasoner train/inference latency、memory 和参数口径。 | A1.10 成本证据；排除在线 Qwen 编码，非 text-CoT Pareto |
| `artifacts/v2-a/a1_10/assessment-summary.json` | overfit、三 run、formal-before-intervention、cross-run、成本、证据边界和总 Gate。 | A1.10 机器可读最终真源；failed，hidden interventions 未执行 |
| `tmp/V2-A1.10 result.md` | A1.10 方法修正、正式结果、成本、完成/未完成项、概率更新和 A1.11 建议。 | 人类可读结果交接 |

## A1.11 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_11/overfit32/boundary/` | learned full-text reader 的 fresh checkpoint/history/results，以及只读 hard-reembed diagnostic。 | strict pointer Gate failed；诊断全通过，formal stopped |
| `artifacts/v2-a/a1_11/overfit32/reasoner/` | exact-symbolic input、fresh anonymous reasoner checkpoint/history/results。 | T1–16 trajectory/final/answer 全 `1.0`；fit-only passed |
| `artifacts/v2-a/a1_11/runs/run-{1,2,3}/reasoner/formal/` | 三组独立 data/reasoner seed 的 checkpoint、history、训练子集诊断和六 split formal eval。 | formal `0/3`；ordinary 后干预未运行 |
| `artifacts/v2-a/a1_11/assessment-summary.json` | A1.9/A1.10 参考格、两臂 Gate、执行顺序、证据边界和一级归因。 | 当前 A1.11 机器真源；combination-only rejected，architecture not validated |
| `tmp/V2-A1.11 result.md` | 方法纠正、两臂结果、可能机制、完成/未完成项和 A1.12 建议。 | 人类可读结果交接 |

## A1.12–A1.17 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_12/overfit32/{bind,cursor,both}/` | 三条唯一变量臂的 length-balanced fit-only checkpoint/results。 | 三臂 overfit 均通过 |
| `artifacts/v2-a/a1_12/runs/run-{1,2,3}/{bind,cursor,both}/` | 九个 fresh formal checkpoint、训练与六 split 评估。 | formal 全 `0/3`；无干预 |
| `artifacts/v2-a/a1_12/assessment-summary.json` | binding × cursor 聚合。 | `binding_and_cursor_insufficient` |
| `artifacts/v2-a/a1_13/overfit32/` | GENERIC-CLOSURE、STRUCTURED-CE、STRUCTURED-CLOSURE fit-only 正控制。 | 三臂均通过 |
| `artifacts/v2-a/a1_13/runs/run-{1,2,3}/` | transition × closure formal 与 Gate 后干预。 | 联合臂 state `3/3`、full `1/3`；run-2 causal 通过 |
| `artifacts/v2-a/a1_13/assessment-summary.json` | state/full 分离的三臂与 A1.12 reference 聚合。 | overall `seed_unstable_inconclusive`；联合 state 稳定 |
| `artifacts/v2-a/a1_13f/runs/run-{1,2,3}/` | GENERIC-CE、GENERIC-CLOSURE、STRUCTURED-CE 的固定 4000-step 审计；复用原本已满预算的配对 run。 | state `0/3`、`0/3`、`1/3` |
| `artifacts/v2-a/a1_13f/assessment-summary.json` | 固定预算与 ordinary formal 的方法学判定。 | `fixed_budget_seed_instability_persists`；early-stop primary=false |
| `artifacts/v2-a/a1_15/` | query-coupled + answer CE overfit、三 fresh formal 与 assessment。 | overfit passed；state/full `0/3` |
| `artifacts/v2-a/a1_16/` | query-coupled + no answer CE overfit、三 fresh formal 与 assessment。 | overfit passed；state/full `0/3` |
| `artifacts/v2-a/a1_17/runs/run-{1,2,3}/{coupled_ce,coupled_noce}/` | A1.13 成功 seed 的两种 paired objective；共享初始化位相等。 | 两臂 state/full 均 `0/3` |
| `artifacts/v2-a/a1_17/assessment-summary.json` | reference、paired seed、共享初始化 tensor identity 与两臂正式结果。 | `independent_answer_auxiliary_gradient_required`；root localized，architecture not validated |
| `tmp/V2-A1.12-A1.17 root-cause result.md` | 故障排除链、最终判断、证据边界、完成/未完成项与 A1.18 建议。 | 当前人类可读根因交接 |

## A1.18/A1.18B 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_18/overfit/{qaux,saux}/` | QAUX 与 FINAL-SAUX length-balanced overfit32、训练checkpoint和辅助剥离部署artifact。 | 两臂严格 overfit Gate 通过 |
| `artifacts/v2-a/a1_18/runs/run-{1,2,3}/saux/` | FINAL-SAUX paired fixed-4000 training、部署formal与通过run干预。 | formal/causal `2/3`；final-only state credit seed 不稳定 |
| `artifacts/v2-a/a1_18b/overfit/tsaux/` | 在 FINAL-SAUX 失败seed上的 TSAUX overfit32 正控制。 | step 4000 严格通过 |
| `artifacts/v2-a/a1_18b/runs/run-{1,2,3}/tsaux/` | TSAUX paired fixed-4000 training、辅助剥离formal与causal。 | formal/causal `3/3` |
| `artifacts/v2-a/a1_18b/fresh/run-{1,2,3}/tsaux/` | model seed `20261821/22/23` 的 TSAUX fresh fixed-4000 training、部署formal与causal。 | formal/causal `3/3` |
| `artifacts/v2-a/a1_18b/assessment-summary.json` | overfit、FINAL-SAUX、TSAUX paired/fresh、完整性、成本和边界的机器总判定。 | `mechanism_solved=true`；当前机制机器真源 |
| `tmp/V2-A1.18 result.md` | A1.18/A1.18B 结果、根因、完成/未完成边界，以及已被 2026-07-17 路线重审后移的原 A1.19 建议。 | 当前人类可读机制交接 |

## P1-H1-WD 非正式机制实验

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `docs/v2-r1r-p1-h1-wd-overlap-residual-design.md` | 正向重合写入、公共残差 target、W/D/J 预算、双 Gate 与证据边界。 | 2026-08-21 冻结的新非正式合同；不恢复旧 H1 |
| `docs/v2-r1r-p1-h1-wd-execution-command.md` | 新 root、只读 predecessor、唯一命令与 fail-stop 规则。 | 只允许 `run-h1-wd`，不调用旧 formal launcher |
| `docs/v2-r1r-p1-h1-wd-failure-review.md` | W/D/J 结果、路径干预、matched-control CI、失败根因与后继设计约束。 | 当前 H1-WD 人类可读判决真源 |
| `src/yggdrasil_v2/r1_revalidation/h1_wd/` | 隔离的 overlap target、WD forward、无 family schedule、W/D/J trainer、评估与 runner。 | 已消耗非正式实现；旧 `h1/` 保持不变，不得用它创建同 identity 后继 |
| `experiments/v2_r1r_h1_write_delete.py` | H1-WD 唯一非正式 CLI。 | 唯一 root 已存在；再次调用必须在 mutation 前拒绝 |
| `tests/v2_r1r_p1_h1_wd/` | forward 等价、重合守恒、无 family schedule、参数冻结与短 W/D 回归。 | 新方向回归测试 |
| `artifacts/v2-r1r/p1-h1-wd-overlap-residual-screen-20260821-1/` | predecessor baseline、W/D/J 进度、四个 checkpoints、干预、matched control 与最终 `result.json`。 | 已消耗；`FAIL_H1_WD_NONFORMAL_MECHANISM`、`authorizes=nothing`，禁止删除后重跑 |
| `docs/v2-r1r-p1-h1-wd-decision-causal-design.md` / `docs/v2-r1r-p1-h1-wd-decision-causal-execution-command.md` | 真实 common-off margin 幅度、projection-output VJP/Fisher 方向、CPU target dataset、增量 W/D Gate、two-path J lock 与负 VJP 方向对照。 | 已消耗非正式合同；不读取旧 WD checkpoint，不授权 H1/F1/P2 |
| `docs/v2-r1r-p1-h1-wd-decision-causal-failure-review.md` | target PASS、W 增量拟合失败、零学习假阳性分析、parameter-reachable transfer 条件与封存 hash。 | 当前 decision-causal 人类可读判决真源 |
| `src/yggdrasil_v2/r1_revalidation/h1_wd_causal/` | 固定 route replay、真实消融、target dataset、无 teacher/family/route-supervision W/D/J、Shapley 与 fail-stop runner。 | 已消耗实现；root 存在后 runner 必须 mutation 前拒绝，不得创建同 identity 后继 |
| `experiments/v2_r1r_h1_wd_decision_causal.py` | decision-causal 唯一 CLI：`run-h1-wd-decision-causal`。 | 唯一命令已消耗；再次调用必须在 mutation 前拒绝 |
| `tests/v2_r1r_p1_h1_wd_causal/` | route replay 位等价、logsumexp margin、VJP/Fisher cap、target 物化、无 teacher/route supervision 与短 W/D/J 回归。 | 专项 `9 passed`；H1/H1-WD 合并回归 `76 passed` |
| `artifacts/v2-r1r/p1-h1-wd-decision-causal-screen-20260821-1/` | 预检、2.685 GB `causal-target-datasets.pt`、训练事件与终态结果；W Gate 前不保存 checkpoint。 | `FAIL_WD_DECISION_CAUSAL_WRITE_FIT`；无 crash/D/J/后继 root，必须原样封存，`authorizes=nothing` |
| `docs/v2-r1r-h1-wd-direction-geometry-screen-design.md` / `docs/v2-r1r-h1-wd-direction-geometry-screen-execution-command.md` | R0 global、R1 route、R2 target-before predictability、R3 shared-parameter Jacobian/Fisher reachability、R4 registered null，以及唯一启动与 fail-stop 合同。 | 已消耗非正式诊断合同；不是旧 WD root 的续跑，不授权训练、formal、F1 或 P2 |
| `docs/v2-r1r-h1-wd-direction-geometry-screen-failure-review.md` | replay Gate 终态、batch-geometry 根因、未测量的 R0–R4 边界、输入 hash 与潜在 successor 前置条件。 | 当前 direction-geometry 人类可读终局真源；机器终态为 `CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY` |
| `src/yggdrasil_v2/r1_revalidation/h1_wd_direction_geometry/` / `experiments/v2_r1r_h1_wd_direction_geometry.py` | v2 已直接切换为 exclusive preflight lease、pre-root 全量 batch-4 replay、R1–R4、active-site macro、target/feature fit null、双 sketch、negative-sign、regularization stability 与 finite-guard；CLI 仅保留 `run-h1-wd-direction-geometry-v2`。 | 已消耗 successor 实现；旧 v1/v2 identity 均不得续跑 |
| `tests/v2_r1r_p1_h1_wd_direction_geometry/` | geometry、ridge、record/active-site bootstrap、target/feature permutation、双 sketch 决策逻辑与 opt-in CUDA/artifact integration。 | 当前自包含回归 `26 passed, 1 skipped`；依赖已冷归档 fixed inputs 的旧工作区 preflight 断言已删除 |
| `artifacts/v2-r1r/h1-wd-direction-geometry-screen-20260823-1/` | contract/preflight/events/run-state/crash；无 `result.json`、checkpoint 或 optimizer/model 写入。 | 已消耗 CRASH root；replay `7.96914e-05/9.50396e-05 > 2.5e-05`，R2–R4 未运行，`rerun_authorized=false`、`authorizes=nothing` |
| `docs/v2-r1r-h1-wd-direction-geometry-v2-design.md` / `docs/v2-r1r-h1-wd-direction-geometry-v2-execution-command.md` | successor 的科学结果/执行 crash 双层状态、exclusive lease、batch-4 全量 preflight、R1–R4 完整控制、三 identity 基础设施预算与唯一命令。 | 已消耗 v2 合同；只求完整可信测量，不预设 signal，始终 `authorizes=nothing` |
| `docs/v2-r1r-h1-wd-direction-geometry-v2-result-review.md` | R1–R4 数值、route/attention 条件方向、projection shared-parameter 失败、双 sketch residual、target prevalence shift 与 null 分辨率边界。 | 当前 v2 人类可读终局真源；结论 `NO_QUALIFIED_R2_R3_COMPONENT` |
| `artifacts/v2-r1r/h1-wd-direction-geometry-screen-v2-20260823-1/` | contract/preflight/replay/events/null/result/run-state；sibling lease 与 stdout/stderr 记录单次启动。 | `COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`；R1–R4 完整，`authorizes=nothing`，禁止重跑 |

## A1.19H 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_19h/h1/` | H1 overfit32、三组 fixed-cardinality formal、structural/address/recurrence 干预与 assessment。 | formal/causal `3/3`；`opaque_handle_hybrid_core_confirmed` |
| `artifacts/v2-a/a1_19h/h2/overfit32/` | N2/N3/N4、T1–16 cell-balanced fit-only checkpoint/results。 | 严格 overfit Gate 通过 |
| `artifacts/v2-a/a1_19h/h2/runs/run-{1,2,3}/data/` | 三组独立 data seed 的 N2–N5、relation/OOD/causal 数据。 | 三组 data audit 通过；跨 run fingerprint 排除生效 |
| `artifacts/v2-a/a1_19h/h2/runs/run-{1,2}/formal/` | 两组 fixed-4000 variable-cardinality checkpoint、部署模型和 formal eval。 | N2–N5 与 relation Gate 全通过；causal intervention 全通过 |
| `artifacts/v2-a/a1_19h/h2/throughput-optimization.json` | 旧逐步 CUDA 编码/同步与新 GPU-cache 管线的同初始化同 batch 序列 60-step 对照。 | `10.744×`；final loss delta `0`；parameter max delta `0` |
| `artifacts/v2-a/a1_19h/h2/runs/run-1/formal/formal-eval-optimized-regression.json` | 优化后 forward/cached evaluator 对旧 run-1 checkpoint 的全 split 回归。 | 原/新 gates 与 splits 逐字段完全相同 |
| `artifacts/v2-a/a1_19h/h2/runs/run-3/formal/` | model/data/mapping seed `20261963/20261953/20261973` 的 fresh optimized fixed-4000 formal。 | formal/causal 通过；训练 `155.52s`、`25.72 step/s` |
| `artifacts/v2-a/a1_19h/assessment-summary.json` | H1、H2 overfit、三数据 audit、三 run formal/causal 与完整性聚合。 | `generalized_hybrid_core_confirmed`；A1.19H 机器真源 |
| `docs/v2-a1.19h-hybrid-core.md` | 架构合同、方法修正、H1/H2 结果、吞吐等价与证据边界。 | A1.19H 人类可读正式记录 |
| `tmp/V2-A1.19H result.md` | 完成/未完成项、异常修正与 A1.20B 交接。 | 当前阶段结果交接 |

## A1.20B 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_20b/overfit32/` | 32 条 N2–N4/T1–16 full-token cache、audit、fresh checkpoint 与逐 cell 结果。 | mapping/trajectory/final/answer 全 `1.0`；fit-only sanity 通过 |
| `artifacts/v2-a/a1_20b/qwen-cache-batch-benchmark.json` | batch 8/12/16 的真实 full-text Qwen 编码 wall time、real tokens/s 与 peak allocation。 | 选择 batch 12，`2602.43 real tokens/s` |
| `artifacts/v2-a/a1_20b/runs/run-1/train-cache/` | train 16,384 + validation 2,048 条 FP16 Qwen last-hidden mmap shards；共 `5,594,417` source tokens。 | full-text Boundary run-1 输入；无 oracle 字段 |
| `artifacts/v2-a/a1_20b/runs/run-1/train-cache-audit.json` | schema/hash/source/shape/finite/fingerprint/overlap/no-oracle/no-truncation Gate。 | 全部通过 |
| `artifacts/v2-a/a1_20b/training-pipeline-benchmark.json` | 同 schedule/initialization 的同步 mmap 与后台 pinned prefetch 20-step 对照。 | prefetch 仅 `0.939×` 且非参数级等价；正式保留同步路径 |
| `artifacts/v2-a/a1_20b/runs/run-1/formal/results.json` | reader seed `20262011` 的 fixed-5000 Boundary training 与 2048 条 heldout validation。 | trajectory/final/answer `0.297363/0.409668/0.608887`；formal eligibility 失败 |
| `artifacts/v2-a/a1_20b/runs/run-1/formal/failure-diagnostic.json` | 15 个选择性 oracle 条件、联合 Boundary/program/address exact 与 N/T 分组。 | full oracle core `1.0`；value/source/target binding 为 primary failure |
| `artifacts/v2-a/a1_20b/runs/run-1/formal/gradient-credit-audit.json` | state CE-only 与 full factorized loss 对各 Boundary logits 的梯度范数。 | state CE 到六类离散控制 logits 全为 `0`；hard-control credit 断裂确认 |
| `artifacts/v2-a/a1_20b/assessment-summary.json` | overfit、cache、run-1、oracle、gradient、未运行阶段与证据边界的机器聚合。 | `a120b_passed=false`；A1.21P/A1.22A 不允许 |
| `docs/v2-a1.20b-full-text-boundary.md` | 正式合同、优化、结果、根因与路线判断。 | A1.20B 人类可读正式记录 |
| `tmp/V2-A1.20B result.md` | 完成/未完成项与停线交接。 | 当前阶段结果交接 |

## A1.20C 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_20c/overfit32/supervision/` | overfit32 train/validation token anchor target 与 manifest。 | 训练专用；不进入 forward |
| `artifacts/v2-a/a1_20c/overfit32/supervision-audit.json` | token offset、shape/range、source 对齐与 no-forward-input 审计。 | 全部 Gate 通过 |
| `artifacts/v2-a/a1_20c/runs/run-1/supervision/` | 16,384 train + 2,048 validation 的 compiler anchor target。 | 已准备；正式 matrix 因 target overfit 失败未使用 |
| `artifacts/v2-a/a1_20c/runs/run-1/supervision-audit.json` | run-1 compiler supervision 审计。 | 全部 Gate 通过 |
| `artifacts/v2-a/a1_20c/smoke/hierarchical__straight_through/` | 2-step CUDA smoke、checkpoint 与 hard-forward equivalence。 | state/answer logit delta `0/0` |
| `artifacts/v2-a/a1_20c/overfit32/hierarchical__straight_through/results.json` | fresh fixed-5000 目标臂 overfit32 结果。 | trajectory/final/state-token/answer `0.875/0.90625/0.964474/1.0`；strict Gate 失败 |
| `artifacts/v2-a/a1_20c/overfit32/hierarchical__straight_through/failure-diagnostic.json` | 四个失败 N4 cell、state-vs-local 梯度夹角、late regression 与停止列表。 | 机器分类 `anchor_localization_solved_but_execution_objectives_conflict` |
| `docs/v2-a1.20c-boundary-repair.md` | 2×2 合同、实现、overfit、梯度冲突、Gate 与下一修复边界。 | A1.20C 人类可读正式记录 |
| `tmp/V2-A1.20C result.md` | 已完成/未完成项与停线交接。 | 当前阶段结果交接 |

## A1.20D/A1.21P 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_21p/routing-{inrange,n5}-entity-capacity-threshold.json` | N4 伪第五实体与 N5 真第五实体在粗/refined section boundary 下的 pair-gain 扫描。 | A1.20D 根因证据 |
| `artifacts/v2-a/a1_21p/joint-shared-family-canonical-formal-bidirectional-probe.json` | 同一 checkpoint 的 canonical 八 split 完整矩阵。 | 全部诊断 Gate 通过；源 artifact `formal_gate=false` |
| `artifacts/v2-a/a1_21p/joint-shared-family-canonical-hidden-causal-audit.json` | normal、zero hidden、batch roll、token reverse 与 core integrity。 | hidden causal Gate 通过 |
| `artifacts/v2-a/a1_21p/joint-shared-family-routing-ood-{bidirectional-probe,seed2,seed3}.json` | 三个 continuation schedule 的 routing OOD/N5 probes。 | 路径稳定性 `3/3`；非 fresh model/data seed |
| `artifacts/v2-a/a1_21p/k1/full-probe/results.json` | K=1 2048 validation 容量基线。 | trajectory/final/answer `0.106934/0.333008/0.550293` |
| `artifacts/v2-a/a1_21p/p{0-current,1-routing}-family-chat-smoke5.json` | official chat template 下 direct/text-CoT/hybrid 在线五条 Pareto smoke。 | candidate 正证据；非正式样本规模 |
| `artifacts/v2-a/a1_21p/routing-data-v2-ood-small/manifest.json` | routing 数据来源、split 和语义变换声明。 | `surface_only_transform=true`；不得作为新代数 |
| `artifacts/v2-a/a1_21p/assessment-summary.json` | 机制门、正式门、来源 SHA-256、完成/缺失项与停机判定。 | `a121p_passed=false`；`a122a_authorized=false` |
| `docs/v2-a1.20d-full-text-mechanism-repair.md` | 三遍双向 section 修复、结果、架构解释和证据边界。 | A1.20D 人类可读记录 |
| `docs/v2-a1.21p-pareto.md` | K1、在线 smoke、六项正式缺口和下一轮合同。 | A1.21P 人类可读正式记录 |
| `tmp/V2-A1.20D-A1.21P result.md` | 完成/未完成项与 A1.22A 停机交接。 | 当前阶段结果交接 |

## A1.5 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_5/data-large/manifest.json` | A1.5 v1 4096/256 split、必要性与泄漏 manifest。 | 当前数据合同证据 |
| `artifacts/v2-a/a1_5/p0-overfit-final/` | 32-example P0 overfit、checkpoint、history 和 results。 | surrogate positive-control |
| `artifacts/v2-a/a1_5/p0-final/` | 4096-example P0 best checkpoint、best-eval、训练曲线和干预结果。 | surrogate probe；A1.5 未通过 |
| `artifacts/v2-a/a1_5/p1-hidden-cache-final/` | Qwen3.5-2B FP16 分片 hidden 与 operation span mask。 | mechanism smoke |
| `artifacts/v2-a/a1_5/p1-final-small/` | P1 small real-hidden underfit 训练与 warm-up/state 诊断。 | surrogate probe；训练预算不足 |
| `artifacts/v2-a/a1_5/p1-hidden-cache-large/` | Qwen3.5-2B FP16 4096/256 全 split hidden cache、显式 span mask 和版本 manifest。 | mechanism-smoke；无静默截断 |
| `artifacts/v2-a/a1_5/p1-final-large/` | P1 4096-cache warm-up/joint formal checkpoint、best reload 和 eval-all。 | surrogate-probe；ordinary validation 通过 |
| `artifacts/v2-a/a1_5/p2-final-large/` | P2 K=8 formal checkpoint、训练历史、结果和 test/composition/length 干预。 | surrogate-probe；composition 失败 |
| `artifacts/v2-a/a1_5/text-test-0-08b-one.json` | A1.5 schema 上 Qwen3.5-0.8B zero-shot、无人工总输出 cap 单样本 text-CoT smoke。 | smoke；final/state `0/0` |
| `artifacts/v2-a/a1_5/text-test-0-08b-formal.json` | Qwen3.5-0.8B matched test zero-shot 128 条；无 token cap，118 semantic terminal、10 safety timeout，parse/final/state `0.1797/0.0625/0.0234`。 | probe；需结合 termination_reason 解读 |
| `artifacts/v2-a/a1_5/text-composition-0-08b-formal.json` | Qwen3.5-0.8B matched composition-heldout zero-shot 128 条；无 token cap，parse/final/state `0.4063/0.3828/0.3750`。 | probe；final 低于 prefix-2 |
| `artifacts/v2-a/a1_5/text-length-0-08b-formal.json` | Qwen3.5-0.8B matched length-heldout zero-shot 128 条；无 token cap，parse/final/state `0.0625/0/0`。 | probe；长度外推失败 |
| `artifacts/v2-a/a1_5/text-test-2-08b-formal.json` | Qwen3.5-0.8B matched test 2-shot 128 条；无 token cap，parse/final/state `0.9766/0.1328/0.0156`，3 条 timeout。 | probe；格式改善但 state 失败 |
| `artifacts/v2-a/a1_5/text-composition-2-08b-formal.json` | Qwen3.5-0.8B matched composition 2-shot 128 条；无 token cap，parse/final/state `0.9219/0.1953/0`，7 条 timeout。 | probe；composition state 失败 |
| `artifacts/v2-a/a1_5/text-length-2-08b-formal.json` | Qwen3.5-0.8B matched length 2-shot 128 条；无 token cap，parse/final/state `0.7656/0.0938/0`，29 条 timeout。 | probe；长度 state 失败 |
| `artifacts/v2-a/a1_5/text-{composition,length}-0-08b-formal.json.partial.json` | no-cap 批次的逐步进度快照；最终 JSON 已完成时仅作终止/进度追溯，不作为质量结果。 | 2026-08-01 已清理；正式 JSON 位于 D 盘外部归档 |
| `artifacts/v2-a/a1_5/p2/smoke.json` | K=8 learned-slot synthetic smoke。 | mechanism smoke |

## 保留的用户资料

| 路径 | 用途 |
| --- | --- |
| `note.txt` | 用户笔记；按要求保留，不删除。 |
| `下一件事.txt` | 用户待办笔记；保留。 |
| `HF token.txt` | 用户本地文件；保留，不在文档中展开其内容。 |
| `thinking/`（不含 `thinking/research/`） | 被 `.gitignore` 排除的用户原始思考稿；五篇近期草稿只读并保留原文，`Project-Yggdrasil 未来多模态潜空间智能体架构.md` 作为项目初期 V1 历史参照；正式整理见 `docs/thinking-draft-synthesis-2026-08-01.md`。 |
| `.gitignore` | 本地环境、缓存、`tmp/` preflight/lease/output、仅本机 `AGENT.md`、生成的 `*.egg-info/` 和未来实验产物的忽略规则。 |

## 旧路线归档

`archive/legacy-proxy-route-2026-07-11/` 是一次完整的非删除归档，保留切换前的原始相对目录：

| 子目录或文件 | 内容 | 当前地位 |
| --- | --- | --- |
| `docs/` | 旧阶段报告、V1 白皮书、旧路线和旧命令清单 | 历史思想与代理证据 |
| `experiments/` | Stage A—AV-J-C 及相关代理脚本 | 不再运行 |
| `src/` | 旧代理公共实现 | 不再作为 V2 基座 |
| `tests/` | 绑定旧契约的测试 | 不再作为当前验收 |
| `artifacts/` | 旧 smoke/probe/formal 输出 | 只用于追溯和失败分析 |
| `thinking/` | 旧路线思考稿 | 历史材料 |
| `pyproject.toml` | 旧实验依赖和 pytest 配置 | 随旧代码归档 |
| `README-legacy.md` | 归档前 README | 历史入口 |

归档中的报告可能仍引用原始根目录路径；这些引用用于还原当时的实验语境，不构成当前可运行入口。后续 V2 实现必须新建代码和测试，不在旧代理代码上增加兼容 wrapper。

## 维护规则

新增 V2 文件应先归入明确的 V2-A/V2-B/V2-C 语义包，再更新本索引。实验结果只能在 evidence level、配置、seed、heldout、消融、成本和失败边界齐全后进入当前证据；旧 Stage 结果不得静默升级为 V2 架构证据。V2-C C0-C4 只提供系统组件归因，不能单独冒充 C5 `integrated-system`。再次废弃的当前文件应移动到带日期的 `archive/`；若用户明确要求移出仓库，则记录等价的 D 盘外部归档路径，不保留并列旧入口。
