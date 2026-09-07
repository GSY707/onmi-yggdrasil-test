# 下一阶段测试任务规划

更新日期：2026-09-02

架构真源：`docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md`

总路线：`docs/Project-Yggdrasil V2 从架构验证到商用路线图.md`

当前状态：A1.19H/A1.20D 的正证据仍集中在 COPY/SWAP 切面，A1.21P 因任务同构、baseline 不公平和 formal 缺口停止。V2-R1R 的 v17 production data 与 P0-M v5 training-path smoke 已 accepted，P1 v6/v6R 已资格化 causal-temporal witness。P1 v7 保留 temporal/state 机制但 CPS causal/OOD 失败；v8R 在 shared v7 checkpoint、完整 rehearsal 与同 mixed schedule 下仍失败。v8D 证明 CPS perturbation 已进入并被 `H_T` 放大，却没有形成跨 pair semantic reduction；v8L 又在 bootstrap 得到 ERE optimization/audit `0.7726/0.7109`、CPS `0.5682/0.5281` 后 fail-stop。匿名 K=8 + lexical-anchor 主线保持关闭，fresh v9 永久未获授权。P1-NR1 已以 N01–N07 全 true 的正式 PASS 关闭 numeric/relation measurement 前置缺口。H1 的完整-transition、互斥完整 FFN、2:1 residual、等分 residual 与 factorized routed projection 五个非正式方向均已否决；最后一个 full-budget screen 的 overall gain 为 `-0.01074`，wrong-route/conditional-write 最大效应仅 `0.00281/0.00781`，`authorizes=nothing`。calibration 与 H1 formal 未启动。K=1、文本基线、完整 P1/Pareto 与 P2 仍未完成。

2026-08-21 的两个 H1-WD 非正式后继也都已 fail-stop：overlap-residual 只实现欧氏输出重参数化；decision-causal 又在 target Gate PASS 后因 W heldout transfer-nMSE `1.00450/0.99247` 停止，D/J 未运行。2026-08-23 唯一启动的 R0–R4 direction-geometry 诊断则在全量 replay identity Gate 以 `CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY` 停止；common/projection 漂移 `7.96914e-05/9.50396e-05` 超过 `2.5e-05`，R2–R4 未运行。相关 roots 均已消耗，仍不授权 calibration、H1 formal、F1 或 P2。

direction-geometry v2 successor 已唯一完成并取得完整 R1–R4，机器终态为 `COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`，科学终态为 `NO_QUALIFIED_R2_R3_COMPONENT`。route-ID pairing 有显著语义，但 heldout route increment 仍为 `-0.586%`；projection trunk 的 `+3.189%` CI 跨零且 lambda 不稳定，exact shared head/sample local-J 分别为 `-0.334%/-5.922%`。R4 双 sketch residual 不与 current null 相容，但未完成 latent structure 归因。最强混杂是 nonzero target prevalence 从 train `4010/4096` 降到 heldout `591/1024`；该诊断当时只支持“若续研先测 target-before write gate”，不授权加强 projection。Closure C0 已在下段直接覆盖这条局部后继，direction-geometry 现在只作历史诊断，仍为 `authorizes=nothing`。

项目随后停止局部 H1/WD 研究并返回整体 V2-A。2026-08-23 的 Closure C0 任务/公平基线资格审计已按 `V2-A-CLOSURE-C0-20260823-1` 单次完成：C001/C002/C003/C005/C007/C008 通过，C004/C006 失败。C004 确认 v17 ordinary ERE relation-query 由 generator 构造为全 TRUE，严格 source-only visible-legend oracle 在 train/validation/language OOD 条件准确率均为 `1.0`；这只污染约 8.33% 子组、最多抬升整 split 约 4.17pp，不等于 ERE/CPS 整体无效。C006 确认历史 compact trace 只有 formatter 与 final-answer parser，没有 full-bank roundtrip、semantic replay 或 fault-kill。机器终态 `FAIL_V2_A_CLOSURE_C0_READINESS`，seal `462317E7…9972`，没有训练或 C1 授权；该失败 root 保持不变。

2026-08-24 的直接 successor C0R 已完成修复并单次正式封存。新 relation-balanced bank 的四个 relation cell 均为 TRUE/FALSE 配平，source-only oracle accuracy `0.5`；26,624 条 compact trace 全部 roundtrip/replay，`394,278/394,278` 定向故障被 kill，token max `389 < 512`。data/trace D001–D012 与 readiness C001–C008 全 PASS，readiness 机器终态为 `PASS_V2_A_CLOSURE_C0R_READINESS`，seal `161FBB36…0BA4`。两阶段均为零训练，`four_arm_results_present=false`、`v2a_passed=false`；其当时唯一授权是 C1 single-seed implementation/eligibility，不再研究 projection write gate，也不得提前运行 C2。

该 C1 授权已经消费。旧 C1 joint answer/trace formal 在 G007 trace credit fail-stop；随后完成的只读归因确认随机 64-token chunk 存在 exposure hygiene 缺口且共享梯度冲突真实，但二者不足以解释整体 poststop 失败。独立 C1R 使用 fresh answer-only Stage A、fixed 6144-step endpoint，并把 complete-target trace probe 放到条件 Stage B；唯一 C1R formal 仍在第一个行为门 G004 失败：ERE/CPS validation accuracy `0.191406/0.156250`，train 为 `0.194092/0.152832`，Stage B 与其余 Gate 全部未运行。标签/legend/join 全量核对 0 mismatch。冻结轨迹的 H0/H1/H10 slot-centered energy 为 `195.31/23.34/0.140`，off-diagonal cosine 为 `0.94096/0.99299/0.999958`；H10 learned-query 与 uniform mean 的预测完全一致、平均 logit L2 差约 `1.13e-6`。准确结论是 final answer path 的 effective K≈1，而不是整个 latent/trace 空间数学上 K=1。C1R seal 完整且 `authorizes=nothing`，不得重跑、换 seed/checkpoint、补跑 Stage B 或直接进入 C2。

C1S Addressed Content Workspace 已按新 package、新合同和新 output identity 直接切换。唯一 S0 zero-update 机制资格通过后，SRW S1 又以 fresh K8 固定 4,000 updates 完整运行：ERE/CPS answer 各 `16/16`，但 no-core overall 仍为 `28/32`，同记录至少两个因果贡献者仅 `7/32`、mean causal effective slots `1.58099`，且 CPS `running_best/final_winner` 与 ERE `changed/operation_source/touched` 未达到 S1 的 `0.95` 动态 BA 下限。机器终态为 `FAIL_V2_A_C1S_S1_QUALIFICATION`，result/seal `9D0B06E1…E38892`/`FAFF301A…68A9D2`，63/63 replay。S2、formal、C2 与 V2-A PASS 均未授权，当前 `authorizes=nothing`。

随后唯一失败归因 diagnosis 的 preflight PASS，但 diagnosis 在 D004 nested readout 处 sealed CRASH：`CPS.running_best` 只有 12/16 条 record-macro eligible records，未分层的哈希 inner fold 出现 0 valid-record cell。D001–D003 的 sealed 部分证据仍按边界保留，旧身份禁止修补或重跑。新的 `TEMPORAL-ATTRIBUTION-REPAIR` 身份现已唯一完成 D004R/D005R：target-only folds、时序运动、固定通道分离、nested cross-fit readout、三类 fit-null、10k score-null 与 frozen decoder 均形成完整证据；CPS/ERE 因没有命中注册的 target/latent/readout 故障而均为 `INCONCLUSIVE`。result/seal `D1603C62…06A496`/`E8A61CF8…1562B0`，103/103 replay；准确结论是“Axis C 未找到主故障”，不是机制 PASS，结果 `authorizes=nothing`。

2026-09-01 已按新 identity 实现并唯一完成 C1T Causally-Partitioned Workspace S0。它不再让 entity/operation/query readers 读取同一份完整 source hidden，而把每个对象卡、操作卡和查询卡作为一次独立冻结编码调用；卡间只能经公开 record-local address、注册 source→target operation 与 target-only transition 相互作用。新的 ERE XOR 和 CPS validity-parity 都是完整 2×2 factorial，翻转任一支持对象都改变 simulator answer，且只改变对应对象卡。preflight P001–P005 与正式 S001–S008 全 PASS；192 张卡由固定 Qwen3.5-2B 执行 192 次独立 forward，8 个 group 的 no-core max delta 全为 `0.0`，cache 写后 readback、对象隔离、目标槽唯一写入、置换等变和 BF16 backward 均通过。正式 result/seal 为 `831BC5…D6A8`/`E4D9B3…A0EE`，optimizer/model writes 为 `0/0`。这只授权 `C1T_S1_CONTRACT_DESIGN_ONLY`，训练仍未授权。

该 S0 授权随后由用户明确启动的 fresh C1T S1 消费。唯一 preflight P101–P105 全 PASS，随后唯一正式 Overfit32 按冻结 schedule 完成 4,000 updates 并只写 `fixed_4000`；运行本身与 43/43 seal replay 完整，但 R103/R105 失败。overall answer 为 `22/32`，完整 factorial group 仅 `1/8` 全对；CPS/ERE two-contributor 分别为 `8/16` 与 `4/16`。no-core accuracy 已降至 `2/16` 与 `4/16`，且 no-core margin-drop/置换等变过门，这证明结构量具有效，却不能覆盖行为拟合和逐记录双对象因果失败。机器终态 `FAIL_V2_A_C1T_S1_QUALIFICATION`，result/seal/endpoint 为 `9B3762…C1DFF`/`89C530…9BCF`/`E7E66F…96F9`，`authorizes=nothing`；同一 identity 禁止重跑、续训、换 seed 或挑 checkpoint。

随后完成的 post-stop 诊断把失败定位为多组共享训练中的 operation-2 dead-gate，而不是数据、route 或结构旁路。失败区域第二次 sigmoid gate 为 `10^-14–10^-6`，初始值约 `0.54`；支持对象 payload 差异仍存在，却在 final target/logits 前被截断。ERE-g00 的单条 hinge 梯度约 `4.95`，完整四格平均后仅 `1.92e-6`；counterpart stop-gradient ablation 将 factor-0 聚合梯度提高约 3,900 倍，但被关闭的 factor-1 仍近零。训练在 update 2198–2564 出现密集 pre-clip spikes 后进入 g01-only/半 parity 双平台，余弦 LR 随后锁定。fresh CUDA BF16 单组 ERE-g00 对照在 update 250 达到 full/support/two-contributor 全 `4/4`，排除了单组不可学习与基础 XOR 表达能力不足。该诊断 `authorizes=nothing`，没有修改正式 root 或创建 successor。

用户随后授权按 Gate 顺序建立 fresh C1U Publicly-Grounded Gate-Free Workspace。唯一 S0 preflight/S0 已 sealed PASS，公开语义桥、真实 192-card Qwen cache、gate-free target overwrite 与 operation-2 梯度机制全部资格化。随后冻结的 fresh S1 保留 C1T 的 causal objective、optimizer、完整 group batch、4,000-step schedule 与行为 Gate，只移除 learned sigmoid gate；唯一 preflight P101–P105 全 PASS，唯一正式 S1 恰好完成 4,000 updates、单一 `fixed_4000` 写入，并在 R101–R107 全 PASS。终局为 answer `32/32`、factorial `8/8`、CPS/ERE no-core `4/16` 与 `2/16`、两个 support flip 与 two-contributor 均为每族 `16/16`；result/seal/endpoint `1FCD2D…6090C`/`A5211D…5350`/`88F538…159F`，44/44 replay。source identity `8BD618…94FD`，授权仅为 `C1U_S2_MATCHED_K1_K8_CONTRACT_DESIGN_ONLY`。

2026-09-02，用户明确授权“只做多 bank，多 seed 先不做”，并启动 C1U S2。冻结 successor 使用六个 fresh、opaque-disjoint banks 和三个 4-train/2-heldout folds；三个 folds 与 K1/K8 两臂都重置到同一 scientific model seed `2026090211`。唯一 preflight P201–P207 sealed PASS；唯一 formal 完成六个 `fixed_4000` endpoints、24,000 updates 与六次 model write，但 R203/R204/R205 FAIL。K8/K1 heldout overall 为 `52/192`/`50/192`，K8 factorial `3/48`，overall paired gain 只有 `+1.04pp` 且 bank-cluster 95% 区间 `[-6.77pp,+9.38pp]`。formal result/seal 为 `F0EABD…6EAA4`/`00B782…25A4B`，76/76 replay，终态 `FAIL_V2_A_C1U_S2_QUALIFICATION`、`authorizes=nothing`。

post-stop 只读归因进一步确认 K8 是 bank-conditional memorization：三个 endpoint 对各自 train banks 为 `128/128,128/128,126/128`，却在 heldout 只有 `18/64,16/64,18/64`。固定已拟合 B2，仅轮换公开 Choices 时，八个非零 rotations 合计 CPS/ERE 只有 `25/128`/`4/128` 等变；保持任务和标签不变、只把 64 个 opaque symbols 一致 alpha-rename 后，CPS/ERE 从各 `16/16` 降为 `0/16`/`4/16`。根因是 pooled query + fixed A–I head 不具 choice-label equivariance，whole-card pooled payload + MLP transition 不具 opaque-renaming invariance。当前建议仅为全新 symbol-bound choice workspace 的合同方向；尚未授权实现、S0、训练或新 root。

## 1. 路线直接切换

当前路线不再继续 AV-J-D，也不继续给 AV-J/AV-J-B/AV-J-C 的 record reward、candidate mask 或 no-source shortcut 打补丁。

旧 Stage A—AV-J-C 保留为代理实验历史，最高只能提供 mechanism/surrogate 证据。新的当前主线从 V2-A 开始：

1. **V2-A：推理介质。**先比较 public/private 离散介质的逐 token 自回归与并行块生成，以及确定性的单向量/多向量 latent recurrence；固定臂资格化后，再判断是否需要带显式转换成本的融合体。
2. **V2-B：静态多模态模型核。**使用 V2-A 资格化的固定介质或融合介质，验证固定输入/输出下的 Boundary-MoE、FFN-MoE、Attention Pump 和文本/视觉/动作表示。
3. **V2-C：多层、多线程、树图混合全双工智能体。**把主动 Boundary、真实工具、工作树/记忆、人类目标连续性和隔离专家演化放在一个系统合同中验证。

在 V2-A 正式通过前，不实现 V2-B；在 V2-B 正式通过前，不实现 V2-C。V2-C 的任务、协议、基线和 Gate 可以预先设计，但不得借此接入真实工具、启动系统训练或宣称整机能力。

## 2. 共同证据口径

### 2.1 证据等级

- `smoke`：前后向、checkpoint、结果 JSON 或单个接口能运行；
- `probe`：单 seed 或缩小数据的方向信号；
- `formal`：预注册配置、多 seed、heldout、消融和成本指标齐全；
- `architecture-fidelity`：成熟基座、连续 recurrence、无旁路、审计和专家边界符合 V2；
- `architecture-formal`：高保真实现相对同预算强基线形成可重复优势。

### 2.2 所有实验必须报告

- 参数量：冻结参数、可训练参数和 active parameters 分开；
- 数据量：unique tokens/examples 与 processed tokens 分开；
- 质量：最终 exact/episode success、heldout、长度外推和分任务指标；
- 因果：no/shuffled input、no/shuffled latent、截短 trajectory 和必要 modality 消融；
- 成本：训练时间、推理延迟（median/p95）、模型前向次数、categorical sampling 次数、输出 token/code 数、latent transitions、介质转换次数及其成本、峰值显存、KV/激活和估算 FLOPs；
- 稳定性：smoke 之外至少记录 seed，formal 默认 3 seeds；
- 恢复：latest/best checkpoint、resume、设备、schema version 和中断原因；
- 边界：失败原因、未测试项和不能外推的能力。

### 2.3 禁止的成功口径

以下指标不能单独判定成功：loss、latent cosine、MSE、单次 answer accuracy、teacher-forced 指标、模型自述或更大参数量。

## 3. V2-A：推理介质实验

### 3.0 当前 Closure 顺序

历史 A0–A1.21P 与 V2-R1R 现在都作为 Closure 输入，不再各自续跑。活动顺序固定为：

1. **C0R 已完成。**新 production generator identity、visible-pattern oracle、strict compact-trace parser/replay/fault-kill、四臂 public-input 与公平合同均已资格化；旧 v17/C0 roots 不修改。
2. **旧 C1 与 C1R 均已消费并失败。**旧 C1 停在 G007；C1R fresh answer-only Stage A 停在 G004。二者均 `authorizes=nothing`，不得重跑、补 Gate 或继续调同一 dense core。
3. **C1S S1 已失败停止。**答案 `32/32` 不能覆盖 recurrence necessity、functional-K 与动态状态 Gate；sealed root 不得重跑、改阈值、换 endpoint 或延长训练。
4. **C1S S2 与 formal 不运行。**matched K1/K8 discovery 的前置资格没有获得，禁止创建其 root/lease；该旧路线不再恢复。
5. **Temporal diagnosis 已关闭 Axis C 缺口。**结果证明时序 target、latent motion 与 readout 都存在，故不再继续修 temporal probe；它没有证明 recurrence necessity 或 multi-address。
6. **C1T S0 是已消费的 predecessor PASS。**真实逐卡 Qwen cache、完整 factorial/no-core 结构量具、target-only transition、source-only forward、CUDA/BF16 与 accounting/seal 均已资格化；它只曾授权 S1 合同设计，不是学习证据。
7. **C1T S1 已 sealed FAIL。**唯一 fresh Overfit32 完成固定 4,000 updates；no-core necessity 通过，但答案 `22/32`、factorial exact `1/8`，CPS/ERE two-contributor 仅 `8/16` 与 `4/16`，故 R103/R105 失败。sealed root 不得重跑、续训、换 seed、选 checkpoint 或降低 Gate。
8. **C1T S1 失败归因已关闭主要缺口。**直接机制是 operation-2 dead-gate；上游是完整 factorial hinge 抵消、未 detach counterpart 的跨 cell 耦合和共享多组优化的 mode switch。单组可学对照排除 task/architecture 基础不可表达，但不构成 full-32 修复资格。
9. **C1U S0 已 sealed PASS。**唯一 preflight/S0 均通过并完整复放；其零训练授权只允许新增 S1 合同与执行层，不是学习或泛化证据，同一 S0 identity 不得重跑。
10. **C1U S1 已 sealed PASS。**唯一 preflight/formal 均已消费且完整复放，同一 identity 禁止重跑、续训或替换 endpoint；其 S2 合同设计授权已经消费。
11. **C1U S2 已 sealed FAIL。**唯一 preflight PASS，唯一 formal 完成全部 24,000 steps 后在 fresh-bank absolute behavior、paired K8>K1 与 heldout causal behavior 三门失败；functional address sensitivity 不能覆盖错误方向。旧 identity 禁止重跑、续训、换 seed 或补 Gate，multi-seed 与 S3 不运行。
12. **后继只形成修复建议，未获授权。**若继续，应以全新 identity 直接切换到 record-local public symbol binder + per-choice shared scorer，先用 zero-training alpha-renaming invariance 与 choice-label equivariance Gate 资格化，再重建 S1→S2 证据链；不得把它叫作 S2 retry。

C0R 的历史授权已经由 C1 消费，不能再次使用。A1.19H 可复用的是 opaque-address/continuous-payload 状态代数、targeted transition 与因果 Gate；A1.20D 可复用的是粗到细双向 Boundary closure。它们不是可直接拼接的旧 checkpoint 或 formal 资格。任何旧 checkpoint、route/family target、旧 K=1 数值或 Pareto smoke 都不得作为 successor 结果输入。

### 3.0.1 V2-A Closure C2：扩展推理介质与融合合同（未来；当前仅设计）

#### 核心判断与阶段边界

C2 不预设所有任务上必须产生一个全局胜者。它先回答六种固定推理介质是否形成可重复、任务条件化的不同 Pareto 区域，再回答一个支付了路由、转换和失败恢复成本的融合体，是否能在未知 heldout episode 上超过最强固定臂。若不同任务确实偏好不同介质，“没有单一全局胜者”是允许的实验结论；但“各有优势”本身不等于融合成功。

本段只更新未来 C2 的活动规划，不授权实现、训练、建 cache、创建 root/lease 或运行 screen/formal。只有 C1U matched learned K1/K8 的全部前置阶段通过，并由新的冻结 successor 合同明确授权 C2 后，才可实施。C0/C0R 的历史四臂合同与 seal 保持原样；它们提供任务、compact trace、public-input、公平性和成本原则，不因这里新增离散臂而被追溯改写为七臂结果。

#### 固定介质矩阵

C2 的第一阶段固定为一个无中间推理介质控制和六个推理介质臂。融合体不与固定臂同时训练，也不计作第八个固定臂。

| 臂 | 中间载体 | 每个 reasoning macro-step | 主评测随机性 |
| --- | --- | --- | --- |
| `direct` | 无中间介质 | 直接进入统一最终答案接口 | 无中间采样；最终答案解码与其余臂锁定相同 |
| `public_ar1` | 人类可读、公开语义 token | 生成一个 categorical distribution，硬采样一个 token；下一步读取已采样历史 | stochastic |
| `public_block_b` | 人类可读、公开语义 token | 一次前向同时生成 `B` 个位置分布并并行硬采样 `B` 个 token | stochastic |
| `private_ar1` | 训练中新建的私有 codebook | 生成一个 code distribution，硬采样一个 code；下一步读取已采样历史 | stochastic |
| `private_block_b` | 训练中新建的私有 codebook | 一次前向同时生成 `B` 个位置分布并并行硬采样 `B` 个 code | stochastic |
| `latent_k1` | 单个连续向量状态 | 一次确定性 latent transition | deterministic |
| `latent_k8` | 八槽连续向量状态 | 一次确定性 multi-slot latent transition | deterministic |

`B > 1`，但本计划不提前把 `B` 与 `K=8` 视为等价。`B` 是单次离散发射带宽，`K` 是同时存在的连续状态槽数；主 `B` 必须在 C2 冻结合同中依据 train-side 容量/成本资格预注册，并在查看 heldout/formal 结果前锁定。`B=8` 可以作为对齐候选，但不能因结果更好而事后选择；其他 `B` 只形成标记清楚的容量曲线，不替代主比较。

#### 离散介质的操作定义

“一次一个 token/code”按逐 token 自回归归类：每次只采样一个离散符号，下一次分布明确以此前硬采样结果为条件。“一次多个 token/code”必须是真正的并行块：一次模型前向返回形如 `[B, |V|]` 或 `[B, |C|]` 的位置分布，所有位置在同一采样事件中完成；块内任一已采样值、ground-truth token 或 teacher token 都不得反馈给同块其他位置。块位置 query 可以在产生 logits 前通过同一受限 block core 交互；后续 macro-step 可以读取已经完整结束的前一块。普通 token-by-token `generate` 即使分 chunk 调用仍是自回归，当前 `transport_chunk_tokens` 也不得记作 parallel block。

public language 不是“tokenizer 能打印字符”即可。它必须使用公开 tokenizer/词义或冻结的公开 compact-trace 语义，能被独立 parser 解析、semantic replay，并对 malformed/fault 输入 fail closed。private language 必须使用 fresh、专用的 codebook、embedding 和 categorical head，不继承预训练词嵌入的词义；code ID 虽可显示为编号，但不能据此宣称人类可读。私有臂必须报告 code utilization、dead-code rate、跨 seed code permutation、rename/permutation 等变性和 causal code substitution，证明它学习的是可用的离散内部语言而不是固定位置、答案或 family 标签。

四个离散主臂在正式推理中都必须从冻结的 categorical policy 硬采样，不得用 soft probability-weighted embedding 冒充离散生成。temperature、top-k/top-p、EOS、最大 macro-step、随机数算法和 sampling-seed ledger 必须预注册；模型/data seed 与 sampling seed 分开报告，并给出均值、置信区间、最差 seed 和 failure rate。配对 greedy/argmax 只作为“分布质量 vs 采样噪声”诊断，不增加一个主臂；best-of-N、自洽投票或 rejection sampling 必须另列全部采样成本，不能作为默认主结果。若训练使用 teacher forcing、straight-through、Gumbel 或其他 soft/biased gradient surrogate，训练暴露账本和 estimator 必须冻结，且另设 inference-consistent hard-forward Gate。

#### 公平监督、预算与成本

扩展 C2 继承 C0/C0R 的同一 Qwen revision、tokenizer、`source_text` 唯一 public forward、任务/split/causal pair、teacher source、episode/order ledger、validation 选择频率、最大 optimizer/search budget，以及 equal-example 与 equal-GPU-hour 双 matched slice。最终答案接口、合法答案 mask 和答案解码必须跨臂相同；family、program AST、reasoning budget、teacher trace、claims、route target 和答案不得进入被测 forward。

七臂逐 record 消费同源的 canonical compact trace、target tokens、loss mask 与 exposure ledger。public 离散臂直接预测可验证 trace；direct、private 和 latent 臂使用信息量匹配的 training-only trace decoder/auxiliary head，并在正式评测前物理删除。parallel public block 对同一 trace 只做冻结的分块与 padding mask，块内不得喂入 ground-truth 前缀。dense state/claim supervision 只能作为 `supervision-advantaged` sensitivity，不能进入介质优越性主结论。

除既有双 matched slice 外，C2 还必须报告完整质量—成本曲线和 matched-quality 成本。equal macro-step、equal emitted-symbol 或 `B=K` 只能作为带宽敏感性诊断，不能单独代表公平。成本账本至少覆盖 teacher 生成/验证、Qwen encode/cache、训练 FLOPs、在线前向、vocabulary/codebook logits、硬采样、输出 token/code、block padding、latent transition、KV/activation、peak VRAM、吞吐、median/p95 latency、重采样/失败恢复，以及每次 token/code↔vector 转换；cached latent/private 延迟不得与在线 public generation 直接比较。

#### 从固定臂到融合体

融合测试必须后置，并按“固定臂 → 转换桥 → 路由器”的顺序推进：

1. 先独立训练、选择并封存七个固定臂；任何臂都不得读取 router 或其他臂的 hidden state。
2. 再资格化显式 typed bridge：public/private token/code→vector 只能通过登记的 embedding + Boundary；vector→public/private 只能通过登记的 categorical head/decoder + 硬采样。bridge 只能读取当前 typed state、位置和允许的公共控制元数据，不能额外读取 raw source、teacher、answer、family 或 oracle budget，也不能扩成第二个隐藏 reasoner。
3. bridge 必须通过 roundtrip/decision consistency、counterfactual substitution、no-bridge/wrong-bridge、介质 provenance、转换次数和净成本审计。仅有低 reconstruction loss 不算转换资格。
4. 只有至少两个固定推理臂在预注册任务 strata 上表现出统计支持、非重叠的优势，且离线 diagnostic oracle 在扣除 bridge/router 成本后仍有正上界，才允许训练 learned router。oracle 只用于资格诊断，永远不得进入模型 forward 或最终模型分数。
5. 先测 episode-level router：每个 episode 只选择一种介质，不做中途切换。它通过后才允许 step-level router 在轨迹中切换介质，并强制经过上述 bridge；不得直接共享未登记 hidden state。router 可以读取允许的 public source representation 和当前状态，但不能读取 family/task 标签、答案、teacher 或 outcome oracle。

融合体必须在计入 router、bridge、额外 cache、转换、失败恢复和所有采样后，超过最强固定臂，而不是超过固定臂平均值；同时报告 worst family/split/seed、router regret、route collapse、转换频率和 route/bridge causal necessity。若一个固定臂在质量—成本上支配其余臂，则保留该臂并停止融合；若固定臂存在条件优势但 learned router 扣费后不能超过最强固定臂，则只允许形成外部静态部署选择，不宣称动态融合成功；若 router 塌缩到单臂，则删除未使用路径，不保留名义融合。

只有固定介质或融合体形成可重复净 Pareto、通过相应介质的因果必要性与审计，并在 private/block/fusion 路线中分别证明 code、并行块、route 和 bridge 不是旁路时，才可成为 V2-A architecture-formal 候选。该资格仍不等于 V2-A PASS，也不授权 V2-B。

下述 3.1–3.4 保留早期 A0–A3 连续分支的实验定义与历史结果；未来 C2 的主比较、离散介质和融合判定以本节为准。

### 3.1 V2-A0：成熟基座与任务基线

#### 目标

选择一个能稳定完成任务的最小成熟文本基座，并建立公平的文本推理基线。

当前默认候选为 Qwen3.5-2B（revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，Apache-2.0）；按用户要求，Qwen3.5-0.8B（revision `2fc06364715b967f1860aea9cf38778875588b17`）已在同一 schema、prompt 和 latent 配置下重跑。0.8B 当前 32-example text-CoT smoke exact `0.75`、同构 latent test `0.2578`，质量明显弱于 2B text-CoT，暂不替换默认基座。V2-A 只从官方 checkpoint 提取 language model 权重，不激活视觉塔。任务 schema 为 `yggdrasil.v2-a.symbolic-state-machine.v4`：三寄存器 `amber/cobalt/jade`、`A-J` 单 token 符号、canonical `SWAP/COPY` 操作、`swap->copy` composition heldout 和 5–6 步 length heldout。

#### 必须完成

- 确定基座、tokenizer、推理模式和许可证；
- 选择确实需要多步状态更新的纯文本任务；
- 建 train/validation/test、composition-heldout 和 length-heldout；
- 跑 direct answer、无显式 CoT 和显式文本 CoT；
- 记录生成 reasoning token 数、KV、延迟和最终质量；
- 排除模板、标签、长度和答案候选泄漏。

#### Gate A0

文本 CoT 基线在多 seed 或可重复 deterministic 配置下稳定；任务不是靠直接映射即可饱和；恢复和成本计量链可用。

### 3.2 V2-A1：连续 latent recurrence smoke/probe

#### 参考结构

```text
成熟文本基座（完全冻结）
-> 输入 hidden states
-> K 个 learned latent queries
-> 复制基座顶部 2 个 block 的独立 recurrent reasoner
-> 固定 T 次 latent transition
-> 只读最终 latent 的答案头
```

第一版固定：

| 参数 | 起始值 |
| --- | --- |
| `D_latent` | 基座 residual width |
| `K` | 8 |
| `T` | 8 |
| recurrent blocks | 2 |
| FFN | Dense |
| 基座 | 完全冻结 |
| 输出 | 单文本答案专家 |
| audit readout | 暂不联合训练 |

#### Gate A1

前后向、checkpoint/resume、无答案旁路、固定 `K/T` 的任务学习和结果 schema 均通过；该 Gate 只证明机制可训练，不提供介质优越性结论。

0.8B 的 `artifacts/v2-a/a1/smoke-k8-t8-qwen3p5/results.json` 只证明早期机制可跑；当前 2B 证据必须以 `artifacts/v2-a/a1/smoke-k8-t8-qwen3p5-2b-current/results.json` 为准。该结果已完成前后向、实际 checkpoint/resume、no-bypass 和干预链 smoke，使用 full-attention layer `[19,23]`，质量仍只能按 smoke 解释。

### 3.3 V2-A2：`K/T` 容量—步骤曲线

#### 对照

1. A0：显式文本 CoT；
2. A1：`K=1` 单向量 recurrence；
3. A2：`K=4/8/16` 多向量 recurrence；
4. 固定 `T=8` 扫 `K`；
5. 选择合适 `K` 后扫 `T=2/4/8/16`；
6. 固定步数与计算匹配两种口径。

#### 必须消融

- answer head 直读输入；
- `no-latent`；
- shuffled latent step；
- trajectory 截短；
- 关键 step intervention；
- 隐藏文本 token 采样检查。

#### Gate A2

至少一个连续方案在 heldout、长度外推和多 seed 中形成稳定 Pareto 改善。若只靠更大 `K/T` 和更多计算获胜，判为容量收益，不判介质通过。

### 3.4 V2-A3：自然语言 audit readout

#### 训练顺序

1. 冻结或大部分冻结已训练 latent reasoner；
2. audit decoder 只读取 `H_1...H_T` 与允许的来源/阶段记录；
3. 不提供标准答案；
4. readout 不足时才以低权重短程联合训练。

#### Gate A3

- 打乱/删除关键 latent step 会同步破坏思维链和结果；
- readout 中间判断能预测后续 state/动作/答案；
- 受控干预产生方向一致变化；
- `no-latent` 与 `shuffled-latent` 明显下降；
- readout 不只是复述问题或最终答案。

### 3.5 V2-A4：architecture-fidelity formal

#### 正式配置要求

- 预注册基座、数据、离散 `B`/codebook 与连续 `K/T`、采样策略、学习率、冻结矩阵和 seeds；
- 同基座 direct、四个离散介质和 K1/K8 固定臂；若测融合，再加入已资格化 bridge/router；
- 至少 3 seeds；
- heldout、长度外推、消融、audit 和成本齐全；
- 原始文本能力 retention 套件；
- latest/best checkpoint 与可复现实验 manifest。

#### V2-A 通过标准

C2 资格化的固定介质或融合体同时满足：

1. 无合同外隐藏推理、答案旁路、teacher/oracle 泄漏；private code、并行块、route 与 bridge 均按所选路线通过专属审计；
2. 最终质量不低于最强适用固定基线，或在同成本下有实质提升；
3. 在同质量下具有可重复成本优势，或在同成本下具有可重复质量优势；融合体必须扣除全部路由与转换开销；
4. block width、codebook、`K/T` 的容量收益与介质收益、采样收益、路由收益被分开报告；
5. 成熟文本能力没有不可接受退化；
6. public trace、private code、latent trajectory 或融合 provenance 的相应 audit 通过因果忠实性 Gate。

若未通过，停在 V2-A，重做 transition、训练监督、任务或基座；不进入 V2-B。

### 3.6 V2-A1.5：潜空间建立与递归正控制（当前执行结果）

A1.5 是在旧 A2 之后新增的分层正控制，不继续旧 K/T sweep。它使用独立 schema `yggdrasil.v2-a1.5.symbolic-state-machine.v1`，要求 train 覆盖 1–4 步，普通 test 与 composition-heldout 同为 2–4 步，length-heldout 为 5–6 步；每一步通过 no-op 反事实检查必要性，并保存 operation span/mask。

当前结果：

- P0 结构化 explicit-register recurrent core 已通过 32-example overfit；4096-example/192k sampled training 的 best ordinary test final/state full exact 为 `1.0/1.0`，说明 shared transition、state CE 和训练链可运行；
- P0 composition-heldout final/state 为 `0.2734/0`，length-heldout final/state full exact 为 `1.0/0.2266`，所以逐步状态与未见组合仍未通过；
- P1 已真实生成 Qwen3.5-2B FP16、分片、无静默截断的 hidden cache；small probe 仅为 underfit 诊断，4096-cache warm-up/joint formal ordinary validation final/state `1.0/1.0`，并支持 best checkpoint reload 与五 split `p1-evaluate`；
- A1.5 matched text-CoT 入口已支持 Qwen3.5-0.8B/2B、zero-shot/2-shot 和无人工总输出 cap；0.8B test/composition/length zero-shot 各128条 formal 分别为 parse/final/state `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`，2-shot test/composition/length final/state `0.1328/0.0156`、`0.1953/0`、`0.0938/0`，test zero-shot 有10条 safety timeout。批量 runner 增加透明 per-example wall-time safety timeout，但512条扩展矩阵仍未完成；
- P2 learned K=8 slots 已完成 formal 训练与干预：ordinary 通过，composition 与 same-answer shuffle 失败，length state full 仅 `0.6484`；
- A1.5 未通过，V2-A3 audit、V2-A4 formal 和 V2-B 均保持停止。

详细记录、命令和 artifacts：`docs/v2-a1.5-latent-foundation.md`、`tmp/V2-A1.5 result.md`、`artifacts/v2-a/a1_5/`。下一轮优先修正 operation-composition binding、same-answer identity dependence 与 length state fidelity；P1 ordinary interface 已建立，不返回旧 A2 的 K/T sweep。

### 3.7 V2-A1.6/A1.7：连续递归核心闭包（当前执行结果）

A1.6 建立了三寄存器 relation-addressed continuous core，但正式 C0 的 relation-heldout trajectory full 仅 `0.765625`。后续只读诊断表明 oracle-reset one-step 与 predicted hard re-embed diagnostic 在 test/length/relation/causal 四个 split 均为 `1.0`，把失败定位到连续 latent 写回的递归闭包；旧 final-vs-trajectory Gate 无效，旧 relation split 也不是严格单变量 holdout。

A1.7 直接切换到独立 schema 和实现，不兼容修补 A1.6：state slot 只含 value content，register key 只负责寻址；唯一新增训练约束是权重 `1.0`、target stop-gradient 的 canonical value prototype cosine closure。唯一 relation holdout 为 `COPY amber→jade`，train/validation/test/length 覆盖其他 11 个 family×directed-pair 组合，relation 每例恰好一次 holdout，background 全在训练支持内。

当前结果：

- data audit 十项 Gate 全部通过，relation heldout 位置 1–4 各 `128`，cross-split fingerprint overlap `0`，causal deletion necessary rate `1.0`；
- overfit32 在 step `200` 达到五项 fit-only 指标全 `1.0`；
- seed `20260715/20260716/20260717` 的 fresh formal C0 与 causal intervention 均通过；目标 seed 的 test/length/relation trajectory full 为 `1.0/0.996094/0.998047`，causal prefix/replacement/deletion/shuffled 为 `0.999349/0.997394/1.0/1.0`；
- 同 seed、同数据、同容量的 2×2 消融中，content-only/address-mixed 与 closure on/off 四种组合均通过 5–6 步 formal 和 causal Gate。因此短程通过不能证明地址/内容分离或 closure 必要，修正后的训练组合覆盖和 relation sequence 是重要贡献；
- fingerprint 零重叠的 8/12/16 步压力评测要求两个 split 的 aggregate 和每个长度 trajectory full 均不低于 `0.95`，三个目标 seed 为 `0/3` 通过。T16 supported 为 `0.894531/0.941406/0.894531`，说明任意加深递归尚未成立；
- closure-off 的 T16 supported 在 content-only/address-mixed 下分别降到 `0.449219/0.679688`，而 closure-on 为 `0.894531/0.953125`。closure 对长程漂移有实质作用；地址/内容分离尚无独立正收益；
- 当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断为 `40%–55%`、中心约 `48%`，完整白皮书 V2 为 `20%–35%`。这些是分层工程概率，不是统计置信区间；
- A1.7 只证明结构化 relation-addressed core 的短程可复现性，不证明 Qwen hidden、匿名 workspace 或完整 V2-A；本轮按合同不进入 C1。

详细记录与 artifacts：`docs/v2-a1.7-core.md`、`tmp/V2-A1.7 result.md`、`artifacts/v2-a/a1_7/core-assessment-summary.json`。A1.7 的长程失败已由 A1.8 后续实验继续归因；不能单独使用本节把结构化正控制升级为完整架构结论。

### 3.8 V2-A1.8：随机深度长程递归（已完成）

A1.8 没有更改 A1.7 的 content-only slots、address keys、shared transition、shared state/answer head 或 closure。唯一主动变量是把训练长度从 T1–4 切换为 T1–16，并在每个 batch 内对 16 个长度严格均衡采样。best checkpoint 以 validation 最差长度优先选择。

三组独立 data/model seed 为 `20260721/20260821`、`20260722/20260822`、`20260723/20260823`。每组数据各 `22,784` 条，彼此以及与 A1.7 data/stress fingerprint overlap 均为 `0`。length-balanced overfit32 通过。

正式结果：

- 三个 run 的 short T1–6 最低 trajectory 均为 `1.0`；
- T16 supported 最低 `0.996094`、relation 最低 `1.0`；
- OOD T24 supported 最低 `1.0`、relation 最低 `0.996094`；
- 诊断性 T32 supported 最低 `0.988281`、relation 最低 `1.0`；
- 三个 run 的 causal prefix/replacement/deletion/shuffled 全部为 `1.0`；
- initial-slot 扰动平均放大率在 T16/T24/T32 均低于 `1.0`，prototype margin 保持为正；
- 三 run 共处理 `204,800` examples、`1,740,800` latent transitions，实测训练 `610.5` 秒；成本 benchmark 与机器可读总表均已保存。

**A1.8 总 Gate 为 3/3 passed。**A1.7 T16 supported/relation mean 从 `0.910156/0.928385` 提升到 A1.8 的 `0.998698/1.0`。由于架构与 closure 不变，结果否定了 shared transition 在 T16 必然内在发散，并强烈支持训练 horizon mismatch；但 A1.8 regimen 同时改变了 horizon coverage 和 transition exposure，没有 compute-matched 地拆开两者贡献。

这个结果仍只属于强结构化 COPY/SWAP core；后续 A1.9 已在不修改 core 的前提下继续验证真实 Qwen hidden boundary，并以 3/3 通过。A1.8 的概率与下一步判断仅保留为阶段快照。

详细记录与 artifacts：`docs/v2-a1.8-long-horizon.md`、`tmp/V2-A1.8 result.md`、`artifacts/v2-a/a1_8/assessment-summary.json`。

### 3.9 V2-A1.9：冻结 Qwen hidden 边界（已完成）

A1.9 冻结 Qwen3.5-2B 和三个已经通过的 A1.8 core，只训练 value、family 和共享 register 边界映射。source、target、query 共用 register adapter；query 只能选择最终 slot；不允许 full-source、answer、query bypass、新答案头、core/Qwen 解冻或旧 A1.6 compatibility wrapper。训练继续使用 T1–16 batch 内均衡长度，正式 Gate 保持 short、T8/T12/T16、T20/T24、relation、trajectory、pointer 与 answer/state identity，并新增 core hash 不变和 boundary mapping Gate。

Qwen cache 只保存 oracle role spans 的 pooled contextual hidden，不保存 full-source hidden。由于 span hidden 仍可能携带全局上下文，A1.9 必须在普通 Gate 通过后执行 replacement/deletion/step shuffle、query swap、same-answer/different-trajectory、no-hidden 和 independent role shuffle；干预必须同时检查对重新计算 oracle 的跟随和对旧 oracle 的放弃，不能只报告 final answer changed rate。

A1.9 的 overfit32 在 step `100` 达到 mapping、pointer、trajectory 与 answer 全 `1.0`。三个正式 run 的 cache audit、formal 和 hidden intervention 均通过；三组 cache 各 `22,784` 条记录且 fingerprint 两两 overlap 为 `0`。short T1–6 最低 trajectory `1.0`，T16 supported/relation 最低 `1.0/0.996094`，T24 均为 `1.0/1.0`，诊断性 T32 为 `0.996094/1.0`，boundary mapping 最低 `1.0`。core hash 全部不变。

prefix、replacement、deletion、operation shuffle、query swap 和 same-answer/different-trajectory 对新 oracle 的跟随率在三个 run 中全部为 `1.0`；replacement/same-answer 对旧 trajectory 的保留率为 `0`，no-hidden 和 independent role shuffle 的 trajectory full exact 为 `0`。因此 **A1.9 总 Gate 为 3/3 passed**，可以声称 oracle-role-segmented frozen Qwen hidden 能因果忠实地驱动 frozen structured core。

三组 Qwen role encoding 共 `3,451.30` 秒、cache 占用 `9,177,138,012` bytes；计时不含模型/tokenizer 首次加载。cached boundary/core benchmark 也不包含在线 Qwen 编码，且未与 text-CoT 对照，因此不是 Pareto 证据。完整合同与结果见 `docs/v2-a1.9-qwen-boundary.md`、`tmp/V2-A1.9 result.md` 和 `artifacts/v2-a/a1_9/assessment-summary.json`。

A1.9 仍使用 oracle character spans、typed value/family/register 输入和显式三寄存器 COPY/SWAP core，不覆盖从完整文本自主发现 role、匿名 K-slot workspace、通用 recurrent Transformer reasoner 或 matched text-CoT Pareto。当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断更新为 `48%–63%`、中心约 `55%`，完整白皮书 V2 为 `24%–40%`；这不是统计置信区间，上调只来自真实 Qwen hidden 接入风险下降。

### 3.10 V2-A1.10：完整文本匿名工作区与通用 reasoner（已完成，失败）

A1.10 同时删除 A1.9 的 oracle character spans、typed role adapter、显式三寄存器 slots/keys 和 COPY/SWAP 专用 transition。Qwen3.5-2B 完全冻结，cache 只保存完整 last hidden 与 attention mask；`K=8`、`D=256` 的匿名 learned queries 先读取完整 source，再由两层共享 pre-norm self-attention/cross-attention/Dense FFN 递归更新。模型前向不接收 operation mask、program length、role tensor 或 register identity。T1–16 统一运行 16 步并在程序结束后要求状态保持。

修正版 overfit32 在 best step `800` 达到 T1–16 trajectory/final/answer 全 `1.0`。三个正式 run 各训练 4,000 steps，cache audit 全部通过且跨 run train fingerprint overlap 为 `0`，但 formal 为 `0/3`：T16 supported trajectory 为 `0.003906/0/0`，relation 为 `0/0/0`；T24 supported/relation 全为 `0`。T16/T24 final state 与 answer 仍约 `0.4–0.6`，说明模型学到部分终态或统计信号，却没有形成逐步状态推进。训练集均衡诊断 trajectory 同样接近 `0`，失败不是只出现在 relation/OOD。

三个 ordinary formal 均失败，按预注册没有运行 hidden intervention。三组 full-token cache 共 `38,358,822,720` bytes，Qwen 净编码 `4,767.62` 秒；三组训练 `3,256.60` 秒、`6,144,000` recurrent transitions。A1.10 只能否定当前联合配置，不能分别否定 full-text boundary、匿名 workspace 或通用 reasoner。完整合同与结果见 `docs/v2-a1.10-anonymous-workspace.md`、`tmp/V2-A1.10 result.md` 和 `artifacts/v2-a/a1_10/assessment-summary.json`。

当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断下调为 `30%–45%`、中心约 `37%`，完整白皮书 V2 为 `14%–28%`。这不是统计置信区间。

### 3.11 V2-A1.11：正交故障定位（已完成，定位成立）

本阶段没有继续联合配置的 K/T/层数/步数 sweep，而是完成两个对称诊断：

1. `A1.11-Boundary`：full-token frozen Qwen hidden → learned typed role reader → frozen A1.8 core，只删除 oracle spans。修正版 overfit32 的 trajectory/answer/mapping 全为 `1.0`，但 source/target pointer 最低均为 `0.9642857143`，严格 Gate failed；只读 hard re-embedding 后所有指标为 `1.0`，formal 按停止规则未运行；
2. `A1.11-Reasoner`：最终使用 exact symbolic typed roles，不加载 Qwen/cache/adapter/core → anonymous `K=8` generic recurrent reasoner。overfit32 在 best step 800 全指标 `1.0`；三组 formal 为 `0/3`，short/in-range/relation/T20–24/causal trajectory 均为 `0–0.003906`，训练集均衡诊断 trajectory 也只有 `0–0.003906`。

Reasoner 在精确输入下独立失败，已经否定“两个健康组件只在组合后失败”。Boundary 另有连续 latent 与 frozen address geometry 的严格接口缺陷，但 task-level formal 因 overfit stop rule 未知。完整合同与结果见 `docs/v2-a1.11-fault-localization.md`、`tmp/V2-A1.11 result.md` 和 `artifacts/v2-a/a1_11/assessment-summary.json`。

当前对 V2-A 形成 matched text-CoT 质量—成本 Pareto 的工程判断调整为 `22%–35%`、中心约 `28%`，完整白皮书 V2 为 `10%–22%`。这不是统计置信区间；下调来自 exact-symbolic 条件仍无法形成大分布逐步算法，保留的上行空间来自 A1.8/A1.9 已证明最小结构化状态与真实 Qwen hidden 分别可工作。

### 3.12 V2-A1.12–A1.17：Reasoner 二级根因定位（已完成）

A1.12 的 BIND/CURSOR/BOTH 三臂均先过 overfit32、随后 formal `0/3`，排除 entity-addressable state 与 aligned operation cursor 的独立/联合充分性。A1.13 的 GENERIC-CLOSURE state `0/3`、STRUCTURED-CE `1/3`、STRUCTURED-CLOSURE state `3/3`/full `1/3`；唯一完整通过 run 的因果干预通过。A1.13F 固定 4000-step 后三条单因素参考仍为 state `0/3`、`0/3`、`1/3`，所以 early-stop 不是主因。

A1.15 query-coupled + answer CE 与 A1.16 query-coupled + no answer CE 均 overfit32 通过、新 seed state/full `0/3`。A1.17 回到 A1.13 的 model seed `20261321/22/23`，逐 tensor 验证共享初始化位相等；paired coupled-CE 和 coupled-noCE 仍分别为 state/full `0/3`，而 independent pooled-answer reference state `3/3`。机器分类为 `independent_answer_auxiliary_gradient_required`：当前 state 算法依赖独立 pooled-answer objective 的全局辅助梯度，但该旁路本身不能成为可靠因果答案接口。

该结论触发并约束了 A1.18：推理答案固定为 query-coupled state，训练期只允许不进入部署图的全局辅助目标，禁止恢复推理答案旁路。

### 3.13 V2-A1.18/A1.18B：训练目标机制解决（已完成）

QAUX 与 FINAL-SAUX 的 overfit32 均通过。QAUX 只作实现阳性控制；FINAL-SAUX 从最终 workspace mean 预测完整三寄存器终态，paired seed formal/causal 为 `2/3`。两个通过run在辅助头删除后通过 relation、T24 与全部因果干预；失败seed从 step `200` 到 `4000` 始终没有进入算法盆地，因此 final-only 全局状态梯度方向正确但初始化不稳定。

A1.18B 只改变全局状态监督密度：TSAUX 使用一组跨步共享训练头，在每个递归步从 global workspace mean 预测当步完整三寄存器 state；正式 answer CE 保持 `0`。它先在 FINAL-SAUX 失败的相同 seed 上于 step `600` 修复 state，随后统一重跑三个 paired seed，再用 model seed `20261821/22/23` 做三个 fresh run。paired 与 fresh 的 formal/causal 都为 `3/3`；六个部署模型的 causal trajectory/answer 都为 `1.0`，全部 counterfactual Gate 通过，disable-recurrence 和 wrong-start trajectory 都为 `0`。

formal 前全部 `training_auxiliary_head.*` 参数均物理删除，正式 loader 会拒绝仍含辅助参数的 checkpoint。机器分类为 `per_step_global_state_credit_assignment_confirmed`、`mechanism_solved=true`、`diagnostic_core_architecture_validated=true`。这证明 answer-specific auxiliary 不是必要条件；缺失机制是每一步 global workspace → complete state 的密集信用分配。

原计划下一阶段直接做 state-target source annealing，现已因路线重审后移。A1.10/A1.11 的匿名通用 reasoner 事实上已经在每个递归步用 workspace mean 预测完整 state 并施加 trajectory CE，但在 exact-symbolic 条件下仍 formal `0/3`；这说明 A1.18B 的成功不能只归因于新增监督，还依赖 relation addressing、显式实体地址和 closure。当前不得先削弱唯一成功配置的监督，也不得直接恢复旧 A2。完整机制结果见 `docs/v2-a1.18-training-scaffold.md`、`tmp/V2-A1.18 result.md` 与 `artifacts/v2-a/a1_18b/assessment-summary.json`。

### 3.14 V2-A1.19H：可泛化混合 core（已完成）

A1.19H 保留 relation-addressed shared transition、soft closure、TSAUX、query-coupled answer 和部署剥离，但分两个顺序子门把固定三寄存器 scaffold 改为任务无关的 addressable hybrid workspace。`H1 opaque-handle` 保持实体数、操作、数据和预算不变，只删除 slot index = register identity，逐样本随机分配 opaque handle 并测试 handle/slot 联合 permutation。H1 formal `3/3` 后才运行 `H2 variable-cardinality`，把实体数量扩展为训练内多个 `N` 与 heldout `N`，其余合同不变。

两个子门均先做 overfit32；通过后使用三个 fresh seed 跑 formal/causal，并测试 OOD horizon、relation holdout、handle permutation/alias、同值不同实体、query swap、旧轨迹拒绝、disable recurrence 和辅助头剥离；H2 另加 OOD entity count。soft continuous addressing 只作 H2 通过后的同预算消融，不是前置 Gate。任一子门 formal 不到 `3/3` 时停止 full-text、Pareto 与 V2-B。新增任务/关系族留到 A1.21P，不在 A1.19H 混入。

A1.19H 已完成。H1 删除 fixed register/slot identity 后 formal/causal `3/3`；H2 在训练 `N=2,3,4`、heldout `N=5` 和 heldout relation 合同下 formal/causal `3/3`。三个 H2 run 的 `N=5` 与 `N=5+relation` trajectory/final/answer 均为 `1.0`，OOD horizon 最低 aggregate trajectory 为 `0.983073`。机器分类 `generalized_hybrid_core_confirmed`、`a119h_complete=true`。部署 artifact 不含训练辅助参数，handle 没有语义 embedding，也不存在 fixed-register 或 task-family-specific executor。

run-3 前发现旧训练管线被 Python→CUDA 标量编码和每步地址校验同步阻塞。新管线使用一次性 validated tensor cache、GPU index-select、预生成采样与 forward 路由预计算；60-step 同初始化对照加速 `10.744×`，final loss 与所有参数差均为 `0`。旧 run-1 checkpoint 的优化后全 split 回归与原 formal JSON 完全相同。完整记录见 `docs/v2-a1.19h-hybrid-core.md`、`tmp/V2-A1.19H result.md` 与 `artifacts/v2-a/a1_19h/assessment-summary.json`。

### 3.15 V2-A1.20B：learned full-text boundary（已停止，失败）

冻结通过的 A1.19H core，使用完整 Qwen hidden 训练 Boundary 自主输出连续 value payload 与必要的 source/target/query handle、relation/type/control。输入不得包含 oracle span mask、oracle role tensor、程序步骤 mask、答案 metadata 或 hard re-embedding 诊断补丁。环境原生 source identity/位置 metadata 可以保留，但任务语义角色必须从完整文本学习。

先做 overfit32，并分别验证 value、source pointer、target pointer、query pointer、relation family 与 handle geometry；全部严格通过后才运行三个 fresh Boundary formal。正式 task Gate 必须覆盖 H2 的 short/in-range/relation/OOD/N5/N5+relation，并在 formal 通过后执行 no-hidden、hidden shuffle、role-preserving text counterfactual、query swap、operation replacement/deletion/shuffle 和 core hash 不变检查。Boundary formal 不到 `3/3` 时停止 A1.21P/A1.22A，不得用 short joint tuning 或 hard re-embedding 掩盖。

A1.20B 通过后进入 A1.21P matched R1 Pareto，再进入 A1.22A natural-language audit。训练目标来源不再是独立硬门：可执行 trace、程序状态和 verifier-filtered teacher record 都允许用于训练，但必须报告生成成本/错误率并在部署前删除辅助路径。零 teacher 与自监督 next-state 降级为可选效率研究。A1.21P/A1.22A 同时通过才关闭 V2-A/R1。路线决策真源见 `docs/v2-a-route-reassessment-2026-07-17.md`。

A1.20B 实际已执行并停止。overfit32 mapping/trajectory/final/answer 全为 `1.0`；run-1 的 full-token cache audit 全通过，固定 5000-step 训练完成，但 best checkpoint 的 2048 条 validation trajectory/final/answer 为 `0.297363/0.409668/0.608887`，最差 cell mapping/trajectory/answer 为 `0.315341/0/0.227273`，没有资格进入 formal split。

逐样本 Boundary 全字段联合 exact 仅 `0.087891`；N4 全部 value 正确仅 `0.007331`，T13–16 的完整 program/source/target 序列 exact 均为 `0`。全 oracle Boundary 使同一 frozen core 恢复 trajectory/final/answer `1.0`，排除 core 失效。state CE 的梯度实测只到 continuous value path，不到 threshold/argmax 后的 entity/operation presence、family、source、target、query logits；完整 loss 只能通过各字段局部 CE 训练这些控制。机器分类 `full_text_entity_binding_and_program_extraction_failure`。

因此 run-1 formal cache/eval、causal、run-2/run-3、A1.21P 与 A1.22A 均按合同未运行。继续前必须重新立项解决 Boundary entity/program binding 与离散控制执行级信用分配；不能直接增加 seed/step，也不能用 hard re-embedding 补丁掩盖。完整记录见 `docs/v2-a1.20b-full-text-boundary.md`、`tmp/V2-A1.20B result.md` 与 `artifacts/v2-a/a1_20b/assessment-summary.json`。

### 3.16 V2-A1.20C：分层编译器 × execution-credit 修复（已停止，失败）

A1.20C 复用 A1.20B 的 frozen full-token Qwen cache、frozen A1.19H-H2 core、数据与 seed，预注册 compiler `flat/hierarchical` × credit `hard-local/straight-through` 2×2。hierarchical compiler 先建立 entity table，再读取 ordered family/source/target 与 query；训练 token anchor target 不进入模型前向。straight-through 模式在 forward 保持 hard prefix mask、family 和 pointer，在 backward 使用 soft surrogate 驱动 frozen core。

目标臂 `hierarchical + straight-through` 的 supervision audit、4 项回归测试和 CUDA smoke 均通过。hard-forward state/answer logit 差为 `0`，core hash 不变。fixed-5000 overfit32 的 best checkpoint 位于 step 4800，anchor all-sequence exact、value、family、answer 均为 `1.0`，但 trajectory/final-state/state-token 只有 `0.875/0.90625/0.964474`，mapping minimum 为 `0.90625`；四条 `N=4` 样本仍有 entity/operation count 与 pointer-validity 联动错误，最差 cell mapping/trajectory 为 `0/0`。

失败根因已进一步定位。pointer local CE 在未执行 predicted entity mask 的 logits 上优化，而实际执行会用 hard entity mask 改写 pointer 候选集合，current ST bridge 未连续化这个 validity-set 决策。best checkpoint 的 state CE 与其余 local mapping/payload/anchor objective 梯度 cosine 全局为 `-0.7366`，entity path 为 `-0.9680`，presence heads 为 `-0.9944`；固定 `3e-4` 学习率又使 total loss 从 step 4800 的 `0.2429` 回升到 step 5000 的 `0.6103`。机器分类 `anchor_localization_solved_but_execution_objectives_conflict`。

因此另外三个 2×2 正式训练臂、heldout validation matrix、三 seed formal/causal、A1.21P 与 A1.22A 均未运行。继续前必须新立项把 entity-count/pointer-validity 联合连续化，分阶段或投影冲突梯度，并引入 joint-stage 学习率衰减；然后从 fresh initialization 重过 overfit32。不能把 anchor 或 answer `1.0` 当作机制通过。完整记录见 `docs/v2-a1.20c-boundary-repair.md`、`tmp/V2-A1.20C result.md` 与 `artifacts/v2-a/a1_20c/overfit32/hierarchical__straight_through/failure-diagnostic.json`。

### 3.17 V2-A1.20D：双向 section 推断（放宽机制修复，通过）

A1.20D 是 A1.20C 严格失败后的 post-stop diagnostic，不追溯改写旧 Gate。容量阈值扫描显示，旧粗 operation boundary 下 N4 伪第五实体和 N5 真实第五实体不可由一个全局 pair-gain threshold 分开；但使用 refined first-operation position 后，两组 gain 分别约为 `5–8` 与 `13–18`，原阈值 `12` 已可分。根因是旧 factorized forward 执行“粗 operation → entity → refined operation”后，没有把 refined program start 反馈给最终 entity decode。

实现改为三遍“粗 operation → entity → refined operation → final entity”。不新增 oracle span、typed role 或 hard re-embedding，不改变 frozen Qwen、frozen A1.19H core 和 hard deployment path。原 joint checkpoint 无需重训：canonical 八 split 诊断矩阵全部通过，N5/N5+relation trajectory/final/answer 全为 `1.0`；normal hidden trajectory 为 `1.0`，zero/batch-roll/token-reverse 为 `0/0.001953/0`。routing 四个 OOD split 在三个 continuation schedule 上都通过，最低 trajectory 为 N5 的 `0.96875`。

该结果将 A1.20C 的部分负梯度冲突重新解释为错误 section 因果图的结果，证明 full-text hybrid mechanism 是强候选。但 canonical artifact 明确为 `formal_gate=false`，三个 schedule 共享 reader/core/data 起点，不能写成 fresh-seed formal。完整记录见 `docs/v2-a1.20d-full-text-mechanism-repair.md`。

### 3.18 V2-A1.21P：matched Pareto（已停止，正式失败）

已完成 K=1 recurrent 容量负基线、canonical/routing 在线五条 Pareto smoke、三 schedule 路径稳定性和机器总判定。K=1 在 2048 条 validation 的 trajectory/final-state/state-token/answer 为 `0.106934/0.333008/0.497406/0.550293`。在线 smoke 包含 Qwen encode：canonical direct/text-CoT/hybrid answer 为 `0/0.4/1.0`、median latency 为 `0.586/45.235/0.489` 秒；routing 表面域为 `0/0/1.0`、`1.540/47.438/1.567` 秒。

这些结果只确认 candidate。正式 Gate 失败六项：joint细粒度 eligibility 为 `0/3`；每个 family 只有五条在线样本；routing manifest 明确 `surface_only_transform=true`，不是第二种可执行代数；三个 continuation 不是 fresh model/data seeds；没有 matched training/teacher-data budget artifact；没有 FLOPs、activation/KV 与 teacher 全成本 artifact。direct/text-CoT 仅 deterministic prompting，而 hybrid 已训练，routing text-CoT 还复用 canonical demonstrations。

机器分类 `full_text_hybrid_candidate_confirmed_but_matched_pareto_not_closed`，`a121p_passed=false`、`a122a_authorized=false`。按硬顺序停止，不运行 A1.22A。重新启动 A1.21P 前必须设计真正改变可执行代数或规划结构的新任务族，补齐三组 fresh data/reader/core seeds、公平 baseline 训练合同、每 family 至少 64 条在线评估和完整成本账本。完整记录见 `docs/v2-a1.21p-pareto.md`、`tmp/V2-A1.20D-A1.21P result.md` 与 `artifacts/v2-a/a1_21p/assessment-summary.json`。

### 3.19 V2-R1R：最小跨任务重验证（P1 v3 停在 K=8 Gate）

当前不再继续完善 COPY/SWAP compiler，也不直接重跑旧 A1.21P。A1.20D 保留为机制正控制；新的 R1R 使用同一共享 Boundary/core 联合验证两个不同任务族：`ERE` 情境规则执行负责单轨长程状态更新，`CPS` 约束计划选择负责并行候选模拟、条件、资源、延迟后果与成本比较。每个 episode 临时定义 nonce 规则/action 语义，禁止固定 operation 枚举和任务专用 transition。

P0-M 已实现的主模型为 `Qwen3.5-2B final hidden -> token-wise 2048→512 Boundary -> K=8 shared two-layer recurrent core -> latent-only 9-label readout`。Boundary 不做 section/entity/candidate compiler，两个任务共享全部 core 参数。P0-M/v3/v4 的训练期 claim verifier 使用 state–query interaction、真假 pair ranking 与同族 owner contrast；v4 已证明任意跨 episode owner negative 会奖励 nonce 身份记忆，因此 v5 活动目标只保留同 episode paired claim，Gate 后仍物理删除 probe并验证答案逐字节不变。AST、span、role、entity/candidate state、正确索引和答案都不得进入 model view。K=1 尚未在新任务的正式 P1 合同中比较。

P0-D 测量链为 R0A–R0D：R0A 只验证 hand-authored semantic oracle 与公共 simulator；R0B 验证 G02–G06；R0C 用已知数值 toy fixtures 验证 G07/G08 statistical learner；R0D 才做完整 integration。v7 的 858-cell lattice 与主审探针关闭 R0A；v9 以固定 profile、独立 typed derivation 和可逆 grammar 关闭 R0B；v10 资格化 source-only parser、条件多数、word/character multinomial NB、grouped CV 与 train-heldout。v11 只组合三层 accepted 实现，以 12 个 case、8 Gate、20 faults、19 raw metrics、4 metamorphic 和 artifact replay 关闭有限 R0D integrated measurement qualification。

v12 已关闭上述两个 entry condition：production renderer 可逆覆盖完整受控 ERE/CPS AST，model view 精确封闭，compact exact scorer 在最长 3,322 characters/984 pinned Qwen tokens 的入口剖面通过。v13 随后完成 1,440-record R1 generator smoke；正式 G01–G11 全 true，claim 局部反事实配平使 heldout word/char NB 全为 0.50，主设计层最终接受固定 seed 的有限 generator smoke。

v14 已完成上述完整 production P0-D 尝试：修复 root provenance、非九整除标签配平、ERE 语义容量与 sparse exact-equivalent audit，并先通过 8,192-record 双运行 preflight。唯一 14,336-record formal 只有 G09 false；两个失败 cell 的 point excess 约 `+0.035/+0.042`，但 98-way simultaneous upper 超 ceiling，所有 14 个 family aggregate 通过。v14 formal 禁止覆盖或重跑。

v15 随后另立 G09 decision/power qualification，并预注册 train 4,096、heldout 1,536。100,000-trial 的 null/local/diffuse 七场景、14 fault、3,264 个 exact 对照、single-bank/batch/progress 与 replay 均通过；但唯一 Q formal 的 Q08 因一次性 speedup `4.7617x < 5x` false，最终 `FAIL_G09_QUALIFICATION`。固定 fresh-seed production root 没有创建。v15 Q artifact 禁止覆盖或重跑，不能用开发期更高速度或事后阈值接受。

v17 已按独立合同完成上述 repair qualification：50,000-seed `if_copy` sweep、13,312-fingerprint capacity、v16 307 条失败回归与 58 项 collision/fault-kill 均通过；fresh seed `2026081702` 的 26,624-record formal 为 G01–G11 全 true，P0-D accepted。P0-M v1–v4 随后依次暴露 cache 判据、BF16 mask/baseline 效率、claim compatibility 与监督时序问题；v5 不降 Gate，以 owner contrast、持续 claim 监督和剥离复载关闭 M01–M08。

P1 v1 随后以 fresh seed `2026081901` 和每族 train `8192`/heldout `1024` 启动；唯一 data formal 只因 `ERE/validation/full_text_char_3_5_nb` 的 simultaneous upper 越线而停止。P1 v2 没有重判 v1，而是先用 100,000-trial Q01–Q09 资格化 train/heldout `8192/4096` 的完整 98/14-cell 测量功效，再以 seed `2026082002` 生成 65,536 records。正式 G01–G11、两次生成、只读 replay、watched inputs 与 12×1024 固定模型子集全部通过；原敏感 cell accuracy/upper 为 `0.29639/0.32034 < 0.37087`。

P1 v2 cache 随后完整生成 28,672-source/191,144-claim、132+26 shards、20,563,413 tokens 的 packed FP16 banks。外层命令在四小时等待上限退出但子进程继续；正式进程在 source 内容审计的 progress callback 中得到 `OSError [Errno 22]`，因此 sealed FAIL。封存后关闭 progress 输出重放同一 full audit，两个 banks 全 entry 通过、`failures=[]`；故障属于 execution telemetry，不是 cache 内容或架构。该历史边界见 `docs/v2-r1r-p1-v2-cache-infrastructure-failure-review.md`。

P1 v3 随后以新合同完成 T01–T06 telemetry decision power 和 R01–R09 immutable cache recovery；两次 full audit 均覆盖全部 entries、canonical 相等，v2 cache 仍保持 formal FAIL，只新增只读训练授权。唯一 K=8 使用 16,384 个 train episodes、38,400 次 episode exposures 完成 2,400 updates，但 validation ERE/CPS 只有 `0.24902/0.19043`，OOD 为 ERE `0.25098–0.29199`、CPS `0.10156–0.16992`，causal flip 为 `0.00391/0`，zero/shuffle/slot-permutation middle 几乎不改变结果；claim accuracy `0.5`、owner-shuffle drop `-0.00195`。K01–K06 false、K07/K09 true；K08 另有 batch-size 写死的非决定性审计误报。正式状态 `FAIL_P1_K8`，后序 roots 全未创建。

训练吞吐为 `191.66 examples/s`、`107,205 tokens/s`，GPU 利用率中位数 `56%`，排除此前低功率基础设施主因。首要合同缺陷是暴露预算没有随数据规模扩展：平均每条 episode 仅 `2.34375` 次，每次只抽 4 个 claim，153,600 次 claim 观察少于 191,144 个唯一训练 claim。下一步只能另立 P1-LQ/P1 v4 training-qualification，以 data scale × exposure ratio 先判断预算与可扩展性，再在必要时比较等计算的 joint-from-start 与 mechanism-warm-start；不得重跑 v3 或进入 P2。完整复核见 `docs/v2-r1r-p1-v3-k8-failure-review.md`。

### 3.20 V2-R1R P1 v4-LQ：机制启动通过，规模扩展因身份记忆失败

P1 v4-LQ 不修改 K=8 架构，也不降低原 K01–K09。B128 在 update 1,200 通过；S512 跑满 4,800 updates 后训练 answer 为 `1.0/1.0`、claim `0.9629`，但 validation 为 `0.2344/0.1885`，正式 Gate 失败并停止。S2048/F8192/assessment 未创建。

只读诊断进一步显示，S512 checkpoint 在见过 episodes 上 claim `0.939/0.965`，在同分布未见 episodes 上为 `0.498/0.506`；zero-state/zero-claim 约 `0.5`。因此不是 claim-only 泄漏，而是 source–claim 联合身份记忆。跨 episode owner-contrast 的真值未定义，却被训练与 Gate 当作负例；该目标和 Gate 已被 v5 直接删除。完整复盘见 `docs/v2-r1r-p1-v4-lq-failure-review.md`。

### 3.21 V2-R1R P1 v5：静态 paired claim 规模迁移失败

v5 不改部署模型，只删除错误 owner negative 并引入 7,168/1,024 optimization-audit。唯一 Q7168 跑满 14,336 updates 后，训练 answer ERE/CPS 为 `1.0/1.0`，隔离 audit answer 为 `0.4746/0.1846`，claim 为 `0.5/0.5`，state-dependency drop 为 `0/-0.0034`，因此 sealed FAIL；F8192/assessment 不存在。梯度、参数移动与小规模可学习性排除了断图，根因是静态平衡 predicate 在形成 predicate-state binding 前梯度相互抵消。完整复盘见 `docs/v2-r1r-p1-v5-semantic-transfer-failure-review.md`。

### 3.22 V2-R1R P1 v6/v6R：causal temporal witness 机制资格已恢复

v6 比较等 source batch、等四个 query-state judgments、同初始化与同 schedule 的 `static_pair` 和 `temporal_witness`。temporal 用同一 prefix-free query 在相邻 T+1 latent states 上提供相反真值；模型仍只见 source hidden、匿名 K=8 workspace、budget 与 choice mask，不见 AST、oracle span、canonical state 或显式寄存器。两臂固定 3,200 updates；checkpoint 只按 optimization seen 选择，128/族 fresh formal audit 只评估一次。Gate 为 seen 每族 `0.70`、audit 每族 `0.65`、audit state drop 每族 `0.10`，temporal 另需 swapped-state drop 每族 `0.20` 和相对 static audit 优势每族 `0.10`。

formal 的 witness/query/mechanism-compare 前四阶段均 sealed PASS：510,401 witnesses 全部重放，query cache 为 18,899 queries；static audit ERE/CPS 为 `0.4805/0.4868` 且 state drop 近零，temporal audit 为 `0.7058/0.8053`、state drop `0.2058/0.3053`、swap drop `0.4117/0.6105`，预注册选择 temporal。原 assessment 唯一 W604 false 是 `reasoning_budgets` 的六个 integer keys 经 JSON 写盘后转 string，而代码错误使用 raw Python equality。

v6R 保留原 FAIL 和五个旧 roots，以两个新 roots完成 canonical/hash independent replay、三类 selection mutation 负控和 18,899/18,899 query content audit；R601–R610 全 true，正式 `PASS_P1_V6R_SELECTION_RECOVERY`。设计、失败复核与主审见 `docs/v2-r1r-p1-v6r-selection-recovery-design.md`、`docs/v2-r1r-p1-v6-ctw-assessment-failure-review.md`、`docs/v2-r1r-p1-v6r-selection-recovery-main-review.md`。

### 3.23 V2-R1R P1 v7 integrated K=8：时序机制保留，CPS 因果答案失败

v7 使用新 model/order/selection seed，不续训 v6 comparator checkpoint。在 full 8,192/族上，每个 episode 40 次 exposure；preflight 与 86,016-query cache sealed PASS，唯一 K=8 跑满 20,480 updates。validation ERE/CPS 为 `0.999023/0.851562`；temporal accuracy `0.766357/0.938965`、zero drop `0.266357/0.438965`、swap drop `0.532715/0.877930`，batch-shuffle-middle 平均答案 drop `0.627441`，说明 recurrent state 与 v6R 机制均真实参与。

失败集中在答案语义：CPS composition/horizon/distractor/language 为 `0.587891/0.754883/0.472656/0.214844`。正式 evaluator 错把数据的 `base/flip` 要求成 `base/counterfactual`；父任务只读修正后 pair flip 为 ERE/CPS `0.705078/0.0078125`，所以 bug 没有隐藏 PASS。共享梯度近正交、K10 强而 CPS causal 近零，支持“temporal 与 answer 在 final state 中形成两条通路，ordinary CE 没把决策绑定到因果变量”的归因。

K04 zero-middle 在每一步仍可读取完整 source，测到的是可恢复性而非 state 必要性；K06 强制 T1 drop 也不属于白皮书必要条件，应转成质量—成本控制。两项保持 v7 历史结果，但不进入后继资格 Gate。三个 v7 roots 均封存，K=1/baselines/P2 未运行。完整边界见 `docs/v2-r1r-p1-v7-integrated-k8-failure-review.md`。

### 3.24 V2-R1R P1 v8 causal-bridge：正式失败与合同归因

v8 不扩大 ordinary 训练。每族 512 个旧 causal pair 按 label-blind hash 划为 384 optimization + 128 sealed audit；ordinary arm 另取 768 train records/族。`ordinary`、`causal_unpaired`、`causal_paired` 三臂使用同一初始化、K=8 模型、batch32、3,072 updates 与每条 record 精确 64 次 exposure。paired 臂只额外使用同对配批和 final-logit 双向 ranking，pair/role 不进入模型前向。

每臂先 768 updates temporal-only，再 2,304 updates 联训；optimization 每 episode 统一取 3 个 witness，audit 取 2 个，共 10,240 个 query。三种长度分桶 schedule padding efficiency 为 `0.88313/0.88783/0.88862`。普通 validation/OOD 使用 256 条诊断子集；batch-shuffle 使用完整 1,024 条 length/horizon 同 cohort 对照，预测有效覆盖为 ERE/CPS `1.0/0.99609`。

B01–B07 要求 causal pair 结构、两族 pair flip `>=0.65`、raw accuracy `>=0.75`、temporal persistence、validation retention、batch-shuffle state dependence、probe strip/architecture/compute 全部成立；被选 causal arm还必须相对 ordinary 两族各提升 `>=0.15`。正式运行中 preflight/query-cache PASS，qualification sealed `FAIL_P1_V8_CAUSAL_BRIDGE`：三臂 B01/B04/B07 true，B02/B03/B05/B06 false，没有 selected arm。

ordinary 臂训练复评 ERE/CPS 为 `1.0/1.0`，validation 仅 `0.25/0.1758`。causal-paired 在 optimization pair 上为 ERE/CPS `0.9974/0.5807`，相对 unpaired 的 `0.9245/0.1458` 证明 pair loss 可优化；development audit 却只有 `0.0859/0.0078`。v8 因而暴露的是覆盖与训练路径错误：它只用 v7 约 9.4% 的独立 ordinary 覆盖，causal 臂又完全没有 ordinary pretrain/rehearsal。完整复核见 `docs/v2-r1r-p1-v8-causal-bridge-failure-review.md`。

### 3.25 V2-R1R P1 v8R causal curriculum：共享能力起点的最小恢复合同

v8R 从 sealed v7 checkpoint 出发，`replay_ce` 与 `replay_pair` 读取完全相同的 mixed batch。每批包含 ordinary ERE/CPS 各 8 条、causal ERE/CPS 各 4 个完整 pair；3,072 updates 使完整 ordinary train 每条精确 3 次、v8 causal optimization 每条精确 32 次。两个臂唯一差异是 pair loss 权重 `0/1`，不再训练 temporal probe。

R01–R06 检查共同起点、同初始化/同 schedule/exact compute、CPS optimization decision power、已揭示 audit 上的 transfer、ordinary retention 与架构完整性。PASS 只授权从未进入 v7/v8 model-view 的 1,536 unused pair/族建立 fresh P1 v9；FAIL 则根据 R03/R04 分离 loss 无效与语义迁移无效。P1/P2 均不因此完成。

正式 preflight sealed PASS，qualification sealed `FAIL_P1_V8R_CAUSAL_CURRICULUM`。`replay_pair` optimization ERE/CPS pair 为 `0.9948/0.3281`，audit 为 `0.8047/0.0078`，validation 为 `0.9990/0.7480`；R01/R02 true，R03–R06 false。CPS audit 中 69/128 对沿用 base 答案，只有 1 对正确切换。paired loss 可在训练 pair 上提高 margin，却没有形成跨 pair 的最终比较算法。原 R06 的 ERE source gradient 为零另被 FP32 重测确认为 BF16 饱和下溢，不能替真实 causal failure 背书。完整复核见 `docs/v2-r1r-p1-v8r-causal-curriculum-failure-review.md`。

### 3.26 V2-R1R P1 v8D causal-decision witness：最终决策状态闭合

v8D 继续从 sealed v7 stripped checkpoint 开始，不继承 v8/v8R 训练权重。训练期重新实例化共享 probe，对同一 base/flip 的 `H_T` 提出相同、真值相反的 final-decision query：ERE/CPS 都监督局部答案标签；CPS 额外监督 unique optimum 与 winner/alternate cost ordering。query 只进入 probe，不进入 recurrent core 或答案头；正式答案评估前物理删除 probe并 strict reload。

先用 384 updates probe-only 判定旧 V7 `H_T` 的可解码性，再以 3,072 updates joint state formation 训练 Boundary/core；前 2,560 updates 冻结 answer readout，最后 512 updates 才允许对齐。D01–D07 分别检查输入/query 真值、精确计算、audit decision-state transfer、answer causal transfer、ordinary retention、probe strip/FP32 gradient/部署完整性和停止边界。使用已揭示 v8 split，因此 PASS 只授权 fresh-seed P1 v9；D03 通过而 D04 失败才允许 addressable readout，D03 失败则回到 state formation/Boundary 设计，不再调同类 loss/exposure。

正式 preflight/query-cache PASS，qualification sealed FAIL。joint audit decision ERE/CPS 为 `0.8887/0.5143`，CPS zero/swapped drop `0.0143/0.0286`；answer audit ERE raw/pair `0.8984/0.7969`，CPS `0.375/0`，ordinary retention 通过。只读 state-path 诊断显示 CPS delta 经 Boundary 保留并在 v8D `H_T` 放大，故根因不是 source blindness，而是 episode-specific delta 没有归约成可迁移的 cost/order/choice algebra。

### 3.27 V2-R1R P1 v8L fixed-anchor causal-state ladder：正式 FAIL 与 K=8 停止判据

v8L 从 sealed V7 stripped checkpoint 重启并移除 ClaimProbe。simulator teacher 只生成 mutation-aligned 的 state index 与互斥语义 claim；冻结 Qwen + 冻结 V7 Boundary 编码 query 后，以与 truth label 无关的 hash 顺序构造 `normalize(q_left-q_right)`。这避免 v8L 初稿中正反自然语言 query cosine `0.96–0.996` 导致的梯度抵消，也不在 cache 中拟合 audit statistics。

正式 preflight 与 anchor-cache sealed PASS，qualification 在 384-update bootstrap sealed `FAIL_P1_V8L_BOOTSTRAP`。ERE optimization/audit direction 为 `0.7726/0.7109`，CPS 为 `0.5682/0.5281`；CPS 五层 optimization 均低于 `0.65`，因此 3,072-update joint、答案 Gate 与 retention Gate 均未运行。预注册 fail-stop 已执行：不延长训练、不降低阈值、不建立 v8 后继，也不授权 fresh-seed P1 v9。

post-stop 分层 geometry 审计同时发现 measurement 缺口：global effective rank `25.61` 掩盖了 CPS cost-trace/final-cost 只有 `3.47/3.84`，重定向后的数值 shared-axis alignment 只有 `0.340/0.380` 且含反向方向。故该结果关闭的是匿名 K=8 与 lexical metric teacher 的组合，不能单独否证 mixed core 或白皮书。若继续项目，必须先另立数值/关系 teacher decision-power qualification，再在新架构合同中比较 shared core 与 task-independent mixed/typed core；不得把这个工作命名为 v8 修复。

若 ERE 通过而 CPS 失败，则只证明状态执行器；若两者可运行但没有 Pareto，则终止 V2 多模态主线；若 K=1 支配 K=8，则删除多 slot 复杂度。完整生成算法、张量结构、loss schedule、CLI、成本和停止规则见 `docs/v2-r1-revalidation-task-design.md`，执行 agent 不得自行修改。

### 3.28 V2-R1R P1-NR1→H1→F1：历史路线（已由 Closure 覆盖）

本路线不建立 P1 v9，也不继承或补跑 V8L。NR1 先资格化没有 answer/label/task id/oracle state 字段的 typed numeric/relation measurement system：numeric 使用迭代累计与 strict-min decision，relation 使用 fixed-point closure；独立 oracle 分别使用 `itertools.accumulate` 与逐 source/query BFS。另有 6+6 个手算 case 与 11 类固定 fault target 的外部 fixture，避免 generator/reference/measurement 同链自证。qualification/heldout 固定为 numeric `384/256`、relation `384/256`；heldout 在 magnitude、horizon、candidate/handle cardinality 上严格超出 qualification，relation corpus 同时强制 branch/merge/multiple-path/multiple-component/redundant-edge 与可变 query 真值布局。

NR1 的 N01–N07 要求 V8L 三根 seal 不变、合同/source/Git/snapshot identity、活动预测试、numeric/relation exact、复杂拓扑、平移/缩放/置换/rename metamorphic、11 类 fault 的预注册 metric/decision kill、fingerprint zero-overlap、唯一 formal 进程树、固定 transport、两根即时复验 evidence seal 与 H1/F1/v9/P2 absence 全部通过。CLI 只暴露 `run-p1-nr1`；formal 前两根 fixed root 与 fixed transport 必须不存在。PASS 只授权 H1 设计，不能把 measurement-system success 写成模型、mixed core 或 P1 success。

该历史合同原规定 H1 获准后比较 anonymous shared core 与 learned/content-routed mixed/typed core，再由 F1 补齐 K=8/K=1/direct/text-CoT。实际 H1 与其 WD 后继均未获授权，当前不再为它设计 successor；Closure C0→C1→C2→C3 已取代其活动地位。以下数值与停线记录仅用于追溯，不是当前执行入口。

2026-08-17，NR1 唯一 formal 已完成：preflight/qualification 分别为 `PASS_P1_NR1_PREFLIGHT` 与 `PASS_P1_NR1_MEASUREMENT_QUALIFICATION`，seals 为 `1825282B…95DAF`、`DADBDDBF…F6FB3`。N01–N07 全 true，source identity 为 `1D837A0F…B78657E`，后继根与残留进程为空；状态只升级为 `p1_h1_design_authorized=true`。下一动作是冻结 H1，而不是重跑 NR1 或直接进入 F1。

同日 H1 development 先后否决三个非正式结构。完整-transition experts 的 heldout 总增益仅 `+0.00684`；纠正为公共 attention 后，互斥完整 FFN screen 虽使 numeric 提高 `+0.12109`，却令 relation 回退 `-0.06055`，总增益 `+0.03027` 且 95% CI 跨零，route flip drop `0.05371` 也未到 `0.10`。第三个 2:1 shared+routed residual 在 current-fingerprint fresh cache 上使总增益升到 `+0.12695`、CI 下界 `+0.09375`，numeric/relation 都为正，但 registered route effect 最大只有 `0.08887`，仍被 H06 否决。三者都不消耗 formal roots，也不授权 F1。等分公共 `H=384` FFN + `H=384` 路由 residual 随后在同 screen seed 跑满 2,400 updates，heldout 总增益 `+0.13867`、CI `[0.10352,0.17285]`，route answer drop `0.13672`，方向 Gate 通过；但预登记全新 `2026081793/2026081794` 独立 calibration 在新 seed 上跑满正式全预算 4,000 updates 后只有 overall `-0.00098`、CI `[-0.03613,0.03516]`、numeric/relation `+0.03125/-0.03320`，最大 route effect `0.03845`。两轮同时改变 seed 与预算，不能纯归因于 seed；calibration 两臂最终 train answer loss 均低于 `0.001`，说明 screen 的 mixed 优势至少含 shared 收敛暂态。calibration `authorizes=nothing`，secondary-floor proposal 无效。进一步复核发现旧 H06 只要求 wrong-route effect，无法排除“错误 expert 有害、正确条件分支却可绕过”：screen 关闭整条 FFN 的 answer drop 约 `-0.00195`。因此活动实现直接切换为共享 `D->H` SwiGLU features + route-selected `H->D` projection，不保留旧 residual expert；旧 `2026081791/1793` 数据身份永久进入 formal 禁用注册表。新 `2026081761/2026081762` screen 从起点固定 4,000 updates，并同时要求 H05、wrong-route `>=0.10`、`disable_routed_projection >=0.05`。

该 factorized screen 已使用预登记 package identity `A52C5222…486DA4` 正常完成，两臂各 4,000 updates，fresh cache、架构、matched active FLOPs、shared no-op、strip/reload、source 与 recurrence 因果检查均通过。heldout shared/mixed 为 `0.55469/0.54395`，overall gain `-0.01074`、CI `[-0.04199,0.01953]`，numeric 回退 `-0.03516`；最大 wrong-route effect 只有 `0.00281`，`disable_routed_projection` 最大效应只有 `0.00781`。H05、wrong-route H06 与 conditional-write necessity 同时失败，机器结果为 `passed=false`、`authorizes=nothing`。因此 `2026081763/2026081764` calibration、H1 formal roots 与 transport 均保持不存在；当前 factorized 方向关闭，不得重跑、换 seed、降低 Gate、冻结 secondary floors 或启动 formal。完整复核见 `artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/SCREEN_REVIEW.md`。

2026-08-21 经用户明确授权，另立 `H1-WD` 非正式机制 screen；它不续跑旧 H1，而以旧 mixed deployment checkpoint 为只读 predecessor，保留 shared FFN 和 routed projection 两条单模型路径。预审在 heldout/train predecessor trajectory 上测得 common/projection 平均余弦约 `0.46`、正向重合 common 能量约 `0.23`，因此 W 阶段只把该正向重合分量写入 projection，D 阶段用冻结 `C0-(PW-P0)` target 残差化 shared FFN，禁止 projection-only。J 阶段与 untouched checkpoint 做同 schedule continuation；family 仅用于结果分层，不进入 W/D/J target。固定 root、预算、Gate 和边界见 `docs/v2-r1r-p1-h1-wd-overlap-residual-design.md`；该 screen 无论结果均 `authorizes=nothing`。

该唯一运行已正常完成并判为 `FAIL_H1_WD_NONFORMAL_MECHANISM`。W/D 的拟合与函数保持成立：W heldout nMSE `0.00196`，D common/total-transition nMSE `0.01393/0.00316`，D free-rollout prediction agreement `0.99121`，common residual energy `0.71348`。J 后 WD/control heldout 为 `0.54785/0.54297`，净增益 `+0.00488`、paired 95% CI `[-0.00195,0.01172]`，未达到 `+0.05`。最终 common-off/projection-off drop 均为 `0.01172`；projection effect 相对 predecessor 的 `0.00781` 仅增加 `0.00391`，故 `common_remains_necessary=false`、`projection_effect_increased=false`。这证明欧氏输出重合可以被无 family target 地搬运并近似守恒，但不能把“重合方向”解释为任务关键能力，也没有产生互补分工。该 root 已消耗，禁止重跑、调 coefficient 或降低 Gate；完整归因见 `docs/v2-r1r-p1-h1-wd-failure-review.md`。

用户随后授权一个隔离的 decision-causal screen，直接修正上一轮“可转移分量”的定义而不修补旧 root。新 target 的幅度来自固定 route 下真实 common-off answer-margin drop，方向来自 selected projection output 的 scalar VJP/Fisher；train/heldout target 按 microbatch `4` 固化为无 family/task target 的 CPU 数据集。W 检查 `P-P0` 是否拟合 `±ΔP`，D 从 common 删除 W 实际写入量，J 用 ordinary answer CE 与 two-path causal-target allocation lock 保持分工，不使用 teacher logits、route supervision 或 family CE。负 VJP matched control 只回答“方向是否重要”，不能证明 shared+routed 相对 shared-only 的架构收益。冻结合同与唯一命令见 `docs/v2-r1r-p1-h1-wd-decision-causal-{design,execution-command}.md`；结果无论 PASS/FAIL 均不授权 H1 formal、F1 或 P2。

该唯一运行已以 `FAIL_WD_DECISION_CAUSAL_WRITE_FIT` 正常 fail-stop。target Gate 全部通过：train/heldout positive fraction `0.97900/0.57715`、realized/request `0.85606/0.93106`、GPU peak `0.32335 GB`。W 两臂各完成 800 updates，但 causal/control heldout transfer-nMSE 为 `1.00450/0.99247`，而 transfer 只占完整 target energy 约 `0.468%`；因此旧式 full-target nMSE 约 `0.0047` 实际是零学习假象。D/J、干预、Shapley 与方向收益均未执行，不能归因于 J 奖励不足，也没有形成架构结论。当前 root 禁止重跑；若未来另立研究，必须先用 projection-parameter Jacobian/Fisher 资格化跨记录共同可达的 `J_θ(x)Δθ`，不能再次把 raw per-record output VJP 直接当作可写入分量。完整复盘见 `docs/v2-r1r-p1-h1-wd-decision-causal-failure-review.md`。

为执行上述可达性诊断，2026-08-23 冻结了独立 R0–R4 screen：R0/R1 分别测 global 与二元 route centroid，R2 只用 target 构造前冻结状态，R3 用跨记录共享的 projection-parameter Jacobian/Fisher，R4 用 unrestricted/within-route permutation、route-conditional sign、谱与 split-half 等注册 null。唯一真实运行完成 train 4096 与 heldout 1024 的 replay 后，在任何 R2 fit、R3 解或 R4 null 前以 `CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY` fail-closed。旧 target bank 由 microbatch `4` 物化，新 capture batch 为 `128`；不同 CUDA reduction/kernel geometry 使 common/projection 尾部最大漂移达到 `7.96914e-05/9.50396e-05`，超过冻结绝对容差 `2.5e-05`。首批 batch-128 spot check 与 batch-4 integration smoke 都未覆盖这一全 split 最坏值。这是执行合同与 preflight 覆盖失败，不是 R0–R4 假设反证；无 `result.json`，也不能把内存中构造但未封存的 R0/R1 centroid 当作结果。root 已消耗且 `rerun_authorized=false`，完整终局见 `docs/v2-r1r-h1-wd-direction-geometry-screen-failure-review.md`。

v2 successor 随后以 sibling exclusive lease、pre-root 全量 batch-4 replay 与 active-site/dual-sketch 控制直接切换，唯一运行完整产出 R1–R4。正确 route pairing 的 permutation `p=9.999e-05`，但 route heldout increment 为 `-0.586%`；projection trunk heldout `+3.189%` 的 record/site CI 均跨零且只 `2/4` lambda 为正；exact shared head `-0.334%`，sampled local-J `-5.922%` 且 CG/finite-difference fidelity 不足。R4 两套 sketch 的 global/route resultant 均拒绝 registered null，所以 residual 不是纯随机兼容，但 spectrum/alignment 的 1024 次 null 无法达到 `0.01/12` 的离散 p 分辨率，终态保守写 `unexplained_under_this_screen`。更关键的是 target-positive prevalence 从 train `4010/4096` 降到 heldout `591/1024`；在 positive heldout 子集上 route absolute gain 仍为 `+6.619%`，整体负 gain 来自大量 no-write 记录的系统性误写。故后继研究应先解决 target-before write gate 与 split shift，不得把当前结果当作加强 projection 训练或“先写后删”的授权。完整终局见 `docs/v2-r1r-h1-wd-direction-geometry-v2-result-review.md`。

P0-D v1 于 2026-08-01 生成每族 train 4096、validation 512、五个规定 OOD/causal split 各 512，旧审计器自判十项为 `true`；主设计层复核随后证明该 conjunction 漏掉关系型位置捷径与监督真值，故当前判定为失败。旧结果见 `docs/v2-r1r-p0-result.md` 和 `artifacts/v2-r1r/p0-v1/`，只保留作 rejected diagnostic。第 16 节已冻结 generator v2、真实 Qwen tokenizer、candidate/action 随机化、ERE 多事件必要性、composition/language 独立性、claim truth 与结构化 heuristic 等 13 项新 conjunction。v2 已使用 seed `20260801` 完成 smoke（13/13 true）和正式规模生成；独立 formal audit 的 13 项中 10 项 true、3 项 false：CPS horizon action-definition position grouped deviation `1.0`，CPS distractor/horizon 缺少 required claim kinds，CPS causal/horizon `longest_plan` heuristic 分别为 `0.310546875/0.2734375`。主设计层独立验收还复现了只数 action definition 即达 `0.83203125` 的 horizon source-only shortcut，确认 ERE length OOD provenance 占比 `1.0`、CPS train/validation 正确计划仅长 1/2，以及多个 heuristic/heldout/claim/provenance Gate 尚未真实实现。正式证据保留于 `artifacts/v2-r1r/p0-v2-failed-audit-20260801-1/`，详细复核见 `docs/v2-r1r-p0-v2-main-review.md`；本轮按规则停止，未启动 P0-M、Qwen hidden cache、Boundary/core 或任何 GPU 训练。

旧 v3 合同曾要求 15-Gate ledger、F01–F15、ERE 五类 provenance/逐样本必要性、CPS 依赖图 hard negative、分层 surface assignment、claim 和 source snapshot。2026-08-01 执行层 machine D0/D1 通过，唯一 D2 `artifacts/v2-r1r/p0-preflight-v3-20260801-1/` 输出 `G01–G04、G09、G10、G12、G14、G15=true`、`G05、G06、G07、G08、G11、G13=false`，随后严格停止。主设计层接受该失败 artifact 与停止纪律，但拒绝 v3 实现：G07/G08 读取不存在的 family aggregate 而恒假，D0 只有负向 fault 没有 known-good 全 conjunction；G06/G08/G13/G15 低于冻结合同；CPS 仍围绕固定五角色候选骨架，non-NONE hard-negative、较长 valid-suboptimal、四类 claim 和位置配额大面积不满足；post-run 文档同步还会让 fresh G03/G15 自失效。旧合同与命令只保存在 D2 source snapshot，完整证据见 `docs/v2-r1r-p0-v3-main-review.md`。

v4/v5/v6 的失败与 sealed artifact 均保留追溯；v6 的 14/4/12/245 窄事实有效，但 query placeholder 十项主审反例证明 coarse coverage tag 不足。v7 不修补 v6：它另立 slot × scope × token 规范并在唯一 formal 得到 508/508 invalid、350/350 acceptance、858/858 coverage；父任务 27/27 新鲜探针通过，R0A 正式 accepted。v8 的机器矩阵、seal 和协议事实有效，但五项 fresh probes 全部错误通过，R0B 主审 rejected；v9/v10/v11 分别接受有限 R0B/R0C/R0D；v12/v13 接受 production entry/generator smoke，v14–v16 保留各自失败或窄组件身份，v17 正式关闭 P0-D，P0-M v5 正式关闭训练通路 smoke。完整合同与判决见各版 design、execution 与 main-review。所有 fixed formal 均停止且不可重跑；只有另立 P1 合同后才可进入泛化验证。

## 4. V2-B：静态多模态与双层 MoE

V2-B 只有在 V2-A 通过后启动。

### 4.1 V2-B0：不可单解的文本 + 视觉任务

- 任务必须同时依赖文本约束和视觉事实；
- 同 prompt 风格但图像事实不同的 hard negatives；
- 冲突文本/图像样本；
- no-text、no-image、shuffled-image 和 text-only 基线；
- 视觉先语言化与直接视觉 latent 两条路径。

### 4.2 V2-B1：Boundary-MoE + Dense core

- 文本/视觉按显式 modality routing；
- 专家内部异构，边界统一 `D_latent`；
- 先 raw concat 保真，再比较 Attention Pump；
- 潜变量核心使用 V2-A 胜出配置和 Dense FFN；
- 输出先固定文本专家。

Gate：Boundary-MoE 相对普通统一接口/早期拼接形成因果正确的质量或成本收益；no/shuffled modality 明显下降。

### 4.3 V2-B2：Attention Pump 与 bypass

比较：

1. raw expert tokens；
2. pump-only；
3. pump+residual/bypass。

`K_pump` 固定并单独报告，route weight 不直接决定输出长度。Gate 同时看重建、任务、消融和成本。

### 4.4 V2-B3：FFN-MoE 2×2 对照

| 外部接口 | 核心 FFN |
| --- | --- |
| 普通接口 | Dense |
| Boundary-MoE | Dense |
| 普通接口 | FFN-MoE |
| Boundary-MoE | FFN-MoE |

FFN-MoE 从 4 experts、top-1 和约 1.25 capacity factor 起步。Boundary router 与 FFN router 分别记录使用率、entropy、collapse 和任务族分布。

Gate：组合相对各单项产生多 seed 可重复净收益；若只增加总参数而没有 active-compute 或质量收益，回退 Dense。

### 4.5 V2-B4：静态文本 + 视觉 + 动作表示

- 固定输入中加入动作状态和动作候选，固定输出使用单一动作或文本专家；
- 检查动作合法性、视觉事实、文本约束和审计一致性；
- 任一单模态不能独立解决任务；
- 只验证动作语义表示与静态路由，不执行真实动作，不允许模型主动 READ/EMIT，不引入超时恢复、工作树或人类目标保持。

Gate：相对同预算统一多模态模型/早期拼接，Boundary-MoE + 胜出 core 在多 seed 上形成可重复质量或 active-compute 净收益；no/shuffled modality、router 干预和固定专家对照形成因果证据。若收益只来自更多总参数，或静态动作合法性不成立，则 V2-B 停止，不进入 V2-C。

## 5. V2-C：多层、多线程、树图混合全双工智能体

V2-C 只有在 V2-A 与 V2-B 正式通过后启动。完整合同见 [`v2-c-hierarchical-full-duplex-agent-experiment.md`](v2-c-hierarchical-full-duplex-agent-experiment.md)。它不是把高保真 I/O、工作树、异步节点、专家晋升和人类目标拆成数轮松散实验，而是围绕一个系统问题组织：同基座、同工具、同环境和同预算下，多层、多线程、树图混合的全双工 Boundary-MoE 智能体，能否相对 Flat 与同步层级基线同时改善目标忠实度、异步任务质量、延迟、恢复和成本。

内部按 C0 协议审计、C1 同步正控制、C2 多线程全双工、C3 树图与主动 Boundary、C4 离线能力演化、C5 整机 formal 推进。C0-C4 只负责构建和因果归因；只有 C5 两类正式任务通过，才能声明 `integrated-system`。人类主权、权限、取消和不可逆动作审批在所有阶段都是代码硬约束；理解、保持、分解、修订和停止人类目标，才是 V2-C 的被测行为能力。

## 6. 当前不执行的内容

以下内容已经聚合进 V2-C，但在 V2-A/V2-B 通过前只有设计地位，不属于最近实现：

- 多输出专家候选 arbiter；
- 图像/视频/音频高保真生成与编辑；
- 工作树、记忆树和 provider KV 联动；
- 长程 Agent、多 Agent 与跨窗口恢复；
- 离线新增专家与授权晋升；
- 渐进生成和动态资源预算；
- 商用 serving、数据治理、SLO、计费和客户试点。

## 7. 旧路线收口

以下路线不再继续：

- AV-J-D；
- 继续优化 AV-J-C 粗粒度 record reward；
- 把 object/cell/count/relation slot 当作目标架构必须采用的内部本体；
- 单阶段 from-scratch Micro-Omni 作为 V2 正式验证；
- Attention Pump 唯一通道；
- 内部宪法自评后直接固化生产权重；
- 工作树 drop 等同 KV 无损回收。
- 把主动调用、真实工具和人类目标能力提前塞进 V2-B；
- 把高保真 I/O、工作树、异步节点、专家晋升和整机评测继续拆成旧 A1-A4 并行路线。

旧脚本、测试、报告和 artifact 只保留历史证据地位。后续若实施 V2，应删除或重写锁定旧契约的代码和测试，不建立兼容 wrapper。

## 8. 最近执行顺序

1. 已完成 V2-A0 基座候选筛选、canonical 任务选择、数据 schema 和可复现数据生成；
2. Qwen3.5-2B 普通/组合 heldout text-CoT probe 已形成正向信号，length-heldout 仍有错误；
3. 已完成当前结构的 2B V2-A1 `K=8/T=8` latent mechanism smoke、checkpoint/resume、no-bypass 和干预链；
4. 已完成 A2 K/T（含 K16/T8 与 K8/T2/T4/T16）、prompt contract、source-mean 初始化、copied/MLP transition、latent-attention probe、mean/flatten readout、source reread、token-wise source adapter、full-data、masked state/query-state supervision、step-level verifier RL 以及 0.8B/2B 基座对照；source-layer bank、token mixer 和 latent-attention 的失败入口已删除；2B K8/T8 full-data MLP test `0.6016`，composition-heldout `0.3672`，source adapter bottleneck=128 为 `0.5859/0.4063/0.5078`（test/composition/length），masked state supervision 为 `0.5703/0.3984/0.4219`，latent-attention 为 `0.5078/0.4141/0.4531` 且 no/shuffled latent 均 `0.0938`，verifier-RL 为 `0.5625/0.3906/0.5547` 且状态 verifier 未学会，0.8B 同构 latent test `0.2578`，均未形成相对 text-CoT 的稳定 Pareto，Gate A2 未通过；
5. A1.6 formal relation Gate 失败后，已通过只读诊断定位 continuous closure 故障；
6. A1.7 已完成 data、overfit32、三个初始化 seed 的 formal/causal、2×2 结构/closure 消融和 8/12/16 步压力；短程 Gate 为 `3/3`，旧严格长程 Gate 为 `0/3`，closure 的长程价值成立，但地址分离的独立收益未成立；
7. A1.8 已完成三组独立 data/model seed 的 T1–16 random-depth training、T20/T24 OOD、T32 诊断、因果、稳定性和成本；总 Gate `3/3` 通过，A1.7 漂移主要归因为 horizon mismatch；
8. A1.9 Qwen hidden boundary 已完成：三组 cache audit、formal 与 hidden causal Gate 均通过，总 Gate `3/3`；
9. A1.10 已完成 full-token + anonymous K-slot + generic recurrent reasoner 联合切换；overfit32 通过但 formal `0/3`，所有 hidden intervention 按 Gate 停止；
10. A1.11 正交故障定位已完成：Boundary 严格 overfit pointer Gate failed；exact-symbolic Reasoner overfit passed、formal `0/3`；纯组合故障解释被否定，所有 formal 后干预按 Gate 停止；
11. A1.12 binding × cursor 已完成：三臂 overfit32 全通过、formal 全为 `0/3`，两项候选不足；
12. A1.13 transition × closure 与 A1.13F fixed-budget audit 已完成：联合臂 state `3/3`/full `1/3`，单因素臂不稳定且不是 early-stop 主导；
13. A1.15/A1.16 fresh-seed query-coupled 两臂均 state/full `0/3`；
14. A1.17 同共享初始化配对审计已完成：coupled-CE/noCE 均 `0/3`，独立 pooled-answer objective 的优化脚手架作用成立，但架构未通过；
15. A1.18 FINAL-SAUX paired formal/causal 为 `2/3`，证明 final-only 全局状态监督可替代答案梯度但 seed 不稳定；
16. A1.18B TSAUX 已完成三个 paired 与三个 fresh seed：两组 formal/causal 都为 `3/3`，部署辅助头全部删除，机制 Gate 通过；
17. A1.19H generalized hybrid core 已完成：H1/H2 overfit32、三 seed formal 与 formal 后 causal 均通过，heldout N5 与 N5+relation 三 seed 全通过，机器分类 `generalized_hybrid_core_confirmed`；
18. A1.20B learned full-text boundary 已完成 run-1 并在 formal 前 eligibility 失败；
19. A1.20C 分层 compiler × straight-through 修复已完成监督审计、实现、smoke、目标臂 fixed-5000 overfit32 与实际 checkpoint 梯度审计；目标臂 strict Gate 失败；
20. A1.20D post-stop 机制修复已完成：双向三遍 section decode 修复 N5 截断，canonical 完整 split matrix、hidden causal 与三个 routing schedule probe 均通过放宽机制门，但不是 fresh-seed formal；
21. A1.21P 已完成 K=1 负基线、两个在线五条 smoke、路径稳定性和机器 assessment；六项正式合同缺口使 `a121p_passed=false`，A1.22A 与 V2-B0 保持停止。
22. 已完成 V2-R1R 高层合同、P0-D v1–v17、P0-M v1–v5 与 P1 v3–v8L 闭环。v6/v6R 已资格化 temporal mechanism；v7 证明该机制和 episode-specific state dependence 可扩展到 full-data 联训，但 CPS causal/OOD 失败；v8R 排除 scratch/coverage，v8D 排除 source/Boundary blindness，v8L 则在 ERE 通过、CPS bootstrap 失败且 lexical numeric teacher 未充分资格化处执行最终停线。K=1、baselines、完整 P1 与 P2/P3 均未运行。
23. P1-NR1 已在独立主审后完成唯一 formal：手工预期、复杂 DAG、oracle-target fault registry、known-handle state 与 fail-closed runner 的 N01–N07 全 true，两根 roots 与 transport 已消耗。状态只授权 H1 设计，不能跳到 F1/P2。
24. 已完成 V2-C 统一系统实验设计：旧 A1-A4 目标已聚合为 C0-C5；当前没有运行时、工具接入、数据、代码、训练或 formal artifact，且不改变 V2-R1R 的当前优先级。

当前已有 A1.8 structured core 到 T24、A1.9 oracle-role-segmented frozen Qwen hidden boundary、A1.10–A1.17 分层失败归因、A1.18B TSAUX 训练机制闭环，以及 A1.19H generalized hybrid core 的多 seed formal/causal artifact。A1.20D 又证明 full-text entity/program compiler 必须双向闭合 section boundary；修复后 canonical N5、长程、relation 与 hidden causal 均形成强诊断正证据，说明当前 hybrid mechanism 已值得继续。可是 learned full-text Boundary 的三组 fresh formal、matched Pareto 与完整 V2-A formal artifact 仍不存在。不得把 A1.20D 单起点诊断、routing 表面改写或五条在线 smoke 写成 A1.21P/V2-A 通过。

## 9. 担忧与不确定性

1. 成熟文本基座可能无法在本机 8GB 显存上按理想配置训练；应优先冻结、缓存 hidden states、使用小型 latent reasoner，而不是退回 from-scratch 代理模型冒充目标架构。
2. 复制顶部 block 形成 recurrent reasoner 是参考实现，不保证最优；R1 失败时应与共享层或独立 reasoner 对照。
3. audit decoder 可能事后合理化；没有因果干预不得判可审计。
4. 计算匹配比参数匹配更重要；latent 方案不能靠更多 step 和更宽状态获得不公平优势。
5. Boundary-MoE 与 FFN-MoE 同时训练可能使归因和优化不稳定，因此必须先 Boundary/Dense，再做 2×2。
6. 商用路线需要首个明确垂直场景；当前文档只覆盖架构和研究路径，不承诺市场已经验证。
7. Qwen3.5-0.8B 在当前 Windows 环境缺少 `flash-linear-attention`/`causal-conv1d` fast path；文本塔可以运行但 baseline latency 较高，不能把该环境差异包装成 latent 成本优势。当前 32-example text-CoT smoke 的 `artificial_output_token_cap` 为 `null`，未设置人为总输出长度上限。
8. 当前 2B text-CoT 在普通 test 与 composition-heldout probe 为正向强基线；修正后的 2B K8/T8 full-data MLP latent 为 `0.6016/0.3672/0.5078`（test/composition/length），K8/T4 为 `0.5234/0.3672/0.4063`，K16/T8 为 `0.5078/0.4063/0.3906`，source adapter bottleneck=128 为 `0.5859/0.4063/0.5078`，K8/T2/T16 也未形成 Pareto；0.8B 同构 latent test 为 `0.2578`。仍没有多 seed 稳定性和 Pareto 优势，不能把训练 loss、teacher/state 辅助 loss、adapter 的单项 heldout 增益或单次准确率包装成 A2 通过。
9. 将 text-CoT 的 2 个 trace demonstrations 注入 encoder 的 probe 反而降至 test 0.0625，说明 demonstrations 不是当前 latent 的稳健修复，后续不能把它当作默认输入合同。
10. Qwen3.5 原生 thinking 的单样本探针在 206 token 后未形成可解析终态；native thinking 不能被当作 A0 visible CoT 的替代 Gate。
11. token-wise source adapter 在 bottleneck=128 的两个 seed 中把 composition-heldout 提升到 `0.4063/0.4609`，但普通 test 为 `0.5859/0.5078`、length 为 `0.5078/0.4766`，没有同时超过 K8/T8 baseline `0.6016/0.3672/0.5078`。bottleneck=512 的普通 test 为 `0.5469`。该方向只能作为下一轮信息保真/正则化设计的候选，不能直接进入 V2-C 能力演化。
12. 将前三个 latent slot 绑定到 amber/cobalt/jade 的 register-slot supervision full512 probe 为 `0.5859/0.3750/0.5078`，没有把 adapter 的 composition 增益转化为答案泛化；过程监督仍不能替代正式 verifier/reward 设计。
13. 首次 step-level verifier self-critical RL full512 probe 使用 v5 有效步骤 mask，结果为 `0.5625/0.3906/0.5547`，test greedy register accuracy `0.0968`、state/final exact 均为 `0`，末尾 policy entropy 约 `0.0017`。RL 已进入反传但发生策略塌缩，不能把 reward loss 或 sampled/greedy reward 当成状态学习证据；后续若重做，必须先解决 reward 信号稀疏和 policy collapse。
14. 为符合白皮书的 Attention + Dense FFN 参考结构，曾加入 identity-initialized latent-attention transition；full512 的 test/composition/length 为 `0.5078/0.4141/0.4531`，no-latent 与 shuffled-latent 都为 `0.0938`。它没有形成因果递归或 Pareto 优势，当前代码与 CLI 已删除，不保留并列旧入口。
15. state/query-state supervision 已改为只计真实程序步骤，masked state full512 为 `0.5703/0.3984/0.4219`，但 test/length 仍低于无监督 MLP baseline，且 shuffled-latent 高于 no-latent；mask 修复解决了计量错误，不等于过程监督形成了有效 verifier。
16. A1.9 的每个 role hidden 都来自 oracle character span，而且 Qwen hidden 本身具有全局上下文；hidden 反事实已经排除多种身份/答案捷径，但不能替代自主 role discovery。
17. A1.18B 已把 A1.17 的辅助梯度拆开：FINAL-SAUX 为 `2/3`，TSAUX paired/fresh 均 formal/causal `3/3`，说明 answer 标签不是必要条件，稳定性来自每步 global workspace → complete state 的密集信用分配。该结论要求后续架构保留 training-only TSAUX 等价机制、state-only checkpoint selection、部署剥离和 causal Gate，不能恢复推理答案旁路。
18. A1.9/A1.10 cache 生成都依赖当前缺少 `flash-linear-attention`/`causal-conv1d` fast path 的 Qwen fallback。A1.10 三组 full-token cache 净编码 `4,767.62` 秒、占用 `38.36` GB，还不含模型/tokenizer 首次加载；cached reasoner 延迟不能与在线 text-CoT 延迟直接比较。
19. A1.11 Boundary 的 hard re-embedding 是只读诊断，不是被验证的新架构；若未来使用 prototype-anchored/discrete addressing，必须直接切换合同并重新做 fresh formal，不能把诊断偷偷变成兼容补丁。
20. 当前 TSAUX 使用每步完整三寄存器 oracle state。它解决结构化 core 的训练机制，不解决开放任务的 target 来源；但白皮书不要求零 teacher 训练。后续允许 executable/verifier-filtered teacher target，必须报告成本、覆盖和错误率，并禁止部署/推理时 teacher 绕过。
21. 纯匿名 `K-slot` 与全连续寻址都不再被当作必须坚持的核心承诺。当前候选 core 是 `S_t=(A_t,H_t)` relation-addressable hybrid workspace；离散 sidecar 只能承载身份、地址、类型和控制，不能承载答案或完整语义 state。
22. A1.19H 已通过，但只形成 hybrid core 证据。完整 R1 仍必须在 full-text、同基座、计算匹配条件下形成 Pareto，并训练自然语言 audit readout 通过因果忠实性 Gate；不得用 symbolic trajectory 指标代替 V2-A formal，更不得提前宣称 V2-C 系统能力。
23. A1.19H-H2 的旧训练路径存在严重 CPU/GPU 同步浪费。后续 A1.20B–A1.22A 的 cached training 必须在正式启动前做 tensor cache、热路径同步、固定 shape 和吞吐基准审计；GPU 功率不是单独 Gate，step/s、端到端延迟和数值等价才是主指标。
24. A1.20B 已按上述要求在正式训练前完成 Qwen cache batch benchmark、mmap cached training 与同步/prefetch 对照。低功率的剩余部分来自约 `2.77M` 参数 Boundary、batch 16 和周期性全 validation，而不是逐样本 Qwen 编码。更关键的失败是 threshold/argmax 控制切断 state CE：局部 mapping CE 在 train batch 可接近零，但 heldout 完整程序联合 exact 只有 `0.087891`。下一轮必须先解决执行级离散信用分配和分层 entity/program binding，不能用更高 GPU 利用率替代机制修复。
25. A1.20C 已把 overfit32 训练推进到约 `4.24 step/s`，GPU 在热路径可达到高利用率；性能不再是本轮失败主因。真正残留的是 pointer-validity 的 hard mask 断点和实际负梯度冲突：state-vs-local 全局 cosine `-0.7366`、presence heads `-0.9944`。下一修复必须改变训练接口和优化顺序，不能只延长固定 `3e-4` 训练。
26. A1.20D 证明上述冲突至少部分来自错误的单向 section 因果图；修复后旧 checkpoint 已可通过强诊断矩阵。因此下一次正式训练应先保持双向 section decode，再判断是否仍需 gradient projection 或 loss schedule。
27. A1.21P 当前最危险的假阳性来自 baseline 不公平与任务同构：五条 smoke、canonical demonstrations 和 COPY/SWAP 表面改写都可能夸大 hybrid 优势。下一轮不得只增加 routing 样本，必须先建立新可执行代数和 matched training/teacher 合同。
28. R1R 不要求模型从 final loss 自行发明地址、mask、预算或信用分配；这些由代码或训练脚手架提供。禁止的不是工程结构，而是推理时 oracle/答案旁路和在模型内硬编码任务语义。
29. R1R 的目标是最小充分证伪，不是把一个切面做到 `1.0`。内部 trajectory exact 降为诊断，正式结论看跨任务行为、因果、最差 seed 和 matched 质量—成本。
30. v6 证明粗粒度 coverage tag 不能替代语义位置的组合覆盖。后继 validator 合同若获准，必须机械枚举 schema 位置 × reserved token × API，并继续保留合同外新鲜探针；不能因 frozen matrix `245/245` 就推断书面不变量完备。
