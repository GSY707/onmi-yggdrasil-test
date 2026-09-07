# V2-A Closure C1S SRW S1 结果复盘

## 1. 核心结论

C1S Semantic-Routed Workspace 的唯一 S1 Overfit32 已完整执行并在机制资格 Gate 正常失败。模型把 32 条训练记录的答案全部拟合正确，也学到了 owner、operation 与多种干预敏感性；但答案仍不依赖 recurrence，单条记录内的多地址因果贡献明显不足，若干注册动态特征也未达到冻结的 `0.95` balanced-accuracy 下限。因此机器终态为 `FAIL_V2_A_C1S_S1_QUALIFICATION`，`authorizes=nothing`；S2、single-seed formal、C2 与 V2-A PASS 均未授权。

这不是 crash、训练不完整、target coverage 缺失或封存损坏。它说明当前 SRW 可以记住答案并形成部分结构信号，但还没有证明“地址化多状态 recurrence 是答案所必需的计算路径”。

## 2. 执行与封存

S1 使用 fresh K8、固定 4,000 updates、batch 8、无 checkpoint selection，完整写出唯一 endpoint。训练完成 4,000/4,000 optimizer steps，stage wall time 为 `5452.985s`，训练器 wall time 为 `5328.911s`；step median/P95 为 `1.1502s/1.2432s`，data-wait fraction `0.1295`，peak CUDA memory `766,601,216` bytes。source identity 始终为 `E3C5D7EC8C75518F9AB9EBE7574459A7DB27990F35D57D153FA63D869A8B77F4`，且 `source_stable=true`。

正式证据：

- result SHA-256：`9D0B06E14C7B2E5E0CDBDC7ED7C35D259BDD540E4BE3F65DA1E7F82D80E38892`
- evidence-seal SHA-256：`FAFF301AB40BEFB3D99106EEA8B0752F63DB96BC5390F7D5068A4D89F668A9D2`
- endpoint SHA-256：`D6F44CFC2C6F0A6F0841AA232E0DE73DA08EB7D0873D26E930C4E28E72AAA2E9`
- gate/evaluation/training SHA-256：`5670347B…E1E0` / `0D88E2AA…5B62` / `68DCC10C…FBF`
- evidence-seal 独立重放：`63/63` matched，missing、unexpected、mismatched 均为空

S1 preflight 也保持独立封存：result/seal 为 `EBEA1FAB…AD85` / `B3F3862E…A8F1`，`68/68` replay；其 100 个 optimizer steps 仅属于 disposable benchmark，formal-stage steps 与 model writes 均为零。

## 3. 学会了什么

答案行为本身全部通过：overall `32/32`，ERE/CPS 各 `16/16`。query/source/target ownership、operation-active、answer-independent query、duplicate payload control、slot permutation、auxiliary strip、S0 单槽仪器重放，以及 no-core margin drop、wrong-start、payload/operation zero/shuffle、relevant replace、target-route shuffle 等注册干预均通过。

这组结果排除了通路断路、无法拟合、owner 完全不可读以及干预量具失效。它也说明模型不是旧 C1R 那种只输出少数 raw label 的行为塌缩：当前失败发生在“答案已经完全正确之后”的机制层。

## 4. 为什么仍然失败

### 4.1 recurrence 不是答案的必要条件

关闭 core 后，overall 仍答对 `28/32 = 0.875`，远高于冻结上限 `0.25`；CPS 为 `12/16 = 0.75`，ERE 为 `16/16 = 1.0`。与此同时 no-core answer-margin drop 仍很大：CPS `10.9119`、ERE `6.1766`。

两者并不矛盾：移除 recurrence 会降低置信度，却没有改变大多数 argmax。也就是说，core 会增强答案，但 Boundary/H0 到 answer readout 的非核心路径已经足以记住大部分答案。故 `no_core_drop=true`、`no_core_near_chance=false`，当前证据不能把 recurrence 写成必要计算。

### 4.2 单记录内的多地址功能贡献不足

按 mean-replace answer-margin degradation 定义，同一记录至少有两个有效内容贡献者的比例仅为 `7/32 = 0.21875`，要求为 `>=0.90`；平均 causal effective slots 为 `1.58099`，要求为 `>=2.0`。分族看，CPS 只有 `1/16`，ERE 为 `6/16`。

跨记录使用了多个不同 owner slot，不能替代同一记录内多个地址共同参与计算。该结果支持“当前答案路径偏单地址/单内容”，因此 matched K1/K8 的 S2 对照没有资格启动。

### 4.3 动态状态质量未达到 S1 严格门槛

S1 对每个注册动态 feature 要求真实正负类、temporal changes 和 balanced accuracy `>=0.95`。target-side coverage 全部通过，但模型侧只有 CPS `processed=1.0` 与 ERE `operation_target=0.98717` 过门；其余为：

- CPS：`running_best=0.94767`，`final_winner=0.78571`
- ERE：`changed=0.94177`，`operation_source=0.87469`，`touched=0.91127`

因此 `dynamic_state_each_feature=false`。CPS `budget_ok`、ERE `present/query_owner` 等静态字段不在注册动态集合中，并未造成此次失败。`final_winner` 是 answer-derived detached auxiliary，不能单独构成机制结论；但 no-core 与 functional-K 两项独立失败已经足以停止 S1。

## 5. 架构含义与后继边界

本轮否定的不是“地址化状态完全学不到”，而是当前训练合同下的 SRW 尚未把已学会的答案约束到多地址 recurrence 路径。继续加 updates、降低 Gate、挑 endpoint 或只提高 state-loss 权重，都不能由现有证据直接授权：前两者违反 single-use 合同，后两者即使改善动态 BA，也未必消除 no-core shortcut 与单地址答案路径。

若继续研究，只能另立新的只读 failure-attribution / contract-diagnosis identity，先把三个问题正交拆开：Boundary/H0 直接答案捷径来自哪里；训练记录是否确实要求同一答案依赖多个地址；以及 temporal auxiliary 的低 BA 是读出、credit assignment 还是 target 表达问题。在该诊断完成前，不应创建新的训练 successor，更不能补跑旧 S1、S2 或 formal。

