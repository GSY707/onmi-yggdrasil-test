# V2-R1R P1 v7 integrated K=8 失败复核

日期：2026-08-11

## 1. 判决

P1 v7 的 preflight 与 query-cache 保持 sealed PASS，K=8 保持 sealed `FAIL_P1_V7_INTEGRATED_K8`。三个 fixed roots 不得修改、重封或重跑。K=1、direct SFT、text-CoT、完整 assessment 与 P2 均未启动。

这次失败不是“模型完全没有学会”。模型在 validation 上达到 ERE/CPS `0.999023/0.851562`，temporal witness 的 ERE/CPS accuracy 为 `0.766357/0.938965`，zero-state drop 为 `0.266357/0.438965`，swapped-state drop 为 `0.532715/0.877930`，batch-shuffle-middle 的平均答案 drop 为 `0.627441`。因此共享 recurrent state 已承载 episode-specific 信息，v6R 的时序机制在联合训练后也没有消失。

正式 FAIL 仍然成立，因为 CPS 的主 OOD 与反事实答案依赖没有建立：CPS composition/horizon/distractor/language 分别为 `0.587891/0.754883/0.472656/0.214844`；修正评估器后，CPS causal pair flip accuracy 只有 `4/512=0.0078125`。该结果说明最终答案路径能利用 latent state，却主要学习了普通训练分布的表面判别，而没有被迫响应决定答案的局部语义变化。

## 2. 已确认的测量缺陷

### 2.1 causal pair role 名称不一致

数据合同使用 `pair_role=base/flip`，v7 evaluator 却要求 `base/counterfactual`，导致正式报告把结构完整的 512 对/族全部登记为 invalid，并把 K03 写成 `0/0`。父任务只读重算 sealed prediction arrays 后得到：

- ERE：pair flip accuracy `361/512=0.705078125`，prediction flip rate `0.78515625`；
- CPS：pair flip accuracy `4/512=0.0078125`，prediction flip rate `0.025390625`；
- 两族 512 对均为完整 `base/flip` 结构。

因此这是报告器缺陷，但没有遮蔽一个本应通过的模型：修正后 ERE 与 CPS 仍都低于原 K03 `0.80`，尤其 CPS 明确失败。

### 2.2 K04 与当前计算图不匹配

v7 的每次 recurrent transition 都可以 cross-attend 完整 source。把中间 state 置零后，后续 transition 可以合法地从 source 重算；因此 `zero-middle` drop 小不能单独推出“答案不依赖 latent state”。本次同一模型对合法同组 batch-shuffle state 的平均 drop 为 `0.627441`，已经证明答案依赖 episode-specific state。

未来若要测量不可恢复的中间状态依赖，应在干预后同时隔离 source，比较 source-isolated suffix 的原状态与零状态，而不是把带完整 source 的恢复能力判作失败。v7 K04 保持历史 Gate 结果，但不再作为后继合同的架构必要条件。

### 2.3 K06 把效率问题误写成必要机制

在冻结 Qwen 的 contextual hidden、全局 source cross-attention 和两层 latent block 下，T=1 可能已经完成大量计算。T1 与 full 接近意味着较浅计算可能足够，它应触发 matched quality-cost control，而不是强迫模型表现出更深 recurrence 依赖。白皮书要求最终 latent-only readout 与质量—成本 Pareto，不要求人为制造 T1 drop。后继合同把 T1/T2/T4 保留为诊断，不再作为资格 Gate。

source-token reversal 同样只重排已经全局 contextualized 的 hidden，不能被解释为自然语言语义破坏，因此也只保留为弱诊断。

## 3. 模型根因

### 3.1 temporal 与 answer 找到了两条近正交路径

共享参数上的 answer/temporal gradient cosine 在 update 100、4096、5120、20480 分别约为 `0.0240/-0.0207/-0.0496/0.0998`。训练后 answer 梯度范数已接近饱和塌缩，而 temporal 梯度仍强。结合 K10 PASS 与 CPS causal FAIL，最符合证据的解释是：

1. temporal objective 建立了可审计的过程表征；
2. ordinary answer CE 另行利用 final state 中的表面统计，足以拟合普通训练和 validation；
3. 两个目标共享参数，但没有约束“改变因果变量时，最终答案必须随之改变”；
4. 所以 state 是必要载体，却不保证其被 answer head 使用的是正确语义坐标。

这不是简单增加 K、update 或学习率能解决的问题。缺少的是训练支持中的反事实闭包，而不是更多相同 ordinary CE。

### 3.2 CPS 的 OOD 命名混入了未支持的单因素外推

当前 CPS train/validation 固定在 candidate count 5、action count 6、plan depth 3，模板只覆盖 plain/reordered。composition 与 distractor split 又同时改变 candidate/action/depth 或干扰规模，language split 使用未在训练中出现的 indirect 表述。因此其中一部分不是“已见因素的新组合”，而是原始结构和语言外推。

这不取消本轮 OOD 失败，但限制解释：不能仅凭这些 cell 区分组合推理失败、规模外推失败和 renderer shift。完整 P1 重做时必须让各单因素先在训练支持中出现，再封存组合；语言也要把训练 renderer 多样性与真正 heldout renderer 分开。

## 4. 后继判决

不直接扩大 v7，不降低原 Gate，也不进入 P2。下一步另立 P1 v8 causal-bridge qualification，用当前 causal pair 的一部分作为诊断训练支持，在相同模型、初始化、样本数和 update 下比较：ordinary、causal-unpaired、causal-paired 三臂。

v8 只回答反事实训练支持是否能把 answer path 绑定到已经存在的 temporal-semantic state。由于 v8 会把旧 causal heldout 的一部分转为优化数据，它不能成为 P1 OOD 或架构通过证据；若成功，只授权用 fresh seed、重新因子化的数据合同重做完整 P1。若反事实视图与显式配对都无法显著改善 sealed causal audit，则应停止训练补丁路线，重新设计 final-state/readout 耦合机制。
