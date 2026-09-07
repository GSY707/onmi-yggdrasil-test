# V2-R1R P1 v8D causal-decision witness 失败复核

日期：2026-08-12  
正式状态：`FAIL_P1_V8D_CAUSAL_DECISION_WITNESS`  
证据等级：sealed development-mechanism failure；不是完整 P1 或架构总否证

## 1. 核心判决

v8D 排除了“CPS 的局部代价变化根本没有进入 latent core”这一解释。模型会显著响应 base/flip 输入差异，但这种响应没有被组织成可迁移的候选成本、唯一最优与答案翻转坐标。准确分类是：

> `causal_perturbation_detected_but_semantic_reduction_not_formed`

因此不能继续增加同类 pair exposure、调整 loss 权重或单独更换 readout。下一步必须改变 state-formation 的信用分配方式；若直接、密集、无可学习旁路的因果状态监督仍失败，当前匿名 K=8 shared core 主线应停止。

## 2. 正式结果

三个正式 root 均已封存：

- preflight：`PASS_P1_V8D_DECISION_WITNESS_PREFLIGHT`，seal `2A2009FE56D5C89D870803FFB365A8E5018764BBA731115C3DCAB6177E4B18E3`；
- query-cache：`passed=true`，seal `EFA1064138DF53852A0EEA39DE68EDE86C1B94EA42367EACC76DD65AEA8FCEFE`；
- qualification：`FAIL_P1_V8D_CAUSAL_DECISION_WITNESS`，seal `674AC61A8FCDFBAC2F762EA11E8AD1E94DBE4AA2087D7417F7711E4BDB91E047`。

Gate 为 D01/D02/D05/D07 true，D03/D04/D06 false。ordinary validation 保持 ERE/CPS `0.999023/0.819336`，所以失败不是 broad competence 坍塌。

decision witness 在 ERE 上成立：joint audit judgment/query-pair 为 `0.888672/0.789063`，zero/swapped drop 为 `0.388672/0.777344`。CPS 则从 probe-only audit `0.498698/0.011719` 只提高到 joint `0.514323/0.087240`，zero/swapped drop 仅 `0.014323/0.028646`。这不是阈值附近波动，而是 CPS final-decision witness 基本没有依赖 episode state。

答案路径同样分裂。ERE audit raw/pair/prediction-flip 为 `0.898438/0.796875/0.8125`；CPS 为 `0.375/0/0.023438`。CPS pair 相对参考反而是 `-0.0078125`。

## 3. 训练后只读定位

以下是 sealed checkpoint 上的 post-stop development diagnostic，不回写正式 Gate：

- CPS audit base/flip 的 mean source hidden 相对差异均值/中位数为 `0.007774/0.007678`；
- 经 Boundary 后为 `0.011540/0.011405`；
- V7 `H_T` 为 `0.087501/0.061981`；
- V8D `H_T` 为 `0.401474/0.148177`。

V8D 因而不是忽略 perturbation，而是在放大它。与此同时，CPS audit 的 128 对中，只有 3 对改变最终预测，0 对同时答对 base/flip；87 对在两侧都预测 base winner。一个在 optimization `H_T` 上拟合的线性诊断头可把 flattened state 拟合到 `1.0`，但 semantic-winner audit 只有 `0.203125`、pair `0.023438`；local-label audit 为 `0.289063`、pair `0.015625`。这说明放大的差异仍是 pair/surface-specific，不是可迁移的决策代数。

## 4. 根因边界

当前证据排除四种简单解释：

1. **不是 source/Boundary 看不见数字变化**：差异进入并被 recurrent state 放大；
2. **不是只缺一个更强 readout**：训练期 query-conditioned probe 与 post-stop 线性诊断都不能迁移；
3. **不是 ordinary competence 不足**：CPS validation retention 通过；
4. **不是训练 pair 太少而未拟合**：optimization state 可被记忆，audit 仍不形成 decision coordinate。

最窄剩余解释是：当前目标只在 `H_T` 末端要求答案或命题翻转，允许 core 把局部数值变化编码为任意 episode-specific delta，却没有逐步约束“动作代价变化 → 候选累计代价 → 候选间比较 → 唯一最优”的同一语义链。可学习 ClaimProbe 又提供了一个共同适配器，使监督并不直接规定 workspace 必须落入稳定的语义坐标。

## 5. 后继决策

活动路线直接切换到 P1 v8L causal-state ladder：

- 从 sealed V7 competent checkpoint 重新开始，不继承 v8D 权重；
- 删除可学习 ClaimProbe；
- 用 frozen Qwen + frozen V7 Boundary 把每组互斥 claim 构造成固定的 lexical semantic-contrast direction，不使用可学习 probe 或 audit 拟合统计；
- CPS 沿真实 mutation path 在相应 `H_t` 监督累计代价，再在 penultimate/final state 监督成本比较、唯一最优与局部标签；
- ERE 使用同一 fixed-anchor scorer 监督变化后的语义答案与局部标签；
- 先做 causal ladder bootstrap，再与 ordinary rehearsal、answer CE 和 coupled switch 联训；部署 checkpoint 不含 anchor encoder、probe 或任务专属参数。

v8L 是当前 K=8 路线的停止判据，不是又一次开放式调参。若其 fixed-anchor audit transfer 或 CPS answer transfer 失败，停止当前匿名 K=8 P1 mainline，接受“当前 shared core 只形成 ERE state-executor 正证据”，不再建立 v8 后继补丁。
