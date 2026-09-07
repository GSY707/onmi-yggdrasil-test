# V2-R1R P1-H1-WD 非正式机制失败复盘

日期：2026-08-21  
机器状态：`FAIL_H1_WD_NONFORMAL_MECHANISM`  
证据边界：`NONFORMAL_WD_MECHANISM_SCREEN_ONLY`  
授权：`nothing`

## 1. 核心判决

“先写后删”在表示层面成立，在能力分工层面失败。

本轮确实构造并训练了一个不含 family/task 分类 target 的残差数据集：先把 predecessor common 输出中与 selected projection 正向重合的分量写入 projection，再从 common 中删除同一分量。W/D 的低误差与自由 rollout 保持证明，这种输出空间重参数化是可实现的，而且没有把 projection 变成单独模型。

但重参数化没有让 shared FFN 与 routed projection 成为各自不可替代的互补路径。最终关闭任一路径都只损失 `0.01172` accuracy；projection 的关闭效应相对 predecessor 仅增加 `0.00391`。同预算 continuation 下，WD 也只比 control 高 `0.00488`，置信区间跨零。因此不能把被搬运的欧氏重合分量解释为“已转移的任务能力”，更不能据此证明 routed architecture 有效。

## 2. 运行完整性

唯一命令正常退出，耗时 `2484.63` 秒；无 `crash.json`。固定 output root、package identity、record order 与 cache audit 均通过。W/D/J 分别完成 `800/800/1200` updates，三个阶段均记录 `uses_family_targets=false`。

predecessor checkpoint 运行前后 SHA-256 均为：

```text
112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D
```

机器真源为：

```text
artifacts/v2-r1r/p1-h1-wd-overlap-residual-screen-20260821-1/result.json
```

## 3. 哪些部分成功

### 3.1 W：重合分量可以写入 projection

heldout write nMSE 为 `0.00196195`，远低于 `0.08` Gate。平均系数为 `0.46022`，正系数样本占 `0.99847`；被写入分量对应约 `0.25070` 的 common 输出能量，写入后 common residual energy 为 `0.71920`。

这说明 projection 的参数确实发生了足以重建目标分量的变化，失败不是“projection 完全没学到”。

### 3.2 D：可以重建公共残差并保持原函数

D heldout common nMSE 为 `0.0139281`，完整 transition nMSE 为 `0.00315984`，common residual energy 为 `0.71348`。自由 rollout prediction agreement 为 `0.99121`，trajectory/final-state relative MSE 为 `0.00175/0.00203`。

因此 `C_D + P_W ≈ C_0 + P_0` 不只在训练 batch 上成立，也在 predecessor heldout rollout 上近似成立。common 仍保留约七成能量，模型没有退化成 projection-only。

### 3.3 J：训练稳定且 matched control 公平

WD 与 untouched predecessor control 使用完全相同的 1,200-update answer-only schedule；两臂最终 checkpoint 都正常生成。WD 最终 free-rollout prediction agreement 为 `0.98047`，heldout accuracy 为 `0.54785`，没有发生能力崩塌。

## 4. Gate 为什么失败

### 4.1 搬运的是向量重合，不是决策关键能力

W 的系数使用普通隐藏空间内积。它回答“common 更新中有多少与当前 projection 同方向”，却不回答“哪一部分会改变答案、哪一部分对当前 route 特有”。recurrent state 存在大量可替代坐标与下游冗余；在欧氏度量下能量较大的方向，可能对最终 decision margin 很小。

本轮最直接的证据是：约四分之一 common 能量被成功写入，但最终 projection-off 只从 predecessor 的 `0.00781` 增至 `0.01172`。能量搬运量与任务必要性没有同步增长。

### 4.2 残差重建目标只要求总和守恒，不要求互补

`C_D + P_W ≈ C_0 + P_0` 对 combined transition 施加约束，却没有规定信息必须如何在两条路径间分配。只要下游能从任一路径的近似冗余信号恢复同一决策，关闭 common 或 projection 的单独效应就可以都很小。

最终结果正是这种可替代性：

| 干预 | WD accuracy drop |
| --- | ---: |
| disable shared FFN | `0.01172` |
| disable routed projection | `0.01172` |
| disable both FFN paths | `0.03516` |
| flip route | `0.00879` |

两条路径合关比单关更差，说明它们合计有用；但单关效应均未达到 `0.02`，说明尚未形成各自必要的互补分工。

### 4.3 J 的目标保持函数，没有制造分工压力

J 的 answer CE 与 predecessor-logit KL 对 WD/control 对称，并且都奖励保持原函数。它能修复 rollout 漂移，却没有 branch-specific counterfactual、route-sensitive margin 或协同必要性目标来阻止两条路径重新变得可替代。

J 后 final allocation common nMSE 从 D 时的 `0.01393` 增至 `0.12829`，完整 transition nMSE 从 `0.00316` 增至 `0.03173`；虽然仍通过预登记 consistency 上限，但说明 answer-only 联合修正部分松动了精确分配。与此同时 common residual energy 仍为 `0.71585`，并没有出现“common 被删空”的错误。

### 4.4 没有产生可区分的架构收益

最终 WD/control heldout accuracy 为 `0.54785/0.54297`。1024 个 paired records 中，WD-only correct 为 `9`，control-only correct 为 `4`，净增益只有 `5/1024 = 0.00488`；bootstrap 95% CI 为 `[-0.00195,0.01172]`，既未达到 `+0.05` Gate，也不能排除零收益。

这说明 reparameterization 本身主要改变了内部坐标，没有打开 control 无法达到的新学习路径。

## 5. 被否决与仍保留的假设

本轮否决：

- 只要 projection 吸收 common 中与它同方向的约四分之一输出能量，就会显著增强 projection 的正常路径必要性；
- `common - projection` 的欧氏残差数据集可以直接代表能力差；
- 函数保持后的 answer-only 联合修正会自然形成 routed specialization。

本轮保留：

- 不使用 family/task target，也可以由模型自身 route 构造稳定的 write/delete transition dataset；
- 在保持单一 shared+routed 模型的前提下，projection 写入和 common 残差化可以低误差完成；
- “先写后删”可作为参数重分配原语，但后继若要研究能力转移，必须把 target 从几何重合升级为决策因果贡献。

## 6. 后继设计约束

该 root 与 identity 已消耗，不得重跑、换 seed、调 coefficient、降低 Gate 或启动 H1 formal/F1/P2。若另立新合同，至少需要同时改变以下原理，而不是继续放大同一欧氏写入量：

1. 用 answer margin、logit Jacobian/Fisher 或真实 counterfactual drop 定义“可转移分量”，不再用普通隐藏空间余弦代替能力；
2. 在始终保留 shared+routed 单模型的前提下，加入双路径互补约束或训练期部分衰减，使正常路径必须联合消费两支，而不是允许任一支近似替代另一支；
3. 用能暴露 shared-gradient interference 的 fresh 数据验证 routed branch 是否带来 matched-control 不能获得的学习收益；
4. router 必须在 fresh、无人工 family label 的训练中形成，并以 route swap/flip 的决策损失验证，而不是继承历史 label-trained router 后声称真实文本资格。

这些是新实验的设计条件，不是本次 FAIL 对后继训练的自动授权。
