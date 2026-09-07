# V2-A Closure C1S：Addressed Content Workspace 提案

日期：2026-08-27

状态：**研究提案已完成对齐，并由 `V2-A-CLOSURE-C1S-ADDRESSED-WORKSPACE-20260828-1` 机制资格合同接替；本文仅保留设计来路，不授权训练或 formal。**

## 1. 实际目标

C1R 已证明当前 anonymous dense K=8 core 的 final answer path 会功能性退化为近似 K=1。下一步的目标不是让八个向量在欧氏空间里“看起来不同”，而是验证以下更强命题：

> full-token public text 能形成带地址和所有权的多实体内容状态；共享 recurrence 只对被选中的状态做因果更新；输入条件 query 能选择最终状态；K>1 相对 matched K=1 具有可复现的行为与因果必要性。

如果只增加 diversity loss、固定 slot bias 或更强 answer head，即使 non-collapse 指标变好，也不能回答这个命题。

## 2. 直接切换的状态代数

推荐架构名为 `Addressed Content Workspace`。它不兼容继承 C1 的 dense all-slot core，而是重新定义状态：

`S_t = ({a_k, h_k}_{k=1..K}, q_t)`

- `a_k`：slot 的地址/所有权侧车，只用于匹配、路由与置换等变性，不携带任务类别 embedding；
- `h_k`：连续内容 payload，保存实体、候选和中间状态；
- `q_t`：由 public source 产生的输入条件 query，不再使用跨所有样本共享的 global `answer_query`；
- Boundary 先形成粗 entity/operation 候选，再用 A1.20D 式双向 section consistency 细化 entity table、operation boundary 与 query binding；
- recurrent transition 使用地址相似度选择 source/target，并以 gated residual 只更新相关 payload；禁止无条件 dense all-slot mixing；
- answer 由 `q_T` 对地址/内容做条件选择，不能退化成固定全局均值；
- 训练期 state/closure 辅助允许读取 simulator 产生的 CT1 target，但 family、route、AST、teacher hidden 和答案 metadata 不进入模型 forward；部署前辅助头物理剥离。

这里复用 A1.19H 的机制原则，不复用其 exact-symbolic oracle handle 接口。地址必须从 public text 的 Boundary 中产生；否则只是在 C1 前重新加入 oracle parser。

## 3. 必要对照

新方案必须与 `K=1 temporal state` 同时存在。K=1 使用相同 source cache、Boundary 参数预算、transition 次数、训练 examples 和 GPU-hour；只删除多地址集合与 slot selection。

若 addressed K=8 的 heldout 行为、query swap、same-value/different-entity、slot deletion 和 recurrence causality不优于 K=1，则多槽架构主张失败，即使 K=8 的 slot cosine 很低也不能通过。

## 4. 建议的分阶段合同

### S0：零训练结构与测量资格

先验证前向只读 public hidden、地址/内容成对 permutation invariance、无固定任务 slot、无 family/route/AST 输入、辅助头可物理剥离，以及 functional-K 测量对 oracle positive control 和 K=1 null 的区分力。任何测量无法区分正负控时停止，不训练。

### S1：Overfit32 机制资格

fresh seed 在 32 条记录上不仅要求 answer exact，还要求：

- 每一步 address ownership 与 state target 可恢复；
- query swap 跟随被查询实体；
- same-value/different-entity 保持地址差异；
- disable recurrence、wrong start 和 target-address shuffle 显著破坏正确行为；
- slot permutation 只改变物理布局，不改变语义输出；
- leave-one-relevant-slot-out 必须改变正确 logit/answer，删除无关 slot 不得产生同等效应。

S1 失败时不得进入全量训练，也不得靠调 non-collapse loss 绕过。

### S2：受限 discovery 与 K=1 matched control

使用预注册且不触碰 formal validation 的 discovery split，同时训练 addressed K=8 与 matched K=1。主判据是 heldout answer、query ownership、state causality和 functional effective K；raw cosine、centered energy与谱只作解释指标。

只有 addressed K=8 在两族都超过 K=1，并通过全部因果 Gate，才允许冻结单 seed formal 合同。若两者同样失败，优先归因 Boundary/任务表示；若 K=1 通过而 K=8 失败，归因多槽绑定/transition；若 K=8 通过而 K=1 失败，才形成多槽必要性的候选证据。

### S3：single-use formal

新 formal 必须使用全新 identity/root/lease/source snapshot。顺序仍为 G004 first：answer behavior 未通过就停止，不能先训练 trace probe。单 seed 全 Gate 通过也只允许 fresh-seed replication，不直接授权 C2、V2-A PASS、V2-B 或 V2-C。

## 5. Gate 口径

non-collapse 不能只用 raw cosine。至少同时报告：

- common-mode energy 与 slot-centered/whitened residual ratio；
- functional leave-one-slot-out、mean-replace、duplicate-slot 与 relevant/irrelevant deletion；
- query/address ownership swap；
- input-conditioned selection entropy 与 matched uniform/K=1 null；
- H0–HT 的 state target、有效秩和跨 split 稳定性；
- addressed K=8 相对 equal-example 与 equal-GPU-hour K=1 的增量。

阈值应先由独立 positive control、K=1 null 和当前 C1R negative control 冻结，不能从 successor heldout 反推。当前 C1R 的 `.999958` cosine 或 `0.140` centered energy只能作为 negative reference，不应直接变成任意绝对阈值。

## 6. 当前决策点

若用户接受本方向，下一轮应先写冻结 S0/S1/S2 合同与测量实现，并只运行零训练 preflight；不直接启动 full training。若不接受显式地址/内容状态代数，则应回到 V2-A 的问题定义，先验证任务是否真的要求 K>1，而不是继续修补 C1 dense core。
