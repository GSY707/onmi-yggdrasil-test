# V2-R1R P1 v8R causal curriculum 失败复核

日期：2026-08-11

## 1. 正式判决

P1 v8R 的 preflight 保持 sealed `PASS_P1_V8R_CAUSAL_CURRICULUM_PREFLIGHT`，qualification 保持 sealed `FAIL_P1_V8R_CAUSAL_CURRICULUM`。两根分别为：

- `artifacts/v2-r1r/p1-v8r-causal-curriculum-preflight-20260811-1`，seal `6891D79C09D147F94D71726B6B56B9501A04D7AE50C1263BB21378CF5026690D`；
- `artifacts/v2-r1r/p1-v8r-causal-curriculum-qualification-20260811-1`，seal `8060854982CB4370A12D0BC26CED28AC8D2C69E8925EF9D71D44402848C0EDE4`。

唯一 formal 没有重启或补跑，两个 seal 均可复算，后继 P1 v9、K=1、direct、text-CoT、assessment 与 P2 均未创建。`fresh_p1_v9_authorized=false`、`p1_completed=false`、`p2_eligible=false` 保持不变。

V7 zero-update baseline 的 ordinary validation ERE/CPS 为 `0.999023/0.851562`。`replay_pair` 最终为：

- causal optimization ERE/CPS pair flip：`0.994792/0.328125`；
- causal audit ERE/CPS pair flip：`0.804688/0.0078125`；
- ordinary validation ERE/CPS：`0.999023/0.748047`。

因此 R01/R02 通过，R03–R06 失败。ERE 的反事实迁移继续成立；CPS 只在参与梯度的 pair 上出现有限改善，未迁移到隔离 audit，并损失约 `0.103516` 的普通 CPS validation。

## 2. 已排除的解释

### 2.1 不是 scratch competence collapse

两个 arm 都从同一个 sealed V7 checkpoint 开始，初始 state hash、mixed schedule、update 和 exposure 完全相同。V7 baseline 已具备普通任务能力，所以 v8 的随机初始化、小覆盖和无 rehearsal 混淆已被排除。

### 2.2 不是 causal optimization 与 audit 的任务类型漂移

只读重放 512 个 CPS pair 证明 optimization 的 384 对与 audit 的 128 对全部属于同一 `cost | resource_final_budget` family。每对的 candidate 顺序、label mapping、valid-choice mask 均相同，base/flip 只提高一项 action cost，且答案确实翻转。audit 失败不能归因于未见 mutation family 或输出重排。

### 2.3 不是单纯 readout 容量不足

冻结 V7 Boundary/core 后，将旧单查询 pooled readout 与 9 个输出地址各自查询 K slots 的 addressable readout 做函数等价初始化；二者用完全相同的 ordinary + causal schedule、3,072 updates 和 coupled switch loss 训练。结果：

- shared readout：optimization pair `0.026042`，audit pair `0.0`；
- addressable readout：optimization pair `0.028646`，audit pair `0.0`；
- 两者 ordinary validation 都约 `0.835`。

另一个冻结表示线性 probe 即使能把 `replay_pair` optimization local-label pair 拟合到 `0.757812`，audit 仍只有 `0.0078125`；改为预测稳定的 semantic candidate，audit 也只有 `0.015625`。这些是固定 seed 的 post-stop development diagnostics，不是 formal architecture Gate，但共同排除了“只换最后一层即可恢复”的主要解释。

## 3. 真实根因

### 3.1 当前 pair loss 仍是可分解的 hard-negative CE

v8R 的 `pair_ranking_loss` 分别要求 base 样本偏好 base label、flip 样本偏好 flip label。它虽然提高了两个目标 label 的 margin，但数学上仍可分解为逐样本分类；没有要求同一个 predicate 在两个 source-conditioned states 上发生可复用的真值翻转，也没有监督 CPS 的跨候选最终比较。

这与结果一致：audit 128 对中，`replay_pair` 有 69 对把 base 答案原样复制给 flip，只有 1 对完成正确切换。训练 pair 上的提升主要是 episode-specific 适配，不是 cost mutation 算法。

### 3.2 temporal state 有局部过程语义，但缺 final reduction closure

V7 已证明 CPS temporal witness accuracy `0.938965`，说明匿名 workspace 能承载候选执行、事实、资源和累计 cost 的局部过程信息；然而旧 witness 没有监督“候选 A 比 B 更便宜”“候选 A 是唯一最优”“局部输出标签 X 正确”这些最终归约命题。普通 answer CE 因而可以从 `H_T` 的表面统计建立另一条捷径。

v8R 又只在答案 logits 上加 margin，没有让 `H_T` 对同一个最终决策 predicate 随 counterfactual source 成对翻转。因此当前缺口不是更多相同 pair exposure，而是 **final-decision state closure**。

### 3.3 R06 的 source-gradient 失败是 BF16 测量假阳性

`replay_pair` 在 integrity audit 选取的 5 条 ERE 样本上全部高置信正确，BF16 CE 约 `5.96e-7`，source gradient 下溢为全零。对完全相同 checkpoint/batch 禁用 autocast 后，FP32 source gradient 有 `4,409,344` 个非零元素；CPS batch 在 BF16 与 FP32 下也都有显著非零梯度。因此原 R06 保留正式 false，但后继 Gate 必须用 FP32 或非饱和目标检查路径，不能把数值下溢解释为 source bypass。

## 4. 后继判决

停止继续调 pair weight、学习率、update、普通 replay 比例或单独替换 readout。下一步另立 P1 v8D causal-decision witness qualification：

1. 继续从 sealed V7 competent checkpoint 开始；
2. 对同一 causal pair 的 base/flip `H_T` 提出相同的、标签相反的 final-decision queries；
3. CPS query 覆盖局部输出标签、unique optimum 与 winner-to-runner-up cost comparison；ERE 使用同构的局部输出标签 query；
4. query 只进入训练期共享 probe，不进入 Boundary/core 或答案头；probe 在正式答案评估前物理删除；
5. 先判断 decision witness 能否在未参与梯度的 pair 上迁移并依赖 state，再判断答案 causal transfer 与 ordinary retention。

v8D 若连 decision witness 都不能迁移，应重新设计 causal state formation 或输入表示；若 witness 迁移而答案仍失败，才有资格测试 addressable latent-only readout。任何结果都不能直接完成 P1 或进入 P2。
