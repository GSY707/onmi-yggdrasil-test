# V2-R1R P0-D v15：G09 判别资格化与新 seed production

日期：2026-08-10

合同版本：`r1r-p0d-v15-g09-qualified-production`

## 1. 核心判定与证据边界

v14 的完整 production formal 只在 G09 失败：两个 OOD char-NB cell 的 point excess 约为 `+0.035/+0.042`，但 512 条样本下的 98-way Bonferroni-Wilson upper 略高于 `mean_random + 0.10`；14 个 family 聚合全部通过。这个结果既不能事后放行，也不足以区分局部弱信号与没有预先匹配样本量的判别规则。

v15 因此包含两个不可倒置的硬阶段：

1. **Q：G09 decision-power qualification**。只用预注册的合成 null/局部/扩散 leak、v14 sealed 数据的只读评分样本和故障矩阵，证明 G09 规则有已知判别能力，并证明快速评分器不改变 accepted exact semantics。
2. **F：fresh-seed production formal**。只有 Q 的唯一 sealed assessment 为 PASS，才使用未预览的新 root seed 生成并验收完整数据。

Q 不重新解释或接受 v14，F 不允许根据新 seed 的结果修改统计规则。两个阶段都只资格化数据与测量；不生成 Qwen hidden cache，不运行模型、GPU 或训练，不判断 recurrent core、Boundary 或完整 V2 架构成立。F PASS 也只允许父任务另立 P0-M 合同。

## 2. Q：合成判别资格化

### 2.1 冻结决策规则

G09 保持 v14 的双层 conjunction，不修改 effect ceiling：

- 98 个 `family × split × baseline` cell 使用 one-sided Wilson upper，Bonferroni `alpha=0.05/98`；point accuracy 和 upper 都必须 `<= mean_random + 0.10`。
- 14 个 `family × baseline` 聚合合并六个 heldout，使用 one-sided Wilson upper，Bonferroni `alpha=0.05/14`；point accuracy 和 upper 都必须 `<= pooled mean_random + 0.05`。
- 缺 cell、重复 cell、非法 success/total/random、错误 split/baseline、预测不完整或非布尔判定全部 fail closed。

合成 qualification 固定 NumPy `PCG64`、seed `2026081501`、100,000 个 Monte Carlo trial。每个 trial 同时生成完整 98-cell 矩阵；train `n=4096`，其余 cell `n=1536`。随机基线使用冻结的 ERE/CPS mask profile；不读取 v14 的 observed success 作为合成概率，也不搜索能让 v14 通过的阈值。

每个场景的经验成功率同时报告 one-sided 95% Wilson lower。Gate 使用 lower 而不是点估计：

| 场景 | 预注册期望 | lower Gate |
| --- | --- | ---: |
| 全 98-cell null：`p=mean_random` | G09 接受 | `>=0.995` |
| ERE composition char 单格 `p=mean_random+0.04` | G09 接受 | `>=0.950` |
| CPS horizon char 单格 `p=mean_random+0.04` | G09 接受 | `>=0.950` |
| 上述任一单格 `p=mean_random+0.10` | G09 拒绝 | `>=0.995` |
| ERE/CPS 各自一个 char baseline 在六个 heldout 全部 `+0.05` | G09 拒绝 | `>=0.995` |

`+0.04` 是 v14 局部信号量级的允许级工程 profile，不表示真实 shortcut 被证明为零；`+0.10/+0.05` 分别对应 cell/family ceiling 的边界挑战。qualification 证明的是冻结决策器在这些 profile 下的操作特性，不是生产数据的独立同分布理论。

### 2.2 快速 exact-equivalent scorer

v15 允许把每次预测都构造 prime-factor expression 的旧路径替换为有界 log fast path，但不得改变 analyzer、Laplace smoothing、class prior、mask、label order tie-break、5-fold group 划分或 train-fit-heldout 语义。

fast path 必须为每个候选计算 score 与保守误差界；只有一个候选的下界严格高于其他候选上界时才直接返回。区间重叠时必须回退到 v12 accepted prime-expression exact comparison。qualification 必须同时满足：

- v14 sealed 数据上固定抽样的 ERE/CPS、三种 NB、至少 1,152 个 prediction 与 accepted dense fit + compact exact scorer 全部逐项一致；
- 至少 2,048 个确定性合成 sparse-model prediction 与 legacy exact sparse scorer 全部一致；
- 至少 64 个 exact tie/near-tie 正控全部一致且确实触发 exact fallback；
- 固定 char-NB batch 上 prediction identity 为 100%，fast 总耗时 speedup `>=5.0x`，普通数据 exact fallback rate `<=1%`；
- production audit 同一时刻只持有一个 analyzer feature bank，并按 family/analyzer 输出进度事件。

性能 Gate 只防止再次出现 v14 的评分实现瓶颈，不改变统计答案；若速度不够但结果正确，Q 仍然失败，不能带着未资格化路径进入 F。

### 2.3 Q Gate 与唯一 artifact

Q 的唯一 root 固定为：

`artifacts/v2-r1r/p0d-v15-g09-qualification-20260810-1/`

| Gate | 判定 |
| --- | --- |
| Q01 | 决策器 schema、98/14-cell 拓扑、Wilson/Bonferroni 与 production G09 调用同一实现 |
| Q02 | null accept lower 达标 |
| Q03 | 两个允许级 local `+0.04` accept lower 达标 |
| Q04 | 两个禁止级 local `+0.10` reject lower 达标 |
| Q05 | 两个 diffuse `+0.05` reject lower 达标 |
| Q06 | 缺失/重复/非法计数、阈值或比较数突变等故障矩阵逐项被拒绝 |
| Q07 | registry/synthetic/tie 三组 scorer prediction identity 全通过，fallback 正控有效 |
| Q08 | 固定 benchmark speedup、fallback rate 与 single-bank/progress 条件全通过 |
| Q09 | v14 输入只读、qualification replay、source snapshot、assessment 与 evidence seal 一致 |

机器状态只能是 `PASS_G09_QUALIFICATION` 或 `FAIL_G09_QUALIFICATION`。root 必须 single-use；Q 一旦失败、异常、中断或 artifact 不完整，立即停止，不得创建 F root。

## 3. F：fresh-seed production formal

### 3.1 数据规模与 seed

Q PASS 后，F 固定 root seed `2026081502`，不得预生成、预览或用其调参。每个 family 固定：train 4,096；validation、composition OOD、length/horizon OOD、entity/distractor OOD、language OOD、causal_pairs 各 1,536。每族 13,312、总计 26,624 records；causal_pairs 为每族 768 对、总计 1,536 对。

唯一 root 固定为：

`artifacts/v2-r1r/p0d-v15-full-production-20260810-1/`

generator 的任务域、renderer/parser、schema/model-view、simulator、claim、fingerprint、permutation、tokenizer 和 1,024-token 上限全部继承 v14；允许改变版本、root seed、heldout 配额及已由 Q 资格化的评分执行路径。4,096/1,536 容量预测试还必须在正式 seed 未生成前证明全局 alpha-invariant uniqueness：ERE 仍固定三个 attribute、六个 value 和同一 query/transition 分布，但将既有二值 claim witness 与四值第三属性 distractor 在 event-role entity 间独立排列。该排列完整渲染、进入 simulator/fingerprint，且不改变答案，是合法状态结构容量而非隐藏 nonce salt。不得针对 v14 两格修改文本模板、标签日程、答案映射或 query/pattern 配额。

### 3.2 F Gate

| Gate | 正式判定 |
| --- | --- |
| G01 | v7/v9/v10/v11/v12/v13 accepted seal、v15 Q PASS seal 与声明只读输入 byte identity 通过 |
| G02 | 14 文件、26,624 条、4,096/1,536 配额、schema/version/model-view、root-to-record provenance 全部精确 |
| G03 | 26,624/26,624 source roundtrip、fresh simulator、answer/label、teacher trace 与全部 claim replay 一致 |
| G04 | 全局 semantic/surface overlap 0；1,536 causal pair 单叶、共享表面因素与答案翻转 100% |
| G05 | ERE primitive/query/depth/composition/entity/length 与 3-attribute/6-value 生成域独立派生 |
| G06 | CPS unique optimum/NONE、suboptimal、hard negative、composition/horizon/distractor 配额独立派生 |
| G07 | 每 split 标签计数差 `<=1`；choice/candidate/action/definition 与 alpha/permutation 正控通过 |
| G08 | language template 分离、marker 禁止、pinned tokenizer 无截断，train/validation p99 `<=900` |
| G09 | 同一 qualified 98/14-cell decision 全部通过；预测完整、fast/exact qualification 引用有效 |
| G10 | claim truth、kind/pair 极性平衡与 24-cell simultaneous claim upper 全部通过 |
| G11 | full regeneration、artifact read-only replay、上游只读、source snapshot、进度 ledger 与 evidence seal 一致 |

机器状态只能是 `PASS_P0D_PRODUCTION` 或 `FAIL_P0D_PRODUCTION`。任一 Gate false、超时、异常或中断都使唯一 F attempt 结束；不得修改或补写 formal root，不得换 seed/suffix 重跑，也不得把局部通过升级为 P0-D PASS。

## 4. 执行与停止顺序

1. 单元、等价、故障和小规模性能测试可在不使用 F seed 的临时输入上反复运行；
2. 只调用一次固定 Q 命令并封印 Q root；
3. Q FAIL 时停止；Q PASS 后直接删除 v14 活动 tests/CLI，验证 v15 contract guard；
4. 不预览 F seed，只调用一次固定 F 命令；
5. F 返回后无论 PASS/FAIL 均停止写 formal artifact，父任务只读复算 seal、关键 Gate 与未知反例，再写主审和索引。

本合同不授权 P0-M、cache、model、optimizer、GPU、训练、Pareto、A1.22A、V2-B 或 integrated-system 表述。
