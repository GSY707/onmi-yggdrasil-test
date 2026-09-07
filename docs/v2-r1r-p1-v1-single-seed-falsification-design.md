# V2-R1R P1 v1：单种子跨任务可否证实验

状态：**冻结，授权执行；P2 明确未授权。**

本合同把已经通过的 P0-D v17 production data/verifier 与 P0-M v5 training-path smoke 接到第一次真正的架构行为检验。P1 的问题不是“模型能否在一个小 selection 上过拟合”，而是：一个没有任务分支、没有 oracle span、只接收完整 Qwen token hidden 的共享 Boundary/recurrent core，能否在 fresh ERE/CPS 数据上同时学到 heldout 组合、长程计算和因果依赖。

P1 是单 seed falsification，不是稳定性结论，也不是 matched Pareto。任何正式 Gate 失败都保留当前 root、停止后续阶段并回到主设计层分析；不得在同一合同内换 seed、调阈值、增大模型或加入任务专用结构。

## 1. 前提与证据边界

执行前必须重新验证以下 sealed PASS：

- P0-D v17：`artifacts/v2-r1r/p0d-v17-full-production-20260810-1/`；
- P0-M v5：`artifacts/v2-r1r/p0m-v5-assessment-20260810-1/`。

P0-D 证明任务、teacher、claim 与 verifier 可用；P0-M 只证明 cache、loss、共享 K=8、小样本 baselines 与 GPU 热路径能运行。它们都不构成 P1 的行为证据。P1 只允许复用 generator/verifier 实现和已经证明可训练的机制，不复用 P0-M selection、模型参数、optimizer state 或数据顺序。

## 2. 冻结随机性与正式 roots

| 项目 | 固定值 |
| --- | --- |
| contract | `r1r-p1-v1` |
| P1 generator seed | `2026081901` |
| K=8 model seed | `2026081911` |
| K=1 model seed | `2026081912` |
| direct model seed | `2026081913` |
| text-CoT model seed | `2026081914` |
| shared data-order seed | `2026081921` |

正式 roots 固定为：

```text
artifacts/v2-r1r/p1-v1-data-20260810-1
artifacts/v2-r1r/p1-v1-cache-20260810-1
artifacts/v2-r1r/p1-v1-k8-20260810-1
artifacts/v2-r1r/p1-v1-k1-20260810-1
artifacts/v2-r1r/p1-v1-direct-20260810-1
artifacts/v2-r1r/p1-v1-text-cot-20260810-1
artifacts/v2-r1r/p1-v1-assessment-20260810-1
```

每个 root 单次使用、拒绝覆盖。失败 root 与 source snapshot 必须保留。预测试只允许写入 `artifacts/v2-r1r/p1-v1-preflight-20260810-1/`，不得提升为正式证据。

## 3. Fresh 数据合同

继续使用 accepted production generator `r1r-p0d-v17-visible-domain-alpha-repair`，但用新的 seed 和新的规模全量生成：每族 train `8192`，validation `1024`，每个 OOD/causal split `1024`。两族合计 `28672` records，其中 causal records `2048`、causal pairs `1024`。

ERE splits 为 `train/validation/composition_ood/length_ood/entity_ood/language_ood/causal_pairs`；CPS splits 为 `train/validation/composition_ood/horizon_ood/distractor_ood/language_ood/causal_pairs`。生成器继续引用 v17 production design hash；本文件定义 P1 对该 accepted generator 的 fresh 实例化，而不伪造一个新的 generator 版本。

数据正式阶段完整运行 v17 G01–G10，并追加 G11：

1. 同 seed 在临时目录全量再生成，逐文件 bytes 相同；
2. 对第一份 artifact 只读重跑 audit，报告 canonical bytes 相同；
3. 生成前后所有 watched inputs hash 不变。

只有 G01–G11 全部通过且 evidence seal 可复算，才允许建 cache。

## 4. 模型与信息通路

四条路径固定使用 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`。latent 路径冻结 Qwen 并缓存 final language-model hidden，Qwen trainable parameters 为零。

K=1/K=8 除 slot 数外完全相同：tokenwise non-affine `LayerNorm(2048) -> Linear(2048,512) -> RMSNorm(512)` Boundary；两层、八头、SwiGLU-2048 的共享 recurrent block；最大公开预算 `T=24`；所有 step 与 ERE/CPS 共享参数；答案只由 `H_T` 的 learned-query pooling 和九类线性头产生。

禁止 task embedding、operator/candidate embedding、task-specific parser/transition/readout、oracle span/role/candidate mask、AST、teacher state、input ids 或答案进入 latent forward。`valid_choice_mask` 与 public reasoning budget 是唯一非 hidden 控制输入。

## 5. Cache 与 8 GiB 工程合同

正式 cache 先对最长的 12 条 source 实测 batch `4/8/12`。每个 batch 要记录 throughput、峰值显存、finite/OOM；在所有 finite、无 OOM 且峰值不超过 `6.5 GiB` 的候选中选择 examples/s 最高者。没有候选则 C01 失败。

source 覆盖全部正式 split；claim 只缓存 train claims。两者都使用相同 frozen Qwen、`float16` final hidden、无 truncation。按 token length 排序并写为 packed mmap shards；每条索引只含 id、shard、offset、length 与内容 hash，不保存 input ids、answer、label、mask、AST、teacher、span 或 pointer。单 shard 上限 `131072` token。

写入前必须精确 tokenization，记录预计 payload bytes、当前磁盘余量和 `1.25 * payload + 30 GiB` 安全需求。正式 cache DataLoader 使用 pinned memory、两个 persistent workers、prefetch factor 4 与 non-blocking H2D；不得把全量 cache preload 到 GPU/RAM。

C01 同时要求 exact revision/layer/dtype、无禁字段、全量 id/length/hash 对应、mmap 可复载、长度排序、容量资格、benchmark 决策和 evidence seal 全部通过。

## 6. Latent 训练合同

K=8 是首个架构 Gate。K=8 未通过前不得运行 K=1 或文本基线。

- joint batch `16`，严格为 ERE 8 + CPS 8；
- `2400` optimizer updates，update 1–2400 全部完成，不用 Gate early stop；
- data-order seed `2026081921`，每族独立 permutation 后循环，四条路径的 episode exposure ledger 必须一致；
- 每样本每 step 选择两组表面匹配真假 claim pair；
- AdamW，Boundary LR `1e-4`，core LR `2e-4`，readout/probe LR `3e-4`，weight decay `0.01`，warmup `5%`，cosine decay，global clip `1.0`，BF16 autocast；
- 每 100 update 完整评测两个 validation；1200 前不得选择为正式 checkpoint；
- checkpoint 以 `min(ERE validation, CPS validation)` 最大化，依次以两者平均值、更早 update 破同分。

### 6.1 监督时序的设计层修订

原高层草案在 update 1801 后关闭 claim，并要求答案阶段 checkpoint。P0-M v4→v5 已经实证该退火会撤掉维持共享机制所需的梯度，使 claim 因果依赖退化而 answer overfit 可暂时维持。P1 v1 因此显式覆盖旧条款：

| updates | objective |
| ---: | --- |
| 1–400 | `0.25 L_answer + 1.0 L_claim` |
| 401–2400 | `1.0 L_answer + 0.5 L_claim` |

`L_claim` 固定为真假 CE、pair ranking 与同族 owner contrast。claim 不进入 core 输入或答案头；选中 checkpoint 后物理删除 claim probe，严格 reload，并以逐 split prediction hash/accuracy delta 验证推理独立性。训练期机制约束可以伴随架构生命周期，部署期信息旁路不能存在。

### 6.2 预注册梯度冲突分支

update `20,40,...,200` 分别计算 ERE/CPS 当前目标在共享 Boundary+core 上的 gradient cosine。update 200 时，仅当十个 cosine 的 median `< -0.15` 且该次两个 validation accuracy 差距 `>0.05`，才从 update 201 启用 core-only symmetric PCGrad；Boundary、readout与 probe 仍使用等权联合梯度。触发条件、全部 cosine、validation gap、启用状态与投影前后 norm 必须入 artifact。未触发不得启用。

## 7. K=8 正式评测与 Gate

正式 checkpoint 在 probe 剥离 reload 后评测。accuracy 按 record；causal flip accuracy 按 `pair_id`，只有 pair 两端都预测正确且预测标签确实不同才算正确 pair。

主 OOD 是 ERE 的 `composition_ood/length_ood/entity_ood/language_ood` 与 CPS 的 `composition_ood/horizon_ood/distractor_ood/language_ood`。因果干预主分数预注册为 `mean(ERE length_ood accuracy, CPS horizon_ood accuracy)`。

| Gate | 判定 |
| --- | --- |
| K01 | ERE、CPS validation 各 `>=0.85` |
| K02 | 八个主 OOD cell 各 `>=0.75` |
| K03 | ERE、CPS causal pair flip accuracy 各 `>=0.80` |
| K04 | `zero_latent@middle` 相对正常的因果干预主分数下降 `>=0.40` |
| K05 | `batch_shuffle_latent@middle` 相对正常的因果干预主分数下降 `>=0.40`；shuffle 只在相同 choice-mask 的样本间确定性轮转 |
| K06 | `T=1` 相对 full T 在 ERE length OOD、CPS horizon OOD 分别下降 `>=0.15` |
| K07 | probe 物理删除严格 reload 后，所有正式 cell 的最大 accuracy delta `<=0.01`，且正常预测 hash 完全一致 |
| K08 | source-to-answer 无 bypass、cache 无 answer leak、state dict/模块签名无任务专用参数 |
| K09 | zero、shuffle、slot permutation、T=1/2/4/full、wrong-definition、source-token-shuffle、aux-stripped、hook/gradient audit 全部产出可复算报告 |

`middle` 对每条样本定义为 `max(1, floor(reasoning_budget/2))`，干预发生在该次 recurrent update 之后。slot permutation 是无阈值完整性诊断并报告 prediction delta；source-token-shuffle 是破坏性诊断，不设语义 Gate。wrong-definition 使用 causal pair 对端 source hidden 替换当前 source，保持当前 answer target，以量化指定定义变化的响应。

K01–K09 任一失败即 `FAIL_P1_K8`，封存后停止；不得运行 K=1、direct、text-CoT 或 P2。

## 8. K=1 与 matched-training 文本对照

仅 K=8 通过后运行。K=1 使用相同数据、episode ledger、2400 updates、loss、optimizer、validation selection 与完整评测；它是容量曲线，不要求失败，也不参与本轮 K=8 Gate 降级。

direct 与 text-CoT 使用同一 Qwen revision、顶部四层 attention/MLP projection 的 LoRA rank 8、alpha 16、dropout 0。两者使用同一 episode ledger、2400 optimizer updates 和相同 accumulated episode count；8 GiB 下允许 microbatch 1/2 资格测试，但 effective batch 固定为 16。direct target 仅 `Answer: <label>`；text-CoT target 是同一 simulator teacher trace 后接同一答案。prompt/target token 限制、official chat template、thinking off、无 truncation 均为硬合同。

P1 只报告 K=1/direct/text-CoT 的 validation、OOD、causal、trainable parameters、teacher tokens、GPU seconds、peak VRAM 与 exposure equality；不在 P1 做在线 Pareto 判决。对照路径运行错误、episode ledger 不同或 seal 无效会使 P1 不完整，但其质量高低不是 K=8 阈值。

## 9. 阶段顺序与最终判定

固定顺序为：prerequisite/contract preflight → fresh data G01–G11 → cache C01 → K=8 K01–K09 → K=1 → direct → text-CoT → P1 assessment。每一步只读取前序 sealed PASS；失败后立即停止。

最终 `PASS_P1_SINGLE_SEED` 要求 data、cache、K=8 全部 Gate 通过，K=1/direct/text-CoT 完整、公平 ledger 一致，所有 roots seal 可复算。它只授权另立 P2 合同；本合同不运行 benchmark-online、不作 Pareto、不启动 P3/OPS/A1.22A/V2-B/V2-C。

