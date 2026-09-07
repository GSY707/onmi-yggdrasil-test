# V2-R1R P1 v7：Integrated K=8 fresh-seed 资格合同

## 1. 核心判断

P1 v6R 只恢复了一个窄结论：在不使用答案损失、teacher state 或跨 episode owner contrast 的条件下，因果时序见证能够迫使共享 latent state 承载可审计的过程信息。它没有证明同一模型能同时完成最终答案任务，也没有完成 P1。

本合同因此只回答一个新的必要问题：**把已通过的时序机制放回完整 K=8 latent reasoner 后，在新的模型种子与数据顺序下，模型能否同时获得答案泛化、因果依赖和时序状态依赖。** 若失败，失败本身是 integrated objective / capacity / optimization 的证据；不得回写为 v6R 机制失败。若通过，只授权另立 K=1、direct SFT、text-CoT SFT 的匹配控制合同；仍不授权 P2。

## 2. 固定证据边界

### 2.1 接受的上游事实

- P1 v2 数据集保持 sealed PASS。
- P1 v2 source cache 的正式状态保持 FAIL；P1 v3 recovery 仅授权对其做只读训练复用。
- P1 v6 witness catalog 保持 sealed PASS，训练部分全量覆盖 8192 ERE + 8192 CPS episode。
- P1 v6 原 assessment 保持 FAIL；P1 v6R 仅以 canonical JSON 等价关系恢复机制资格，且明确记录 `p1_completed=false`、`p2_eligible=false`。

上游 artifact 不得修改、重封或改判。v7 使用新的固定根；同名根一旦存在不得覆盖或补跑。

### 2.2 本轮不声称的内容

- 不声称白皮书整体成立，不声称 K=8 优于控制组。
- 不运行 K=1、direct SFT、text-CoT SFT，也不运行 P2。
- 不把训练期 temporal probe 当成部署架构的一部分。
- 不使用验证集选择 checkpoint；唯一候选是固定最终 update。

## 3. 任务与数据

答案任务继续使用 P1 v2 冻结的 ERE/CPS：每族 8192 个训练 episode，以及 validation、四类主 OOD、causal-pairs 六个评估 cell。

时序任务由冻结的因果见证构成。每个见证包含同一个自然语言谓词、相邻前缀 `(p,p+1)` 和一真一假的标签。查询文本没有 prefix 地址、真假标签或答案字段。

### 3.1 固定选择

- train：每个 16384 个训练 episode 选择 5 个唯一查询，共 81920 个见证。
- validation：每个 2048 个 validation episode 选择 2 个唯一查询，共 4096 个见证。
- 总查询数固定为 86016；query identity 是 `(example_id, query_text)`。
- 选择算法固定为 per-episode、按 witness kind 的 deterministic round-robin；同一查询若在多个迁移处翻转，只缓存一个由固定排序选出的迁移。
- 选择种子固定为 `2026082701`；预注册选择 preimage SHA-256 为 `8AFD3D04067FE0D52EC0CCA75866D7EBA0323966D20FD16C2F0E88AE5D38577F`。

query cache 必须使用与 source cache 相同、完全冻结的 Qwen3.5 text-only backbone 最终 hidden，float16、无截断、无可训练 Qwen 参数。cache 构建后逐 entry 检查 hash、finite、offset、shape 和全集 identity。

## 4. 模型切面

部署切面仍是通用 K-slot recurrent reasoner：

- source hidden 2048，经 tokenwise boundary 投影到 latent width 512；
- K=8 匿名 slot，固定 slot Fourier，不含 ERE/CPS/task/operator/candidate embedding；
- 两层共享 recurrent block，每次包含 source cross-attention、slot self-attention 和 SwiGLU；
- public 最大语义迁移数为 24；
- 最终答案只从 latent `trajectory[T]` 读出，不能直接读取 source width；
- 训练时附加可物理删除的 claim probe，probe 对 query embedding 与指定 latent state 做二分类；正式答案评估前必须删除并 strict reload。

### 4.1 T+1 状态语义

旧实现把 witness prefix 0 错位到了第一次 recurrent 更新后的状态。本轮直接切换到明确的 T+1 映射：

- runtime 第 0 次更新形成读入/初始化态 `trajectory[0]`；
- 第 `p` 个语义迁移之后的状态是 `trajectory[p]`；
- reasoning budget 为 T 的 episode 运行 T+1 次共享 block；
- temporal witness `(p,p+1)` 只读取 `trajectory[p]` 与 `trajectory[p+1]`；
- 答案只读取 `trajectory[T]`。

中间干预发生在语义迁移 `max(1,floor(T/2))` 之后，即对应的 `trajectory[m]`。T1/T2/T4 代表初始化态后分别保留 1/2/4 次语义迁移，而不是旧数组索引。

## 5. 固定训练路径

### 5.1 种子与预算

- model seed：`2026082717`
- data-order seed：`2026082721`
- batch：32，ERE/CPS 各 16
- 总 update：20480
- 总 episode exposure：655360，即训练集平均 40 次
- 前 4096 update：只优化 temporal paired objective
- 后 16384 update：答案 CE 与 temporal paired objective 联合优化
- 答案权重在联合阶段前 1024 update 从 0 线性升到 1
- temporal 权重：bootstrap 为 1.0，joint 为 0.5

每次 episode exposure 从其 5 个训练见证中按确定性轮换取 2 对。因此每个 batch 有 32 个答案监督和 128 个 temporal judgment；轮换不依赖损失或验证结果。

数据顺序使用完整的 40 个 epoch，每个 episode 恰好出现 40 次。每族先按冻结 source token 长度分桶、桶内固定种子打乱，再将相同长度分位的 ERE/CPS 半 batch 配成 16+16；完整 batch 顺序每个 epoch 再固定种子打乱。该调度不改变样本权重或任务比例，只减少 padding、随机 mmap 放大和 GPU 等待，并在 ledger 中记录 padding efficiency 与完整 schedule hash。

### 5.2 允许与禁止的目标

temporal loss 固定为 pair CE 加 pair rank loss。联合阶段的总损失只包含 answer CE 与该 temporal loss。

明确禁止：teacher state reconstruction、oracle span/role mask、显式三寄存器 scaffold、跨 episode owner contrast、static query pair、任务专属参数、PCGrad、验证集 checkpoint selection、训练后换 seed 或补跑。

训练会在四个固定 update 记录答案梯度与 temporal 梯度在共享参数上的 cosine/norm；这些只用于解释，不参与更新、调权或选择。

优化器为 AdamW；boundary/core/answer/probe 的初始学习率分别为 `1e-4/2e-4/3e-4/3e-4`，weight decay 0.01，5% warmup，cosine decay 到 10%，global gradient clip 1.0。周期性 `latest.pt` 只用于崩溃取证；正常完成后删除，正式 checkpoint 固定为 update 20480。

## 6. 正式评估与 Gate

训练期 probe 先在冻结 validation witness 上评估，然后从 checkpoint 物理删除。删除后的模型 strict reload，并运行全部 12 个答案 cell 及干预。unstripped/stripped 答案预测 hash 必须完全一致。

### 6.1 K=8 Gate

- K01 validation：ERE、CPS 各 `>=0.85`。
- K02 主 OOD：ERE 的 composition/length/entity/language 与 CPS 的 composition/horizon/distractor/language，八项各 `>=0.75`。
- K03 causal flip：ERE、CPS 各 `>=0.80`，且 pair 结构完整。
- K04 zero-middle：length/horizon 相对 full 的两族平均 drop `>=0.40`。
- K05 batch-shuffle-middle：两族平均 drop `>=0.40`。
- K06 T1：ERE length 与 CPS horizon 各自 drop `>=0.15`。
- K07 probe strip：删除 probe 后所有答案预测 hash 与删除前相同，最大 accuracy delta `<=0.01`。
- K08 architecture integrity：边界/答案头 shape、source-to-answer gradient、参数命名和 latent-only readout 全通过。
- K09 report completeness：zero/shuffle/slot、T1/T2/T4、source-token-shuffle、wrong-definition 均有报告，batch-shuffle 有效覆盖两族各 `>=0.99`。
- K10 temporal persistence：ERE、CPS 的 validation witness accuracy 各 `>=0.65`，zero-state drop 各 `>=0.10`，within-pair swapped-state drop 各 `>=0.20`。

所有 Gate 同时通过才是 `PASS_P1_V7_INTEGRATED_K8`。不得因接近阈值而四舍五入、降低 Gate 或改用 best checkpoint。

## 7. 阶段与停止规则

正式命令依次创建：preflight、query-cache、K8 三个固定根。

1. preflight 冻结源码身份，核验上游 seal/hash、合同 hash、选择 preimage、预测测试、GPU 联合反传和磁盘/GPU 条件。
2. query-cache 只在 preflight PASS 且源码身份未变时创建；构建后全量内容审计并封存。
3. K8 只在 query-cache sealed PASS、源码身份未变且 full reaudit PASS 时训练。

任一阶段 FAIL 立即封存并停止；后序根必须不存在。运行时异常也算该阶段 FAIL，不得在同名根重启。K8 PASS 仍停止在 P1 控制组之前，等待新的控制合同。

## 8. 可复现性与解释边界

每个根保存 attempt、progress、result、运行环境、git dirty fingerprint、完整 active-source snapshot 和 evidence seal。选择、数据顺序、模型初始化、update 数、checkpoint 与 Gate 都是预注册常量。

若 K8 失败，首先按三层解释：

1. infrastructure / contract 是否完整；
2. temporal mechanism 是否在联合训练后仍通过 K10；
3. 在 K10 通过时，答案失败才归为 integrated optimization/capacity/generalization；在 K10 失败时，归为联合目标破坏了机制。

该分层避免再把一个技术性合同错误、一个局部机制结果和完整架构结论混为一谈。
