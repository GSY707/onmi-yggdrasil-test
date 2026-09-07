# V2-A Closure C1U S2 失败诊断、归因与修复方向

## 1. 核心判断

S2 的主故障不是训练时长、dead gate、K8 容量不足或统计功效不足，而是当前模型没有把 public card 中的两类 record-local 绑定实现成结构不变性：

1. query 中的 `semantic meaning → A–I label` 仍由单个 pooled query vector 和固定九类 head 隐式学习，不具 choice-label permutation equivariance；
2. object/operation 中的 record-local opaque symbol 仍进入 whole-card pooled payload 与 MLP transition，不具 opaque alpha-renaming invariance。

K8 因而能在四个训练 bank 上建立正确且双 support 因果的 bank/group 记忆，却不能把相同算法迁移到新名字、新 query mapping 的 heldout bank。R206 通过只说明多地址敏感性存在，不说明地址中的内容计算可迁移。

## 2. 已排除的解释

formal 与 post-stop 只读证据排除了以下近因：

- **运行或封印故障**：24,000/24,000 updates、6/6 fixed endpoints、76/76 seal replay 全部完整。
- **4,000 steps 不足以拟合 K8**：F0/F1 K8 训练集为 `128/128`，F2 为 `126/128`；末步 total loss 分别约 `0.0011/0.0006/0.0120`。
- **learned sigmoid dead gate 复发**：C1U/S2 使用 gate-free target-only overwrite；训练 bank 的完整 factorial 与 two-contributor 直接证明两步可工作。
- **只是 K1 过强**：K1 训练集只有 `60/128, 67/128, 70/128`，CPS 在三个 folds 均为 `32/64`；K8 heldout 失败独立成立。
- **只是不懂某个固定 label rotation**：heldout prediction 的 canonical-label 命中接近随机，错误 offset 没有单一模式。
- **raw/valid-choice 口径错配**：正式 CE、margin 与 evaluator 都使用 raw A–I logits；`valid_choice_mask` 只做目标合法性验证，未在主 CE/Gate 中 mask logits。
- **public 信息欠定义**：ERE 明示 bit 0/1 legend，CPS 明示 candidate index，query 明示 Choices；public-only parser/replay 在 preflight 已通过。

## 3. Train–heldout 重放

使用 sealed endpoints 与正式 cache，只读重放每个 endpoint 自己的四个训练 bank。没有 optimizer、权重写入或 checkpoint 选择。

| Endpoint | K1 train | K8 train | K8 heldout |
| --- | ---: | ---: | ---: |
| F0 | `60/128` | `128/128` | `18/64` |
| F1 | `67/128` | `128/128` | `16/64` |
| F2 | `70/128` | `126/128` | `18/64` |

F0/F1 的 K8 训练 factorial 为 `32/32`、two-contributor `128/128`；F2 只有两条 ERE 训练记录未满。与此同时三 fold heldout 合计只有 `52/192`、factorial `3/48`。这是 bank-conditional memorization，而不是通常意义上的轻微 distribution shift。

K1 的末步 total loss 都约 `1.11–1.12`，其中 raw answer CE 约 `0.87`、support hinge 约 `0.49–0.50`。K8 的末步两项均接近 0。matched K1 因单状态压缩无法完成训练任务，符合控制臂预期；但它没有给 K8 的泛化提供正证据。

## 4. Choices-only permutation 反事实

### 设计

固定 sealed F0-K8、B2 的全部四个 CPS groups 与四个 ERE groups。object cards、operation cards、地址、语义状态和权重完全不变；只把 query `Choices:` 中的 label 整体循环移动 0–8 位，并用同一冻结 Qwen3.5-2B revision 对 72 个 query cards 独立重编码。正确答案也按公开映射同步移动。

若模型真实执行 `computed semantic → current choice meaning → current label`，则 logits/prediction 应随 label permutation 同步等变。

### 结果

| Family | 原 mapping | 八个非零 rotations 合计正确 / 应正确 | 等变结论 |
| --- | ---: | ---: | --- |
| CPS | `16/16` | `25/128` | FAIL |
| ERE | `16/16` | `4/128` | FAIL |

逐 rotation 的 prediction-shift match 与 correctness 完全相同，因为原 mapping 为 16/16。ERE 在 +1、+2、+3、+6、+8 五个 rotation 上均为 `0/16`；CPS 在 +3、+6 为 `0/16`。其中部分 rotation 值在 F0 的其他训练 bank 出现过，仍不能迁移到同一 B2 内容上的新 choice binding。

该实验直接证明 query legend 被模型当作整体上下文特征，而不是结构化的 record-local choice table。

## 5. Opaque alpha-renaming 反事实

### 设计

仍使用 sealed F0-K8 与 B2 的 32 条训练记录。保持 factors、task semantics、object order、choice labels、答案与 causal ledger 不变，只对记录中 64 个 opaque symbols 做一致一一重命名；新旧 token 都为 64 个且交集为 0。所有 192 张 public cards 由同一冻结 Qwen revision 重新独立编码，随后只做推理。

正确的 record-local binding 应把一致 alpha-renaming 当作语义恒等变换。

### 结果

| Family | 原 B2 | Alpha-renamed | Factorial exact | Two-contributor |
| --- | ---: | ---: | ---: | ---: |
| CPS | `16/16` | `0/16` | `0/4` | `0/16` |
| ERE | `16/16` | `4/16` | `0/4` | `0/16` |

这不是 heldout bank 中多个变化混在一起的观察，而是对一个已被完全拟合任务的单因素反事实。它证明 hard address routing 只保证“字符串相等的地址路由到同一槽”，没有使 payload/operation 对 opaque surface identity 不变。

## 6. Latent probe 的次级证据

对 sealed K8 的 selected state、query state 与 fused state 进行了固定 alpha ridge-probe sweep。训练 probe 只使用各 fold 的四个 train banks，并在两个 heldout banks 上预测 canonical semantic class 或 raw A–I label。

selected state 的 heldout semantic accuracy 不稳定：CPS 三 fold 的最好观察值约 `75%/59%/53%`，ERE 约 `50%/63%/50%`，而对应多数/随机基线约为 50%。fused/raw-label probe 也在 train 接近饱和时于 heldout 大幅下降。由于样本少、维度高，该 probe 只作次级证据；它足以否定“所有 folds 都已有清晰、统一的可迁移 semantic state”，但不单独承担根因判决。

根因主要由 Choices-only 与 alpha-renaming 两个确定性反事实建立。

## 7. 为什么当前结构会失败

当前 `IndependentCardEncoder` 对每张卡做 token mean pooling；`TargetOnlyTransition` 直接在两个 payload 与一个 pooled operation state 上运行 MLP；`QueryReadout` 再把 selected payload、单个 pooled query state 及乘积送入固定 A–I linear head。

这个图具有 object-slot permutation equivariance 和 exact address routing，却没有两项 S2 真正需要的对称性：

- label 名称变化会改变整张 query embedding，但输出 head 仍是九个固定类别，没有逐 choice 的共享评分与公开 scatter；
- opaque 名称同时参与 object/operation/query hidden，surface token 变化会改变 payload 与 operator，图中没有 record-local symbol table 把“名字相等”与“内容语义”分离。

within-group factorial CE/hinge 只约束训练组上的答案和 counterfactual。一个按 bank/group identity 记忆的解也能取得 128/128、32/32 factorial 和 128/128 two-contributor，因此 S1 的正证据不能外推到 alpha-renaming 或 fresh-bank execution。

## 8. 建议的直接修复

不能在旧 C1U S2 上加 augmentation、换 seed 或续训。建议建立全新 identity，并直接替换接口为 **symbol-bound choice workspace**；旧 pooled-query/fixed-head 路径不保留兼容分支。

### 8.1 Public symbol binder

把公开字符串分为两条通道：

- opaque address/symbol 只进入 record-local equality table 和 hard routing，不进入内容 MLP；
- card 中显式公开的 typed fields（bit 0/1、yes/no、candidate index、cost、goal/effect、operator type）进入共享 field encoder。

ERE 的公开 value legend 应在记录内建立 `opaque value ↔ bit role` 绑定；CPS 的 candidate index 与 availability 应成为 typed payload。binder 只能读取 public card text，禁止读取 `answer_index`、`semantic_answer`、AST target、support ledger 或 label metadata。

对于当前 Closure 目的，优先使用可 fault-kill 的确定性 public-card compiler 来隔离 workspace 机制；如果目标改为端到端自然语言 parser，应作为后续独立 stage，不应与 multi-address 资格混在同一 Gate。

### 8.2 Choice-set scorer

删除固定 `Linear(width, 9)` raw-label head。模型先产生 semantic answer state，再对每个公开 choice meaning 独立编码，并用同一个 shared scorer 计算 `score(answer_state, choice_state)`；最后仅按 query 中公开的 label index scatter 到 A–I logits。

这样 label permutation 只改变 scatter 位置，模型 logits 在结构上严格随 permutation 等变，而不是依靠六个 bank 学会近似轮换。

### 8.3 新 S0 必须先证明精确对称性

任何新训练前，zero-update S0 至少应包含：

- 全量 opaque alpha-renaming 后 typed payload、trajectory 与 semantic choice scores 严格不变；
- choice label permutation 后 semantic scores 不变、raw A–I logits 按同一 permutation 精确移动；
- object order permutation 继续等变；
- 两个支持对象的正控制分别改变注册 semantic result；
- public-only、target-free cache、fault-kill 和 K1/K8 参数/初始化 parity；
- source、optimizer steps 与 model writes 均为 0。

这些必须是 architecture-by-construction Gate，而不是训练后统计门。

### 8.4 重新建立阶段证据

若用户另行授权，顺序应为：fresh successor 合同设计 → zero-training S0 → fresh S1 Overfit32 → 同规格六-bank、单-seed S2。架构已经实质变化，不能把下一次称为旧 S2 retry，也不能复用六个 endpoint。

multi-seed 仍不应成为下一步。当前两个确定性反事实已经定位结构缺口；在精确 alpha/choice Gate 通过前，增加 seed 只会重复测量同一个错误图。

## 9. 当前权限边界

本文是 sealed FAIL 后的只读诊断与修复建议，不创建 successor 身份，不授权实现、训练、cache、root/lease、S3 或 C2。下一步需要用户对“全新结构合同设计”作明确授权。
