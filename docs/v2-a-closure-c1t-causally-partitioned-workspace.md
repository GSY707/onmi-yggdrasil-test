# V2-A Closure C1T：Causally-Partitioned Workspace successor 合同

## 1. 核心判断

C1S 已证明模型能够形成并读出时序状态，但没有证明这些状态对答案因果必要。决定性缺口来自 Boundary：entity、operation 与 query readers 都读取同一份完整、全局上下文化的 source hidden，因此 H0 的任一槽都可能携带整题答案。仅增加 no-core loss 或 slot-deletion loss，会把“不要走捷径”当作偏好，而没有消除捷径通道。

C1T 直接切换到 **Causally-Partitioned Workspace（CPW）**。它同时改变任务合同与模型边界：

1. 每个对象卡、操作卡和查询卡由冻结 source model **独立调用**编码；禁止从整篇上下文化 hidden 切片伪装成独立卡。
2. 模型只接收公开卡 hidden、mask 与由公开地址字符串确定的 record-local address vectors；family、AST、answer、semantic label、support ledger 均不得进入 forward。
3. 初始 query-owner 卡不含其他对象状态。答案只在注册操作执行后，从 query address 选中的最终对象状态读取。
4. ERE 与 CPS 都使用完整 2×2 factorial。对任一记录翻转任一支持对象，simulator answer 必须改变；两个对象卡以外的 public cards 必须保持逐字节不变。
5. 训练目标直接包含 full answer、no-core confusion、两个支持对象各自的 margin degradation；Gate 与 loss 使用同一个 gauge-invariant answer margin。

这是一条新 package、新 cache、新 checkpoint、新 output identity 的直接切换。C1S checkpoint、全篇 hidden cache、旧 S1/S2/S3 runner 和旧授权都不得复用为 successor evidence。

## 2. 冻结身份与范围

- identity：`V2-A-CLOSURE-C1T-CAUSALLY-PARTITIONED-WORKSPACE-20260901-1`
- package：`src/yggdrasil_v2/v2_a/closure_c1t/`
- CLI：`experiments/v2_a_closure_c1t.py`
- S0 preflight：`tmp/v2-a-closure-c1t-cpw-s0-preflight-20260901-1`
- S0 root：`artifacts/v2-a/closure-c1t-cpw-s0-20260901-1`
- S1 preflight：`tmp/v2-a-closure-c1t-cpw-s1-overfit32-preflight-20260901-1`
- S1 root：`artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1`

本轮用户授权实现 successor，不授权消费固定 preflight/root，不授权训练。S0 只做结构与数据量具资格；S1 必须另获明确授权。S1 FAIL 后 S2/S3/formal 全部 `NOT_RUN`。

## 3. 公共卡输入合同

每条记录公开以下文本卡：

- `object_cards[K]`：每张只描述一个公开地址及其局部初始内容；
- `operation_cards[T]`：每张描述一个公开 source address、target address 与局部操作；
- `query_card`：只描述 query address、问题与 record-local label legend。

每张卡由相同冻结 source model 独立编码。cache ledger 必须记录 `(example_id, card_kind, card_index, text_sha256, token_sha256, hidden_sha256)`，并证明一次 encoder call 中只有该卡文本；同一 `text_sha256` 在不同记录中的 `token_sha256/hidden_sha256` 必须逐字节一致，否则说明 encoder 未冻结、仍有随机态或调用合同漂移，立即 fail-closed。旧 C1 的 68GB full-source cache 可作 checkpoint/tokenizer 身份参考，但其 hidden 字节不得作为 C1T 输入。

公开地址向量由 `NFKC + casefold` 后的地址字符串与冻结 salt 做确定性 SHA-256 展开，再 L2 normalize。它只表达同一记录内“这些公开名字相同”，不编码 family、semantic slot 或 answer。地址碰撞、归一化碰撞、非有限值均 fail-closed。

模型 forward 只允许：

- `object_hidden`, `object_mask`, `object_addresses`, `object_present`
- `operation_hidden`, `operation_mask`, `operation_source_addresses`, `operation_target_addresses`
- `query_hidden`, `query_mask`, `query_address`

answer、valid-choice mask、support slots 与 counterfactual group 仅存在于 model 外部的 loss/evaluator。valid-choice mask 只审计正确答案确属公开 legend，不得在 CE、top-1 或 margin 前屏蔽 raw logits；模型必须从 query card 自行学会压低不存在的 A–I 标签。

## 4. task-side causal arity

### 4.1 ERE XOR

每条 ERE 记录含两个支持实体 `A/B` 和 query target `T`。两个注册事件依次：

1. 把 `A.bit` 复制到 `T.bit`；
2. 用 `B.bit` 与当前 `T.bit` 计算 XOR，再写回 `T.bit`。

查询 `T.bit`。四个 `(A.bit,B.bit)` 组合必须全部存在，答案为 XOR。翻转 A 或 B 都必然翻转 simulator answer；只改变对应对象卡，operation/query/另一个对象卡保持不变。target 初始 bit 与 factorial factors 独立配平，避免 no-core 从 T 初态取巧。

### 4.2 CPS parity-of-validity

每条 CPS 记录含两个等成本候选与 decision object。候选 A/B 各自是否有效只由各自对象卡中的局部 prerequisite 决定：

- 仅 A 有效 → A；
- 仅 B 有效 → B；
- 都无效或都有效且等成本 tie → NONE。

四个 validity 组合全部存在。翻转任一候选的局部 prerequisite 都必然改变 simulator answer；另一个候选卡、operation cards 与 query card保持不变。raw A–I label mapping 按 factorial group 独立随机化，并在同组四个 cell 内固定，既保证 counterfactual query 卡逐字节不变，也避免跨组固定 raw label 语义。

### 4.3 arity certificate

每个 factorial group 必须满足：四个 cell 完整、simulator replay 一致、两个单因子 flip 均改变 semantic answer、每个 flip 只改变对应对象卡、公开 address/order/operation/query 不变。满足后才可写 `task_causal_arity=2`。这一定义不使用“query-reachable certificate”代替 answer causality。

## 5. 模型结构

`PartitionedBoundary` 对每张卡独立做 masked pooling 与共享 local encoder。对象内容之间没有 attention，操作和查询也不能读取对象 hidden。公开地址向量用于无参数 cosine 路由：每个 operation 只能读一个 source object、写一个 target object；query 只能在操作结束后读取一个 final object。

`TargetOnlyTransition` 不更新 source slot，只根据 source payload、target payload 与 operation card 更新 target。最终 `QueryReadout` 将 query card state 与 final query-owner payload 合并后输出 raw A–I logits。

no-core 路径跳过全部 registered operations，但仍执行相同 query readout。因为 query-owner 初始卡被 task contract 配平且不能读取支持对象，这一路径应接近 record-local chance；它不再能通过全篇 hidden 获得答案。

对象卡排列每条记录随机化。同步置换 object hidden/address/presence 后，logits 与 trajectory 的语义状态必须保持不变；固定 slot identity 不得成为证据。

## 6. S1 objective 与 Gate 对齐

训练损失固定包含：

1. `full_answer_ce`：完整运行的 raw A–I 九类 CE；不得用外部 valid-choice mask 替模型删除错误类；
2. `no_core_confusion`：no-core 在 raw A–I 九类上接近均匀，禁止直接 answer path；
3. `support_margin_hinge`：batch 必须包含完整 factorial groups；对 A/B 分别从同组单因素翻转记录取回**同一公开地址**的 counterfactual payload，只替换这一张对象卡，要求原答案 margin degradation 达到冻结下限。其他 present-slot 均值替换只作 binding-sensitive 旁证，不进入 primary Gate；
同一 group 四个 cell 的 simulator raw answer 全部正确是数据与 Gate 约束，不另造一项与 CE 重复的伪 loss。

因果 Gate 统一使用 `correct - logsumexp(all eight raw wrong)` margin。S1 Overfit32 固定由 4 个 ERE factorial groups 与 4 个 CPS factorial groups组成，每组四个 cell，共 32 条；训练 batch 固定为两个完整 group（8 records），不允许拆散 group 或只抽一个 baseline cell。

S1 必须同时满足：32/32 full answer、8/8 factorial groups 全 cell exact；同组四个 cell 的 no-core logits 逐项相同；两族 no-core accuracy 不高于 factor-blind 基线容差 `0.55`（ERE 的无信息确定性基线本来就是 `0.50`，不能错误冻结为 `0.35`）；full−no-core margin 的 bootstrap 下界过门；两支持对象各自 margin-drop 的 bootstrap 下界和双 contributor Wilson 下界过门；对象置换不变、support flip effect 与 simulator 一致、0 forbidden forward fields。任一失败均停止，不进入 matched K1/K8 discovery。

## 7. 阶段与停止规则

### S0：零训练资格

验证 simulator factorial、public-card locality、独立编码 ledger、address collision、forward 边界、card isolation、object permutation、no-core topology、intervention evaluator 与 artifact fail-closed。S0 不能用随机网络的 accuracy 宣称机制。

### S1：fresh Overfit32

只读复用 S0 新生成并封印的 C1T 逐卡 cache，使用 fresh 模型初始化、固定 endpoint，无 checkpoint selection。只有 S0 sealed PASS、S1 合同/实现/回归/一次性 preflight 全部通过且用户再次授权后才可运行。

### S2：matched learned K1/K8 discovery

只有 S1 全 Gate PASS 后另立合同。K1/K8 必须共享数据、card cache、预算、optimizer family 与参数预算；旧 C1S K1/K8 数字不进入证据。

### S3：single-seed formal

只有 S2 在 heldout causal/OOD 与 matched K1 对照上取得稳定增益后才设计。当前不预注册 root，避免把未获资格的路径误写成授权。

## 8. 证据边界

C1T 显式公开对象/操作/查询卡，因此它首先验证“因果分区后的工作区机制”，不同时证明自然文本到 cards 的 learned compiler。若 C1T 成功，下一阶段才比较 deterministic public cardizer 与 learned text compiler；若 C1T 都失败，就没有理由把失败归咎于自然语言分割。

成功也不能由 loss、answer accuracy、route accuracy或时序可读性单独宣称。最小架构证据仍是：真实 task causal arity、no-core 受抑、两个独立 support interventions、matched learned K1/K8、heldout/OOD 与 sealed reproducibility。

## 9. 当前实现状态

唯一 S0 preflight 与唯一 zero-training S0 已封存 PASS。固定 Qwen3.5-2B 对 32 条记录的 192 张卡分别执行独立 forward，持久 cache、source ledger、factorial causality、no-core topology、target-only write、置换等变和 CUDA/BF16 backward 的 S001–S008 全部通过；全过程 optimizer/model writes 为 `0/0`。S0 结果与边界见 `docs/v2-a-closure-c1t-s0-result-review.md`。

2026-09-01 用户以“开始S1阶段”给出 C1T S1 的一次性启动授权。当前 S1 合同冻结为：只读复用 S0 sealed cache，fresh K8，完整 factorial-group batch，固定 4,000 updates 与唯一 `fixed_4000` endpoint；训练每步计算 raw A–I answer CE、no-core confusion 和同地址单因素 counterfactual support hinge。正式 root 只可在独立 S1 preflight 通过后消费，结果无论 PASS、FAIL 或 CRASH 均封存且禁止重试。精确身份、schedule、Gate 与停止边界见 `docs/v2-a-closure-c1t-s1-execution.md`。
