# V2-R1R P1 v6 causal-temporal-witness 机制资格合同

日期：2026-08-11

证据目标：验证一种可长期复用、真值有定义的 recurrent-state 信用机制；不是完整 P1、K 容量、文本基线、跨 seed 或 P2 证据。

## 1. 判决问题

v5 证明静态 paired claim 在 7,168/族的平衡 nonce episodes 上没有启动。v6 不改匿名 K=8 部署模型，不再依赖跨 episode negative，也不继续堆暴露；它比较等 source batch、等 query-state 判断数、等 seed 的两种 claim-only 信号：

1. `static_pair`：两个不同 predicate query 在同一 prefix 上一真一假，即 v5 的合法目标；
2. `temporal_witness`：同一个 prefix-free predicate query 在同 episode 的两个相邻 recurrent states 上一真一假。

只有未见 episode 的 temporal accuracy 和 state dependence 通过，才说明新的目标打破了对称信用问题。

## 2. Causal temporal witness

每个 witness 固定包含一个 query、同一 episode 的 `before_prefix/after_prefix` 和相反 labels。query 文本不得出现当前 prefix、before/after 或目标标签；时间地址只由选择哪个 latent state 提供。`query_id` 由 `episode_id + query_text` 派生，因此相同的通用自然语言 predicate 可以跨 episode 复用表面语义，却不会在监督账本中合并 owner。

ERE witness 包含：event-position processed、attribute old/new value、relation added/removed。CPS witness 包含：candidate plan-position processed、candidate remains executable、fact added/removed、resource old/new amount、accumulated cost old/new value。所有 labels 由 accepted simulator 对原 AST 重放得到；model view 仍只有既有 source hidden、reasoning budget 与 choice mask。

这不是显式寄存器或 oracle span：模型从不接收 AST、symbol table、canonical state、source pointer 或 teacher hidden。训练期 probe 只接收自然语言 predicate hidden 与匿名 workspace state，并在后继完整 P1 的部署 checkpoint 前物理删除。

prefix 与递归状态采用唯一映射：每条 source 先经过一次共享 recurrent source-conditioning 得到 `H0`，逻辑 prefix `p` 读取 `trajectory[p]`；所以运行内部迭代数固定为 `reasoning_budget + 1`。这一步不新增寄存器、task embedding 或外部状态，只补齐 T 次语义 transition 必须对应 T+1 个状态的定义；两条 comparator arm 使用完全相同的映射和计算。

## 3. 固定数据与比较预算

继续只读复用 P1 v2 source/claim cache与 P1 v3 recovery 授权。使用 formal `SELECTION_SEED=2026082601` 对每族 8,192 train rows 按 label × pattern × reasoning budget 确定性平衡排序；前 256/族为 comparator optimization，随后 128/族为永久隔离 audit。两集合不相交，选择不得读取任何模型结果。

两臂均从同一 formal `MODEL_SEED=2026082617` 初始化，以 `DATA_ORDER_SEED=2026082621` 共享 K=8、batch32、source schedule、AdamW、学习率，并固定完成 3,200 updates。每 episode 每次只取两个 pair：static 为 4 个不同 query-state 判断；temporal 为 2 个 query 各在两个 states 上判断，也是 4 个判断。每 400 updates 只在 optimization rows 上评估；候选 checkpoint 必须不早于 update 1,600，并按两族 seen accuracy 的最小值、均值和 loss 预注册排序。两臂都跑满、每个 optimization episode 平均暴露 200 次后，才对各自选中 checkpoint 做唯一一次隔离 audit；audit 不参与 checkpoint 选择。

训练只使用 paired CE + rank，不使用 answer loss、owner contrast、teacher state reconstruction 或跨 episode negative。该阶段隔离训练信号本身，不把 answer memorization 混入判决。

## 4. Formal 阶段与 Gate

顺序固定：

1. `preflight`：来源 hash/seal、v5 失败身份、后序 root absence、预测试、GPU batch32 temporal backward；
2. `witness-audit`：全 16,384 train episodes 生成、重放与覆盖；
3. `query-cache`：只为 comparator 256+128/族 rows 编码 prefix-free query hidden；
4. `mechanism-compare`：运行 static 与 temporal 两臂；
5. `assessment`：复算 seals、selection、matched compute、Gate 与禁止项。

Witness Gate 为：每个 query ID 只属于一个 episode 语义、两个 labels 恰为 `{false,true}`、prefix 相邻且严格递增、query prefix-free、全部 simulator replay 一致；每族所有 rows 至少一个 witness；ERE 每个 event 有 progress witness且至少 90% event 有 state-change witness；CPS 每个实际执行或首次失败 transition 有 progress/validity/state/cost witness，每个 candidate 至少一个 witness。

单臂资格条件为：两臂均完成固定 3,200 updates、选中 checkpoint 不早于 1,600；seen accuracy 每族≥`0.70`；隔离 audit accuracy 每族≥`0.65`；audit accuracy−zero-state accuracy 每族≥`0.10`；所有值有限。temporal 另需 swapped-state accuracy drop 每族≥`0.20`。

选择规则预先固定：若 `static_pair` 自身通过，选择更简单的 static；否则只有 temporal 通过且其 audit accuracy 相对 static 每族至少高 `0.10`，才选择 temporal。其他结果一律 FAIL，不允许父任务事后按最好看的 arm 改判。

### 4.1 formal 前开发探针与 seed 退役

正式 root 创建前，旧开发 seed `SELECTION=2026082501 / MODEL=2026082517 / ORDER=2026082521` 做过一次 64/32 每族、等计算、非 formal 探针。100 次暴露时 static audit 为 `0.479/0.503` 且 state drop 近零；temporal audit 为 `0.704/0.722`，state drop 为 `0.204/0.222`，swapped-state drop 为 `0.408/0.445`。temporal 在 200 次暴露时 seen 为 `0.736/0.836`，audit 为 `0.697/0.768`，两项因果 drop 继续通过。

该结果说明机制迁移已经出现，但 `seen≥0.80` 会把机制资格误写成完整状态库存拟合。因此在任何 formal root 创建前，seen Gate 校正为高于随机 20 点的 `0.70`，audit 与因果 Gate 不降低。由于开发探针已经读取旧 audit，三枚旧 seed 永久退役；上述新 formal seeds 和其 128/族 audit 在合同冻结后才派生，开发结果不得计入正式通过。

## 5. 证据边界

v6 PASS 只授权另立 integrated P1 v7，把选中的训练机制与 answer objective、full 8,192/族、原 K01–K09、K=1/direct/text-CoT 和 fresh seed 纳入完整合同。v6 FAIL 表示当前无 oracle、无结构输入的 query-supervision 机制仍不能规模迁移，应重新评估 frozen-Qwen Boundary 或匿名 workspace，而不是继续调权重。

任一 root non-PASS 后立即停止，后序 roots 不得创建；所有 roots single-use，禁止修改、覆盖、重封或重跑。不得启动 P2。
