# V2-R1R P1 v8：Causal-bridge 等算力资格合同

## 1. 目标与证据等级

P1 v7 已证明 K=8 shared latent core 能保留强 temporal mechanism，也能让答案依赖 episode-specific final state；它没有让 CPS 答案响应决定结果的局部代价变化。v8 不再重复扩大 ordinary training，而是直接检验最小机制假设：**只要把合法的反事实视图加入训练支持，final-answer CE 是否会接上已有语义状态；若仍不足，显式同对配批与 pair ranking 是否提供额外必要桥梁。**

本轮是低成本、单种子的 mechanism qualification，不是完整 P1。它复用 P1 v2 的 frozen source cache，并把旧 causal-pairs 的一部分转作 optimization；因此任何 PASS 都不能称为 causal OOD、跨分布或架构通过，只能授权另立 fresh-data P1 v9。

## 2. 固定数据分区

输入仍是 P1 v2 model subset，且不修改任何上游 artifact。

### 2.1 causal optimization/audit

每族原有 512 个完整 `base/flip` pair。按固定 seed `2026082811` 对 `(family,pair_id)` 做 domain-separated SHA-256 排序：

- 前 384 pair/族为 causal optimization，即 768 records/族；
- 后 128 pair/族为 sealed causal audit，即 256 records/族；
- optimization 与 audit 的 pair_id、example_id 必须完全不交叠；
- 每对必须恰有 `base/flip` 两个 role、相同 label mapping、不同正确答案。

模型前向不得读取 pair_id、pair_role、AST、certificate、span、task/operator identity 或答案。pair metadata 只允许进入确定性 sampler、pair loss 和 evaluator。

### 2.2 ordinary optimization 与诊断评估

ordinary arm 从各族 8192 个 train records 中按冻结 balanced order 取前 768 条，与 causal optimization 的 record 数完全相同。三臂共享同一个 causal audit。

普通能力诊断固定为每族 validation 与四个主 OOD 各 256 条，均由不看标签、文本、长度或模型输出的独立 SHA-256 排序取得。它们不参与梯度、checkpoint 选择或 arm 选择，只用于检测 causal bridge 是否以普通能力坍塌为代价。batch-shuffle 另使用冻结 model subset 中完整的 1,024 条 ERE length 与 1,024 条 CPS horizon；干预 drop 必须与同一完整 cohort 的 full 结果比较，从而保持两族有效覆盖各 `>=0.99`。

## 3. 模型与训练路径

三臂都从同一 model seed `2026082817` 重新初始化完全相同的 v7 K=8 模型：Qwen3.5-2B frozen final hidden、tokenwise 2048→512 Boundary、8 个匿名 slot、两层共享 recurrent block、T+1 状态映射、final `trajectory[T]` latent-only readout，以及可物理删除的 temporal probe。没有 oracle span mask、显式寄存器、任务专属参数、teacher state reconstruction 或跨 episode owner contrast。

每臂固定：

- batch 32，ERE/CPS 各 16；
- 3072 updates，98,304 episode exposures；每个 optimization record 精确 64 次；
- 前 768 updates 只训练 temporal paired objective；
- 后 2304 updates 联训 answer 与 temporal，answer 权重在前 384 updates 线性升到 1；
- 每个 optimization episode 选择 3 个 temporal witnesses，每次 exposure 确定性轮换 2 个；3 是 causal ERE 全体都能满足的最大统一下界，避免按样本难度改变监督量；
- AdamW、参数组学习率、weight decay、warmup、cosine floor 与 gradient clip 沿用 v7；
- 固定 final update，不使用 audit/validation 选择 checkpoint。

每族先按冻结 source token count 排序为 8 个等大长度桶，桶内按 arm-specific 固定种子打乱，再把相同长度分位的 ERE/CPS half-batch 对齐；causal-paired 以 pair 的最大长度分桶，causal-unpaired 在同一 pair bucket 内拆成两个不共批的 role wave。该调度保持每条 record 精确 64 次 exposure，并在 ledger 中冻结 padding efficiency 与完整 schedule hash。

三臂唯一允许的差异是 answer training support 与 pair objective：

1. `ordinary`：768 ordinary records/族，普通确定性 shuffle，answer CE；
2. `causal_unpaired`：768 causal records/族，打散单条 record，answer CE，不保证 pair 同批；
3. `causal_paired`：与 unpaired 完全相同的 records 和 exposure，8 pair/族同批；除 answer CE 外，加入双向正确标签 pair ranking：每个 view 的正确标签 logit 必须高于同对另一答案标签，margin 固定 `1.0`、权重固定 `0.5`。

pair ranking 只作用于最终答案 logits，不重建 teacher state，也不向部署模型添加结构。它的作用是将单变量输入差异与答案决策差异显式绑定。

## 4. query cache 与可复现性

optimization 每 episode 固定 3 个 witness，causal audit 每 episode 固定 2 个 witness。query identity 仍是 `(example_id,query_text)`；选择使用 kind round-robin 和固定 seed，不依赖模型结果。所有 query 用与 source cache 相同的 frozen Qwen backbone 编码为 FP16 packed bank，无截断，并逐 entry 重算 content hash、finite、offset 与 shape。

preflight 必须冻结：合同 hash、上游 seal/hash、数据分区 preimage、query selection preimage、三种 schedule、同初始化 hash、预测测试、GPU 三种 objective backward，以及磁盘/GPU 条件。正式运行的每阶段保存 source snapshot、dirty fingerprint、progress、result 与 evidence seal。

预注册值：

- partition SHA-256：`8EFFE0A087493753F5630349DEECBD27487DA585CA6D39764E1E06686412ED85`；
- query selection SHA-256：`17A5ABC206D92993244C1991F2DEA24E96D85B57502AE35F8E759CE408D866B3`，共 10,240 个唯一 query；
- ordinary schedule SHA-256：`C55C6A16B308F6FBAECA62A500B8E617C002CA76936FC84BD1AE5C8287071CB6`，padding efficiency `0.88313`；
- causal-unpaired schedule SHA-256：`2C46C19BB1100EF0965E4E57B4FD34ACAE0B34CDBD17ED172EAE5A64555A5F8B`，padding efficiency `0.88783`；
- causal-paired schedule SHA-256：`1EBE27BA1D48542DFB7CEA26DADE04AC7A3CABC58F24CD58E319FA3B8FE32E3D`，padding efficiency `0.88862`。

## 5. 正式评估

每臂在固定 final update 评估：

- sealed causal audit 的 raw answer accuracy、pair flip accuracy、prediction flip rate 与 pair 结构完整性；
- causal audit temporal accuracy、zero-state drop、within-pair swapped-state drop；
- ordinary validation 与主 OOD 256-record diagnostics；
- length/horizon 上的 batch-shuffle-middle；
- probe 物理删除、strict reload、答案预测 hash 不变；
- model integrity、finite、训练 exposure、吞吐与 GPU telemetry。

`zero-middle`、T1/T2/T4 与 contextual hidden token reversal 只报告，不作为 v8 Gate。pair evaluator 的合法 role 集固定为 `base/flip`，并通过破损 pair 的负控测试。

## 6. Gate 与最简机制优先级

单臂资格 Gate：

- B01 causal pair structure：两族各 128 对完整，零 invalid；
- B02 causal pair flip：ERE、CPS 各 `>=0.65`；
- B03 causal raw accuracy：ERE、CPS 各 `>=0.75`；
- B04 temporal persistence：两族 accuracy 各 `>=0.65`、zero-state drop 各 `>=0.10`、swapped-state drop 各 `>=0.20`；
- B05 ordinary retention：validation ERE、CPS 各 `>=0.60`；本轮只防止灾难性坍塌，不把它误称为完整泛化；
- B06 state dependence：length/horizon batch-shuffle-middle 两族平均 drop `>=0.30`，有效覆盖各 `>=0.99`；
- B07 probe strip、architecture integrity、finite、exact compute 与报告完整性全部通过。

相对 decision-power Gate：被选 causal arm 相对 ordinary 的 causal pair flip 必须在 ERE、CPS 各提升 `>=0.15`。否则即使绝对值偶然越线，也不能归因于 causal bridge。

选择遵循最简机制优先：

1. 若 `ordinary` 自身通过全部绝对 Gate，则结论是额外 bridge 未被证明必要，v8 不选择 causal arm；
2. 否则若 `causal_unpaired` 通过绝对与相对 Gate，选择 unpaired，说明关键是反事实训练支持而非 pair scaffold；
3. 只有 unpaired 未通过、且 `causal_paired` 通过全部 Gate并在两族 pair flip 上相对 unpaired 至少一族提升 `>=0.10`，才选择 paired；
4. 其余情况均为 `FAIL_P1_V8_CAUSAL_BRIDGE`。

## 7. 停止与后继

正式顺序只有 preflight、query-cache、qualification 三阶段。任一阶段 FAIL 立即封存并停止，同名根不得重启或补跑。

PASS 只授权设计 P1 v9：生成 fresh causal train/audit、补齐 CPS 单因素训练支持、封存真正 composition/language OOD，并重新运行完整 K=8 与 matched controls。FAIL 则停止向 ordinary loss 叠加训练补丁，转向 final-state/readout 耦合机制的架构重设计。无论 PASS/FAIL，本轮都不运行 P2。
