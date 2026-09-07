# V2-R1R P1-H1 共享特征—路由状态写入投影 Core 开发对照设计

日期：2026-08-17

状态：当前 factorized routed-projection 开发方向已在预登记 full-budget screen 中失败并停线；calibration、合同哈希冻结与唯一 single-use H1 均未获授权。本文不是 H1 formal 结果，也不授权 F1、P2 或 P1 完成。

## 1. 判断与用途

P1-NR1 已证明 numeric/relation measurement surface 本身可独立复算、可杀死注册故障，但它没有训练模型。V8L 则只关闭了“匿名 shared K=8 + lexical anchor teacher”的具体组合。H1 因而不再修补 V8L，也不复用其 checkpoint；它只回答一个介于 measurement 与完整 P1 之间的问题：在相同 frozen text representation、相同 fresh data、相同 latent state、相同 loss 与相同 active compute 下，保留所有样本共享的非线性 feature trunk，再把从内容学得的类型 sidecar 只用于选择最终 state-write projection，是否比完全共享 projection 产生可重复的 OOD 与因果净收益。

H1 是窄的 architecture-development comparison，不是完整 P1 formal。PASS 只允许另立并冻结 F1；FAIL 原样封存并停止。两种结果都保持：

```text
p1_completed=false
p2_eligible=false
p2_started=false
```

## 2. 证据边界与直接切换

H1 只借鉴三类已存在事实：P0-M v5 的 mmap cache/训练封存工程、A1.19H 的 `S_t=(A_t,H_t)` 分层思想，以及 NR1 已资格化的离线 target。它不继承 P0-M/V7/V8D/V8L 权重，不使用 A1.20D checkpoint、section oracle、hard re-embedding 或 post-stop 结果，也不把 A1.19H exact-symbolic 成功提升为 full-text 证据。

活动入口直接切换为一个 `run-p1-h1`。NR1 roots、transport、source snapshot 与结果保持只读历史；旧 formal 命令不得继续暴露。H1 使用独立 `h1/` 模块、测试、两根 fixed roots 与 transport，不在旧 `p1/` 或 `p0m/` 上加兼容层。

## 3. Fresh 数据与离线 teacher

### 3.1 数据身份

正式数据 seed 固定为 `2026081801`、`2026081802`、`2026081803`；相应 model seed 为 `2026081811`、`2026081812`、`2026081813`。每个 seed 固定：

| split | records | 用途 |
| --- | ---: | --- |
| train | 4096 | 两类平衡训练 |
| validation | 512 | 仅报告，不选 checkpoint |
| supported | 1024 | 分布内能力与机制 floor |
| heldout | 1024 | 预注册 primary endpoint |
| causal | 512 | 变形、替换与干预配对 |

五个 split、三个正式 seed 之间的 semantic/source fingerprint 必须全部零重叠，并与 NR1、P0-D v17、P0-M v5 的已登记 fingerprint 零重叠。生成顺序冻结为 `history -> consumed balanced screen -> consumed balanced calibration -> current screen -> current calibration -> formal -> smoke`，七个 identity domain 两两零重叠。已消耗的 `2026081791/2026081792` 与 `2026081793/2026081794` 永久保留为历史禁用身份；当前 factorized projection screen 固定为 `2026081761/2026081762`，只有它通过后才允许一次 `2026081763/2026081764` calibration。只有 formal 三包与 smoke 进入正式 23,456-record cache，四个非正式 package 只登记 identity，禁止混入正式训练。不得从 validation、supported、heldout 或 causal 选择 checkpoint、训练轮数、route 方向；阈值只允许按 6.3 的预登记机械规则从 calibration 导出。

semantic fingerprint 覆盖完整监督语义，而不是只 hash 裸 component：numeric 纳入 label-to-delta choices、prefix/final target；relation 纳入全部 distractor handles、directed topology、label-bound query pairs、answer 与 fixed-point trace，并对 opaque-handle rename 和 query-order 采用结构不变量。大规模 relation 抽样的有限简单拓扑可能跨 seed 重复，因此生成器按上述注册顺序执行确定性全局拒绝采样；它只推进发生重复的 record attempt，不把 seed 或 split 标签掺入 fingerprint。2026-08-17 的只读复算重现旧 screen package `CFCE5DAE…5650F8` 与旧 calibration `47D36E7D…6BBD77`，并预注册当前 screen `A52C5222…486DA4`、current calibration `21CBD922…761E9`；四包各含 7,808 records。history、四包非正式身份、formal 三包与 smoke 的 semantic/source overlap 全为 `0`，实现级 formal+smoke 仍只覆盖 `23,456` records；这些都是预测试事实，不是 H1 formal 结果。

### 3.2 两类中性内容记录

numeric record 含四个 opaque handles、每个 handle 的同长度 signed non-zero delta 序列、随机 label-to-handle choices 与“选择严格最小终值”的自然语言问题。relation record 含 opaque handles、非链式 DAG direct edges、四个 query pair、随机 label-to-query choices，并保证最终恰有一个 reachable query。source 可以包含表达实际内容的 changes/link/path 等普通词，但不得出现 `numeric`、`relation`、`task`、family 名或专用 route token。

supported 只使用训练范围内但未见的内容与 surface；heldout 同时扩大 numeric horizon/magnitude、relation handle/path/topology，并使用未进入 train 的 paraphrase template。causal 为 heldout 语义建立 handle rename、clause/choice/query permutation、numeric cancelling-pair、relation transitive-redundancy 与 surface paraphrase 配对。

### 3.3 target 与 forward 隔离

NR1 `numeric_oracle`/`relation_oracle` 只允许在 fresh-data 构建与离线审计侧产生 final decision。numeric prefix winner 与 relation fixed-point round decision 形成 8-step、五类的 trace target，其中第五类表示当前尚无唯一命中。target 文件可以包含这些监督，但 model-facing source/cache 只能包含 record id、source text/atoms、frozen hidden 与 mask。

以下字段或等价编码禁止进入 source、cache index、forward 参数或 `A_t`：

```text
task_id / family label / winner / final / prefix / closure / decision
candidate_index / oracle span or role / simulator state / answer / teacher trace
```

训练代码可以把独立 target 送入 loss；model forward 不得接受它们。模型模块不得 import simulator、NR1 reference/measurement 或 production generator。

## 4. Frozen Qwen 与 learned Boundary

唯一 teacher 为 `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。Qwen 完全冻结，只对完整 source 运行一次 final hidden，并保留最多 384 个 contextual token 及 mask，不做 line/role pooling。通用换行 atom 只用于 source 完整性和分隔泄漏审计，不进入模型张量，也不给 token 附加 role/type/span 标签。两个 arm、所有 loss 与所有干预读取同一份 cache、相同 record 顺序、相同 token/atom 计数 ledger 和相同 no-truncation policy。

learned Boundary 对每个 contextual token 做无任务标签的 norm + projection，形成统一 `D_latent=256` source memory。初始 latent slots 与 source 无关；source memory 只作为 recurrent cross-attention 的 K/V。每个 recurrent Transformer block 先用公共 cross-attention 与 slot self-attention 更新统一 residual，再执行独立公共 FFN；第二条 residual 分支先由所有记录共享同一 SwiGLU feature trunk，最后才由 route 选择一个 `H=384 -> D=256` state-write projection。这仍属于白皮书的“dense attention + FFN-MoE”边界：route 不控制 Boundary、attention、公共 FFN、feature trunk、answer 或外部动作，只控制最终状态写入投影。Qwen LM head、raw input ids、token logits 与答案映射不进入 reasoner。cache 必须记录 model/revision、source hash、token/atom count、dtype、shape、file hashes、VRAM 与编码时间，并证明 Qwen trainable parameter 为 0、无 silent truncation、无 forbidden field。

## 5. 共同 latent state

两个 arm 共享以下结构：

- `K=8` learned/generic latent slot queries，不绑定候选、handle、query 或任务语义；每个固定 step 另有不可训练的 Fourier step address；
- `T=8` 固定 recurrence，不使用 dynamic K、dynamic stop 或样本 reasoning budget；
- 两层跨 step 共享的 recurrent Transformer layer；每层依次执行公共 source cross-attention、公共 slot self-attention、`H=384` 公共 SwiGLU FFN、record-level content router、共享的 `D=256 -> H=384` residual SwiGLU feature trunk 与一个 `H=384 -> D=256` state-write projection；route 只选择最后的 projection 参数；
- source K/V 每层每次 forward 只预计算一次；
- router 只读已经 cross-attended 的当前 latent residual 的 record-level content summary，输出两类 content route；route 在一个 record 内不依赖 slot position；
- 最终四类 answer head 只读取 normalized `H_T` 的 learned-query pool，不能读取 source、raw tokens、router label、`A_t`、teacher target 或中间 trace；
- 五类 trace head 只用于训练和 state audit，正式 deployment checkpoint 前物理删除；剥离前后 answer logits/prediction 必须逐值一致。

状态解释为：

```text
S_t = (A_t, H_t)
A_t = 从 source content 学得的 route/type sidecar 与非语义 slot address
H_t = 统一宽度连续 residual workspace
```

`A_t` 不保存数值、closure、答案或完整 teacher state。所有专家输出都必须返回同宽 `H_t` residual，不能直接写 answer logits。

## 6. 唯一差异：route 是否控制计算

### 6.1 Shared arm S

S 在每层公共 cross/slot attention 后先执行独立 `SwiGLU(D=256,H=384)` 公共 FFN，再由所有记录共享的第二个 norm + gate/up 产生 `H=384` 条件特征，并始终通过同一个 generic `Linear(H=384,D=256)` 写回状态。generic projection 使用标准非零初始化，避免旧 zero-output 分支在训练早期成为可忽略旁路。S 仍训练并报告同构 content router，使两臂拥有相同类型监督和 router FLOPs，但 route assignment 不改变 generic projection。`flip_route`、`force_route0/1` 与 `swap_experts` 对 S 的 state、trace 与 answer 必须逐值 no-op（被干预的观测 route assignment 本身除外）。

### 6.2 Mixed arm M

M 执行与 S 逐值同初值的 `H=384` 公共 FFN，并与 S 共享同构、逐值同初值的 residual feature trunk；唯一结构差异是最后保留两个 `H=384 -> D=256` projection heads。content router 使用 record-level hard top-1 sparse dispatch；每个 record、每层、每步只执行其中一个 projection，未选 projection 不执行且无梯度。route 不读取 raw source、target 或答案，也不改变 attention、公共 FFN、feature trunk 或 answer head。

两臂都接受相同的外部 type-target CE，但该 target 只进入 loss：S 用它验证“模型能从内容识别计算类型但共享 FFN 仍不利用类型”；M 用同一 router 的预测结果选择 expert。正式 forward 从第一步起只用预测 route，不以 target warm start，不用 teacher forcing route。

S 与 M 的 Boundary、cross/slot attention、公共 FFN、residual feature trunk、control、router、pooler、answer/trace head 初始化逐值相同；M 的两个 projection heads 都从 S 的 generic nonzero projection 复制。M 多出的静态参数只是一个未激活的 `H=384 -> D=256` projection，必须单独报告，不能称为 parameter-matched gain。每个 record 在两臂中都执行相同的公共 FFN、同一个完整 feature trunk 和恰好一个 projection，因此 active forward/backward FLOPs 相同；tokens、updates 与 batch schedule 也必须完全相同。实际 wall time、memory 与 dispatch overhead 另行报告。

### 6.3 冻结前非正式校准导致的直接切换

2026-08-17 的 disjoint probe seed 曾检验“两个完整 transition experts 各自复制 slot attention + FFN”的开发版本。该版本在 7,808 条记录、4,000 updates 上证明 route、source、recurrence 与 trace 因果通路存在，但 heldout mixed-shared 总增益只有 `+0.00684`，paired-bootstrap 95% CI 为 `[-0.02637, 0.04004]`；numeric 为 `-0.01758`、relation 为 `+0.03125`。这不是 formal，也不消耗 H1 fixed roots，却足以否决该开发结构：复制完整 attention 使两个 expert 各自只获得一半训练族，混合臂失去公共 attention 的跨族表示迁移。

随后同一 disjoint data/model seed 的纠正版 2,400-update screen 检验“公共 attention + 两个互斥的完整 `H=768` FFN experts”。route accuracy 为 `1.0`，numeric heldout 获益 `+0.12109`，但 relation 回退 `-0.06055`；总增益仅 `+0.03027`，95% CI `[-0.00586, 0.06641]`，route flip decision drop 也只有 `0.05371`。因此它同时违反 H05 的 `+0.05`/无 family regression 和 H06 的 `0.10` 因果门槛。该结果位于 `artifacts/v2-r1r/p1-h1-nonformal-ffn-moe-screen-20260817-2/`，不是 formal，也不消耗 fixed roots。

证据说明失败点不在类型识别，而在互斥完整 FFN 让每个 expert 只接收一半训练族、丢失公共 FFN 的跨族迁移。第三轮因此使用“公共 `H=512` FFN 主干 + `H=256` generic/routed residual 分支”。在当前 fingerprint 真源重新构建的 fresh cache 上，该 2,400-update screen 取得总增益 `+0.12695`、95% CI `[0.09375, 0.16113]`，numeric/relation 分别为 `+0.22266/+0.03125`，H05 强通过；但四个注册 route intervention 的最大 decision/trace drop 只有 `0.08887/0.06494`，仍低于 H06 的 `0.10`。结果位于 `artifacts/v2-r1r/p1-h1-nonformal-shared-routed-screen-20260817-2/`，所以只证明公共 FFN 恢复了净收益，仍不允许校准或 formal。

合同据此直接切换为等分 `H=384` 公共 FFN + `H=384` generic/routed residual：总 active inner width 仍为 768，既不增加 FLOPs，也不引入基于结果精调的 scale 超参数；它用 H05 的充足收益余量换取更强的 route decision power。旧完整-transition、互斥完整 FFN 与 2:1 residual 结构及其断言均不保留兼容入口。

等分结构随后在原 screen 数据上正常完成 2,400 updates：heldout shared/mixed 为 `0.43750/0.57617`，增益 `+0.13867`，95% CI `[0.10352, 0.17285]`；numeric/relation 分别为 `+0.26172/+0.01563`，无 family regression。`force_route1` 的 answer drop 为 `0.13672`，超过 H06 的 `0.10`，shared no-op、两臂 active FLOPs、cache、架构与 strip/reload 也全部通过。因此 `artifacts/v2-r1r/p1-h1-nonformal-balanced-routed-screen-20260817-1/` 的方向 Gate 只授权一次全新、正式全预算 4,000-update calibration；它仍不冻结阈值、不资格化 H1，也不授权 F1。

在读取 calibration 结果之前，阈值派生规则已冻结：H04 的 supported/heldout answer/trace 取 arm、macro 与逐 family 中的最差注册 cell；H07 metamorphic 取 aggregate、七 transform、两 family 的 relation-consistency/transformed-exact 最小值；source 取 zero/shuffle 各自 decision/trace 较大 drop 后的较小值，recurrence 取 decision/trace 较大 drop；route floor 同时覆盖 heldout route accuracy 和全部 metamorphic route cells。competence、source/recurrence 与 metamorphic 的固定 margin 都为 `0.05`，route accuracy margin 为 `0.01`，减 margin 后统一向下取整到 `0.01`。任一导出 floor 不大于零则 calibration FAIL。H05 primary `0.05`、family regression `0.02`、H06 wrong-route causal `0.10` 与 conditional-write necessity `0.05` 是 claim Gate，绝不由 calibration 导出或降低。只有 calibration 自身再次通过 H05/H06、完整预算/身份/非资格声明和该机械派生检查，才授权合同 freeze audit；仍不等于 H1 PASS。

预登记 calibration 随后在新 seed 上正常跑满正式全预算 4,000 updates，但没有复现 screen：heldout shared/mixed 为 `0.51367/0.51270`，overall gain `-0.00098`，95% CI `[-0.03613,0.03516]`；numeric 为 `+0.03125`，relation 为 `-0.03320`，违反 family regression；最大 route effect 只有 `0.03845<0.10`。fresh cache、架构、matched active FLOPs、shared no-op、source/recurrence causality 与 strip/reload 均通过，所以这不是运行故障。screen 同时更换 seed 并从 2,400 增至 4,000 updates，不能把差异纯归因于 seed；但 calibration 两臂到 4,000 时 train answer loss 都低于 `0.001` 且 heldout 几乎相同，说明早期 mixed 优势至少包含 shared 收敛较慢的暂态，zero-output typed specialization/route decision power 并未在注册最终预算稳定成立。`artifacts/v2-r1r/p1-h1-nonformal-balanced-routed-calibration-20260817-1/` 因而是 `authorizes=nothing` 的非正式失败证据；机械产生的 secondary-floor proposal 无效且不得冻结。当前等分 zero-output routed residual 方向关闭，正式 roots 保持未创建。

post-failure 机制复核进一步发现旧 H06 的盲点：screen 虽能让 wrong-route 下降 `0.13672`，但关闭整条 FFN 写入时 answer drop 约为 `-0.00195`；calibration 关闭 FFN 也只有 answer/trace `0.02441/0.05048`。因此旧 Gate 只证明某个错误 expert 可以有害，没有证明正确条件分支是正常推理所必需；参数分化也不能替代功能性消费。当前合同据此直接删除 zero-output full residual experts，切换为 6.1–6.2 的“共享 nonlinear features + routed final state-write projection”，不保留旧模块或干预名的兼容层。

新方向在读取任何结果前固定：screen data/model 为 `2026081761/2026081762`，root 为 `artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/`，从第一步起跑满 `4,000` updates，不再用 2,400-update 暂态筛选。除 H05 与 wrong-route `>=0.10` 外，`disable_routed_projection` 必须在保留公共 attention/FFN 与共享 feature trunk 的条件下令 decision 或 trace 至少下降 `0.05`。只有三者全通过，才允许一次 `2026081763/2026081764` calibration，目标 root 固定为 `artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-calibration-20260817-1/`；任一失败即封存并停止，不能换 seed、缩短预算、降低阈值或启动 formal。

该 screen 随后以注册 package identity `A52C5222…486DA4` 正常完成：两臂各 4,000 updates，active FLOPs 同为 `12,250,611,712`；fresh cache、架构、初始化、schedule、shared no-op、strip/reload、source 与 recurrence 因果检查均通过。heldout shared/mixed answer macro 为 `0.55469/0.54395`，overall gain `-0.01074`、95% CI `[-0.04199,0.01953]`，numeric 回退 `-0.03516`。最大 wrong-route effect 只有 `0.00281`，单独 `disable_routed_projection` 的最大效应只有 `0.00781`；H05、wrong-route H06 与 conditional-write necessity 三门同时失败。机器 Gate 为 `passed=false`、`authorizes=nothing`。这将失败定位为 route-selected projection 未形成正常路径的必要计算，而不是 cache、算力不匹配、source blindness、recurrence bypass 或进程故障。按预登记规则，当前 factorized 方向关闭，calibration 与 formal roots 保持不存在，secondary floors 与合同 hashes 不得冻结。人类可读复核见 `artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/SCREEN_REVIEW.md`。

## 7. 训练合同

每个 seed 对两个 arm 使用同一 train data、同一 cache、同一 4000-step batch index ledger 与相同 common initialization：

```text
dtype              FP32 trainable core; FP16 frozen cache
optimizer          AdamW
batch              32
updates            4000 exactly
early stopping     forbidden
selection          final update only
gradient clipping  1.0
loss               answer CE + trace CE + identical content-route CE
```

正式 development 先在独立 smoke data/model seed `2026081891/2026081892` 上做 32-record overfit/gradient/strip smoke；两类各 16 条、batch 32、800 updates，最终 answer/trace/route floor 分别为 `0.95/0.90/0.99`。它只决定通路是否可运行，不提供正式指标，也不作为任何正式 checkpoint 初始化。任一 smoke Gate 失败即封存并停止；通过后按 seed 顺序串行训练 S/M，不并行常驻多个模型或 cache 副本。

## 8. H01–H08

### H01：历史、合同与身份

独立复算 NR1 preflight/qualification 两根 seal、status 与 `p1_h1_design_authorized=true`；确认 V8L 仍为 sealed FAIL，未出现 F1/v9/P2 successor。H1 两份合同哈希、source snapshot、Git identity、fixed-root freshness、single-use transport 与唯一 CLI 全通过。H1 新 source identity 不要求等于 NR1 旧 identity；要求的是 NR1 snapshot 自洽且 H1 start/after/snapshot identity 三者相同。

### H02：fresh data、cache 与 leakage

全部 seed/split fingerprint 零重叠；与历史登记 fingerprint 零重叠；range/topology/template audit 符合 3.2；source/cache forbidden scan 零命中；Qwen/revision/hash/finiteness/no-truncation 全通过；S/M record ids、hidden、mask、tokens、atoms 与 batch ledger 逐值相同。

### H03：架构与无旁路

AST、参数名、forward signature 与 runtime intervention 共同证明：模型不 import/call oracle/simulator；route 从经过公共 attention 的 content latent 预测；公共 FFN 与共享 nonlinear feature trunk 对所有记录执行，M 只稀疏选择最终 state-write projection；answer 只读 `H_T`；禁止 raw-source/sidecar/trace bypass；trace head 可物理删除且剥离 reload 等价；公共 attention/FFN/feature trunk 参数初始化一致；S route intervention 严格 no-op；M 未选 projection 不执行且无梯度；两臂都只执行一个 `H=384 -> D=256` projection，完整 active parameter/FLOP ledger 相同。

### H04：两臂可训练性 floor

三 seed、两 arm、两类分别报告 supported answer/trace 与 heldout answer/trace。最终阈值只按 6.3 的独立 calibration 最差 cell、固定 margin 与向下量化规则冻结，formal 后不得修改。任何注册 arm/seed/family floor 失败都使 H1 FAIL，不能靠 macro 平均掩盖。

### H05：typed primary benefit

primary endpoint 是 heldout numeric/relation macro decision exact 的逐 record paired gain。PASS 至少要求 M-S 绝对增益 `>=0.05`、预注册 paired-bootstrap 95% CI 下界 `>0`、三个 seed 方向一致，且任一 family 不得下降超过 `0.02`。supported 或训练内收益不能替代 heldout endpoint；两臂都高但无 typed gain 仍判 FAIL。

### H06：route 因果性

M 在 heldout 上执行 predicted-route flip、projection swap、force-route0/1 与 `disable_routed_projection`。四个 wrong-route intervention 中至少一个必须使 decision 或 trace 下降 `>=0.10`；单独关闭 routed projection、同时保留公共 FFN 与共享 nonlinear trunk，还必须使 decision 或 trace 下降 `>=0.05`，证明正确条件写入本身不可绕过。S 的四个 route control 必须在 logits、`H_T`、全 trajectory、trace 与 answer 上逐值不变。normal route accuracy、两 projection load、跨 seed route orientation，以及七类 surface/handle/choice/clause/query/cancelling/transitive 变形的逐 transform、逐 family route consistency 与 transformed route exact 均使用冻结 route floor；固定第一 projection、position shortcut、route collapse、错误 projection 仅具破坏性或只有额外参数但无正常路径 causal power 均失败。

### H07：内容与状态忠实性

causal/metamorphic ledger 覆盖 handle rename、clause/choice/query permutation、cancelling pair、transitive redundancy、surface paraphrase、same-answer/different-trajectory、zero/shuffled source 与 disable recurrence。等义变形应保持答案，改义 choice 应随新 target 改变；aggregate、七个 transform 与两个 family 的 relation consistency 和 transformed exact 必须分别达到冻结 floor，不能用 macro 掩盖单项失败。zero 与 shuffle source 两项都必须分别造成 decision 或 trace 的冻结幅度下降，disable recurrence 也必须独立达到冻结下降，从而证明输出真实消费 source 与 recurrence。

### H08：预算、运行与工件完整性

三 seed × 两 arm 必须恰好 4000 updates；tokens、atoms、batch indices、active FLOPs 相同；总参数、active 参数、VRAM、吞吐、wall time、router/load 单独记录。单一 launcher/process ancestry、stdout/stderr binding、source/Git stability、later-root absence、即时 evidence seal 与 seal replay 全通过。任一 Gate FAIL 原样停止，不建立 F1 root。

## 9. 结果状态机

H1 PASS 的唯一合法状态：

```text
status=PASS_P1_H1_MIXED_CORE_DEVELOPMENT
p1_f1_design_authorized=true
p1_completed=false
p2_eligible=false
p2_started=false
```

H1 FAIL 的唯一合法状态：

```text
status=FAIL_P1_H1_MIXED_CORE_DEVELOPMENT
p1_f1_design_authorized=false
p1_completed=false
p2_eligible=false
p2_started=false
```

PASS 后也必须先由主设计层复算 seals、paired endpoint、route causality 与 source/compute identity，另立 F1 design/execution 合同；不得在同一进程、同一命令或同一 root 中自动启动 F1。
