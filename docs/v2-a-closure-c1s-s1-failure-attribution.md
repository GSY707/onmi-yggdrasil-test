# V2-A Closure C1S SRW：S1 失败归因诊断合同

状态：已唯一消费；preflight sealed PASS，diagnosis 在 D004 fail-closed 并 sealed CRASH，禁止重跑

日期：2026-08-31

诊断身份：`V2-A-CLOSURE-C1S-SRW-S1-FAILURE-ATTRIBUTION-20260831-1`

终态复盘：`docs/v2-a-closure-c1s-s1-failure-attribution-result-review.md`

固定实现真源：
`src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/contract.py`

## 1. 这轮诊断要回答什么

C1S SRW 的 S1 已经把 32 条 Overfit32 记录的答案全部答对，但没有通过“答案必须由 recurrence 和多个地址共同产生”的机制资格。已知证据中，no-core 仍有 `28/32` 答案正确，至少两个有效内容贡献者只有 `7/32`，而动态状态的若干族内 balanced accuracy 低于 `0.95`。本轮不再猜测“多训练一点”或“把某个 loss 调大”，而是把失败拆成可独立证伪的五层：

1. 训练目标是否真的要求了 Gate 要求的行为；
2. Boundary/H0 是否已经能直接回答，以及 core 是否只是提高置信度；
3. 任务本身是否真的需要多地址，模型的多地址影响是否只是单槽复制；
4. temporal target 是没有形成、读出不足，还是 target 本身不足；
5. 按 ERE/CPS 分族后，以上证据能否给出互不混淆的归因。

这是一轮 diagnosis，不是模型修复实验。诊断的目的，是决定下一轮研究应该改 objective、改数据/任务合同、改 readout，还是承认当前 latent 轨迹没有形成。直接训练一个新模型会同时改变这些变量，无法告诉我们 S1 到底为何失败。

## 2. 身份、输入与不可越过的边界

固定输出目录为
`artifacts/v2-a/closure-c1s-srw-s1-failure-attribution-20260831-1/`，preflight 目录为
`tmp/v2-a-closure-c1s-srw-s1-failure-attribution-preflight-20260831-1/`。
两者只属于本身份；目录或 sibling lease 已存在时，在任何写入前拒绝。

只读 pin 包括：

- S1 root `artifacts/v2-a/closure-c1s-srw-s1-overfit32-20260829-1/` 的 result、evidence seal、endpoint、source identity；
- S1 preflight 的 target bank gzip 与 split ledger；
- C1 hidden cache 与 C0R data/trace source identity。

合同代码内保存完整 SHA-256。尤其不能把 preflight 的 target bank 当成可以重新生成的输入，也不能用另一个 endpoint、另一个 seed 或未封存的 target bank 替代 pinned evidence。D000 必须先重放 result/seal/tree/source identity，并检查旧 S1 的终态确实是 `FAIL_V2_A_C1S_S1_QUALIFICATION`、`authorizes=nothing`。任一 hash、schema、source identity 或旧 root 内容漂移，诊断立即 `INCONCLUSIVE`/STOP。

D002–D004 的端点行为只允许 `model.eval()` 和 `torch.no_grad()`。D001 为复现 sealed optimizer 当时看到的 straight-through backward，单独允许 `model.train()`/`heads.train()` mode 下的内存 `torch.autograd.grad`；这不是行为评分，也不创建 optimizer、不调用 `backward()`/`optimizer.step()`、不写参数的 `.grad`。两种模式必须分别记账，禁止把 train-mode 路由输出混入自然端点指标。preflight 封存的 source identity 必须在领取 diagnosis lease 前与当前源码逐文件重算结果精确相等；诊断结束还要再次重放 C1 cache、C0R source、S1/S1-preflight seal，防止长时只读运行中的输入漂移。全程不写模型、checkpoint、旧 root 或训练数据。终态无论如何均为 `authorizes=nothing`，且 `v2a_passed=false`。本合同不授权 S2、S3、single-seed formal、任何新训练、C2、V2-A PASS、V2-B 或 V2-C。

## 3. 共同测量规则

主判定按 `ERE`、`CPS` 分族完成；overall 只能作为摘要，不能用平均数掩盖一族的失败。所有答案因果效果使用同一个 gauge-invariant margin：

```text
m = logit(correct) - logsumexp(logits(wrong))
effect = m(base) - m(intervention)
```

原始 correct-logit drop、raw cosine、slot 数量、owner 直方图和 answer exact 都是辅助描述，不能单独构成机制证据。所有干预先做对象、slot、mask、shape、finite 和语义合法性检查；无法证明干预只改变注册因素时，结果为 invalid 并 fail-closed，不能填成零效应。

bootstrap 固定 seed `2026083101`、10,000 次、按 record cluster 重采样；阈值、候选 rank/lambda、抽样哈希和 split 在 preflight 前封存，看到结果后不得改变。

## 4. D001：objective 与 Gate 是否对齐

这一层必须先于架构归因。S1 的 no-core 训练项只要求

```text
relu(0.5 - (full_margin - no_core_margin))
```

因此它在 margin drop 达到 `0.5` 后即可为零；这不能推出 no-core accuracy 必须低于 `0.25`。同理，functional-K Gate 没有直接训练项；一个间接受 operation/payload loss 影响的模型，不会因此自动满足“同记录至少两个独立 contributor”。state loss 是全局 masked micro-BCE，而 Gate 是按族、按 feature 的 balanced accuracy `>=0.95`；微平均 loss 低不能证明稀有 feature 的宏平均 BA 达标。

D001 固定报告：Gate→loss 的静态映射、端点各 loss 的饱和状态、每项 hinge active fraction、loss/component 梯度 norm 与 shared-core cosine、macro/micro 与稀有 feature 的权重差异，以及最终失败 Gate 是否有直接有效的训练压力。若失败 Gate 无对应 loss，或对应 loss 已饱和但 Gate 仍失败，登记 `OBJECTIVE_GATE_MISMATCH_CONFIRMED`；这只是目标合同归因，不等于模型已被证明有能力。

## 5. D002：Boundary/H0、core 与 residual 路径

令 `B=(P0,A,Q,O,S)`，自然 rollout 为 `hT=R(P0,O,S,A)`。对同一 endpoint 测量：

- `full`：完整 SRW rollout 后的答案 margin；
- `H0=f(P0,A,Q)`：只用 Boundary 初始 payload 与 query 的直接读出；
- `core-only=f(R(mean_present(P0),O,S,A),Q)`：保留公开 boundary 的 operation/source/query 条件，把 present payload 替成行均值后运行 core；
- `H0-relevant`：只把 query-owner 的 pre-recurrence payload 替换为“其他 present 槽”的 leave-one-out mean；它的因果量严格定义为 `margin(H0)-margin(H0-relevant)`，不得把 owner 自身混入替换均值，也不得用 `margin(full)-margin(H0-relevant)` 代替；
- 每个 `h0..h10` 的 margin 曲线；
- `final-delta=f(hT-P0,A,Q)`：只作 residual-removed architectural counterfactual，绝不称为自然 rollout。

同时保留 zero payload、只清零 operation state（route 权重与 active mask 保留）、zero query、boundary operation 顺序逆转和 answer-permutation metric-null。后两者是注册的操作顺序/评分对照，不得改称已经形成的 trajectory state permutation。判定必须分族：

- H0 高而 full 更高：`H0_DIRECT` 或 `MIXED`，只能说 core enhancement/residual retention，不能说 recurrence necessary；
- H0 低、full 高、core-only 高：支持 `CORE_REQUIRED` 的 state computation 路径；
- H0 低、full 高、core-only 低且 residual 保留有效、delta-only 崩溃：支持 `CORE_RESIDUAL`；
- 路径证据互相冲突或控制无效：`INCONCLUSIVE`。

预注册参考门为 H0 point `0.75` 且 Wilson lower `0.65`、H0-required point 上限 `0.25`、相关 margin-drop bootstrap lower `0.50`。这些是归因门，不是把失败 S1 改判 PASS。

## 6. D003：任务阶数与模型 functional arity

D003 先不读模型，用 immutable source simulator/AST 分析任务真实需要的依赖阶数。对每条支持记录区分三种概念：

1. `answer_support`：改变对象后答案是否改变；
2. `certificate`：完整答案/状态是否仍可由该对象集合证明；
3. `counterfactual_influence`：合法删除、置零或交换时，目标输出是否按语义改变。

对象子集最多枚举 8 个对象；任何 simulator 重放不一致、对象 mask 非法或 counterfactual 不可解释都 fail-closed。本轮只解释已经消费的 S1 Overfit32（每族 16 条），不借诊断名义读取未来 S2/formal 保留样本，也不把 post-hoc 结果外推为泛化结论。模型侧只统计 baseline 正确且 margin `>0.5` 的记录；每族少于 8 条有效记录时该族为 `INCONCLUSIVE`。

这三个量必须保持独立。`certificate` 只证明注册 schedule 中哪些对象可达 query，不等于答案因果依赖阶数；尤其 ERE 没有合法的对象删除反事实时，B 轴必须为 `INCONCLUSIVE`，不能用 certificate 补成任务阶数。CPS 的 AST 只能来自单独 pin 的 immutable source record，禁止从 target-bank row 回读监督信息。AST candidate index 必须通过 target object 的 `semantic_slot`/`candidate:n` 双射映射到置换后的物理 model slot，不能用物理槽排序冒充 AST 顺序；映射不完整、重复或互相矛盾时整条记录 `INVALID`。NONE、tie、非法 subset 分别计数；baseline 没有唯一 winner 时 answer-support 为 `NOT_ASSESSABLE`。

然后在 frozen endpoint 上测 full、逐槽 pre-recurrence mean-replace、`copy_all`、`global_mean`。逐槽干预必须把被测 content slot 替换为“其他 content 槽”的 leave-one-out mean；不得把被测槽自身或 CPS decision 槽混入均值，非 content/absent slot 保持原值。`copy_all` 只复制 present slot，absent slot 保持原值。`two_contributor` 只统计 `content_mask` 中的槽，CPS decision slot 永不计入；其分母为 `full_margin-H0_margin`，分母非正的记录不准通过夹小常数制造贡献者。只有 task-side answer support/counterfactual influence 证明多对象因果依赖后，模型侧结论才有资格进入 Gate；静态 certificate 只能作为独立旁证。每族还必须有至少 8 条 baseline 与 contributor denominator 均有效的记录。

`SINGLE_SLOT_COPY` 要求 copy-retained point `>=0.80`、Wilson lower `>=0.65`，同时 two-contributor point `<=0.50`；`MULTI_ADDRESS_SUPPORTED` 要求 two-contributor Wilson lower `>=0.65`，且 full-minus-copy margin drop 的 record-cluster bootstrap lower `>=0.50`。`TASK_LOW_ORDER` 只在可评估的答案支持、counterfactual influence 与低阶注册路径一致时成立；证据未评估或相互冲突一律 `INCONCLUSIVE`。复制或 duplicate 只能证明地址/内容读出控制成立，不能单独证明多地址计算。

`CPS.final_winner` 是由答案派生的 detached auxiliary，永远不得进入 D003 的机制 Gate。

## 7. D004：non-answer temporal target、latent motion 与 readout

硬 Gate 只使用不由答案直接构造的动态 target：

- ERE：`touched`、`changed`、`operation_source`、`operation_target`；
- CPS：`processed`、`running_best`。

`CPS.final_winner` 和 `ERE.query_semantic_match` 可以报告，但不是 temporal mechanism 证据。D004a 先审计每个 feature 是否至少有每族 32 个正例、32 个反例、16 个跨 step change，并能从 sealed source AST 重新通过同一个 `materialize_target` 实现生成一致结果；这是 source-to-bank consistency replay，不是独立 semantic oracle，无法发现 materializer 与 bank 共享的实现错误。不足时是 `TARGET_INSUFFICIENT`，不准把模型 BA 低归因于 latent。

通过 target adequacy 后，直接记录 `h0..h10` 的相对 delta、active/inactive target 条件的 motion separation，并做 fixed-channel AUC。为了区分“latent 没形成”和“latent 已形成但 readout 不会读”，另做只在内存中的 cross-fitted ridge decoder：rank 只从预注册 `[8,16,32]` 选择，lambda 只从 `[1e-4,1e-3,1e-2,1e-1,1]` 由 inner fold 选择；不使用 answer、margin、VJP、projection output、target norm 或任何 target 派生量，不调用 optimizer。

time-shuffle、temporal-mean static null 和 target-label shuffle 是必要对照。inner rank/lambda 选择、自然 ridge、三个 null 与 frozen decoder 的主指标全部使用 record-macro balanced accuracy；observation-micro BA 只作旁证，禁止两种单位相减。只有 cross-fitted ridge record-macro BA `>=0.80`、超过最强 record-macro null 至少 `0.05`，而 frozen decoder record-macro BA `<0.80`，才登记 `READOUT_INSUFFICIENT`；若运动、cross-fitted channel AUC、ridge 与 null gap 都低，登记 `LATENT_NOT_FORMED`；target audit 失败登记 `TARGET_INSUFFICIENT`。final_winner 单独失败不得升级为机制结论。

## 8. D005：独立的分族结论

D005 不生成 composite PASS，而是输出三个轴的 family-level 分类，并附 overall 摘要：

- A（路径）：`H0_DIRECT`、`CORE_RESIDUAL`、`CORE_REQUIRED`、`MIXED`、`INCONCLUSIVE`；
- B（阶数）：`TASK_LOW_ORDER`、`SINGLE_SLOT_COPY`、`MULTI_ADDRESS_SUPPORTED`、`INCONCLUSIVE`；
- C（时间状态）：`TARGET_INSUFFICIENT`、`READOUT_INSUFFICIENT`、`LATENT_NOT_FORMED`、`MIXED`、`INCONCLUSIVE`。

任何一轴的结果都不能覆盖另一轴。例如 no-core ERE `16/16` 与 CPS `12/16` 不能用 overall `28/32` 掩盖；functional-K 的 CPS `1/16` 与 ERE `6/16` 也必须分别呈现。诊断完成后仍 `authorizes=nothing`，不得以“找到了原因”自动启动修复训练。

## 9. 终态与停止条件

D000 infrastructure 或任一必要 null malformed：原样 STOP、封存 `INCONCLUSIVE`。任一 D001–D004 发现输入漂移、非法干预、答案派生量误入硬 Gate 或写入 optimizer/model：立即 STOP，不继续追逐假设。

只有 D000–D005 全部完成、每族分类单元齐全、D002 答案置换 control 有效、D003 无结构性 `INVALID` 记录、D004 source replay 与三个 record-macro null 齐全、D001 零 mutation，以及输出树可完整 seal replay，才允许写 `COMPLETE_V2_A_C1S_S1_FAILURE_ATTRIBUTION`。某一科学轴可以诚实地得到 `INCONCLUSIVE`，但注册证据本身不能缺失。这仍然只表示归因诊断完成，不代表 S1 PASS，也不授权任何 successor。若输入或测量不完整，写 `INCOMPLETE_...`；异常写 `CRASH_...`；这两者同样 `authorizes=nothing`。
