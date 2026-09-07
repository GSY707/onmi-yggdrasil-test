# V2-A Closure C1 单 seed 架构资格合同

状态：历史冻结合同；cache qualification 已 sealed PASS，唯一 C1 formal 已在 G007 FAIL-stop，`authorizes="nothing"`

执行后说明：本文件以下阈值与顺序保持启动前冻结版本不变。正式结果见 `docs/v2-a-closure-c1-result-review.md`；不得据此修改旧 root、重跑、换 seed 或创建未授权 successor。

cache 正式 root 的 `result.json` / `evidence-seal.json` SHA-256 已冻结为 `A058687125FDC1315871733DD1499036A040636864665793B8D32B8C9D758D4A` / `68CA34BB18F889405F875BF54B7CA2E55C3E9E549DE5DFE1C63C4F47E2B4DC34`；K001–K008 全 PASS，seal replay `867/867`。这只授权本合同的 C1 single-seed launcher，不构成模型或 V2-A PASS。

日期：2026-08-25

## 1. 本轮只回答什么

C0R 已资格化新的 ERE/CPS bank、strict CT1 compact trace 与四臂公平输入合同，但没有训练模型。C1 只检验一个候选介质：一个 fresh、`K=8`、共享 dense recurrence，能否仅从完整 `source_text` 派生的冻结 Qwen hidden 中形成可泛化、可因果干预的推理状态。

C1 不是 direct/text-CoT/K=1/K=8 Pareto，也不比较介质优劣。单 seed PASS 只授权另外两个完全相同合同的 fresh C1 seed；三 seed 全部 PASS 后才允许另立 C2。C1 不授权 C2、C3、V2-B、V2-C，也不恢复 H1/WD、routed projection、write-delete 或“先写后删”。

唯一输入 prerequisite 为：

- C0R data/trace root：`artifacts/v2-a/closure-c0r-data-trace-20260824-1/`；
- C0R data result SHA-256：`B2502F66CB5D9ECEB2CA0547CB280CC5346D1A24D2F8901F94E617E1B258FFF9`；
- C0R data seal SHA-256：`5AFB37AB5950D04298FED4D2DAFA78102260A728101A5EDE4972D1EC0EEBB570`；
- C0R readiness root：`artifacts/v2-a/closure-c0r-20260824-1/`；
- C0R readiness result SHA-256：`391C846D2150FD27D2016A79A9C360F3CB4E1455EC71240720F772CF58A28D5E`；
- C0R readiness seal SHA-256：`161FBB367DEE40D18CDD23521E7DA9EBA3257EE8BF57C1FA194886E43D140BA4`；
- frozen encoder：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`。
- frozen runtime：项目 `.venv` 的 `transformers==5.13.1`；global Python 的旧版 Transformers 不属于本合同执行环境，loader 必须 fail-closed。

## 2. 目标架构

正式 public forward 固定为：

```text
source_text
  -> frozen Qwen full-token hidden
  -> learned full-text Boundary
  -> K=8 latent workspace H0
  -> one source-closed shared dense core, recurrently reused for T=10
  -> answer logits from H10 only
```

`T=10` 是对所有记录相同的公开常量，覆盖 C0R bank 的最大 trace/program depth。模型不读取每条记录的 `reasoning_budget`，也不实现从 metadata 推导的 active-step mask。

Boundary 先对完整 Qwen token hidden 做 learned projection，再由八个匿名 slot 通过两层 cross-attention/self-attention/FFN 压缩为 `H0`。进入 `H0` 后，shared recurrent core 只能读取上一时刻 latent state；它在任何一步都不能重新读取 source hidden。这一 source-closed 约束使 recurrence intervention 测量“状态传递是否必要”，而不是测量模型能否从全文重新计算。

core 由两层相同结构的 dense self-attention/SwiGLU block 组成；这两层参数在十个时间步完全复用。没有 route、expert、family branch、operator branch、task embedding、离散执行器、source span、role tensor、answer mask 或旧 checkpoint。

八个 slot 使用一个共享 learned seed 与固定 slot Fourier identity。所有 transition 参数对 slot 轴置换等变，answer readout 对 slot 轴置换不变。slot identity 只承担匿名地址，不携带任务、答案或 teacher state。

## 3. CT1 训练信用与部署边界

CT1 是训练期 dense credit，不是模型输入，也不是部署时的第二个模型。

训练期 trace probe 只读取 `H1...H10`、固定 token position 与由 canonical CT1 字符 offset 确定的 step alignment。它不得读取 source hidden、source token、AST、family、label mapping、teacher previous token、teacher prefix、claim、answer index或任何旧 target bank。probe 是单层 latent cross-attention + FFN + token head，不含自回归 self-attention 或独立 recurrent state，因而不能在 latent 之外再次执行任务。

每条记录的监督字节严格等于 C0R `format_compact_trace(record)`；使用同一 pinned Qwen tokenizer、`add_special_tokens=false` 与完整 loss mask。每个 CT1 item 的 token 对齐到相同 index 的 recurrent state；固定前缀对齐 `H1`，固定后缀与 `Answer:` 对齐 `H10`。offset 跨边界 token 使用 token midpoint 所属区间，规则在 materialization 前冻结。

probe 使用一个输入可构造的受限词表：

1. C0R 全 bank 的 public `source_text` tokenizer ID 并集；
2. CT1 v1 固定语法 token ID：`[58,92,487,1089,1123,1143,1288,1293,1666,1797,1802,2129,2456,2685,3147,4851,4891,5046,5702,7664,8631,8783,11534,14522,15050,15666,16352,17709,20691,22357,22642,25312,31928,32817,34764,40775,45404,46793,55558,80620,86451,93482]`。

这 42 个 ID 只编码已资格化 CT1 schema 的固定标点、字段和枚举字面量；不包含逐记录 target 值。冻结统计为：Qwen tokenizer 总词表 `248,077`，public source 并集 `5,555`，最终 trace 词表 `5,597`。materialization 必须在不扩词表的条件下证明 26,624 条 CT1 token 覆盖率为 `1.0`；任何 OOV 直接阻断。

probe 不进入 answer head。选出 checkpoint 后，先在固定 validation 子集完成 trace-credit 资格测试，再物理删除 probe，重新实例化 deployment model 并 strict reload。正式 validation、OOD、causal 与 intervention 全部只能使用 stripped deployment model。stripped artifact 中不得出现 trace probe 参数、optimizer state、trace logits或 target buffer。

## 4. 数据、cache 与禁止项

C1 使用 C0R 的全部 26,624 条记录：ERE/CPS train 各 4,096；validation 各 1,536；每族四个 OOD cell 各 1,536；causal_pairs 各 1,536，即 768 对。

fresh cache 只保存由 public `source_text` 得到的 FP16 full-token final hidden、attention mask、token length、example ID、source byte hash、shard identity 与 encoder/tokenizer identity。最大 source token 长度固定 `1024`，禁止 truncation。cache 不保存 input IDs、AST、family embedding、answer、choice mask、reasoning budget、trace、claim、role/span mask 或旧 latent。

family/split/pair metadata 只允许用于 balanced schedule、分层评测、paired bootstrap 与报告；不得进入模型 forward。正式 answer logits 固定为 A-I 九类 raw logits，不使用 `valid_choice_mask`。合法 label 与 semantic answer 只在 evaluator 中解析。

`source_mask` 是唯一允许进入 public forward 的 mask：它只表示 public source tokenizer padding，和 cache 中对应 hidden 构成同一条 source row；它不编码任务、程序长度、选择合法性或答案。`within-family shuffled-hidden` 必须同时置换整条 hidden 与其 padding mask，但保持 offline target/metadata 不动；zero-hidden 保持原 padding mask。任何 `valid_choice_mask`、operation/role/span mask 仍属硬禁止项。

以下输入或复用一律阻断：

- P0-M、P1、H1/H1-WD、A1.9、A1.20D 或其他旧 hidden cache/checkpoint/optimizer；
- route ID、projection expert、FFN-MoE、task/family branch、write-delete 路径；
- `reasoning_budget`、`valid_choice_mask`、program length、operation mask；
- AST、teacher trace、training claims、dense state、margin/VJP/Fisher target；
- Qwen LM head、input-token bypass、source-to-answer skip connection；
- heldout/OOD/causal 指标参与优化、超参数或 checkpoint 选择。

## 5. 固定训练合同

### 5.1 Overfit32 正控制

正式 single-seed launcher 先从 train 中按 family 与 trace depth 预注册选择 32 条，使用独立 fresh initialization。batch size `8`，最多 `4,000` updates，每 `100` updates 评估；首次同时达到以下条件即停止：

- answer exact `32/32`；
- 全 token trace accuracy `>=0.99`；
- content-token accuracy `>=0.99`；
- correct-owner trace NLL 至少比 within-family cyclic wrong-owner 低 `0.50` nat/token；
- trace probe 物理剥离后 answer prediction `32/32` 保持且 logits FP32 max-abs-diff `<=1e-6`。

Overfit32 只证明通路可训练。任一条件未在 4,000 updates 内达到，C1 原样 FAIL-stop，不进入 primary training。

### 5.2 Primary single seed

- model seed：`2026082501`；data/order seed：`2026082502`；bootstrap seed：`2026082503`；
- `K=8`，latent width `512`，Boundary depth `2`，core depth `2`，attention heads `8`，FFN width `2048`，`T=10`；
- family-balanced batch size `8`，每 batch ERE/CPS 各 `4`；
- 六个完整无放回 epoch，共 `6,144` optimizer updates；
- trace 每次暴露每记录最多 `64` 个连续 token；chunk 随 epoch 循环，episode/order/trace exposure ledger 全部保存；
- AdamW，weight decay `0.01`，gradient clip `1.0`；Boundary LR `1e-4`，core LR `2e-4`，answer/trace heads LR `3e-4`；
- 线性 warmup `256` updates，之后 cosine decay 到零；
- updates `1...2,048`：`0.5 * answer CE + 1.0 * trace CE`；updates `2,049...6,144`：`1.0 * answer CE + 0.5 * trace CE`；不训练 wrong-owner contrast loss；
- BF16 autocast，FP32 optimizer，finite/gradient/peak-VRAM/throughput ledger；
- 每 `512` updates 在固定 validation trace 子集上评估并保存候选 checkpoint；不得 early stop primary budget。

checkpoint 选择只使用 validation trace：在 updates `2,048...6,144` 的候选中，最小化 `max(ERE trace NLL, CPS trace NLL)`；数值相同取最早 checkpoint。validation answer 只记录，不参与选择。OOD 与 causal 在 checkpoint 固定、probe 剥离后才首次读取。

## 6. 机器 Gate

所有 answer 指标按 family/cell 独立报告，不允许 aggregate 掩盖最差 cell。普通比例同时报告 point estimate 与 Wilson 95% CI。paired effect 按 record 或 causal pair 做 `10,000` 次 fixed-seed cluster bootstrap；Gate 是全部条件的交集，不用单个显著结果替代失败条件。

| Gate | 冻结要求 |
|---|---|
| G001 identity/no-bypass | C0R result/seal、cache result/seal、Qwen revision、source hashes、fresh initialization、root/lease、禁止字段与参数图全部通过；正式启动前 later-root 不存在。 |
| G002 overfit32 | 第 5.1 节全部条件通过；否则不开始 primary training。 |
| G003 train/select/strip | 恰好 6,144 updates、全 finite、固定 schedule/exposure ledger 完整；只按冻结 trace rule 选 checkpoint；probe 物理删除并 strict reload。 |
| G004 validation behavior | stripped model 的 ERE 与 CPS validation answer point 各 `>=0.75`，Wilson lower 各 `>=0.70`。 |
| G005 OOD behavior | ERE composition/entity/length/language 与 CPS composition/distractor/horizon/language 八个 cell 各自 point `>=0.65`，Wilson lower `>=0.62`。 |
| G006 causal behavior | 每族 causal base 与 flip answer 各 point `>=0.70`、Wilson lower `>=0.66`；每族 pair-both-correct 与 simulator-semantic-flip 各 point `>=0.60`、pair bootstrap lower `>=0.56`。 |
| G007 trace credit | probe 剥离前固定 validation 256/族上，全 token accuracy 每族 `>=0.80`、非固定语法 content-token accuracy每族 `>=0.60`；correct-owner 相对 within-family cyclic wrong-owner 的 NLL 改善 point `>=0.15` nat/token、record-bootstrap lower `>=0.05`。point 固定为该族全部有效 trace token 的 pooled nat/token；bootstrap 单位仍是 record，另报 record-mean 仅作诊断。这只资格化 training credit，不冒充部署 trace 输出。 |
| G008 hidden necessity | stripped validation 上 zero-hidden 与 within-family shuffled-hidden：每族 normal-minus-intervention paired drop 的 bootstrap lower 均 `>=0.20`，intervention accuracy Wilson upper 均 `<=0.35`。 |
| G009 recurrence necessity | stripped validation 上 `H0/no-core` 每族 paired drop lower `>=0.15`；在 step 5 做 within-family latent-state shuffle 后继续共享 core，每族 paired drop lower `>=0.05`。core 不可重新读取 source。 |
| G010 slot/strip integrity | 固定全局 slot permutation 后 prediction invariance `1.0` 且 FP32 logits max-abs-diff `<=1e-5`；strip 前 deployment-mode 与 strict-reloaded stripped model prediction invariance `1.0`、max-abs-diff `<=1e-6`；参数图零 trace-probe 项。 |
| G011 accounting/seal | examples/tokens/exposures、Qwen cache time/bytes、trainable/active params、updates、latent transitions、wall time、peak VRAM、latency、checkpoint/strip hashes、source stability、artifact seal 与 replay audit完整。 |

`content-token` 定义为 target tokenizer ID 不属于本合同冻结的 42 个 CT1 固定语法 ID；该分类在看到模型结果前固定。

## 7. 顺序与停止树

基础设施先使用独立 single-use identity 构建并资格化 full C0R Qwen/trace cache：

```text
V2-A-CLOSURE-C1-CACHE-20260825-1
artifacts/v2-a/closure-c1-cache-20260825-1/
```

cache Gate 未全部通过时，不得启动 C1 training root。cache PASS 只授权本合同的 single-seed launcher。

cache qualification 的八项机器 Gate 固定为：

| Gate | 冻结要求 |
|---|---|
| K001 prerequisite/source identity | 两个 C0R root 的 result/seal 精确命中本合同 pin，C1 source snapshot 完整且构建前后不变。 |
| K002 bank identity | 14 个 cell、26,624 个唯一 example ID、各 cell 数量、public source byte hash 与 immutable C0R JSONL 完全一致。 |
| K003 tokenizer/length | pinned fast tokenizer、`add_special_tokens=false`；全 bank source 长度 `1..1024`，无截断；最长八条的真实 Qwen batch-8 前检 finite。 |
| K004 trace materialization | source lexicon `5,555`、source+grammar lexicon `5,597`、42 个冻结 grammar ID、26,624 条 CT1 target OOV=0；local-ID、step/global/local position 与 grammar mask 全部可复验。 |
| K005 hidden cache | fresh pinned Qwen text-only final hidden，FP16、width `2,048`、全 token、26,624 条全覆盖且全部 finite；Qwen 参数全冻结。 |
| K006 payload isolation | hidden shard 只含 packed hidden、attention mask、length/offset、example ID/source hash 与 encoder/tokenizer identity；禁载荷、旧 cache/checkpoint/optimizer 数均为零。 |
| K007 indexed replay | manifest/hash/shape/token count 全部复验；固定跨 cell 样本按 example ID 随机 mmap 读取，与 shard 顺序读取逐元素一致，mask/length 正确。 |
| K008 accounting/seal | cache bytes/time/shards/tokens、trace bank hash、source identity、result/run-state 与 evidence seal 完整；训练、optimizer step 与 learner model write 均为零。 |

正式 root 前必须先运行并封存独立 tmp identity 的 read-only/小样本 preflight；preflight 最多真实编码全 bank 最长的八条 source，不得生成正式大 cache、训练 learner 或消费正式 lease。正式 cache launcher 在领取 lease 前必须复验 preflight PASS、当前 source identity、零训练/零 optimizer step/零 model write 与 evidence seal；缺失或过期时 before-mutation REFUSE。preflight PASS 不替代 K001–K008。

single-seed identity 与 root 固定为：

```text
V2-A-CLOSURE-C1-SINGLE-SEED-20260825-1
artifacts/v2-a/closure-c1-single-seed-20260825-1/
artifacts/v2-a/closure-c1-single-seed-20260825-1.preflight-lease.jsonl
```

cache PASS pins 写回冻结合同后，还必须运行一次独立 C1 forward/backward preflight。它只在最长八条真实 cache row 上分别覆盖 trace chunk 64 与 512，验证完整参数梯度、显存可行性、probe 可剥离和 source stability；不得执行 optimizer step。正式 C1 launcher 同样在领取 lease 前复验该 preflight 的 sealed PASS 与当前 source identity，禁止绕过。

正式顺序固定：

```text
preflight/G001
  -> overfit32/G002
  -> primary train + trace-only checkpoint selection
  -> trace credit/G007
  -> physical strip + strict reload/G003（G010 的 strip 子条件）
  -> validation/G004
  -> OOD/G005
  -> causal/G006
  -> hidden/G008；仅 PASS 后 recurrence/G009
  -> slot permutation + final strip integrity/G010
  -> accounting, replay and seal/G011
```

任一阶段 FAIL，立即写明尚未运行的后续阶段、封存已产生的证据并自然退出；不得换 seed、延长 updates、降低阈值、选择另一个 checkpoint、删除 root 重跑或创建未授权 successor。输入缺失、seal 不可复验、异常退出或证据不完整为 `INCOMPLETE`/`CRASH`，不改写为模型 FAIL 或 PASS。

终态只有：

- `PASS_V2_A_C1_SINGLE_SEED_ELIGIBILITY`：G001–G011 全 true；`authorizes="two additional fresh C1 seeds only"`；
- `FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY`：输入完整但任一能力/机制 Gate false；`authorizes="nothing"`；
- `INCOMPLETE_V2_A_C1_SINGLE_SEED_ELIGIBILITY` 或 `CRASH_V2_A_C1_SINGLE_SEED_ELIGIBILITY`：证据链不完整；`authorizes="nothing"`；
- root/lease 已存在时 `REFUSE_V2_A_C1_SINGLE_USE`，before-mutation 退出。

即使单 seed PASS，也必须保持 `c2_authorized=false`、`v2a_passed=false`。只有另立合同且三个 fresh C1 seed 全部通过后，才允许开始 C2 matched Pareto。
