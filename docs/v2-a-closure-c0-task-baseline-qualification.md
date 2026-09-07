# V2-A Closure C0：任务与基线资格合同

日期：2026-08-23

身份：`V2-A-CLOSURE-C0-20260823-1`

状态：冻结后执行；C0 只做只读任务/基线资格审计，不训练模型。

## 1. 核心判断

V2-A 当前缺的不是另一个 projection 局部修复，而是一场能直接判定“连续 latent recurrence 是否值得保留”的端到端实验。C0 先回答更基础的问题：现有任务是否真的需要两种不同计算过程，四条比较路径是否能在相同信息与透明成本口径下公平比较，以及历史 V2-A 证据中哪些可以复用。

C0 不要求 direct、text-CoT、K=1、K>1 四条正式结果已经存在；那是后续 C2 matched Pareto 的职责。C0 PASS 只表示任务套件和比较合同可用，并且只授权 C1 的单 seed 实现与资格测试。

## 2. 固定输入与禁止输入

主任务输入固定为已经封存的 V2-R1R P0-D v17：

- `artifacts/v2-r1r/p0d-v17-full-production-20260810-1/`
- evidence seal SHA-256：`453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`
- dataset manifest SHA-256：`E2D2DE707C456F2FF7F3382B7EF8ED50DE57DE81C36497576A894EDF7490AD77`
- Qwen：`Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`
- 数据：ERE/CPS 各 `4096` train、`1536` validation、四类 `1536` OOD 和 `1536` causal-pair records，总计 `26624` records。

训练通路正控制固定为 P0-M v5 assessment，seal SHA-256 为 `17F45EAC8A3DF129B236688D3D1E2B22BD9B634D4098F903142CA42E7FEEF09E`。它只能证明 direct、text-CoT 与 K=8 的 overfit/training path 可运行，不提供 heldout、K 容量或 Pareto 结论。

历史 V2-A summary 只用于证据分级，不作为新模型初始化、teacher、训练样本或 C0 PASS 的替代品。H1、H1-WD、decision-causal、direction-geometry 与其 causal-target bank 全部排除在 C0/C1 活动路线之外；不得把 route、family、common/projection target、margin、VJP 或旧 checkpoint 送入新模型。

## 3. 为什么保留 ERE + CPS

ERE 与 CPS 共享“从完整自然语言中理解 episode-local 临时语义并递归更新状态”，但不是同一代数的表面改写：

- ERE 单轨执行 nonce rule/event，最终查询 attribute 或 relation；
- CPS 并行模拟多个 candidate plan，检查前置条件、资源、延迟后果、最终约束和成本，再选择唯一最优或 NONE。

C0 必须机械确认两族 AST 结构不同、各自具有独立 composition/length-or-horizon/language/规模 OOD、causal pair、零跨 split fingerprint overlap、弱捷径不过线、teacher/claim 可验证且正负平衡。除 v17 原有七类 shortcut 外，新增 source-visible pattern Gate：每个含至少 `128` 条 ERE relation query 的 split，TRUE/FALSE 各自至少 `64` 条，dominant semantic answer mass 不高于 `0.80`；同时直接运行只读 `source_text`、从公开 legend 选择 TRUE 的条件 oracle，其单侧校正 Wilson upper 不得超过条件随机率 `+0.10`。旧 direction target bank 的 train/heldout zero-target shift 与这里无关，因为 C0 直接审计原始任务记录，不使用派生 write target。

## 4. 四条路径的冻结公平合同

四臂为：

1. `direct`：部署时只输出答案；训练期可使用与其他臂同源的 verifier/trace distillation 辅助头，但正式评测前必须物理删除。
2. `text_cot`：输出可验证文本 trace 与答案；trace 必须来自同一个 simulator/teacher record，不得额外人工挑选 demonstrations。
3. `latent_k1`：单向量 recurrent latent；除 `K=1` 外与多向量 latent 使用同一 Boundary、transition、readout、数据顺序、优化与评测协议。
4. `latent_k8`：最小多向量 candidate；C1 只允许一个共享 dense core，不含 route、projection expert、FFN-MoE、task/family branch 或“先写后删”。

四臂共同约束：

- 同一 Qwen checkpoint、tokenizer、source text、answer label、split、causal pair 和 teacher source；四臂唯一 public forward 输入为 `source_text`，答案合法范围统一从 source 的 choices 解析。`reasoning_budget`、`valid_choice_mask`、family、program AST、trace 与 claims 均不得进入 forward；family 只允许用于分层采样与报告。latent 的递归次数必须由固定公开上限或从 source 学出的停止机制决定，不能读取 oracle budget。
- 同一 train episode 集、预注册 episode/order ledger、validation selection 频率、最大 optimizer/search budget；不得按 heldout 为某一臂单独选超参数。C2 必须同时报告 equal-example 与 equal-GPU-hour 两个 matched slice，不能用“相同 updates”代替公平算力。
- teacher 信息必须有一条真正 matched 的主比较：四臂逐 record 消费完全相同的 compact trace target、target tokens、loss mask、answer/trace exposure ledger；direct 与 latent 的 training-only trace decoder 在评测前物理删除。主比较禁止 dense state/claim oracle。允许另跑 dense state/claim sensitivity，但必须标记 `supervision-advantaged`，不得用于介质优越性结论。
- direct/text-CoT 使用相同 LoRA target、rank、alpha、dropout、chat template、thinking 与截断策略；K=1/K=8 除 slot 数外保持结构与训练合同相同。
- 训练 FLOPs、teacher 生成/验证 tokens 与 wall time、在线 Qwen encode、输出 tokens、latent transitions、peak VRAM、activation/KV、吞吐和端到端 latency 全部进入成本账本。cached latency 不得与在线文本生成直接比较。
- compact trace 在成为主比较监督前，必须对完整 bank 通过 formatter→parser roundtrip、source simulator semantic replay、malformed rejection 与定向 fault-kill；生成 trace 的机器指标在训练前冻结。`max_new_tokens=512` 与 semantic EOS 固定，不读取 heldout target length。
- P0-M 的 64-record-per-family smoke cache 不得复用为 C1 full cache。Qwen encode、cache build wall time 与 cache bytes 必须计费；single-seed 先行，后续 fresh cache 顺序构建、封存和归档，禁止默认三套同时常驻。
- 统一报告 validation、各 OOD cell、causal-pair flip、最差 family、最差 seed、答案质量与 trace/latent 因果必要性。内部 state/trajectory 只作诊断，不能替代行为 Gate。

## 5. C0 Gates

| Gate | 要求 |
| --- | --- |
| C001 | P0-D v17 seal、assessment、manifest 和全部 sealed 文件逐字节未变；当前 generator/renderer/schema 与 v17 source snapshot 相同。 |
| C002 | ERE/CPS 是两个 AST 结构不同的可执行任务族，不是 surface-only transform。 |
| C003 | 每族 train/validation、四类 OOD、causal-pair 数量与 fingerprint 隔离满足固定 profile；答案、mask、teacher trace 均完整。 |
| C004 | v17 shortcut-report 与 claim-report 通过；claim 正负严格平衡；新增 ERE relation-query visible-pattern/legend oracle Gate 通过。旧报告未覆盖的新组合捷径不得由旧 G09 自动豁免。 |
| C005 | P0-M v5 M01–M08 与七个子 root seal 可复验；解释严格保持 training-path smoke。 |
| C006 | direct/text-CoT/K=1/K=8 的信息、训练、选择、双 matched-slice、评测和全成本合同已冻结；compact trace parser/semantic verifier/full-bank roundtrip/fault-kill 已资格化。四臂结果缺失不算 C0 不完整。 |
| C007 | 历史证据按“可复用机制 / 负向诊断 / 不可复用结果”分级，并逐项写明禁止外推。 |
| C008 | H1/WD 当前机器结果仍为 `authorizes=nothing`，活动 C1 明确排除 routed projection 与旧 checkpoint。 |

全部 Gate 通过时机器状态为 `PASS_V2_A_CLOSURE_C0_READINESS`。输入缺失或无法复验为 `INCOMPLETE_V2_A_CLOSURE_C0_READINESS`；输入完整但任一任务/公平性硬门失败为 `FAIL_V2_A_CLOSURE_C0_READINESS`。这些状态均不得改写任何旧 formal。

## 6. C0 后的唯一执行顺序

C0 PASS 后只允许直接切换到新的 C1 package：

1. 先实现单 seed `C1-S`：完整文本、learned Boundary、共享 dense recurrent latent core、training-only dense credit assignment、部署剥离和因果干预；不兼容复用 H1/WD。
2. C1-S 通过后才运行总计三个 fresh model/data seeds 的 `C1-F`；若单 seed 失败，先判定 Boundary、transition 或训练机制，不并行浪费另外两 seed。
3. C1-F 通过后才运行 C2 四臂 matched Pareto；C2 需要 fresh sealed bank，不能把已揭示的 v17 heldout 当最终确认集。
4. C2 通过后才运行 C3 natural-language audit；C2/C3 同时通过才可能关闭 V2-A。

C0 不授权训练、C2、C3、V2-B 或 V2-C；不证明 latent、多 slot、hybrid core 或白皮书成立。

## 7. 唯一执行身份

固定输出：

- root：`artifacts/v2-a/closure-c0-20260823-1/`
- sibling lease：`artifacts/v2-a/closure-c0-20260823-1.preflight-lease.jsonl`
- CLI：`python experiments/v2_a_closure_c0.py run-v2-a-closure-c0`

CLI 不接受 seed、阈值、输入 root 或输出 root 覆盖参数。lease 或 root 已存在即拒绝；C0 只读输入，不加载 checkpoint、不创建 optimizer、不训练或写模型。
