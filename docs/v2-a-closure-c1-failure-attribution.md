# V2-A Closure C1 失败归因合同

状态：冻结实现合同；尚未启动唯一诊断 root

日期：2026-08-26

诊断身份：`V2-A-CLOSURE-C1-FAILURE-ATTRIBUTION-20260826-1`

固定 root：`artifacts/v2-a/closure-c1-failure-attribution-20260826-1/`

## 1. 目标与证据地位

本轮只解释 sealed C1 为什么在 G007 trace credit 失败，并为一个全新 C1 repair 合同选择修复层级。它不修改、续跑或重新判定旧 C1 formal，不把 post-stop measurement 回填为 G003–G011，也不授权 C2、C3、V2-B、V2-C 或 V2-A PASS。

诊断必须区分五种可能：checkpoint 选择错误、trace/step alignment 错误、训练 token 曝光不足、probe/metric 与 latent 信息之间的错配、以及 Boundary/latent/core 的整体能力不足。已知 owner-relative NLL margin 通过只证明样本相关轨迹信号存在，不能替代精确内容恢复或架构行为证据。

旧 root 始终只读。诊断允许在内存加载 checkpoint、前向、反向计算独立 loss gradient 和物理 strip/reload；禁止 `optimizer.step()`、参数更新、模型/checkpoint 写入、Qwen 重编码以及向旧 root/cache 写任何字节。新 root 只保存 JSON/JSONL、source snapshot、stdout/stderr、result 与 evidence seal。

## 2. 冻结输入

| 输入 | SHA-256 / 身份 |
| --- | --- |
| C1 formal `result.json` | `FE9F22B89E89092EDD0DCE1CC4FCBA926A3B8D4AB79123F237F4B9EDF379972C` |
| C1 formal `evidence-seal.json` | `BEB46576A91ADC08B9E7A1D51249E4CA5F03BFFFC7069D1E791FBBAC8749FA19` |
| `primary-training.json` | `1B4711E1D2839A61E2619F1683954F6106B217B9DAF817A0686AB44886C44B3A` |
| `trace-exposure-ledger.json` | `0FDB026A005D237729CB663FC61F8C200713F4D672A41783A5AA88B656DA25CC` |
| `trace-credit.json` | `E74EF5D27113C88352BC9E458B40E8F98507ED4A11B797A9A10DF60BB6CF6BFC` |
| `g007-trace-gate.json` | `4E6C6A98B27A3BD34A795BCA080CCBF9DF54120CC1DEA1D7C257A99547568FEE` |
| selected update 5,120 | `3FEC2FF80CC65FAEEB5E033F77D686D35E42B113730BC555F0FF1E47B8FEF46B` |
| final update 6,144 | `1AB9D8683D099BF64457C9B37DD1851B1BD956FB05867EBAE16A8D57404C6FC8` |
| cache result / seal | `A058687125FDC1315871733DD1499036A040636864665793B8D32B8C9D758D4A` / `68CA34BB18F889405F875BF54B7CA2E55C3E9E549DE5DFE1C63C4F47E2B4DC34` |
| trace target bank | `C3EFBC11E2C1DE2ED4BA29829771B23A3B9B709C9C538E89BB29416FF531EC0B` |
| formal source identity | `F910F5F47EE066606BF2A7AB65F65B875B9E143EC7C3929D7D1BCD8FB992729C` |

当前 `closure_c1` 可执行实现与专项测试必须与 formal `source_snapshot` 逐文件相同；新诊断 package 与本合同另算 source identity。旧合同文档在 formal 后仅新增了两行历史状态/结果链接，因此允许这一个已知 provenance delta，但当前文档必须精确匹配固定 SHA-256 `7D104A80E0FCAB034BE1C6D3865E06181DFDF0FF56D201C2B9B1EDBE4C0A43F8`，formal snapshot 中该文档仍保持原字节。除这一个显式例外外，任何 pin、snapshot 或 cache/source identity 不匹配都在领取 root 前 REFUSE，不能降级为模型结论。

## 3. 冻结测量

### A001 identity 与 selection

复验上述 pins、旧 formal seal `55/55`、selected/final checkpoint schema/config，以及十二个 candidate 的保存 hash。用保存的 validation trace NLL 重新执行冻结 selection rule；update 5,120 必须仍是 `min max(ERE,CPS)`。这一步只判断 selection 实现是否自洽，不因其他 checkpoint 的 post-hoc accuracy 更高而改写旧选择。

### A002 alignment 形式与语义边界

对完整 target bank 重放 tokenizer offset、item span、midpoint、step/global/local position 与 grammar mask；形式必须 100% 一致。另报告每族/每 step 的 token 数、trace depth 和 prefix/suffix 分配。CPS 顶层 candidate 与内部 action 没有独立 step 标签，因此当前 mapping 只标记为“top-level candidate alignment”，不能冒充 action-level causal state。诊断只允许对 selected checkpoint 做预注册的 `step-1`、`step+1`、constant-H1、constant-H10 对照；若任一对照相对注册 alignment 的 pooled NLL 改善 `>=0.05 nat/token` 且 top-1 提升 `>=0.02`，登记 alignment suspect，不据此选择新 checkpoint。

### A003 exposure coverage

由 sealed target bank 与 exposure ledger 为每条 train record 的每个 token position 重建 exposed count。统一报告：record micro/macro unique coverage、grammar/content coverage、step coverage、global-position bins、exposed-count histogram、从未暴露 token 数，以及 H1/H10 与两族最差 step。

“随 epoch 循环”的实现资格要求每条训练 target token 至少暴露一次；任一族达不到 `1.0` 即登记 `EXPOSURE_CYCLE_BREACH`。同时对 validation 固定 256/族按 step/position 报 selected checkpoint 的 top-1/top-5/top-10、NLL、target rank/MRR；训练覆盖只用于解释同一位置族的监督机会，不把 validation token 偷并入训练。

为直接检验“窗口未命中导致解码更差”，还固定从 train 按 example ID 排序取 256/族，在 selected checkpoint 上按实际 `exposed_count=0`、`=1`、`>=2` 分层报告同一组 top-k/NLL/rank 指标，并同时给出 grammar/content、step 和 position 条件层。该测量只比较已封存模型的训练 token，不继续优化；暴露关联仍可能受 token/position 难度混杂，因此它与新 repair 的单变量结果共同构成因果证据，不能单独冒充随机试验。

### A004 selected/final trace 与 metric audit

对 update 5,120 与 6,144 使用同一固定 validation 256/族，报告 exact top-1/top-5/top-10、pooled NLL、per-record macro、grammar/content、step 与 position bins。若 final 在两族均同时把 top-1 提高至少 `0.02` 且 NLL 不恶化，登记 selection-metric mismatch；否则不能把换 checkpoint 当修复。

top-k/NLL 强而 top-1 弱只能标记 metric/probe mismatch，不能把旧 G007 改成 PASS。若 top-k、NLL、content 与位置分层全部弱，则支持真实解码不足。

### A005 shared-gradient credit

固定使用 primary epoch 0 schedule 的前 16 个 family-balanced batch，在 selected checkpoint 上分别对 answer CE 与 trace CE 做 `torch.autograd.grad`，不调用 optimizer。对 Boundary 与 shared core 分别报告 norm、weighted norm ratio、cosine、负 cosine 比例与 bootstrap 区间。若 late-phase `0.5*trace` 的 shared-gradient norm 中位数仍超过 answer `3x`，或负 cosine 比例超过 `0.5`，登记 credit imbalance/conflict；否则不得靠直觉修改 loss weight。

### A006 post-stop architecture measurements

在内存中 strip trace probe 并 strict reload；strip 前后必须在完整 validation 上 prediction invariance `1.0`、FP32 max-abs-diff `<=1e-6`，且剥离后零 trace-probe 参数。随后仍用原 batch size 8 完整测量：

- ERE/CPS validation answer；
- 八个 OOD cell；
- 两族完整 causal pairs；
- zero-hidden 与 within-family shuffled-hidden；
- H0/no-core 与 step-5 state shuffle；
- 全 validation slot permutation。

允许复用旧 G004–G010 评分函数，但字段必须命名为 `poststop_*`，只能报告“若按旧阈值会通过/失败”，不得写成正式 Gate 状态。所有 baseline、cell、pair 与 bootstrap 单位保持原合同。

## 4. 归因决策

结果按以下优先顺序登记，不允许看到 post-stop 结果后新增分支：

1. identity/selection/alignment 形式不自洽：`MEASUREMENT_OR_SELECTION_FAILURE`，停止 repair 训练；
2. A003 不满足完整 cycle，且最差 step/position 的 selected trace 指标同步更差：`EXPOSURE_PRIMARY`；
3. exposure 完整但 alignment 对照越过双阈值：`ALIGNMENT_PRIMARY`；
4. post-stop validation/OOD/causal 与 hidden/recurrence 大体成立，而 exact trace 仍失败：`TRACE_METRIC_OR_PROBE_PRIMARY`；
5. answer 与 trace 都弱，或 hidden/recurrence 没有必要性：`LATENT_OR_TRAINING_PRIMARY`；
6. 多项同时成立：按证据分别登记 `contributing_causes`，但 repair 只先处理最早的必要失败，禁止一次混改多个层级。

当前已知的 ledger 探索性复算显示 ERE/CPS mean unique coverage 约 `0.920/0.791`，H10 约 `0.408/0.097`。这些值是冻结 A003 的动机，不是最终机器结果；唯一 root 必须从 pinned JSON 独立重算。

## 5. 单次运行与终态

唯一命令只能在专项测试、CPU fake integration、CUDA selected-checkpoint preflight 与 later-root absence 全部通过后启动一次。正式 diagnostic 可以读 GPU/cache，但必须保持：`training_started=false`、`optimizer_steps=0`、`model_writes=0`、`old_root_writes=0`。完整 alignment tokenizer replay 固定使用最多 16 个 CPU worker；模型与 trace logit 测量使用唯一 CUDA-visible 的离散 GPU。Windows 同时枚举出的核显不冒充第二张 CUDA 计算卡。

终态固定为：

- `COMPLETE_V2_A_C1_FAILURE_ATTRIBUTION`：A001–A006 全部完成、归因分支唯一或显式并列、seal replay 完整；只授权设计一个新 C1 repair 合同；
- `INCOMPLETE_V2_A_C1_FAILURE_ATTRIBUTION`：输入/测量不完整；`authorizes="nothing"`；
- `CRASH_V2_A_C1_FAILURE_ATTRIBUTION`：异常退出并封存；`authorizes="nothing"`；
- root/lease 已存在时 before-mutation `REFUSE_V2_A_C1_FAILURE_ATTRIBUTION_SINGLE_USE`。

无论 complete 与否，本诊断都保持 `c2_authorized=false`、`v2a_passed=false`。repair 必须使用新 identity、fresh initialization 与新的 preflight/formal root；不能继续旧 optimizer、延长旧 run、修改旧 checkpoint 或降低旧 G007 阈值。
