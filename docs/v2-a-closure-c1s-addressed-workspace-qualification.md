# V2-A Closure C1S：Addressed Content Workspace 机制资格合同

冻结日期：2026-08-28

合同身份：`V2-A-CLOSURE-C1S-ADDRESSED-WORKSPACE-20260828-1`

状态：本合同只授权一次 S0 zero-update 机制资格运行。S0 通过最多授权一个独立 S1 Overfit32 身份；本合同不授权 S1 自动启动、S2、single-seed formal、C2、V2-A PASS、V2-B 或 V2-C。

## 1. 研究问题与直接切换

C1R 已证明旧 anonymous dense K=8 的 final answer path 功能上退化为近似 K=1，但没有证明整个 latent workspace 数学上 K=1。C1S 不给旧 core 增加 diversity loss、slot penalty 或更大的 answer head，而是直接切换状态代数：

`S_t = ({a_k, h_k}_{k=1..K}, q_t)`

- `a_k` 是地址/所有权 sidecar，只用于身份持续、相似度寻址和置换等变，不承载 family、答案或完整 teacher state；
- `h_k` 是连续 payload，承载实体、候选和中间状态；
- `q_t` 由 public source 产生，只用于选择最终地址，不是跨样本共享的 global answer query；
- Boundary 从完整 frozen-Qwen hidden 自主产生 entity/candidate、operation 和 query 表征；family、route、AST、answer、teacher trace 与任何 oracle pointer 都不进入模型 forward；
- recurrence 只通过 source/target address route 对相关 payload 做 gated update，禁止 dense all-slot self-attention；
- 答案只读取最终 query 所选 payload；训练辅助头不参与部署答案，并在部署模型中物理删除。

A1.19H 只提供 `S_t=(A_t,H_t)`、定向共享 transition、query-coupled answer 与因果 Gate 的机制原则；其 exact-symbolic handle、旧 checkpoint 和 formal 资格不进入 C1S。A1.20D 只提供 coarse section → refined operation → entity feedback 的双向闭环原则；旧阈值和 post-stop 数值不进入 C1S。

## 2. 公共接口与禁止信息

部署模型的唯一公共输入为：

```text
source_hidden [B,S,2048]
source_mask   [B,S] bool
```

输出为 raw A–I 九类 logits，以及调用者显式要求时才返回的地址、payload trajectory 和诊断量。模型 API 和参数名均不得出现 `family`、`route`、`AST`、`answer_label`、`teacher_state`、`trace_input` 或 oracle source/target/query pointer。训练目标只进入模型外部 loss/materializer。

冻结参考配置为 `D_payload=512`、`D_address=64`、`K=8`、`T=10`、8 attention heads、Boundary reader depth 2、FFN width 2048。matched K=1 使用完全相同参数化、宽度、T、输入、答案头和训练辅助头，只把 fixed slot count 改为 1；slot Fourier 是 buffer，不得令 K 改变 trainable parameter count。

Boundary 使用 shared slot seed + fixed physical Fourier 仅作交换对称破缺；它不赋予任何固定任务槽语义。entity、operation、query 先分别读取 public source，再由 predicted address route 做 operation↔entity 双向 refinement。source/target/query route 在 forward 中使用 hard value、soft gradient 的 straight-through 选择；部署为 hard route。

## 3. S0：zero-update 结构与测量资格

固定 root：`tmp/v2-a-closure-c1s-s0-preflight-20260828-1/`

固定 sibling lease：`tmp/v2-a-closure-c1s-s0-preflight-20260828-1.preflight-lease.jsonl`

唯一命令：

```powershell
python experiments/v2_a_closure_c1s.py run-s0-preflight
```

运行前 root 与 lease 必须均不存在。runner 必须先复验 C1 formal、C1 attribution 与 C1R formal 的 result/seal pin，再取得 lease。S0 可执行 forward/backward 数值 smoke，但 `optimizer_steps=0`、`model_writes=0`、`training_started=false`；不得加载训练集、validation、旧 checkpoint 或创建模型 checkpoint。

S0 全部 Gate 必须同时通过：

| Gate | 冻结要求 |
| --- | --- |
| S001 identity/pins | source identity 稳定；三组旧 root result/seal hash 与 seal replay 全部一致；旧 root 写入 0 |
| S002 public boundary | forward 只接受 public hidden/mask；禁止字段/参数名为空；source 在 Boundary 后关闭 |
| S003 direct-switch core | 无 dense all-slot attention、无 global answer query；address 保持，唯一 shared transition 重用 10 步；答案来自 query-selected final payload |
| S004 K parity | K=8 与 K=1 trainable parameter count 必须完全相等；除 slot count/buffer 外配置相同 |
| S005 slot permutation | 成对置换 address、payload 与 route 后，FP32 logits max-abs delta `<=1e-5`，trajectory 等变 delta `<=1e-5` |
| S006 targeted update | one-hot source/target 控制下未选 slot delta `<=1e-6`，inactive transition delta `<=1e-6`，至少一个被选 slot delta `>1e-8` |
| S007 query ownership | 注册 positive control 的 query swap 必须改变被选 slot并使 logit L2 delta `>=1.0`；duplicate-content null 的 query swap delta `<=1e-6` |
| S008 functional-K instrument | unique addressed positive control 的 relevant mean-replace effect `>=1.0`、irrelevant max `<=1e-6`；legacy uniform-mean null 的 ownership contrast `<=1.01`；K=1 被识别为单槽 null |
| S009 auxiliary stripping | 训练辅助参数物理删除；strip 前后部署 logits max-abs delta `<=1e-6`；部署 state 不含 auxiliary key |
| S010 numerics/device | 注册 RTX 4070 CUDA BF16 forward/backward finite；梯度存在且 finite；optimizer/model write 均为 0 |
| S011 seal | result 写明所有 NOT_RUN/授权边界，evidence seal 全量 replay |

S0 任何 Gate FAIL 或异常都必须封存原样停止，`authorizes=nothing`。S0 PASS 只授权设计并实现一次全新身份的 S1 Overfit32；父进程不得在同次命令继续训练。

## 4. S1：Overfit32 机制资格（条件授权，当前不执行）

S1 必须另立 identity/root/lease，并在训练前先资格化新的 ownership/state target bank。target 可由 immutable C0R `program_ast`、teacher trace、training claims 与 tokenizer offsets 离线生成，但这些信息只进入 loss/materializer，不进入模型 forward。target bank 至少为每个 record 给出：public-span ownership、每步 owner/state target、operation source/target ownership、query ownership、positive/negative claim 配对与完整 provenance/hash。

冻结抽样为 `sha256("C1S-S1-OVERFIT32-V1|example_id")` 每族 16 条；fresh seed，batch 8（4+4），最多 4000 optimizer steps，固定 endpoint，不做 checkpoint 选择。S1 必须同时满足：

- answer exact `32/32`；
- valid slot/address owner accuracy、query owner accuracy、per-step state/closure accuracy均 `>=0.95`；
- query-swap、same-value/different-entity 跟随率均 `>=0.90`；
- disable-recurrence 与 wrong-start 的 answer 或 registered state metric drop 均 `>=0.50`；
- relevant-slot mean-replace 的 correct-logit drop `>=0.50`，irrelevant-slot最大 drop `<=0.10`；
- slot permutation logit delta `<=1e-5`；
- auxiliary strip 前后 logits delta `<=1e-6`。

任一项失败即停止，不得靠放宽 non-collapse 阈值、增加 diversity loss、延长同一 run 或替换 seed 继续。S1 PASS 只授权 S2 discovery 实现与一个 paired K=8/K=1 身份。

## 5. S2：paired discovery 与 matched K=1（条件授权，当前不执行）

S2 从 C0R train 中排除 S1 的 32 条后，按冻结 hash 每族选择 3072 条 discovery-train 与 512 条 source-disjoint discovery-eval；现有 validation/test/OOD 不进入 C1S discovery。K=8/K=1 采用相同 public cache、target bank、初始化公共参数、batch 顺序、examples、optimizer schedule、6 epochs、batch 8、固定 4608-update endpoint 和同一 RTX 4070；两臂按相同 GPU-hour 上限裁切，并同时报告 equal-example 与 equal-GPU-hour 结果。任何后续 formal 必须使用新 data identity，不能复用 discovery heldout。

K=8 只有同时满足以下条件才形成 formal 候选：

- ERE/CPS heldout answer point accuracy 均 `>=0.75` 且 Wilson lower 均 `>=0.70`；
- 相对 K=1 的两族 paired accuracy gain 均 `>=0.05`，record bootstrap 95% lower `>0`；
- query ownership 与 state/closure heldout 均 `>=0.80`；
- disable recurrence、wrong start、query swap、relevant/irrelevant deletion 全部跨两族通过注册因果门；
- K=8 functional-K 证据超过 K=1/null，而不是只凭 raw cosine、centered energy 或谱；
- 参数、examples、processed tokens、GPU-hour、峰值显存和吞吐账本完整。

若 K=8 与 K=1 都失败，优先判 Boundary/任务表示不足；若 K=1 通过而 K=8 失败，判多槽绑定/transition 失败；若两者表现相当，删除多槽必要性主张。只有 K=8 通过全部门且稳定优于 K=1，才允许另行冻结 single-seed formal 合同。

## 6. 永久边界

- C1/C1R roots、seed、checkpoint、Gate 与失败结论不可修改或重跑。
- C1S 不继承 trace-probe Stage B、随机 trace exposure、global answer query、dense all-slot core 或旧 h0/state-shuffle Gate。
- S0/S1/S2 均不是 V2-A PASS，也不授权 C2、V2-B、V2-C 或“先写后删”。
- raw cosine、loss、单次 answer accuracy、slot entropy 或参数量都不能单独证明 addressable workspace 成立。
- 若 target bank 不能在不向 forward 泄漏 oracle metadata 的前提下定义跨 ERE/CPS 的 ownership/state target，必须在 S1 训练前停止并回到任务/监督设计。
