# V2-A Closure C1U S1 结果复核

## 1. 终局判断

C1U Publicly-Grounded Gate-Free Workspace 的唯一 fresh S1 Overfit32 已在 2026-09-01 按冻结合同完整运行并 sealed PASS。P101–P105 与 R101–R107 全部通过；正式训练恰好完成 4,000 optimizer steps，只写一次 `fixed_4000.pt`，32/32 raw A–I answer、8/8 factorial groups、两族 no-core necessity 与逐记录双 support causality 全部满足预注册门槛。

机器终态为 `PASS_V2_A_C1U_S1_OVERFIT32`，授权逐字为 `C1U_S2_MATCHED_K1_K8_CONTRACT_DESIGN_ONLY`。这只允许设计并审计 matched learned K1/K8 的 S2 合同，不授权 S2 训练、S3、single-seed formal、C2、V2-A PASS、V2-B 或 V2-C。

## 2. 单次执行与封印

S1 preflight：

- identity：`V2-A-CLOSURE-C1U-PGF-S1-OVERFIT32-PREFLIGHT-20260901-1`
- root：`tmp/v2-a-closure-c1u-pgf-s1-overfit32-preflight-20260901-1`
- result SHA-256：`E62E1C7575158C0011DC5F4155E766B22403CF19AE13DE81E5B57DFB536E2034`
- evidence-seal SHA-256：`484AF8EFBD93ECA51FFAA8096214A7EF472B44179823B709235A96DEBC02D309`
- P101–P105 全 PASS；32 disposable steps；formal steps/model writes 为 `0/0`
- benchmark p95 `0.0800 s`，peak CUDA `430,366,208` bytes，估算 formal `320.0 s`

正式 S1：

- identity：`V2-A-CLOSURE-C1U-PGF-S1-OVERFIT32-20260901-1`
- root：`artifacts/v2-a/closure-c1u-pgf-s1-overfit32-20260901-1`
- result SHA-256：`1FCD2DEF44F0324631FC3ECE4D1F552DC6D66181A2F6E97F6894A3FE5816090C`
- evidence-seal SHA-256：`A5211D46340365E7B9DB4B3300C39034C89AA6C51942FDEF5A2911D152635350`
- endpoint SHA-256：`88F538F7B0E5AD0A0C77759EE2D72C8866D69F36F5D4E8F7AB951FFEC80F159F`
- source identity：`8BD618F1960FFCCF1BBD6DEA50EA57051B48D16955AA3AF40EF194230D9194FD`
- schedule SHA-256：`372A48060DAF2C6D684C95AD47F372603112D24252C4642A6B6F4A3B7F7C1AB5`
- 44 个封印条目完整复放，无 missing、mismatched 或 unexpected 文件

## 3. 联合行为 Gate

| 指标 | CPS | ERE | Gate 结果 |
| --- | ---: | ---: | --- |
| full answer | 16/16 | 16/16 | PASS |
| factorial exact groups | 4/4 | 4/4 | PASS |
| no-core correct | 4/16 | 2/16 | PASS，均低于 0.55 ceiling |
| no-core margin-drop lower95 | 12.4725 | 12.1525 | PASS，均高于 0.50 |
| support-0 flip | 16/16 | 16/16 | PASS |
| support-1 flip | 16/16 | 16/16 | PASS |
| two-contributor | 16/16 | 16/16 | PASS，Wilson lower95 均为 0.8064 |
| support-0 margin-drop lower95 | 22.1505 | 21.1673 | PASS |
| support-1 margin-drop lower95 | 22.1868 | 21.1480 | PASS |

所有 8 个 factorial groups 的 no-core logits 在四格间最大差均为 `0.0`。反转 slot 顺序后的 logits 与 trajectory 最大绝对差也均为 `0.0`；endpoint integrity 明确保持 gate-free target overwrite、无 learned write gate、无 optimizer state 与 checkpoint selection。

## 4. 训练与会计

正式运行使用 model seed `2026090122`、order seed `2026090131`，每个 CPS/ERE group 各出现 1,000 次。训练 wall time 为 `348.782 s`，compute/data time 为 `323.100/24.318 s`，peak CUDA 为 `431,742,464` bytes，step median/p95 为 `0.0586/0.1902 s`。最终 total loss 为 `3.4458e-5`，其中 support hinge 为 `0.0`。

会计为 `formal_optimizer_steps=4000`、`disposable_optimizer_steps=0`、`model_writes=1`、`intermediate_checkpoints=0`、`checkpoint_selection=false`。唯一 endpoint 为 `fixed_4000.pt`，重新加载后的 identity、update、schedule、file hash、gate-free model integrity 与 optimizer absence 全部通过。

## 5. 修复判断与证据边界

C1T 在相同完整-group batch、旧 causal objective、优化器与 4,000-step 预算下停在 22/32、1/8 factorial 和不完整的双贡献；C1U 保留这些训练条件并直接移除 sigmoid write gate，同时补全公开语义桥，最终达到所有联合 Gate。结合此前 bounded 三臂 screen 中“gate-free + 原 hinge”在 1,000 updates 达到 32/32、而“旧 sigmoid + direct counterfactual CE”在 2,000 updates 仍失败，可确认 direct CE 不是必要修复，learned gate 饱和是被成功切除的主要故障机制。

正式 C1U 同时包含 public bridge 修复，因此该 formal 不能单独量化 bridge 与 gate removal 各自的效果大小；它证明的是整个 C1U 修复包在封存 Overfit32 合同下成立。它仍是单 bank、单 seed、训练内 32 records 的能力资格，不是 fresh-group 泛化、K8 优于 K1、formal 统计稳定性或 V2-A 终局证据。

## 6. 后继边界

下一步若继续，只能先设计 matched learned K1/K8 S2：同 source/cache/task/schedule/budget/optimizer/seed family，真实 learned K1 与功能性 K8 都必须通过 owner-only shortcut、seal replay 与完整因果 Gate。S2 训练需新的明确授权；不得因为本次 S1 PASS 直接消费任何 S2 root/lease。
