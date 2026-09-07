# V2-A Closure C1U S0：public-grounding 与 gate-free zero-training 资格合同

更新日期：2026-09-01

## 1. 单次身份与授权

本文件与 `src/yggdrasil_v2/v2_a/closure_c1u/contract.py` 构成 C1U S0 的执行真源。用户授权原文为“没问题，按这个顺序做”。授权只按 Gate 条件生效，不改变 C1T `authorizes=nothing` 的历史终态。

固定身份与路径：

| 阶段 | identity | root / lease |
| --- | --- | --- |
| S0 preflight | `V2-A-CLOSURE-C1U-PGF-S0-PREFLIGHT-20260901-1` | `tmp/v2-a-closure-c1u-pgf-s0-preflight-20260901-1` 及 sibling preflight lease |
| S0 | `V2-A-CLOSURE-C1U-PGF-S0-20260901-1` | `artifacts/v2-a/closure-c1u-pgf-s0-20260901-1` 及 sibling S0 lease |

任一 root 或 lease 一旦出现，该身份即已消费。CRASH/FAIL 后禁止删除、覆盖、补文件或同身份重跑。S0 全过程 optimizer/model writes 必须为 `0/0`。

## 2. 前驱与 fresh identity

启动前必须重放 C1T S1 result/seal，确认终态仍为 `FAIL_V2_A_C1T_S1_QUALIFICATION`、旧 endpoint/source hashes 不变、没有旧 `run-s1` 进程，也不存在 C1T S2/S3/formal root。该重放只证明 fail-closed predecessor，不向 C1U 提供 checkpoint、cache 或训练授权。

C1U 使用：

- task seed `2026090121`；
- zero-training model seed `2026090122`；
- address salt `yggdrasil-c1u-public-address-v1`；
-同一冻结 Qwen3.5-2B revision/assets、CUDA 设备和逐卡独立编码纪律；
- 新的 public cards、token/hidden cache、ledger 与 source snapshot。

不得复制 C1T `cards.pt`、ledger 或 task bank。相同 source model 资产只作为冻结编码器 identity，不表示 hidden cache 可复用。

## 3. S0 preflight

preflight 在领取 root 前完成当前专项/代表性回归、固定路径、C1T seal、source assets 与 RTX 4070/CUDA 检查。全部通过后才能单次领取 preflight root，并只对 CPS/ERE 各一个完整 factorial group 编码真实 cards。

P001–P006：

1. C1T predecessor seal 与 C1U paths/user authorization；
2. source closure、测试、assets、device 与版本 identity；
3. public-identifiability 和 public-only replay，含 bridge fault-kill；
4. 一卡一 forward 的真实 cache/ledger/readback；
5. gate-free target-only、permutation/no-core 与 operation-2 gradient-connectivity structure；
6. CUDA BF16 loss/backward finite、optimizer/model writes `0/0`、source hashes 前后不漂移。

任一失败只写 preflight FAIL/CRASH 与 seal，不得启动 S0。

## 4. 正式 S0 Gate

正式 S0 对 CPS/ERE 各四个 2×2 group，共 32 records/192 cards，执行一次 fresh Qwen cache 与 zero-training measurement。U001–U010 必须全部通过：

1. `U001_predecessor_source_device_identity`：preflight PASS、C1T seal replay、source assets/device/source snapshot一致；
2. `U002_task_factorial_and_public_identifiability`：四格完整、单因素翻转、factor-independent bridge、public-only semantic/raw-answer replay全过；
3. `U003_bridge_fault_registry`：ERE value legend 删除/交换/冲突及 CPS candidate index 删除/交换/冲突均被审计器拒绝；
4. `U004_independent_card_cache`：192 次独立 encoder call、target-free manifest、持久化 readback、ledger/hash全过；
5. `U005_source_only_forward_boundary`：forward 仅接收注册 hidden/mask/address，不含 answer/family/factor/AST/support ledger；
6. `U006_gate_free_target_only_transition`：无 learned gate 参数/输出，inactive identity，active 只改 registered target，非 target max delta不高于 `1e-6`；
7. `U007_equivariance_and_no_core_topology`：object permutation logits/trajectory max delta不高于 `1e-5`，operation 不进入 no-core，组内 no-core logits invariant；
8. `U008_operation2_gradient_connectivity`：两个 support 的 intervention objective 均连接到 shared transition/readout，operation-2 source payload 与 proposal 的有效梯度有限且非零；断开 proposal/source 的 fault 必须杀死此 Gate；
9. `U009_bf16_forward_backward_finite`：真实 CUDA BF16 full/no-core/counterpart loss 与全部已有 gradients finite；
10. `U010_accounting_source_and_seal_integrity`：optimizer/model writes `0/0`、training never started、source hashes稳定、result tree seal完整重放。

结构量具不得读取 answer 作为 model input；梯度量具可在 model 外部使用已注册 answer/support ledger构造 loss。S0 不读 accuracy 作为模型选择信号。

## 5. 唯一命令与停止纪律

在仓库根目录按固定顺序：

```powershell
uv run python experiments/v2_a_closure_c1u.py inspect
uv run pytest tests\v2_a_closure_c1u -q
uv run python experiments/v2_a_closure_c1u.py preflight-s0
uv run python experiments/v2_a_closure_c1u.py audit-s0-preflight
uv run python experiments/v2_a_closure_c1u.py run-s0
uv run python experiments/v2_a_closure_c1u.py audit-s0
```

`preflight-s0` 与 `run-s0` 各只能启动一次。任一前置命令非零、任一 Gate false、seal replay异常、root/lease 冲突、source drift、CUDA fallback 或非有限值都必须原样停止。不得继续到后续命令、修补已消费 root、降低门槛或换 seed。

## 6. S0 PASS 边界

S0 PASS 只授权冻结 C1U S1 合同与实现。S1 合同必须写入本次 S0 result、seal、source、task bank、cards cache 与 ledger 的真实 SHA-256，并在启动前重放。S0 PASS 不证明 C1U 学会任务，也不授权跳到 S2。

用户本轮的条件式授权允许在 S0 PASS 且 S1 合同/实现/测试全部审计通过后继续消费一次 fresh S1 preflight 与 S1 root；若 S0 FAIL/CRASH，S1/S2/S3 全部 `NOT_RUN`。
