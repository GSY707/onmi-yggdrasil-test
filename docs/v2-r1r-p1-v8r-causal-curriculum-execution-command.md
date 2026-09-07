# V2-R1R P1 v8R causal curriculum recovery 执行命令

日期：2026-08-11  
工作目录：`C:\Users\24408\Documents\onmi yggdrasil test`

## 1. 固定 roots

- preflight：`artifacts/v2-r1r/p1-v8r-causal-curriculum-preflight-20260811-1`
- qualification：`artifacts/v2-r1r/p1-v8r-causal-curriculum-qualification-20260811-1`

两个 root 在唯一正式启动前必须都不存在。v7 与 v8 roots 只读，不得修改、重封或覆盖。

## 2. 唯一正式命令

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-causal-curriculum
```

该编排命令只能启动一次。不得预先单独运行会创建固定 root 的子命令，不得在失败后重启、补跑或拼接 artifact。

## 3. 编排顺序

1. preflight：验证 source/seal/hash、v7 checkpoint、v8 FAIL 保持、数据分区、两臂共同 batch ledger、预测测试、CUDA smoke 与固定 root 新鲜性；
2. qualification：先生成 zero-update v7 baseline，再依次运行 `replay_ce` 与 `replay_pair`，完成 train/dev/retention/integrity 评估并封存；
3. 任一步失败即停止；不得创建 fresh v9 或 P2 root。

## 4. 固定实现要求

- 两臂必须加载同一个 v7 stripped checkpoint；
- 两臂必须使用完全相同的 episode-ID batch sequence；
- 每个 batch 固定 8 ordinary ERE、8 ordinary CPS、4 causal ERE pair、4 causal CPS pair；
- 两臂唯一差异是 pair loss 权重 `0.0` 与 `1.0`；
- 3,072 updates，batch 32；ordinary 每 record 3 次，causal 每 record 32 次；
- 固定 final update，不得按 development audit 选 checkpoint；
- 不创建 temporal probe，不使用 validation/audit/OOD 训练；
- warning 不自动等同失败，runtime error、非 finite、hash mismatch 或 Gate false 必须失败。

## 5. 正式回传

正式回传必须包含：

- 唯一启动时间、PID/父子链、命令、退出码与 wall time；
- preflight/qualification status、seal SHA-256 与 `verify_seal`；
- zero-update v7 baseline；
- 两臂 train pair、development audit、ordinary validation 与状态干预指标；
- R01–R06；
- initial-state、batch-sequence、checkpoint、source snapshot hash；
- 明确说明是否创建 successor root。

PASS 只授权 fresh P1 v9 设计；不完成 P1，不进入 P2。
