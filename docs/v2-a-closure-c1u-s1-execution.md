# V2-A Closure C1U S1 Fresh Overfit32 冻结执行合同

## 1. 目标与授权边界

本合同评估一个单变量修复：在保留 C1T 的完整 counterfactual causal objective、batch 结构、优化器、学习率和行为 Gate 的同时，移除 learned sigmoid write gate，改为 public-grounded gate-free target overwrite。ERE/CPS 的公开语义桥已由 C1U S0 资格化；S1 不再改 loss，也不把诊断 screen 的中间 endpoint 当作选择依据。

唯一前驱是 sealed PASS 的 C1U S0。用户以“没问题，按这个顺序做”授权按 `S0 preflight → S0 → S1 contract → S1 preflight → S1` 的 Gate 顺序继续。本授权只允许本身份的一次 S1 preflight 和、在 preflight sealed PASS 后、一次正式 S1。它不授权重跑、调参、换 seed、加载旧 checkpoint、进入 S2 训练或任何 formal 阶段。

执行链固定为：

```text
sealed C1U S0 PASS
  -> S1 contract/source/test/inspect
  -> S1 preflight（一次；32 disposable optimizer steps）
  -> 仅当 preflight sealed PASS：fresh S1 Overfit32（一次；4,000 formal steps）
  -> sealed PASS：只授权 matched learned K1/K8 S2 合同设计
  -> 任一 FAIL/CRASH/INCOMPLETE：原样停止
```

## 2. 固定身份、路径与前驱哈希

S1 preflight identity 为 `V2-A-CLOSURE-C1U-PGF-S1-OVERFIT32-PREFLIGHT-20260901-1`，root 为 `tmp/v2-a-closure-c1u-pgf-s1-overfit32-preflight-20260901-1`，同级 lease 后缀为 `.preflight-lease.jsonl`。

正式 S1 identity 为 `V2-A-CLOSURE-C1U-PGF-S1-OVERFIT32-20260901-1`，root 为 `artifacts/v2-a/closure-c1u-pgf-s1-overfit32-20260901-1`，同级 lease 后缀为 `.s1-lease.jsonl`。任一 root 或 lease 一旦存在，即视为对应身份已消费。

启动前必须同时复放：

- S0 result：`39070BCD12AA15B5EBA6F55E84A55CE348A69992BBD5087BB6DCE526CABA714A`
- S0 evidence seal：`F224DB2101897442742FD621BAA838A78D42ACAD5B6DAE68597E7B968D8A9ADE`
- S0 source identity：`2068B1E54F18C82AC3984A1B01C483E4A3B2297C191861DD962DB925A883572E`
- `cards.pt`：`7CC6E33B73DFFF057C77489676B0A3AFB904507254FC55328242F79B6F82CAAD`
- `ledger.json`：`74DFB049B56E4D013C150732E11BF11EB65988078ECF6D7DFAB728541A784929`

S0 资格化的 C1U cache/cards/model/objective/runtime/source/tasks 与共享 simulator/cache contract 必须和 S0 source snapshot 逐文件同 hash。S1 可新增或修改的范围仅为本合同、S0 review、S1 evaluator、trainer、runner、CLI、source-closure registry 与测试。

## 3. 数据、模型与目标

数据固定为 S0 封存的 32 records：CPS/ERE 各 4 个 2×2 factorial groups，共 8 groups。训练只读取封存的 target-free 192-card Qwen cache；模型 forward 仅接收 public card hidden/mask 与公开地址。答案、valid-choice、factor、family、support ledger、counterfactual indices 和 AST 只能在外部 objective/evaluator 使用。

模型使用 fresh seed `2026090122`，不读取任何模型或 optimizer state。transition 在每一步只写注册 target，直接采用 proposal overwrite，不存在 learned write gate；同一 transition 跨 step 与 family 共享。目标保持不变：full raw A–I CE 权重 1.0、no-core confusion 权重 0.5、双 support margin hinge 权重 0.5、margin floor 0.5。完整 factorial counterpart 仅由封存外部 ledger 构造。

## 4. 固定 schedule 与 optimizer

每个 batch 由一个完整 CPS group 与一个完整 ERE group 组成，共 8 records；组内顺序固定 `00,01,10,11`。每四个 updates 为一个 cycle，order seed 固定为 `2026090131`，分别置换四个 CPS groups 与四个 ERE groups 后逐项配对。正式训练固定 4,000 updates，因此每个 group 精确出现 1,000 次；完整 schedule 在 preflight 写盘并以 canonical SHA-256 冻结，formal 必须逐字复用。

optimizer 为 AdamW：boundary/transition/readout LR 分别 `1e-4/2e-4/3e-4`，weight decay `0.01`，global gradient clip `1.0`。前 256 steps 线性 warmup，随后 cosine 到固定 update 4,000；CPU intra/inter-op 为 2/1，CUDA BF16 autocast，TF32 关闭。禁止 early stop、预算延长、训练后阈值调整或 checkpoint selection。

唯一允许写出的模型文件为 `fixed_4000.pt`，不含 optimizer state。禁止中间 checkpoint、best checkpoint、resume 或 endpoint 替换。formal optimizer step 必须逐步追加 crash-recovery journal；异常时在已领取 root 内按实际 steps 写 CRASH result 并 seal，不得补跑。

## 5. S1 preflight Gate

preflight 在领取 root 前运行 C1U 独立测试、S0 前驱复放、task/cache audit、固定路径、source closure、CUDA/BF16 与资产检查。全部通过后才执行固定 32 个 disposable BF16 optimizer steps；preflight model/order seeds 固定为 `2026090193/2026090194`。这些权重不保存、不进入正式训练，也不以 accuracy/loss 选择参数。

P101–P105 必须全部为真：

1. S0 前驱与用户授权逐字匹配；
2. 全套测试、固定路径、source/device/assets PASS；
3. task/cache/schedule、public forward、gate-free model integrity PASS；
4. 32-step BF16 loss/gradients/optimizer state 全有限，peak CUDA 不超过 7 GB、step p95 不超过 3 秒、估算 formal 不超过 4 小时；
5. formal steps/model writes 为 0，disposable steps 恰为 32，执行前后 source hashes 完全不变。

preflight 结果必须 seal 并由 `audit-s1-preflight` 复放；只有 sealed PASS 才允许正式 S1。

## 6. 正式行为 Gate

正式结果 R101–R107 必须全部为真：

1. S0/preflight/source/cache/schedule/设备身份完全匹配；
2. 恰好 4,000 formal optimizer steps、一个 `fixed_4000.pt` 写入，无 optimizer state、中间 checkpoint 或 selection；
3. 32/32 raw A–I answer 正确，8/8 factorial groups 全四格正确；
4. CPS/ERE 各自 no-core accuracy 不高于 0.55，no-core margin-drop bootstrap lower95 不低于 0.50，组内 no-core logits 最大差不超过 `1e-6`；
5. 每族 two-contributor point 不低于 0.80 且 Wilson lower95 不低于 0.65；两个 support 各自 margin-drop lower95 不低于 0.50，counterpart answer flip 为 100%；
6. slot permutation logits/trajectory 最大差不超过 `1e-5`，endpoint integrity 明确保持 gate-free target overwrite，参数结构与 fixed endpoint metadata 完整；
7. 结果写出前后 source/cache 哈希、formal schedule 与会计 journal 完整一致，最终 evidence seal 可复放。

loss 下降、route 命中、单个 family/支持项改善、部分 factorial 成功、单纯 32/32 answer，或诊断 screen 在较早 update 的通过都不能替代上述联合 Gate。

## 7. 终局语义与停止规则

PASS 只证明该封存 32-record、单 seed、单固定预算下，C1U gate-free workspace 能形成 no-core necessity 与双 support 因果贡献。它不证明 fresh-group 泛化，不证明 learned K8 优于 matched learned K1，也不构成 V2-A PASS。PASS 仅授权另行设计、审计 matched learned K1/K8 S2 合同；S2 训练仍未授权。

任何 Gate FAIL、CRASH、INCOMPLETE、非有限值、CUDA fallback、source/cache/schedule 漂移、root/lease 冲突或 seal 不完整，都使 S2、S3 与 formal 保持 `NOT_RUN`。禁止重跑本身份、补 steps、换 seed、降 Gate、挑 checkpoint 或修复后沿用已消费 root。

## 8. 唯一命令顺序

```powershell
uv run pytest tests/v2_a_closure_c1u -q
uv run python experiments/v2_a_closure_c1u.py inspect
uv run python experiments/v2_a_closure_c1u.py preflight-s1
uv run python experiments/v2_a_closure_c1u.py audit-s1-preflight
uv run python experiments/v2_a_closure_c1u.py run-s1
uv run python experiments/v2_a_closure_c1u.py audit-s1
```

其中前两条可在未消费状态下复核；四条阶段命令必须严格按 Gate 顺序。`preflight-s1` 与 `run-s1` 各只能启动一次。
