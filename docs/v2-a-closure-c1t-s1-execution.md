# V2-A Closure C1T S1：fresh Overfit32 执行合同

## 1. 合同地位与授权边界

本文件与 `src/yggdrasil_v2/v2_a/closure_c1t/contract.py` 共同构成 C1T S1 的冻结执行真源。旧 C1S S1 已 sealed FAIL，禁止重跑；本合同只执行新的 C1T Causally-Partitioned Workspace fresh Overfit32，不兼容、不继承旧 checkpoint，也不把旧 S1 数字带入 Gate。

C1T S0 的唯一正式结果为 `PASS_V2_A_C1T_S0_QUALIFICATION`，其授权仅到 S1 合同设计。2026-09-01 用户以“开始S1阶段”给出本身份的一次性启动授权。该授权只在 S1 preflight 全部通过后允许消费一个 S1 root；它不授权 S2 训练、single-seed formal、C2、V2-A PASS、V2-B 或 V2-C。

阶段顺序固定为：

```text
sealed S0 PASS + S0 scientific-core pins + 用户授权
  -> S1 preflight（一次；32 个 disposable optimizer steps）
  -> 仅当 preflight PASS：S1 fresh Overfit32（一次；4,000 formal steps）
  -> PASS 仅授权 S2 合同设计；FAIL/CRASH 后序全部 NOT_RUN
```

preflight 的 disposable steps 不是 formal steps，不保存权重，也不能用于 checkpoint、参数、阈值或 endpoint 选择。任何 root/lease 一旦存在即视为该身份已消费。

## 2. 固定身份、前驱与 source closure

S1 preflight 身份为 `V2-A-CLOSURE-C1T-CPW-S1-OVERFIT32-PREFLIGHT-20260901-1`，root 为 `tmp/v2-a-closure-c1t-cpw-s1-overfit32-preflight-20260901-1`。正式 S1 身份为 `V2-A-CLOSURE-C1T-CPW-S1-OVERFIT32-20260901-1`，root 为 `artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1`。二者各有同级 single-use lease。

前驱必须同时满足：

1. S0 result SHA-256 为 `831BC5C2…41D6A8`，evidence seal 为 `E4D9B3DC…5FA0EE`，全树重放无 missing、unexpected 或 mismatch；
2. S0 optimizer/model writes 为 `0/0`，授权逐字等于 `C1T_S1_CONTRACT_DESIGN_ONLY`；
3. simulator、C1 cache/source loader、C1T cards、task、model、objective、runtime、source 与 persistent cache 实现逐文件匹配 S0 snapshot；
4. S0 task bank 与当前确定性重建完全相同；`cards.pt` 与 `ledger.json` 分别匹配 `44232C7D…A4723` 与 `287CB720…FF55B`，持久化 readback 与 target-free audit 全过。

S1 可新增合同、evaluator、训练器、runner、测试和文档，但不得修改上述 S0 scientific core。preflight 写入本轮完整 source snapshot；正式启动必须逐字重放 preflight source hashes，防止在 32-step benchmark 后调参或修代码。

## 3. 数据、cache 与完整 group schedule

S1 固定使用 S0 的 32 条记录：CPS/ERE 各 4 个 2×2 factorial group、每族 16 条。每组四个 cell 均保证单独翻转任一 support 对象就改变 simulator answer，且公开输入只改变该对象卡。S1 不重新编码 Qwen，也不复制旧 C1/C1S hidden；只读使用 S0 新生成并封印的 192-card C1T cache。

每个 batch 必须由一个完整 CPS group 与一个完整 ERE group 组成，共 8 条。组内 cell 顺序固定为 `00,01,10,11`；每四个 update 为一 cycle，在 seed `2026090111` 下分别置换四个 CPS group 与四个 ERE group后逐项配对。固定 4,000 updates，因此每个 group 精确出现 1,000 次。完整 4,000-row schedule 在 preflight 写盘并以 canonical SHA-256 冻结；正式 schedule 必须完全同 hash。

## 4. 模型、objective 与训练 endpoint

模型配置保持 S0 的 source width 2048、payload width 512、FFN width 2048、8 slots、10 operations 与 raw A–I 九类 head。模型只能接收注册的 object/operation/query hidden、mask 与公开地址；answer、family、factor、support slot、counterfactual index、valid-choice mask 与 simulator target 全在 forward 外部。

模型 seed 固定为 `2026090102`，fresh 初始化，不加载任何 model/optimizer state。训练使用 AdamW：Boundary/transition/readout LR 分别为 `1e-4/2e-4/3e-4`，weight decay `0.01`，global gradient clip `1.0`，前 256 step 线性 warmup 后 cosine 到固定 4,000 endpoint；CPU intra/inter-op 为 `2/1`，CUDA BF16 autocast，TF32 关闭，完整 counterfactual causal loss 每步计算。

loss 固定为：

- raw A–I `full_answer_ce`，权重 1.0；
- raw A–I no-core-to-uniform KL，权重 0.5；
- 两个同地址 single-factor counterpart payload 的 answer-margin hinge，floor 0.5、权重 0.5。

只允许写一个 `fixed_4000.pt`。它不含 optimizer state，不允许中间 checkpoint、best selection、early stop、endpoint 延长或训练后调阈值。正式 optimizer step 逐步追加到 crash-recovery journal；任何异常在已领取 root 内按实际步数写 CRASH result 并 seal。

## 5. S1 preflight Gate

preflight 在领取 root 前先运行当前 C1T 全套测试、S0 前驱重放、task/cache audit、固定路径与 CUDA/BF16/资产检查。全部通过后才消费 preflight root，并在 disposable fresh model 上执行固定 32 steps；不保存权重，不读取 accuracy 作为选择信号。

P101–P105 必须全部通过：前驱与用户授权；源码/测试/路径/设备；task/cache/schedule/forward integrity；BF16 loss、gradient、参数与 32 steps 有限且 p95 step 不高于 3 秒、估计 4,000-step wall 不高于 4 小时、peak CUDA 不高于 7 GB；最后 source hashes 不漂移、formal steps/model writes 仍为 `0/0`。任一失败都不启动正式 S1。

## 6. 正式 Gate

正式评价从唯一写出的 endpoint 重新 strict load，在 CUDA BF16、完整 32-record batch 上执行。answer margin 固定为 `correct - logsumexp(all eight raw wrong)`，外部 valid-choice mask 只审计答案合法性，不能屏蔽 raw logits。

R101–R107 必须联合通过：

1. 前驱、source、cache 与 4,000-row schedule identity 完整；
2. 恰好 4,000 formal optimizer steps、一个 fixed endpoint write、无 optimizer state/中间 checkpoint/selection；
3. 32/32 raw answer exact、CPS/ERE 各 16/16、8/8 factorial group 全 cell exact；
4. 每组四个 no-core logits 在 `1e-6` 内相同；两族 no-core accuracy 均不高于 `0.55`，full−no-core margin bootstrap lower95 均不低于 `0.50`；
5. 对两个 support 分别换入同组、同公开地址、单因素 counterpart payload：两族、两 support 的 counterpart raw answer 都必须 16/16 exact，margin-drop bootstrap lower95 均不低于 `0.50`；每条记录两个 drop 同时不少于 `0.50` 的 point 至少 `0.80`、Wilson lower 至少 `0.65`；
6. object permutation 的 logits/trajectory max delta 不高于 `1e-5`，forward 字段和模型参数命名无 target 泄漏，target-only transition 与部署结构保持完整；
7. endpoint、S0 predecessor、source 与 cache 在结果写入前再次复核，随后完整 result tree 进入 evidence seal。

loss 降低、route 命中、slot cosine、单个 support 的平均效应或 answer 32/32 均不能单独判 PASS。

## 7. 停止与结论边界

PASS 只证明该 32-record、固定 source/cache、单 seed、单预算下 C1T 能在 Overfit32 形成 no-core necessity 与双 support 因果贡献；它不是泛化证据，也不证明 K8 优于 learned K1。PASS 仅授权另行设计、审计 S2 matched K1/K8 合同，S2 训练仍需新授权。

FAIL、CRASH、INCOMPLETE、非有限值、CUDA fallback、source/cache/schedule 漂移、root/lease 冲突或 seal 不完整都使 S2、S3、single-seed formal、C2 继续 `NOT_RUN`。禁止重跑、补 steps、换 seed、挑 checkpoint、降低 Gate、修复后沿用已消费身份，或把失败写成整个多向量架构的否定。
