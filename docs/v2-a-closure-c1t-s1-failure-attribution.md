# V2-A Closure C1T S1 失败诊断与归因

更新日期：2026-09-01

## 1. 核心判断

C1T S1 的直接失败机制已经定位：共享 `TargetOnlyTransition` 在多组训练中形成了“第一次操作有效、第二次操作按组或按第一因子饱和关闭”的 dead-gate 局部解。第二次写入关闭后，第二支持对象虽然在 card payload 中仍有清晰差异，却无法进入最终 target state；CPS 因而只学到条件化的半 parity，ERE 则只在 g01 学到完整 XOR。R103 与 R105 的失败是这一机制的行为投影，而不是两个互不相关的故障。

dead-gate 的上游原因也有直接梯度证据。当前 support-margin hinge 在完整 factorial 上只要求“替换 support 后原答案 margin 下降”，不直接监督 counterfactual 正确答案；当两个 forward 已相同，hinge 虽保持正损失，却可获得近零梯度。counterpart payload 未 stop-gradient 又使梯度同时穿过原记录与 counterpart 记录，在对称四格上发生强抵消。训练约在 update 2198–2564 经历密集的 pre-clip gradient spikes，随后进入局部模式；余弦学习率衰减不是最初故障，但把近零梯度和饱和 gate 锁定到固定 endpoint。

结论链为：

```text
完整多组 factorial 共享训练
  -> hinge 梯度抵消 + 跨 cell 梯度耦合
  -> 中后期 mode switch / 第二操作 gate 向负饱和
  -> 第二支持对象的写入与恢复梯度同时消失
  -> CPS 半 parity、ERE 单组 specialization
  -> R103 full/factorial exact FAIL + R105 two-contributor FAIL
```

这不是 schedule 曝光不平衡、support swap 换错地址、card 信息缺失、no-core 伪开关、运行崩溃或“C1T 数学上不能表达 XOR”。一个 fresh、非正式、单组 disposable screen 用同一模型、同一目标和同一 AdamW 在 ERE-g00 上 250 updates 即达到 4/4 与双 contributor 全通过，已经排除单组不可学习与基础表达能力不足。该 screen 不是新 S1，也不授权 successor。

## 2. 诊断边界与证据来源

正式 S1 root `artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1/`、source identity `07E8707E…5900E`、`fixed_4000.pt`、result/seal `9B3762…C1DFF`/`89C530…9BCF` 均保持不变。本轮没有从 endpoint 继续 optimizer step，没有修改正式 source、schedule、Gate、seed 或 artifact，也没有创建 S2/S3/formal root/lease。

诊断使用四类证据：

1. sealed `evaluation.json`、`training-progress.jsonl`、`schedule.json` 与 task/cache；
2. 对 `fixed_4000.pt` 的无 optimizer CPU FP32 前向、梯度和 gate 干预；
3. source snapshot 中 `model.py`、`objective.py`、`runtime.py` 的数学与信息流审计；
4. fresh identity `C1T-DIAG-SINGLE-ERE-G00-20260901-1` 的内存-only CUDA BF16 bounded screen。

前三类是正式 endpoint 的 post-stop 只读证据；第四类只用于可辨识性，不属于 formal 或 qualification evidence。

## 3. 错误不是随机的，而是第二操作条件失活

正式 endpoint 的完整答案拓扑为：

| 范围 | 正确数 | 稳定模式 |
| --- | ---: | --- |
| CPS g00/g01/g02/g03 | 每组 `3/4` | 第一因子为 1 时两格都对；第一因子为 0 时忽略第二因子，每组错一格。 |
| ERE g00 | `2/4` | 四格固定预测 raw index 0。 |
| ERE g01 | `4/4` | 唯一完整 XOR group。 |
| ERE g02 | `2/4` | 四格固定预测 raw index 4。 |
| ERE g03 | `2/4` | 四格固定预测 raw index 0。 |

所有 12 条高 `base_margin` 记录都正确；其余 20 条的 margin 约为 0，其中 10 条正确、10 条错误。CPS support-1 有 `8/16` 条 margin drop 为 0，且全部位于第一因子为 0 的 cell；ERE 两个 support 都只有 g01 的 `4/16` 条具有非零效应，其余 `12/16` 为 0。族级 support mean 因而被少量大 effect 抬高，不能代表逐记录双 contributor。

no-core 的 32 条 raw prediction 全部是 index 0。CPS `2/16`、ERE `4/16` 的 no-core“正确”只是 target 恰好也为 0，不是 no-core 具备解题能力。R104 仍能证明 recurrence 对已学会的部分行为有必要性，但不能单独证明完整 factorial 机制已经学会。

## 4. 冻结 endpoint 的因果定位

### 4.1 support swap 没有实现错误

同组单因素 replacement 与 counterpart full forward 在 CPU FP32 重放中数值等价：

- initial payload max delta：`1.67e-6`；
- operation/query state max delta：`1.31e-6`；
- support-intervention logits 对 counterpart full logits max delta：`1.91e-6`；
- trajectory max delta：`2.86e-6`。

因此没有发现换错 slot、遗漏 card、替换多个地址或 route 漂移。反过来，这也说明 `support_flip_correct` 在当前 2×2 构造中几乎复放 counterpart 的 full prediction，不能被当作与 full-answer 完全独立的成功证据；真正额外的信息来自逐边 margin 和 two-contributor 分布。

### 4.2 输入信息存在，第二 transition gate 将其截断

失败 ERE 组两个支持对象的 endpoint initial-payload 平均配对距离均显著非零：g00 为 `7.49/18.66`，g02 为 `5.06/4.97`，g03 为 `10.68/12.85`。card encoder 因而没有把两种 factor 值编码成同一个 payload。

但第二次 transition 的 sigmoid gate 已进入极端负饱和：

| 区域 | operation-2 gate |
| --- | ---: |
| CPS，第一因子为 0 | `3.34e-14` 至 `6.19e-11` |
| CPS，第一因子为 1 | `0.340` 至 `0.9999` |
| ERE g00 | `2.84e-13` 至 `5.00e-10` |
| ERE g01 | `0.263` 至 `0.964` |
| ERE g02 | `3.12e-10` 至 `1.50e-6` |
| ERE g03 | `3.54e-14` 至 `2.25e-11` |

fresh 初始化时同一 operation-2 gate 位于 `0.535–0.552`，所以关闭不是模型默认状态，而是训练形成的。失败 ERE 组第二因子的 initial payload 虽有 `4.97–18.66` 的差异，到最终 selected payload 只剩约 `3e-6–2.8e-5`，logit delta 约 `1e-6`。

冻结 gate-floor 干预进一步确认了这条路径：把第二 gate 从原值强制提高到 1 后，ERE g00/g02/g03 的第二因子 logit 配对距离从近 0 恢复到 `10.02/3.12/3.89`。但 overall accuracy 从 `22/32` 降到 `11/32`，因为被关闭的 proposal branch 本身没有得到充分训练。故 dead-gate 是直接阻断点，却不能靠推理时改阈值事后修复。

### 4.3 正损失不等于存在恢复梯度

ERE-g00 support-0 四条 edge 的 margin drop 为：

```text
+0.006291, -0.006250, +0.006250, -0.006291
```

四个 hinge 都处于 active 区间，单条参数梯度范数为 `4.947–4.959`；但四格平均后的梯度范数只有 `1.92e-6`。这不是 loss 已经满足，而是对称 edge 的梯度互相抵消。

被 gate 关闭的 support-1 更严重：四条 hinge 都等于 `0.5`，单条梯度范数已经为 0，聚合梯度只有 `3.04e-8`。在完整 32-record batch 上，support hinge 仍为 `0.25`，其总梯度范数却只有 `4.14e-6`；同期 full CE 为 `0.43444`、梯度范数 `0.01970`。因此 causal loss 在 endpoint 上几乎不能推动模型离开错误模式。

源码中的 counterpart payload 没有 stop-gradient，hinge 同时穿过原 row 与 counterpart row。只把 counterpart 分支 detach 的无训练梯度 ablation 使 ERE-g00 support-0 聚合梯度从 `1.92e-6` 恢复到 `7.51e-3`，约增加 3,900 倍；support-1 仍约为 `3e-8`，因为 sigmoid dead-gate 继续阻断该路径。这证明未 detach 的跨 cell 耦合显著放大了 factor-0 抵消，而 gate 饱和是 factor-1 无梯度的更近端原因。

## 5. 训练动力学与共享优化

schedule 并不偏向成功的 g01：每个 group 都精确出现 1,000 次，16 种 CPS×ERE pairing 各出现 240–267 次。正式训练前期下降，约 update 1250–2100 停在 `total≈0.914`；update 2198–2564 的 367 步中，pre-clip gradient norm 有 310 步超过 1、122 步超过 10、19 步超过 100，最高在 update 2456 达 `864.375`。global clip 把这些 step 限制到 1，但轨迹表明这一段是明显的 mode-switch 区，而不是平滑收敛。

约 update 2600 后 batch loss 分裂为两个固定平台：ERE-g01 pairing 约 `0.205`，其余 ERE pairing 约 `0.678`，且几乎不再依赖 CPS group。从 update 2939 起，后续所有 pre-clip gradient norm 都低于 `0.1`；最终余弦 LR 到 0。没有中间 checkpoint，故不能证明经典“先学会再遗忘”，但可以确认固定 endpoint 已被锁在 group-specialized 局部解。

endpoint 的 group-gradient cosine 同时存在正负值，例如 CPS-g00 对 ERE-g03 为 `-0.206`、ERE-g00 对 ERE-g02 为 `-0.258`。这支持共享参数竞争，但数值来自 post-stop CPU FP32 几何诊断，不能替代正式 CUDA BF16 行为 Gate。

## 6. 决定性 disposable 可辨识实验

为区分“架构不可表达”与“多组共享优化失败”，本轮运行了一个全新、无文件输出的 bounded screen：

| 项 | 固定值 |
| --- | --- |
| identity | `C1T-DIAG-SINGLE-ERE-G00-20260901-1` |
| seed | `9100001` |
| 数据 | 仅 ERE-g00 四格完整 factorial |
| 模型/目标 | 当前 `C1TModel` 与 `compute_training_loss` |
| optimizer | 同 S1 AdamW 参数组、LR、weight decay 与 clip 1.0 |
| schedule | max 2,000；warmup 256 后 cosine；每 250 step 检查 |
| device | RTX 4070，CUDA BF16，TF32 关闭 |

第一个周期检查点 update 250 即满足提前停止条件：

- wall time `79.4 s`，LR scale `0.9765625`；
- full CE 约 `4e-5`；
- support hinge `0.0`；
- full accuracy `4/4`；
- support-0/support-1 flip 均 `4/4`；
- two-contributor `4/4`；
- operation-2 gate 范围 `[0.05664062, 1.0]`。

这证明 ERE-g00、当前 transition/readout 形式和现有 objective 在隔离条件下可形成完整解；它排除了“该组不可学习”和“C1T 没有 XOR 表达能力”作为本次主因。它没有证明 32-record 共享训练的修复方案、跨组泛化或任何 S2 资格。

## 7. 归因分级与排除项

| 归因 | 置信度 | 判据 |
| --- | --- | --- |
| operation-2 sigmoid gate 饱和关闭是直接机制 | 高 | gate、trajectory、support effect 与 gate-floor 因果干预一致。 |
| 完整 factorial hinge 梯度抵消是上游主因 | 高 | 正 hinge 下单条大梯度、聚合近零的直接 autograd 证据。 |
| counterpart 未 detach 放大跨 cell 抵消 | 高 | stop-gradient ablation 将 g00 factor-0 梯度提高约 3,900 倍。 |
| 多组共享参数竞争与中后期 mode switch 促成局部解 | 中高 | 平衡曝光下仅 g01 成功、负 gradient cosine、集中 pre-clip spikes 与双平台。 |
| opaque symbol / per-group legend binding 增加共享学习难度 | 中 | group specialization 与任务结构一致；单组可学，故不是硬性不可表达。 |
| cosine LR 是初始触发原因 | 低 | mode switch 发生时 LR 尚非 0；它更像后续锁定器。 |
| schedule 曝光不平衡、support swap bug、card 信息缺失、运行故障 | 已排除 | exact counts、swap equivalence、payload delta、seal/accounting 均反证。 |
| 整个 C1T/多向量架构无效或任务数学上不可学 | 已排除为本次结论 | 单组 250-step 完整拟合，且本轮只有单 seed/单预算。 |

最准确的失败标签是：`MULTI_GROUP_CAUSAL_OBJECTIVE_GRADIENT_CANCELLATION_WITH_SECOND_TRANSITION_DEAD_GATE`。

## 8. 路线边界

本诊断只关闭“为什么这个 frozen S1 失败”的主要证据缺口，不修复源码，也不恢复 S1。当前正式终态仍是 `FAIL_V2_A_C1T_S1_QUALIFICATION`，`authorizes=nothing`；S2、S3、single-seed formal、C2、V2-B 与 V2-C 全部保持 `NOT_RUN`。

若 owner 后续要求设计修复，新的 fresh 合同至少必须把以下内容变成启动前 Gate：非饱和/可恢复的 transition 写入、逐 edge counterfactual correct-answer 信号、counterpart gradient ownership、worst-cell 而非均值掩盖、operation-2 gate health，以及 isolated→multi-group 的梯度冲突 screen。它们是下一合同的资格要求，不是对当前 root 的补丁授权。
