# V2-R1R P1 v8R causal curriculum recovery 设计

日期：2026-08-11  
阶段：P1 v8R 机制恢复资格  
证据等级：development qualification；使用已揭示 audit，不是 fresh formal

## 1. 实验问题

本实验只回答一个问题：在同一个已经具备 broad ERE/CPS 能力的 K=8 latent reasoner 上，显式 causal pair objective 是否能比普通 CE 更好地把 CPS 成本变化绑定到最终答案，同时保留 ordinary competence。

它不再同时训练基础能力、temporal probe 和 causal binding。v7 sealed checkpoint 提供共同基础能力；v8R 只比较两种 matched causal fine-tuning objective。

## 2. 固定输入

- 初始模型：`artifacts/v2-r1r/p1-v7-integrated-k8-20260811-1/checkpoint.pt`；两个臂必须逐字节加载同一 stripped checkpoint。
- ordinary rehearsal：P1 v2 model-view 中 ERE/CPS 全部 train，各 8,192 条。
- causal optimization：沿用 v8 的 384 pair/族，共 1,536 条记录。
- development audit：沿用 v8 的 128 pair/族，共 512 条记录。
- source cache：沿用已由 recovery 授权只读训练复用的 P1 v2 cache。
- v7 与 v8 三个正式 root 必须 seal 有效；v8 FAIL 必须保持不变。

由于 v8 audit 指标已经被观察，development audit 只用于判断训练机制是否值得进入 fresh v9，不产生独立泛化结论。

## 3. 两个 matched 臂

### 3.1 `replay_ce`

从 v7 checkpoint 开始，在 mixed batch 上只优化 answer CE。它控制继续训练、ordinary rehearsal、causal records、batch 配对、数据顺序和计算量。

### 3.2 `replay_pair`

读取与 `replay_ce` 完全相同的 batch，并在相同 answer CE 上增加 causal answer pair-ranking loss。除了该 loss，模型初始化、样本、样本顺序、优化器、学习率与 updates 必须一致。

两个臂都 co-batch causal pair。这样实验变量严格收缩为“是否使用 pair objective”，不再把 co-batching 与 loss 混在一起。

## 4. mixed batch 与计算合同

每个 batch 固定 32 条：

- ordinary ERE 8 条；
- ordinary CPS 8 条；
- causal ERE 4 个完整 pair，共 8 条；
- causal CPS 4 个完整 pair，共 8 条。

每臂 3,072 updates，共 98,304 exposures。调度必须满足：

- 16,384 条 ordinary record 每条恰好 3 次，共 49,152 exposures；
- 1,536 条 causal record 每条恰好 32 次，共 49,152 exposures；
- 两臂 episode-ID batch sequence 完全相同；
- 每个 causal batch 中所有 pair 完整，pair role 为 `base/flip`；
- 不允许 validation、audit 或 OOD 进入训练。

## 5. 训练目标

v8R 不创建或训练 removable temporal probe。v7 已提供 temporal mechanism 正证据；本轮避免让新随机 probe 再次与答案目标竞争。

两个臂共同使用：

`L_common = CE(answer, target)`

`replay_pair` 额外使用：

`L_pair = softplus(margin - own_base + flipped_base) / 2 + softplus(margin - own_flip + base_flip) / 2`

`L_replay_pair = L_common + 1.0 * L_pair`

优化器使用 v7 参数分组的十分之一学习率：Boundary `1e-5`、core `2e-5`、answer readout `3e-5`；AdamW，weight decay `0.01`，gradient clip `1.0`，5% warmup 后 cosine decay至 20%。固定 final update，不按 development audit 选 checkpoint。

## 6. 评估

两个臂都评估：

- causal optimization train：ERE/CPS raw accuracy、pair-flip accuracy；
- v8 development audit：ERE/CPS raw accuracy、pair-flip accuracy、prediction-flip rate；
- ordinary validation 全量 1,024 条/族；
- ERE length OOD 与 CPS horizon OOD 全量；
- middle-state batch shuffle，作为状态持续性诊断，不再作为必要机制 Gate；
- checkpoint probe absence、source-to-answer gradient、finite、相同初始化、相同 batch sequence、exact exposure。

同时用 zero-update v7 checkpoint 在相同 evaluation cells 上生成 baseline，禁止引用旧 evaluator 的错误 pair-role 数值。

## 7. Gate

所有 Gate 预注册，不因结果修改。

### R01 输入与共同起点

- v7/v8 source roots seal 全部有效；
- v7 checkpoint hash 与冻结值一致；
- zero-update v7 ordinary validation ERE `>=0.95`、CPS `>=0.80`；
- zero-update v7 audit ERE pair flip `>=0.60`、CPS pair flip `<0.10`，确保修复靶点存在。

### R02 matched compute

- 两臂 initial-state hash 相同；
- 两臂 batch-sequence hash 相同；
- 每臂 3,072 updates、98,304 exposures；
- ordinary/causal exposure 分别严格为 3/32 次每 record；
- 全部 loss、gradient、logit finite。

### R03 objective decision power

- `replay_pair` 的 CPS optimization pair flip 比 `replay_ce` 高至少 `0.20`；
- `replay_pair` 的 CPS optimization pair flip 至少 `0.70`。

### R04 development causal transfer

- `replay_pair` audit ERE pair flip `>=0.60`；
- `replay_pair` audit CPS pair flip `>=0.30`；
- `replay_pair` audit CPS raw accuracy `>=0.50`；
- `replay_pair` audit CPS pair flip 比 zero-update v7 与 `replay_ce` 中较高者至少高 `0.20`。

R04 只证明已揭示 split 上的机制恢复，不能宣称 fresh generalization。

### R05 ordinary retention

- `replay_pair` validation ERE 不低于 zero-update v7 超过 `0.05`，且绝对值 `>=0.90`；
- `replay_pair` validation CPS 不低于 zero-update v7 超过 `0.05`，且绝对值 `>=0.75`。

### R06 integrity

- checkpoint 不含 probe；
- source-to-answer gradient 非零；
- architecture/config 与 v7 相同；
- v8 sealed artifacts 未改变；
- 没有 fresh v9 或 P2 root。

## 8. 判定与停止

只有 R01–R06 全 true 才记为 `PASS_P1_V8R_CAUSAL_CURRICULUM`。PASS 只授权建立 fresh causal source/cache 与 P1 v9 合同。

任一 Gate 失败即记为 `FAIL_P1_V8R_CAUSAL_CURRICULUM` 并停止。如果 R03 通过而 R04 失败，结论是 pair objective 仍只记忆训练 pair，下一步应改变语义状态约束；不得继续堆 exposure。如果 R03 失败，先审计 loss/batching/optimization，不得进入 fresh v9。

无论 PASS 或 FAIL，P1 都未完成，P2 都不得启动。
