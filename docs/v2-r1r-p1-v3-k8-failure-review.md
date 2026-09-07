# V2-R1R P1 v3 K=8 失败复核

日期：2026-08-11

证据等级：P1 v3 single-seed formal 的 cache-recovery PASS 与 K=8 falsification FAIL。本文是父任务只读验收与路线判决，不修改 frozen 合同、活动源码或任何正式 artifact。

## 1. 核心判决

父任务接受 P1 v3 的执行完整性、telemetry qualification、immutable cache recovery 和 K=8 失败证据。正式状态为 `FAIL_P1_K8`；K=1、direct、text-CoT、P1 assessment 与 P2 均未运行且不获授权。

这次失败不是 P1 v2 cache 故障的延续，也不是 GPU 吞吐不足。cache recovery 已证明旧 banks 可在不改判 v2 sealed FAIL 的前提下只读复用；K=8 随后完整训练、评测、干预并封存，但 validation、OOD、causal 和 recurrent-middle dependence 都接近选择先验或随机水平。当前候选及其训练路径被 P1 拒绝。

该结果尚不足以否定白皮书中的抽象 mixed latent core。它强烈否定的是当前这条从 P0-M 小集合 overfit 直接跳到 16,384 个 fresh episode、仅 2.34 次平均暴露的端到端学习路径。下一步不能进入 P2，也不应只把 `2400` updates 机械放大后重跑；应另立训练资格阶段，先分离数据规模、暴露预算和目标 bootstrap 三类因素。

## 2. 正式证据与停止纪律

唯一 `run-p1` 由 `2026-08-11 10:22:15 +08` 的一次隐藏 `Start-Process` 启动；launcher PID `46156`、child Python PID `33868`，父子链一致，未重启。stdout/stderr 保留在 `tmp/p1-v3-transport/`。正式结束后无残留进程。

| 阶段 | fixed root | 正式结果 | seal SHA-256 | wall |
| --- | --- | --- | --- | ---: |
| preflight | `artifacts/v2-r1r/p1-v3-preflight-20260811-1/` | PASS | `8340307E7745A68D39D04885424826EA9BB9808C31A735C61E91CBE69710A6BF` | `2.144s` |
| telemetry | `artifacts/v2-r1r/p1-v3-telemetry-qualification-20260811-1/` | T01–T06 全 true | `F18458A95A5450496357B58E665127FA4C68B28120AB6FF9294B551F0F5D9C5B` | `0.242s` |
| recovery | `artifacts/v2-r1r/p1-v3-cache-recovery-20260811-1/` | `PASS_P1_CACHE_RECOVERY`，R01–R09 全 true | `FC763C6B131BEB6A27745D1A54F26F4A063F6A9E1735E258553F52F6F44497C2` | `485.903s` |
| K=8 | `artifacts/v2-r1r/p1-v3-k8-20260811-1/` | `FAIL_P1_K8` | `8D17215939D13C3BD0BB64AD0EA3E388CE2797F63665BDCF7EEDA6862BDFFCE5` | result `1118.952s`；run metadata `1191.319s` |

recovery 对 28,672 个 source entries、191,144 个 claim entries 执行两次全量内容审计，结果 canonical 相等、`failures=[]`，审计前后 v2 cache 文件集合和 stat fingerprint 不变。它只写出 `training_cache_authorized=true`；v2 cache 的 `passed=false`、`OSError [Errno 22]` 与原 seal `55C00FDF43757F5E4B958FA0E6C672AC033B3181D480859D7F314F731083E315` 均未改变。

K=8 失败后，`p1-v3-k1-*`、`p1-v3-direct-*`、`p1-v3-text-cot-*` 和 `p1-v3-assessment-*` roots 均不存在。执行层按 frozen 顺序停机，没有用后序结果补救 K=8，也没有启动 P2。

## 3. K=8 结果

最佳 eligible checkpoint 按冻结规则选在 update `1200`。它不是学习曲线峰值的可信证据：`1200` 正好是最早允许参与选择的 update，之后到 `2400` 没有形成持续上升。

| Gate | 结果 | 关键证据 |
| --- | --- | --- |
| K01 validation | `false` | ERE `0.24902`、CPS `0.19043`，阈值各 `0.85` |
| K02 八个 OOD cells | `false` | ERE `0.25098–0.29199`；CPS `0.10156–0.16992`，阈值各 `0.75` |
| K03 causal flip | `false` | ERE `0.00391`、CPS `0`，阈值各 `0.80` |
| K04 zero-middle | `false` | 两任务平均 drop `-0.00391`，阈值 `0.40` |
| K05 batch-shuffle-middle | `false` | 两任务平均 drop `0.00391`，阈值 `0.40` |
| K06 T=1 | `false` | ERE length/CPS horizon drop 为 `-0.00781/-0.01953`，阈值各 `0.15` |
| K07 probe stripping | `true` | 所有正式 cell prediction hash 相等，最大 accuracy delta `0` |
| K08 architecture integrity | formal `false` | 两个 hook shape 子检查存在 batch-size 写死导致的审计误报；见第 5 节 |
| K09 报告完整性 | `true` | 全部指定干预、hook、gradient 和 stripped 报告均产出 |

ERE/CPS validation 上按每条记录合法 choice 数均匀随机的期望分别约为 `0.26953` 和 `0.16667`。K=8 的 `0.24902/0.19043` 与这一先验同量级；尤其不能把 CPS 高于 `1/6` 的约 2.4 个百分点解释成推理形成。causal pair 只有 ERE `2/512` 对、CPS `0/512` 对同时答对翻转两端；zero/shuffle/slot-permutation 对结果几乎没有影响，说明输出尚未因果依赖 recurrent middle。

训练期 claim 诊断为 accuracy `0.5`，owner-shuffled accuracy `0.50195`，owner-shuffle drop `-0.00195`；claim CE/rank 长期约为 `0.693`。因此 P0-M v5 中存在的 claim/state 对齐在 full-scale P1 没有建立。ERE/CPS 的共享 core 梯度 cosine 中位数为 `+0.29547`，PCGrad 按合同未触发；当前没有证据支持“跨任务负梯度冲突”是主因。

## 4. 根因分层

### 4.1 已排除的主因

吞吐不是瓶颈。前 100 steps 为 `191.66 examples/s`、`107,205 source tokens/s`，data-wait fraction `0.0267`；GPU 利用率中位数 `56%`、功率中位数/95 分位 `41.72/45.95W`，最大显存 `2528 MiB`，数值 finite。相比此前个位数功率问题，本轮训练管线已经实际占用 GPU；继续优化 loader 或功率不会解释接近随机的机制指标。

容量和基本接线也没有被直接否定。P0-M v5 曾在 ERE/CPS overfit64 与 joint overfit128 上达到 answer `1.0`，claim 约 `0.997/0.999/0.907`，owner-shuffle drop 约 `0.346–0.372`。这证明 12.89M 参数的当前 Boundary/core/readout 可以记忆小固定集合，但不证明它能从低复用的大型 fresh 集合学出通用算法。

### 4.2 首要合同缺陷：暴露预算没有随数据规模扩展

P1 训练集为每族 8,192 条，共 16,384 个 episode；冻结训练只有 `2400 × 16 = 38,400` 次 episode 暴露，即平均每条 `2.34375` 次。每次暴露只抽取 4 个 claim，总 claim 观察上限为 `153,600`，甚至少于训练 cache 中 `191,144` 个唯一 claim；考虑 episode 重复后，实际覆盖更低。

P0-M 的成功来自对 64/128 条固定 selection 的高复用，而 P1 合同没有先做 scale × exposure 学习曲线，就从“能记忆”跳到了“少于一个 claim-bank epoch 的跨任务组合泛化”。因此当前 `2400` updates 不是一个经过资格化的充分训练预算。`4096 heldout` 只解决了 G09 测量功效，不会增加模型训练暴露，也不能修复这一问题。

### 4.3 伴随机制：目标 bootstrap 没有发生

稀疏预算并非唯一可能因素。当前端到端目标同时要求 Boundary/core 从冻结文本 hidden 中发现临时语义绑定、状态更新、claim—trajectory 对齐和最终答案；claim 监督又只有在 latent state 已携带正确语义时才会变得可学。P1 曲线中 claim 始终为 chance、CPS answer loss 长期接近六选一熵，说明这个 bootstrap 在当前课程与预算下没有启动。

这不等于应恢复显式三寄存器或 oracle span mask。正确的后继问题是：增加同一目标的有效暴露是否足以启动机制；若不足，再以同总训练计算比较“从一开始 joint”与“通用 claim/mechanism warm-start 后持续 joint”的训练课程。只有这种受控比较才能区分预算不足、目标组织不良和架构表达失败。

## 5. K08 审计误报

formal K08 必须保持 `false`，不得事后改判。但父任务复核确认其中两个失败子项是 batch-size-dependent audit bug：`integrity_audit` 要求 answer-head hook 精确等于 `[[2, 512]]`，并要求 Boundary 输入第一维精确为 `2`；实际被审计的首个 grouped batch 大小为 `1`，hook 分别为 `[[1, 512]]` 与 `[[1, 128, 2048]]`。它们仍分别是 rank-2 latent-only readout 与一次 Boundary 调用。

因此 K08 是需要在新合同中改成 batch-size invariant 的测量缺陷，而不是已观察到的 bypass 或任务专用参数。该误报不影响本轮停止判决：即使 K08 被正确测量为 true，K01–K06 仍全部大幅失败，K=8 仍为 `FAIL_P1_K8`。

## 6. 架构证据边界与后继

本轮已经证明：数据、cache、GPU、前后向、checkpoint、probe 剥离和完整干预链可运行；当前端到端训练没有形成可泛化、可干预的共享 recurrent mechanism。它没有证明 K=1 优于 K=8，也没有形成 latent 与 direct/text-CoT 的 matched Pareto，因为这些后序对照依法未运行。

推荐下一阶段暂名 `P1-LQ`；若获授权，应另立 P1 v4 training-qualification 合同，而不是修补或重跑 v3。合同至少应包含：

1. 固定 v2 data/cache 只读身份，以 512、2,048、8,192 每族的数据尺度与预注册的 episode-exposure ratio 形成学习曲线；v3 的 `2.34×` 作为 sealed reference，新增至少 `8×` 与 `24×` 高复用臂。
2. 先用 answer/claim/owner-shuffle 的早期学习 Gate 判断机制是否启动；只有通过才支付完整 OOD/causal 成本。若同一目标随 exposure 明显上升，归因为预算；若小尺度可学而大尺度不随预算改善，归因为 sample/optimization scalability。
3. 只有在高复用 joint 目标仍不启动时，才比较等总 updates、等 episode/claim observations 的 joint-from-start 与通用 mechanism warm-start→joint 两臂；辅助监督必须贯穿后段，部署时仍物理剥离。
4. K08 改为检查 rank、末维、调用次数及当前 batch 一致性，不再写死 batch `2`；该修复只修测量，不计作模型收益。
5. P1-LQ 只资格化学习路径，不运行 K=1、文本基线或 P2。只有 full-scale K=8 在预注册训练预算下重新通过 K01–K09，才另立完整 P1/P2 后继合同。

## 7. 已完成与未完成

已完成：P1 v3 preflight、telemetry qualification、immutable cache recovery、K=8 全训练与正式评测、seal 复算、停机审计、学习曲线/暴露预算/审计误报复核。

未完成且未授权：K=1、direct、text-CoT、P1 assessment、P2 matched Pareto、P3 fresh seeds、A1.22A、V2-B 与 V2-C 实施。当前应保持这些阶段停止，等待新的训练资格合同。
