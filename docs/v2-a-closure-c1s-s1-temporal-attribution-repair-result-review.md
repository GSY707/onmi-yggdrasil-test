# V2-A Closure C1S S1 时序归因修复结果复盘

## 核心判断

`V2-A-CLOSURE-C1S-SRW-S1-TEMPORAL-ATTRIBUTION-REPAIR-20260831-1` 已唯一完成并封存。机器终态为 `COMPLETE_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR`，D004R/D005R 证据完整，但 CPS 与 ERE 的 Axis C 分类均为 `INCONCLUSIVE`。

这里的 `INCONCLUSIVE` 不是“没有观察到时序结构”。恰好相反：两族所有注册 feature 的 target adequacy、active-step motion、固定通道分离、nested cross-fit readout、fit/score null 和原 frozen decoder 都满足诊断阈值。预注册分类器只负责在 `TARGET_INSUFFICIENT`、`LATENT_NOT_FORMED`、`READOUT_INSUFFICIENT` 与 `MIXED` 等故障之间归因；当 target、latent、可学习 readout 和 frozen decoder 全部足够时，它保守返回 `INCONCLUSIVE`，而没有事后增加一个“机制通过”标签。

因此，当前证据排除了“时序 target 不足”“时序 latent 根本没有形成”和“现有 decoder 根本读不出”作为 S1 失败的主要解释。它没有证明 recurrence 必要、多地址工作区成立或 V2-A 架构通过。

## 封存身份

| 项目 | 封存值 |
| --- | --- |
| diagnosis root | `artifacts/v2-a/closure-c1s-srw-s1-temporal-attribution-repair-20260831-1` |
| status | `COMPLETE_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR` |
| source identity | `C4FAD98AEA80AE502B41B070FFA2967BAB15321543B10FE242E8A8466E4883E5` |
| `result.json` SHA-256 | `D1603C625B7854DB6D96DD1034388478F8006709347950A83C9E4D869606A496` |
| `evidence-seal.json` SHA-256 | `E8A61CF8E7EA96AB8F618253D56F6C4B64E62928370C12C515C16F797A1562B0` |
| seal replay | `103/103` matched，0 missing，0 mismatch，0 unexpected |
| wall time | `514.3048127999064` 秒 |
| peak CUDA memory | `229,029,376` bytes |
| optimizer/model writes | `0/0` |
| authorizes | `nothing` |

模型参数状态在诊断前后均为 `BE143CAEA99A55EE3287AA9B57CC234C4BEABC29BABC19A1E7DF54AEB59A2E05`。`final_winner` 与 `query_semantic_match` 等 answer-derived feature 已从 hard evidence 中排除。

## D004R：时序状态确实存在且可读

active step 的相对运动量在 CPS 为 `0.11101`、ERE 为 `0.19809`，明显高于注册下限 `0.05`；inactive step 仅为 `0.000114` 与 `0.000261`。这说明变化集中在注册更新步骤，而不是所有 step 上均匀漂移。

| family / feature | eligible records | natural cross-fit BA | strongest fit-null BA | natural-null gap | fixed-channel AUC | frozen decoder BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPS `processed` | 16 | 1.0000 | 0.7813 | 0.2188 | 1.0000 | 1.0000 |
| CPS `running_best` | 12 | 0.9507 | 0.8982 | 0.0525 | 0.9894 | 0.9542 |
| ERE `touched` | 16 | 0.9117 | 0.7396 | 0.1721 | 0.9786 | 0.9064 |
| ERE `changed` | 16 | 0.9510 | 0.6648 | 0.2861 | 0.9846 | 0.9362 |
| ERE `operation_source` | 16 | 0.9199 | 0.8108 | 0.1092 | 0.9776 | 0.8780 |
| ERE `operation_target` | 16 | 0.9896 | 0.6619 | 0.3277 | 0.9722 | 0.9861 |

六个 feature 的 10,000-replicate fixed-prediction score-null 均得到最小可分辨右尾 `p=1/10001≈0.00009999`。其中 `running_best` 是最弱证据：其 natural-minus-strongest-fit-null 为 `0.05246`，只略高于预注册 `0.05` 下限；应把它称为过门但边界较薄，而不是强稳健余量。

source replay 为同一 materializer 实现的一致性复核，不是独立语义 oracle。诊断对象也只有 S1 的 32 条 Overfit32 记录，不构成 heldout/OOD 泛化证据。

## D005R：为什么仍是 INCONCLUSIVE

两族均满足：

- target 与 nested-fold coverage 足够；
- motion 与固定通道 AUC 证明轨迹含时序关联；
- cross-fit readout 超过 strongest fit-null，且 score-null 显著；
- 原 frozen decoder BA 也超过本诊断用于判定“可读”的 `0.8` 下限。

因此它们不符合 `TARGET_INSUFFICIENT`、`LATENT_NOT_FORMED` 或 `READOUT_INSUFFICIENT`。`INCONCLUSIVE` 的准确语义是“Axis C 未找到已注册的故障”，不是“Axis C 数据无法解释”，更不是“C1S 已通过机制资格”。原 S1 的逐 feature `0.95` 资格门仍然真实失败；本诊断的 `0.8` 是区分“是否存在可读状态”的归因阈值，不能反向改写旧 Gate。

## 合并 D001–D005 后的路线结论

当前最可信的整体解释是：模型确实形成了丰富、可读的时序状态，但答案仍可大量经 Boundary/H0 直接取得，训练目标也没有直接优化 recurrence necessity 与 multi-address functional-K。旧 sealed evidence 中 no-core 仍答对 `28/32`，D002 两族均为 `MIXED`，D003 又没有建立任务侧 multi-object causal arity。因此“状态存在”没有转化成“状态对答案因果必要”，这才是当前架构验证的核心缺口。

下一步不应继续修 temporal probe、降低旧 S1 Gate，或恢复 S2/formal。合理的新 successor 应先在任务侧建立可验证的多对象因果阶数，再让训练目标直接对应 no-core、owner-specific deletion 与至少两个独立 causal contributors；同时切断或严格限制 H0 answer shortcut。它必须使用 fresh identity、fresh Overfit32 和新的资格链，不能复用当前 checkpoint 作为通过证据。

本诊断 `authorizes=nothing`：S2、S3、single-seed formal、C2 与 V2-A PASS 继续保持 `NOT_RUN`。
