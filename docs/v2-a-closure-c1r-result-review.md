# V2-A Closure C1R 结果复盘

日期：2026-08-27

正式身份：`V2-A-CLOSURE-C1R-STAGED-CREDIT-20260827-1`

## 1. 核心判断

C1R 的 zero-training preflight 已封存通过；唯一 formal 在 Stage A 的第一个行为门 G004 明确失败并按合同停止。Stage B、trace probe 修复以及 G005/G006/G008/G009/G010 均未运行，机器授权为 `nothing`。

失败不再主要归因于旧 C1 的 trace exposure 或 answer/trace 共享梯度。C1R Stage A 根本没有 trace probe、trace loss 或 trace exposure，却仍在完整训练集和 validation 上同时只得到约 15%–19% accuracy。只读轨迹诊断进一步表明，当前 dense recurrent answer path 会在第一步迅速形成 common-mode consensus，最终的八槽 answer readout 功能上退化为近似均值池化。准确措辞是：**final answer path 的 effective K 接近 1**；不能把它升级成“所有 latent slot 数学上完全相同”或“未训练的 trace path 也必然 K=1”。

因此当前最强失败链是：

`full-population answer representation/optimization 未建立 → dense recurrent core 强化 common mode → final answer readout 的 K=8 功能退化 → E/I label-mode collapse`

## 2. 正式结果与封存边界

| 项目 | 结果 |
| --- | --- |
| preflight | `PASS_V2_A_C1R_PREFLIGHT`；零训练、零 optimizer/model write；13/13 seal replay |
| preflight result / seal | `E1C2030AEC7E6C639430416590097EB3B08944CDAF436390DFF1B6C8A12B02A1` / `89387EBAB7FA0FA2E754C51C9DD3A8CC3634ED89EA3C9F930EBD0DC5883D0C49` |
| formal | `FAIL_V2_A_C1R`，`stopped_at=G004`，exit code `1` |
| formal result / seal | `54F426D5825A00A3DE9C57F16657F9DE499C007D447986007373FCA8C5F3590D` / `28B4FD9366FABCABCDBC0A92F86AE9008548DE34FD4B7F380E3CD12D903D6C7E` |
| formal seal | 20/20 replay，0 missing/mismatch/unexpected |
| accounting | answer Overfit32 200 steps + Stage A 6144 steps = 6344 optimizer steps；2 次 checkpoint write；旧 roots 写入 0 |
| source/input | C1R source identity 稳定；旧 C1 formal/cache/attribution roots 前后 metadata 完全相同 |
| 后续 | G005/G006/G008/G009/G010、freeze、trajectory cache、Stage B、G007、strip/reload 全部 `NOT_RUN` |

G004 的正式结果为：

| family | correct / total | accuracy | Wilson lower | 合同要求 |
| --- | ---: | ---: | ---: | --- |
| ERE | 294 / 1536 | 0.191406 | 0.172511 | point ≥ 0.75 且 lower ≥ 0.70 |
| CPS | 240 / 1536 | 0.156250 | 0.138952 | point ≥ 0.75 且 lower ≥ 0.70 |

Stage A 的 6144-step answer loss 均值为 `2.224284`，接近九分类均匀交叉熵 `ln(9)=2.197225`，且没有形成持续下降曲线。参数、loss 与梯度 finite；fixed final endpoint 消除了 checkpoint selection 解释。相同代码的 answer-only Overfit32 在 200 steps 达到 `32/32`、loss `0.00014025`，所以不是 forward/backward 断路、head 绝对无容量或优化器完全失效。

## 3. 失败归因

### 3.1 不是 validation-only 泛化失败

对封存 `stage-a-final.pt` 的 train split 做同口径只读复算：

| split/family | accuracy | 主要预测模式 |
| --- | ---: | --- |
| train overall | 1421 / 8192 = 0.173462 | E/I |
| train ERE | 795 / 4096 = 0.194092 | E=2618，I=1469，A=7，B=2 |
| train CPS | 626 / 4096 = 0.152832 | E=2855，I=1170，A=65，B=6 |

train 与 validation 几乎一致地低，因此不是“训练集已拟合、validation 才崩”的 split generalization 问题。validation 的九个 target label 基本均衡，但预测集中为 E/I：ERE 为 E=1027、I=507、A=2；CPS 为 E=1084、I=428、A=21、B=3。

### 3.2 不是 legend、候选顺序或 evaluator 映射错误

3072 条 validation row 与 C0R source 按 `example_id` 全量 join 后：missing、duplicate、family、split、行序 mismatch 均为 0。逐行解析公开 `LABEL KEY` 并核对 `label_mapping`、`valid_choice_mask`、`semantic_answer → target_label`、`answer_index → A-I`、`predicted_semantic` 和 `correct`，所有 mismatch 均为 0。因此 E/I 是模型 raw nine-logit head 的输出模式，不是 post-hoc label ledger 或 evaluator 制造的假象。

### 3.3 core 在第一步后形成 answer-path consensus

冻结 discovery 使用 `sha256("C1R-POSTSTOP-T0-V1|example_id")` 排序，每族取 512 条，共 1024 条；与 sealed batch-8 G004 的 prediction/correct 逐行 mismatch 均为 0。它只作 post-stop diagnosis，不回填 G008–G010。

| readout site | overall accuracy | target NLL | 与 H10 预测一致率 | 主要预测 |
| --- | ---: | ---: | ---: | --- |
| H0 | 0.188477 | 2.533838 | 0.663086 | A/B/E/I |
| H1 | 0.173828 | 2.468476 | 0.994141 | 几乎立即变成 E/I |
| H10 | 0.173828 | 2.145616 | 1.000000 | E/I |

后九步继续改善 NLL 和 margin，但几乎不再改变 argmax。它们主要把已经形成的共识模式重新标定，而没有恢复九类可分结构。

逐 slot 几何给出同一方向：

| site | within-record slot-centered energy | mean off-diagonal slot cosine | answer score std across slots |
| --- | ---: | ---: | ---: |
| H0 | 195.311 | 0.940963 | 0.005901 |
| H1 | 23.339 | 0.992993 | 0.001798 |
| H10 | 0.139949 | 0.999958 | 0.000153 |

Boundary 的 `output_norm` 不会制造该现象；独立 128-row 反证中，Boundary norm 前后 cosine 从 `0.95275` 降至 `0.94088`，而 H0→H1→H2→H5→H10 才从约 `0.9284→0.9862→0.9979→0.99975→0.999939`。final LayerNorm 前后也几乎不改变 H10 cosine。因此 common-mode 放大主要在 recurrent core 内形成。

### 3.4 final answer readout 的 effective K 接近 1

H10 的 answer-query 平均 entropy 为 `2.99999999 / log2(8)=3` bits，平均最大 slot weight 为 `0.125024`。把 learned query pooling 替换成八槽均值后，1024 条预测一致率为 `1.0`，平均 logit L2 差仅 `1.13e-6`。bias-only 则统一预测 D、accuracy 约 `0.1006`，说明 head 不是纯 bias；它使用了 latent 的公共均值，只是没有功能性使用多槽差异。

独立 128-row leave-one-slot-out 控制的 logit 绝对变化均值为 `0.000343`、P95 `0.000823`、最大 `0.03135`，0 条改变 argmax。final core self-attention 的跨槽 score std 约 `0.00435`，平均最大权重 `0.1256`，归一化 entropy `0.999995`，不是 softmax 下溢。

必须保留边界：slot-centered residual 仍非零；减去 slot mean 后 residual pairwise cosine 约 `-0.13`。所以 raw cosine 受强 common-mode/gauge 影响，不能单独证明八槽完全相同。现有证据足以支持“final answer readout 功能上近似 K=1”，不足以声称整个 latent workspace 或未运行的 trace probe 都是 K=1。

## 4. 已排除与仍未区分的因素

| 因素 | 当前判断 |
| --- | --- |
| 旧随机 trace chunk exposure | 是旧 C1 的合同缺陷，但 C1R answer-only Stage A 不含 trace，故不是整体失败的充分原因 |
| answer/trace shared-gradient conflict | 旧 C1 中真实存在；C1R 移除后仍失败，故不是充分原因 |
| checkpoint selection | C1R 使用 fixed final endpoint，排除 |
| 数值崩溃/断路/绝对容量不足 | finite checks 与 Overfit32 排除 |
| label mapping、候选顺序、join | 全量 0 mismatch，排除 |
| train→validation 泛化 | train 与 validation 同低，排除为近端主因 |
| Boundary 完全无信息 | 未被证明；H0 与跨记录 state 仍有结构，需由下一代 readout/地址机制对照拆分 |
| 整个 latent/trace effective K=1 | 未被证明；当前只资格化 answer-path effective K≈1 的诊断结论 |

## 5. 审计性能缺陷

归因过程中发现 `audit_target_bank()` 曾在每个 target token 上调用会重建完整词表字典的 `TraceLexicon.local_id()`。生产 bank 约 125 万 token、local vocab 5597，因此形成不必要的单核超线性开销。实现已改为每次 audit 只物化一次 global→local mapping，并增加“一次访问”回归测试。

修复后 26,624-record production target bank：JSON load/hash `6.63s`，audit `5.59s`，总计 `12.21s`，仍为 PASS、0 failures。该修复不改变 target、hash、Gate、checkpoint 或任何封存 root，也不改变模型失败判断。

## 6. 对整体 V2-A 路线的含义

C1R 已否定“先修 trace exposure 就能恢复整体 C1”的路径，也表明继续延长同一 dense-core answer-only schedule 缺乏依据。下一步不应给旧 C1 增加 epoch、loss 权重、slot-diversity penalty 或 answer-head 容量；这些做法最多阻止某个几何指标塌缩，不能证明多槽具有地址、所有权与查询功能。

可复用的历史正证据来自 A1.19H 的状态代数而非其旧代码：`S_t=(A_t,H_t)`，opaque address 只负责相等性寻址，连续 payload 承载状态，transition 只更新被 source/target 地址选中的实体，query 选择最终实体。A1.20D 可复用的是 Boundary 的粗到细双向 section closure。下一代候选应直接切换为 addressable content workspace，并以 K=1 temporal state 作必要对照；详细提案见 `docs/v2-a-closure-c1s-addressed-workspace-proposal.md`。

本身份最终仍为 `authorizes=nothing`。不得重跑 C1R、换 seed/checkpoint、补跑 Stage B，或把 discovery 诊断回填为正式 G008–G010。任何 successor 必须在用户确认新合同后使用新 identity、root、source manifest 和先机制后 formal 的停止链。
