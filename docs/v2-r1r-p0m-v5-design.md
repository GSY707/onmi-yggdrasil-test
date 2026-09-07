# V2-R1R P0-M v5 训练通路 Smoke 合同

## 1. 目标与不变量

P0-M v5 只验证 P0-D v17 固定数据上的训练通路，并在全 PASS 后停在 P1 入口。它不证明 OOD、组合泛化、因果、K 容量差或 Pareto。v5 保留 v4 已验证的 cache、Boundary/shared core、interaction probe、baseline 与全部阈值，只修复 owner 依赖没有进入训练目标以及 claim 监督提前关闭的问题。

唯一输入 `artifacts/v2-r1r/p0d-v17-full-production-20260810-1`，seal `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`。selection seed `2026081801`；ERE/CPS 各 64，joint 128；baseline 每族32且 direct/text-CoT 同64 episodes。immutable cache component 与 v4 相同，v5 qualification 重新验证 seal、内容、exact Qwen、无禁字段/截断及 batch8 吞吐/≤6GiB 判据。

## 2. 模型与 claim 目标

主模型固定：2048→512 tokenwise Boundary；K=8 generic slots；两层共享 recurrent cross/self/SwiGLU；8 heads、FFN2048、24 steps；固定 slot 和 `(t,T)` Fourier；source K/V 每层每 forward 一次；answer head 只读 `H_T`。禁止任务/算子/候选 embedding、oracle slot、task-specific transition 与答案旁路。

claim probe 为 query attention 读取对应 `H_t`，再用 `concat(q⊙pooled, |q−pooled|)` 分类；它不进入 core/answer head并在 checkpoint 物理删除。每个 scheduled pair 的 loss 包含 binary CE 与 positive-over-negative ranking。

新增 owner contrast：对每个正 claim，计算正确 owner 与错误 owner 的 truth logit difference，最小化 `softplus(-(score_correct-score_mismatch))`。单任务 batch 循环置换全部16 owners；joint 只在 ERE 0–7、CPS 8–15 各自循环置换，杜绝跨任务族捷径。其权重固定1.0。full evaluation 用同族/同批 owner rotation，M04 继续要求 claim≥.90、correct-vs-shuffled accuracy drop≥.15。

latent seeds `2026081811/12/13`、batch16、最早1200/最多2400、AdamW/分组LR/warmup/cosine/clip 不变。1–400 loss 为 `.25 answer + 1.0 claim`，401–2400 为 `1.0 answer + .5 claim`；不得在 Gate 前关闭 claim 梯度。最早1200前只算 answer，避免无决策价值的全量 claim sweep。

## 3. Baseline 与 Gates

baseline 完全沿用 v4：seeds `2026081814/15`；官方 chat template、thinking disabled；顶部四层 native LoRA rank8/alpha16；batch1；left padding；target-only `logits_to_keep`；FP16 autocast/FP32 masters；不截断；greedy-generation Gate。

| Gate | 要求 |
| --- | --- |
| M01 | cache qualification PASS |
| M02 | ERE answer≥.95、update≥1200、stripped reload相同 |
| M03 | CPS answer≥.95、update≥1200、stripped reload相同 |
| M04 | joint ERE/CPS各≥.90、claim≥.90、owner-shuffle drop≥.15、共享参数唯一 |
| M05 | direct greedy answer≥.95 |
| M06 | text-CoT greedy answer≥.95 |
| M07 | joint stripped checkpoint 连续100 optimizer steps CUDA finite，无OOM/fallback，并记录吞吐/VRAM/utilization/power |
| M08 | fixed roots seal、seed/source hash、同episode、公平LoRA、no-truncation、conjunction可重放 |

root 一次性；首个失败停止并封存。修复必须另立版本，不得降 Gate、换 episode、扩数据、加任务分支/答案旁路。全 PASS 只登记 `p1_eligible=true`、`p1_started=false`。
