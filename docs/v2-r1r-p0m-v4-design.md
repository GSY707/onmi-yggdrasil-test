# V2-R1R P0-M v4 训练通路 Smoke 合同

## 1. 目标与冻结边界

本阶段只验证 P0-D v17 固定数据上的工程可训练性，并在 PASS 后停在 P1 入口。它不证明 validation/OOD、组合泛化、因果、K 容量差或 Pareto。v4 继承 v3 已验证的 cache、Boundary/core、target-only baseline logits 与所有数据/seed/Gate，只修复 v3 已定位的可剥离 claim probe。

唯一数据为 `artifacts/v2-r1r/p0d-v17-full-production-20260810-1`（seal `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`）。selection seed `2026081801`：ERE/CPS 各 64，joint 128；baseline 每族 32，direct/text-CoT 为相同 64 episodes。不得换样本、扩数据或降阈值。

immutable cache component 为 `artifacts/v2-r1r/p0m-v1-cache-20260810-1`；v4 资格 root 必须重新验证 component seal、bank hash/shape/finite/mask、exact Qwen revision、selection、无截断/禁字段，以及 4/8/12 finite、peak≤6 GiB、batch8 throughput>batch4。

## 2. Latent core 与 claim 修复

主模型不变：2048→512 tokenwise Boundary；K=8 generic slots；两层共享 recurrent cross/self/SwiGLU；8 heads、FFN 2048、24 steps；固定 slot 与 `(t,T)` Fourier control；每层 source K/V 每 forward 一次；answer readout 仅取 `H_T` 输出 9 local labels。禁止 task/operator/candidate embedding、oracle slot、task-specific transition 与答案旁路。

claim 经同一 Boundary。辅助 probe 用 claim query attention 读取相应 `H_t`，但分类必须保留绝对交互：`concat(q⊙pooled, |q−pooled|)` → RMSNorm → Linear/SwiLU/Linear。训练 schedule 每个 episode 每 update 轮换两个既有正负 pairs；claim loss 为 individual binary CE 加 pairwise positive-over-negative softplus ranking。probe 不进入 core/answer head，PASS checkpoint 必须物理删除整个 probe。

为排除 claim-only shortcut，full claim evaluation 同时将每个 claim 的 owner 循环置换到 batch 中另一 episode，在相同 prefix 读取错误状态。joint Gate 除 accuracy≥.90 外，正确 owner 相对 shuffled owner 的 accuracy drop 必须 ≥.15。

seeds 仍为 ERE/CPS/joint `2026081811/12/13`，batch16，最早1200、最多2400；AdamW、分组 LR、loss phase 与 v3 相同。最早判定前只评估答案，full claim 与 intervention 延迟到 update 1200，避免无决策价值的 GPU 开销。

## 3. Baseline

direct/text-CoT seeds `2026081814/15`；官方 chat template、thinking disabled、同 episodes。只对顶部四层 attention/MLP Linear 注入 native LoRA rank8/alpha16/dropout0，其他 Qwen 冻结。batch1、最早1200、最多2400。left padding 保持 assistant target 在尾部，`logits_to_keep=maximum_target+1` 只计算精确 loss window；FP16 autocast、FP32 LoRA master parameters。prompt≤1024，direct target≤16，CoT target≤512，禁止截断。

## 4. Gates

| Gate | 要求 |
| --- | --- |
| M01 | v4 cache qualification 全通过 |
| M02 | ERE answer≥.95，update≥1200，stripped reload prediction 相同 |
| M03 | CPS answer≥.95，update≥1200，stripped reload prediction 相同 |
| M04 | joint ERE/CPS各≥.90，claim≥.90，owner-shuffle drop≥.15，单一共享参数集 |
| M05 | direct greedy answer≥.95 |
| M06 | text-CoT greedy answer≥.95 |
| M07 | joint stripped checkpoint 100 optimizer steps CUDA finite，无 OOM/fallback，记录吞吐、VRAM、utilization/power |
| M08 | fixed roots seal、seed/source hash、同 episode、公平 LoRA、no-truncation 与 conjunction 可重放 |

每个 root 只创建一次；首个失败即停并封存。允许另立新 version 修复实现/训练问题，但禁止降低 Gate、换 episode、加任务分支或答案旁路。全 PASS 只写 `p1_eligible=true`、`p1_started=false`。
