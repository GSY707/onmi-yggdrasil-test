# V2-R1R P0-M v1 训练通路 Smoke 合同

## 1. 目标与边界

P0-D v17 已由 `docs/v2-r1r-p0d-v17-repaired-production-main-review.md` 接受。本阶段只验证 frozen-Qwen cache、Boundary、K=8 recurrent core、claim loss、direct SFT、text-CoT SFT 和 GPU throughput 的实现能否在固定小样本上过拟合。

P0-M PASS 不证明 validation/OOD、组合泛化、因果干预、K=1/K=8 容量差或 Pareto；这些属于 P1/P2。本阶段结束后必须停在 P1 之前。

## 2. 冻结输入与选择

唯一数据输入是：

```text
artifacts/v2-r1r/p0d-v17-full-production-20260810-1
seal SHA-256 = 453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D
```

selection seed `2026081801`。ERE/CPS 各从 train 确定性选择 label-balanced `64` 条，联合模型使用同一 128 条；文本 baseline 从上述集合各取 32 条，合计 64 条，direct/text-CoT 必须使用完全相同 episode 与 shuffle 约束。不得因训练结果换样本。

## 3. Frozen Qwen cache

固定 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` final language hidden，source 与 claim 均用同一 frozen Qwen 编码。cache 只保存 FP16 full-token hidden、attention mask、token length 和 opaque id；禁止答案、label mapping、AST、teacher、oracle span/role/entity/operation mask、input ids 或 pointer。

持久化采用 `.npy` mmap；训练开始时一次性复制到 pinned CPU 后 nonblocking preload GPU，optimizer hot path 只做 GPU index-select。Qwen cache batch probe 固定比较 4/8/12；本机预探针分别约 `1.93s/1.53s/2.27s`，batch 8 最快，正式 cache 固定 batch 8。任何 token 超过 1024 都硬失败，不截断。

## 4. R1R-Latent v1

模型严格实现高层合同：

- Boundary：non-affine LayerNorm(2048) → Linear(2048,512) → RMSNorm(512)，逐 token、无任务分支；
- K=8 generic slots：一个 learned shared seed 加固定 Fourier slot position；
- 两层 recurrent block：gated cross-attention、gated self-attention、gated SwiGLU，8 heads、FFN 2048、dropout 0；两层 step 内不同，跨最多 24 步和两个任务共享；
- 每层 source K/V 每次 forward 只预计算一次，attention 用 SDPA；固定 `(t,T)` Fourier control 经共享 linear 注入；
- readout 只接收 `H_T`，learned-query pooling 后输出 9 local labels并应用 public choice mask；
- 不存在 task/operator/candidate embedding、oracle slot、task-specific transition 或 source-to-answer bypass。

claim hidden 经同一 Boundary masked mean；共享 probe 用 claim query attention 读取相应 `H_t`。claim 不进入 core 或答案头。

## 5. 训练

latent seeds：ERE `2026081811`、CPS `2026081812`、joint `2026081813`。batch 16；joint 每 batch ERE/CPS 各 8。AdamW、weight decay 0.01、warmup 5%、cosine、clip 1.0；Boundary/core/readout+probe LR 分别 `1e-4/2e-4/3e-4`。loss schedule 保持高层合同：1–400 为 `0.25 answer + 1.0 claim`，401–1800 为 `1.0 answer + 0.5 claim`，之后 answer-only。每样本每 update 确定性轮换两个 claim pair。

smoke 最早 update 1200 判断，最多 2400；通过后保存时物理删除 claim probe，并用 stripped checkpoint 独立 reload，答案 prediction hash 必须相同。smoke checkpoint 不是 P1 正式 checkpoint。

文本 seeds：direct `2026081814`、text-CoT `2026081815`。两者使用官方 chat template、关闭 thinking、同一 64 episodes；direct target `Answer: <label>`，CoT target 为 simulator trace 的紧凑无损结果摘要加同一答案。prompt ≤1024，direct target ≤16，CoT target ≤512，不截断。只在顶部四层 attention 与 MLP 的所有 Linear projection 注入原生 LoRA rank 8、alpha 16、dropout 0，其余 Qwen 参数冻结；loss 只计算 assistant output token。

## 6. Gate

| Gate | 要求 |
| --- | --- |
| M01 | P0-D seal、selection、cache schema/hash/shape/finite/no-forbidden 与 4/8/12 benchmark 全通过 |
| M02 | ERE K=8 overfit64 answer ≥0.95，update ≥1200，aux-stripped reload prediction 相同 |
| M03 | CPS K=8 overfit64 answer ≥0.95，update ≥1200，aux-stripped reload prediction 相同 |
| M04 | shared K=8 overfit128：ERE/CPS 各 ≥0.90，claim ≥0.90，只有一套共享参数 |
| M05 | direct SFT overfit64 greedy generation answer ≥0.95 |
| M06 | text-CoT SFT overfit64 greedy generation answer ≥0.95 |
| M07 | joint stripped checkpoint 连续 100 个 optimizer steps：CUDA、finite、无 OOM/fallback，并记录 median/p95、examples/s、peak VRAM、GPU utilization/power samples |
| M08 | 所有 fixed root seal、seed、source hash、同 episode、公平 LoRA、no-truncation 与结果 conjunction 可重放 |

## 7. 失败与修复

每个 route root 只允许创建一次。失败 root 永久保留；允许分析 loss/mask/cache/数值/吞吐并升 P0-M contract version/seed 重新运行，不允许扩大数据、降低阈值、换 episode、给 latent 加 task 分支或给 baseline 改成分类头。父任务已授权在 P1 前持续进行这种修复循环。

P0-M 全通过后只登记 `P1 eligible` 并停止，不创建或运行 P1 数据、checkpoint、OOD 或干预。
