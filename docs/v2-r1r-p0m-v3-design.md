# V2-R1R P0-M v3 训练通路 Smoke 合同

## 1. 目标与边界

P0-D v17 已接受 production data 与 verifier。P0-M v3 只检验固定小样本上的工程可训练性：frozen-Qwen cache、Boundary、K=8 shared recurrent core、claim objective、direct SFT、text-CoT SFT 与 GPU throughput。PASS 只登记 `P1 eligible`；不得启动 P1，也不产生 validation/OOD、组合泛化、因果、容量差或 Pareto 结论。

v1 因 latency/throughput 判据错误停止；v2 修复 cache qualification 后在非正式 GPU 预检中发现 BF16 mask bug 并优化 baseline target-only logits，未进入训练。两轮根均封存，详见各自 review。v3 是训练使用的第一份完整冻结合同。

## 2. 固定数据与 cache

唯一数据为 `artifacts/v2-r1r/p0d-v17-full-production-20260810-1`，seal SHA-256 `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`。selection seed `2026081801`：ERE/CPS 各 label-balanced 64 条；joint 使用相同 128 条；baseline 每族 32 条，direct/text-CoT 使用完全相同 64 episodes。不得换样本、扩数据或降低 Gate。

immutable cache component 为 `artifacts/v2-r1r/p0m-v1-cache-20260810-1`，固定 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` final hidden、FP16 full-token `.npy` mmap、inference batch 8。v3 qualification 必须复核 component seal、所有 bank hash/shape/finite/mask、exact selection/model、无截断和无 oracle/answer fields。4/8/12 benchmark 均须 finite 且 peak ≤6 GiB；batch 8 throughput 必须高于 batch 4。训练 preload GPU 后 hot path 只做 GPU index-select。

## 3. Latent 模型与训练

模型固定为 non-affine LayerNorm(2048) → Linear(2048,512) → RMSNorm(512) Boundary；K=8 generic slots；两层共享 recurrent gated cross-attention/self-attention/SwiGLU，8 heads、FFN 2048、24 steps；固定 Fourier slot 与 `(t,T)` control；source K/V 每层每 forward 只预计算一次。readout 仅接收 `H_T` 输出 9 local labels。禁止 task/operator/candidate embedding、oracle slot、task-specific transition 与 source-answer bypass。

claim 经同一 Boundary，probe 只从相应 `H_t` 读取；claim 不进入 core/readout。seeds 为 ERE `2026081811`、CPS `2026081812`、joint `2026081813`；batch 16，joint 每批两族各 8。AdamW、weight decay .01、warmup 5%、cosine、clip 1.0；Boundary/core/readout+probe LR 为 `1e-4/2e-4/3e-4`。1–400 updates 用 `.25 answer + 1.0 claim`，401–1800 用 `1.0 answer + .5 claim`，之后 answer-only。最早 1200、最多 2400 updates；通过后物理移除 claim probe，独立 reload 的 prediction hash 必须相同。

## 4. Baseline 与训练优化

direct/text-CoT seeds 为 `2026081814/2026081815`；官方 chat template、thinking disabled、同 episodes。只对顶部四层 attention/MLP Linear 注入 native LoRA rank 8、alpha 16、dropout 0，其他 Qwen 参数冻结；batch 1，最早 1200、最多 2400 updates，loss 只覆盖 assistant output。

训练 batch 采用 left padding，使 assistant target 保持在尾部；`logits_to_keep` 只保留 `maximum_target_tokens+1` 个 causal logits，之后与尾部 labels 精确移位对齐。FP16 autocast 只改变计算 dtype，LoRA master parameters 保持 FP32。此优化不得改变 chat tokens、targets、teacher-forced token 集或 greedy-generation Gate。prompt ≤1024，direct target ≤16，CoT target ≤512，禁止截断。

## 5. Gates 与停止

| Gate | 要求 |
| --- | --- |
| M01 | v3 cache qualification 全部检查通过 |
| M02 | ERE overfit64 answer ≥.95，update ≥1200，stripped reload prediction 相同 |
| M03 | CPS overfit64 answer ≥.95，update ≥1200，stripped reload prediction 相同 |
| M04 | shared overfit128：ERE/CPS 各 ≥.90，claim ≥.90，单一共享参数集 |
| M05 | direct SFT overfit64 greedy answer ≥.95 |
| M06 | text-CoT SFT overfit64 greedy answer ≥.95 |
| M07 | joint stripped checkpoint 连续 100 optimizer steps CUDA finite，无 OOM/fallback，记录 latency、examples/s、VRAM、utilization/power |
| M08 | 所有 fixed roots seal、seed/source hash、同 episode、公平 LoRA、no-truncation 与 conjunction 可重放 |

route root 只创建一次。首个失败停止当前正式顺序，保留失败根；可分析后另立新 version/root/seed 修复实现、数值或训练问题，但不得降低 Gate、换 episode、加 task 分支/答案旁路或改 baseline 为分类头。M01–M08 全 true 后写 `PASS_P0M`、`p1_eligible=true`、`p1_started=false` 并停止。
