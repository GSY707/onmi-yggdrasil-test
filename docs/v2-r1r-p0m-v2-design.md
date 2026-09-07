# V2-R1R P0-M v2 训练通路 Smoke 合同

## 1. 目标与停止边界

P0-D v17 已接受 production data 与 verifier。P0-M v2 只验证固定小样本上的工程可训练性：frozen-Qwen cache、Boundary、K=8 shared recurrent core、claim objective、direct SFT、text-CoT SFT 与 GPU throughput。PASS 只登记 `P1 eligible`，不得启动 P1；不产生 validation/OOD、组合泛化、因果干预、容量差或 Pareto 结论。

## 2. 固定输入与选择

唯一数据输入为 `artifacts/v2-r1r/p0d-v17-full-production-20260810-1`，evidence seal SHA-256 `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`。selection seed 固定 `2026081801`：ERE/CPS 各 label-balanced 64 条，联合模型使用相同 128 条；baseline 各取 32 条，direct/text-CoT 使用完全相同 64 episodes。不得因结果换样本、扩数据或降阈值。

## 3. Cache component 与 v2 资格判据

P0-M v1 在 `artifacts/v2-r1r/p0m-v1-cache-20260810-1` 已生成完整 frozen `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` FP16 full-token hidden cache，并通过内容 audit；其 aggregate FAIL 来自把绝对 batch latency 当作吞吐的错误规则，详见 `docs/v2-r1r-p0m-v1-failure-review.md`。

v2 把该 root 作为 immutable cache component，只读复核其 evidence seal、manifest/bank hashes、shape、finite、mask、selection、exact model revision、无截断和无 oracle/answer fields，并在新 root `artifacts/v2-r1r/p0m-v2-cache-qualification-20260810-1` 封存引用与判决。固定 inference batch 仍为 8。4/8/12 benchmark 必须全部 finite、peak allocation 均 ≤6 GiB，且 batch 8 examples/s 必须高于 batch 4；batch 12 只作为压力观测，不用一次 wall latency winner 决定合同。

## 4. 模型与训练冻结项

R1R-Latent 使用 non-affine LayerNorm(2048) → Linear(2048,512) → RMSNorm(512) Boundary；K=8 generic slots；两层共享 recurrent gated cross-attention/self-attention/SwiGLU；固定 Fourier slot 与 `(t,T)` control；最多 24 步；每层 source K/V 每次 forward 只计算一次。readout 只读取 `H_T` 并输出 9 local labels。禁止 task/operator/candidate embedding、oracle slot、task-specific transition 与 source-to-answer bypass。

claim hidden 经同一 Boundary，probe 只从相应 `H_t` 读 claim；claim 不进入 core/readout。latent seeds 固定 ERE `2026081811`、CPS `2026081812`、joint `2026081813`，batch 16，joint 每批两族各 8。训练 schedule、AdamW、分组 LR、最早 1200/最多 2400 updates 与 aux-stripped reload 规则保持 v1，不因失败调样本或阈值。

direct/text-CoT seeds 固定 `2026081814/2026081815`，官方 chat template、thinking disabled、同 episodes；只向顶部四层 attention/MLP Linear 注入 native LoRA rank 8、alpha 16、dropout 0，其余 Qwen 冻结。prompt/target 不截断，loss 只覆盖 assistant output。

## 5. Gates

| Gate | 要求 |
| --- | --- |
| M01 | v2 cache qualification 的 component seal、内容 audit、固定 batch 8 吞吐/显存判据全部通过 |
| M02 | ERE K=8 overfit64 answer ≥0.95，update ≥1200，stripped reload prediction hash 相同 |
| M03 | CPS K=8 overfit64 answer ≥0.95，update ≥1200，stripped reload prediction hash 相同 |
| M04 | shared K=8 overfit128：ERE/CPS 各 ≥0.90，claim ≥0.90，只有一套共享参数 |
| M05 | direct SFT overfit64 greedy-generation answer ≥0.95 |
| M06 | text-CoT SFT overfit64 greedy-generation answer ≥0.95 |
| M07 | joint stripped checkpoint 连续 100 optimizer steps CUDA finite，无 OOM/fallback，记录 latency、examples/s、VRAM、utilization/power |
| M08 | 所有 v2 fixed roots seal、seed、source hash、同 episode、公平 LoRA、no-truncation 与 conjunction 可重放；cache component seal 由 M01 嵌套验证 |

## 6. 修复权限

route root 只创建一次。失败 root 永远保留；首个失败即停止当前版本，分析后可另立新合同/root/seed修复训练、数值、mask、吞吐或实现问题。禁止降低上述 Gate、扩大数据、换 episode、加入 task 分支/答案旁路或把 baseline 改成分类头。
