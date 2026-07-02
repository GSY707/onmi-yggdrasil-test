# Stage S：Scorer 与信息还原诊断实验

## 目的

Stage R 显示 `MoE Attention-Pump latent` 只稳定解决 L1 单颜色识别。Stage S 按新的怀疑点拆解：

1. 先把最终 scorer 从 `mean(latent) dot mean(answer)` 改成候选答案 token cross-attend context tokens。
2. 再移除 Attention Pump，让候选答案直接 cross-attend 加权后的专家 token。
3. 如果仍差，就从各层 representation 训练 reconstruction probe，逆向恢复答案类别，定位信息在哪一层丢失。

## 代码与产物

- `experiments/omni_transformer_stage_s_scorer_reconstruction.py`
- `artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_results.json`
- `artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_runs/`
- `artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_runs/*/*/samples/`

## 模型对照

| 名称 | 改动 |
| --- | --- |
| Stage R MoE mean-dot | 原始路径：expert tokens -> Attention Pump -> thought latent -> mean pool -> dot answer |
| `cross_pump_latent` | 保留 Attention Pump，但最终 scorer 改成 answer cross-attention over latent tokens |
| `cross_no_pump_expert_tokens` | 移除 Attention Pump，answer 直接 cross-attend weighted expert tokens |

Reconstruction probe：

- no-pump 模型上测试 `vision`、`prompt`、`fusion`、`expert_concat`、`weighted_expert_concat`。
- pump 模型上测试 `pump_latent`、`thought_latent`。

## 正式命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_s_scorer_reconstruction.py --sweep --seeds 20260701,20260702,20260703 --levels l1_color,l2_object,l3_position,l4_count,l5_relation --train-size 1536 --val-size 384 --test-size 512 --batch-size 64 --image-size 64 --prompt-len 96 --answer-len 96 --d-model 96 --layers 2 --heads 4 --latent-tokens 8 --candidate-count 8 --no-pump-steps 360 --pump-steps 600 --probe-steps 160 --eval-every 180 --output-dir artifacts\omni_transformer_stage_s_scorer_reconstruction\sweep_runs --aggregate artifacts\omni_transformer_stage_s_scorer_reconstruction\sweep_results.json
```

正式 3 seed 平均每 seed 约 378.44 秒。`cross_pump_latent` 约 1,261,829 trainable params。

## Scorer 结果

Top-1，3 seeds：

| Level | Random | Stage R MoE mean-dot | Cross scorer + pump | Cross scorer + no pump | No-pump no image |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 color | 12.50% | 100.00% | 16.47% | 100.00% | 11.59% |
| L2 object | 12.50% | 46.03% | 9.96% | 94.40% | 13.22% |
| L3 position | 12.50% | 16.60% | 12.30% | 26.95% | 11.98% |
| L4 count | 20.00% | 19.40% | 19.40% | 19.99% | 19.99% |
| L5 relation | 12.50% | 25.20% | 12.96% | 77.67% | 12.17% |

补充校准：L3 no-pump 单 seed 从 360 steps 的 12.11%-26.95% 区间，拉长到 1000 steps 后达到 77.9%。所以 L3 不是完全不可学，而是 no-pump cross scorer 对颜色+形状+位置组合需要更长训练。

## Reconstruction 结果

Exact answer-class reconstruction，3 seeds：

| Level | no-pump vision | no-pump fusion | no-pump expert concat | pump latent | thought latent |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 color | 100.00% | 100.00% | 100.00% | 32.49% | 16.08% |
| L2 object | 96.94% | 92.51% | 93.16% | 2.73% | 3.19% |
| L3 position | 8.14% | 4.30% | 7.10% | 0.72% | 0.85% |
| L4 count | 19.99% | 19.99% | 19.34% | 21.03% | 20.25% |
| L5 relation | 55.92% | 69.79% | 83.14% | 12.37% | 13.22% |

Prompt-only reconstruction 在 L1/L2/L5 接近随机，说明 no-pump 的提升来自图像/融合 token，不是答案先验。

## 判断

Stage S 明确支持“信息在 Attention Pump / latent 路径丢失”的怀疑：

1. 只改最终 scorer、仍走 pump，不但没有修复，L1/L2/L5 基本掉到随机附近。
2. 去掉 pump，让答案 cross-attend weighted expert tokens，L2 从 Stage R MoE 的 46.03% 提到 94.40%，L5 从 25.20% 提到 77.67%，且 no-image 回到随机附近。
3. Reconstruction probe 显示 no-pump 的 vision/fusion/expert_concat 在 L2/L5 中能恢复大量答案信息，但 pump/thought latent 基本不能恢复。
4. L4 counting 仍然所有路径随机，说明它不是 pump 单点问题，而是缺对象级计数归纳偏置。

## 下一步

- 不要继续沿用当前 Attention Pump 作为唯一压缩器。
- 下一轮应测试保信息 latent：增加 latent token 数、去掉 expert scalar gating、使用 residual passthrough、Perceiver-style 多层 cross-attention、或者让 answer cross-attend latent 前保留 object/slot tokens。
- 对 L3 位置任务，应继续用 no-pump 或保信息 latent 做长训练，确认是否能稳定逼近 Stage R direct。
- 对 L4 counting，应单独加入 object-slot/counting expert，否则 scorer/pump 修改不会解决。
