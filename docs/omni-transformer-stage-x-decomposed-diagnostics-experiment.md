# Stage X：信息存在、对象化、跨专家互读与最终读头分解诊断

## 目的

Stage T 证明 `wide_residual_latent` 能保住答案所需信息；Stage W 证明更严格的跨 expert 语义互读仍然很弱。Stage X 把问题拆成四层：

1. **信息是否存在**：冻结 expert tokens，训练 answer-class probe，看答案信息能不能被读出。
2. **场景语义是否可读**：训练 4x4 semantic grid probe，读 occupancy/color/shape。
3. **不同 expert 是否互通**：source expert 上训练 semantic probe，直接评估 target expert。
4. **最终 scorer 是否会用**：冻结 expert tokens，只重新训练 candidate scorer，看读头是否是瓶颈。

同时测试参数量和 Transformer 读头：

- `grid_mlp_small`：316,719 参数。
- `grid_mlp_large`：839,279 参数。
- `grid_transformer`：338,703 参数，2-layer Transformer reader。
- answer probe 也对比 MLP 和 Transformer。

## 重要口径

Stage X 的 semantic `cell_info_accuracy` 必须和 empty-cell baseline 一起看。本任务平均每张图约 6.5 个对象，16 个 cell 中多数为空；永远预测 empty cell 的 baseline 约为 **59.1%**。所以 55%-60% 的 cell info 并不代表真正读懂了场景。

真正有意义的是：

- occupied color/shape 是否高。
- scene exact 是否高。
- patch/wide 与 object/spatial/count 之间的 transfer 是否能成立。
- object table 的 set recall/exact 是否能恢复对象集合。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_x_decomposed_diagnostics.py --sweep --seeds 20260701,20260702 --modes ranking_only,shared_grid_decoder --train-size 1024 --val-size 256 --test-size 384 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --fusion-tokens 8 --output-tokens 8 --candidate-count 8 --train-steps 320 --eval-every 160 --shared-loss-weight 0.30 --semantic-probe-steps 70 --answer-probe-steps 90 --object-probe-steps 110 --scorer-probe-steps 110 --probe-hidden-small 192 --probe-hidden-large 512 --transformer-probe-layers 2 --output-dir artifacts\omni_transformer_stage_x_decomposed_diagnostics\sweep_runs --aggregate artifacts\omni_transformer_stage_x_decomposed_diagnostics\sweep_results.json
```

正式 sweep：2 seeds，`20260701,20260702`。

## 主结果

候选答案 ranking：

| Model | Full Top-1 | No patch | Frozen scorer best |
| --- | ---: | ---: | ---: |
| ranking only | 13.02% | 12.76% | 15.76% patch tokens |
| shared grid decoder | 22.53% | 20.70% | **27.08% raw expert concat** |

`shared_grid_decoder` 的 frozen scorer 明显高于原模型 scorer，说明最终读头/训练也有瓶颈：相同冻结 token 上，单独重训 candidate scorer 可以从 22.53% 提到 27.08%。

## 语义读头与跨专家互读

整体 transfer：

| Model | Reader | Diagonal cell info | Offdiag cell info | Gap |
| --- | --- | ---: | ---: | ---: |
| ranking only | MLP small | 58.49% | 40.84% | 17.65% |
| ranking only | MLP large | 58.11% | 38.03% | 20.08% |
| ranking only | Transformer | 57.80% | 46.98% | 10.82% |
| shared grid decoder | MLP small | 55.95% | 42.80% | 13.15% |
| shared grid decoder | MLP large | 55.33% | 44.74% | 10.59% |
| shared grid decoder | Transformer | **63.82%** | 39.32% | 24.49% |

只看平均值会误导。shared + Transformer 的 diagonal 提升，主要来自 patch/wide：

| Source -> Target | Cell info | Occupancy | Occupied color | Occupied shape | Scene exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| patch -> patch | 89.45% | 98.82% | 99.56% | 77.26% | 16.02% |
| wide -> wide | 86.36% | 97.01% | 99.35% | 73.90% | 8.85% |
| object -> object | 49.78% | 66.81% | 71.38% | 45.84% | 0.00% |
| patch -> wide | 86.60% | - | - | - | - |
| wide -> patch | 81.17% | - | - | - | - |
| patch -> object | 11.71% | 46.57% | 38.53% | 30.74% | 0.00% |
| wide -> object | 34.75% | 60.16% | 50.03% | 36.75% | 0.00% |

结论：Transformer reader 能从 patch/wide token 读出真实语义，但不能让 object slots 变成同一种语言。问题不是单纯参数量不够；更像 object/spatial/count expert 本身没有形成对象级表示。

## Answer Reconstruction

answer-class 是 16 类全局答案，随机约 6.25%。

| Model | Reader | Wide latent | Raw expert concat | Object combo | No-patch visual concat |
| --- | --- | ---: | ---: | ---: | ---: |
| ranking only | MLP | 10.81% | 11.85% | 10.55% | 10.55% |
| ranking only | Transformer | 11.46% | 11.98% | 11.20% | 12.11% |
| shared grid decoder | MLP | 18.75% | 17.71% | 20.05% | 20.31% |
| shared grid decoder | Transformer | 18.36% | **20.96%** | 20.05% | 18.10% |

共享语义监督让答案信息从接近随机提高到约 18%-21%，但仍然不高。这说明 Stage W 的提升不是因为 latent 已经完整保存任务答案，而是获得了一些弱可用语义信号。

## Object Table

set recall/exact 忽略 slot 顺序，按真实对象数取 top-k predicted slots。

| Model | Source | Positive cell | Positive color | Positive shape | Set recall | Set exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ranking only | object slots | 19.48% | 15.08% | 25.59% | 1.71% | 0.00% |
| ranking only | patch tokens | 22.17% | 16.98% | 27.38% | 1.79% | 0.00% |
| shared grid decoder | object slots | 31.49% | 41.89% | 34.09% | 8.36% | 0.00% |
| shared grid decoder | patch tokens | **35.66%** | **44.61%** | **36.16%** | **11.45%** | 0.00% |
| shared grid decoder | wide latent | 32.09% | 43.31% | 34.30% | 8.15% | 0.00% |

这是最硬的负结果：即使 shared decoder 让 patch/wide 可读，object table 仍然几乎没有成型。Set exact 全部为 0，最高 set recall 也只有 11.45%。

## 结论

Stage X 支持“问题不止一个方面”：

1. **不是单纯参数量不够。** MLP large 没有稳定优于 MLP small。Transformer 能显著读取 patch/wide，但不能修复 object slots。
2. **patch/wide token 里有真实语义。** shared + Transformer 能把 patch/wide 读到 89.45% / 86.36%，occupied color 接近 99%。
3. **object/spatial/count experts 没有对象化。** object table set exact 为 0，set recall 最高 11.45%。这解释了为什么 no-object 消融不稳定掉点。
4. **跨 expert 语言仍不通。** patch/wide 互读较好，但 patch/wide -> object slots 很差，object slots 自身也低于 empty baseline 附近。
5. **最终 scorer 也弱。** 冻结 token 上单独训练 scorer 可把 shared 模型从 22.53% 提到 27.08%，说明 readout/training 还有损失。
6. **answer 信息仍不足。** answer reconstruction 最高 20.96%，远没到 Stage T 那种 L2/L5 90%+ 的保信息状态。

## 下一步

1. 先不要单纯加参数。更大的 MLP 没解决，Transformer 只证明 patch/wide 可读。
2. object slots 应直接切成 DETR-like set prediction：Hungarian matching、objectness、cell/box、color、shape，全程用 set loss，不再用排序 slot 作为主监督。
3. contrastive 要改成 object/cell-level：同一对象或同一 cell 在 patch/wide/object/spatial token 中拉近，不同对象/cell 拉远；scene-level pooled contrastive 太粗。
4. scorer 应换成 Transformer latent reader 或至少加入更强的 answer cross-attention 训练；但仍保持直读 latent，不引入推理期 common bus。
5. common semantic bus 更适合当 teacher：用它监督 patch/wide 已可读语义，再蒸馏到 object slots，而不是让 bus 成为最终运行时路径。
