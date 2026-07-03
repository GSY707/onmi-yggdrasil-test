# Stage AA：token 级对齐与潜变量互读诊断

## 目的

Stage Z 证明显式监督和训练期 teacher 能降低信息丢失，但没有解决专家之间的 latent 语言互通。Stage AA 在同一个 Stage Y `parallel_direct` 推理拓扑上加入 token 级对齐：

- `patch_tokens`、`object_slots`、`spatial_tokens`、`count_tokens`、`reasoning_tokens` 都重采样到 4x4 cell-token 位置。
- `token_aligned_teacher` 对这些 cell token 施加同位置 teacher 对齐：MSE + in-batch same-position contrastive。
- 同时对不同专家之间的同位置 cell token 做 cross-expert contrastive。
- teacher、token aligner 和所有监督头仍然只在训练期/诊断期使用；推理路径仍是各专家直出 latent，经 latent reasoner 和 answer scorer 输出。

本轮还修正了 Stage Z 诊断中的一个控制变量：probe 读头和各 mode 基础模型显式设置 torch seed，使新增 mode 不会改变旧 mode 的读头初始化序列。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_aa_token_alignment_diagnostics.py --sweep --seeds 20260701,20260702 --modes baseline_parallel,supervised_no_teacher,supervised_teacher,token_aligned_teacher --train-size 1024 --val-size 256 --test-size 384 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --output-tokens 8 --reasoning-layers 1 --candidate-count 8 --train-steps 320 --eval-every 160 --semantic-probe-steps 55 --count-probe-steps 55 --answer-probe-steps 55 --token-teacher-align-weight 0.08 --token-cross-align-weight 0.04 --token-alignment-temperature 0.07 --output-dir artifacts\omni_transformer_stage_aa_token_alignment_diagnostics\sweep_runs --aggregate artifacts\omni_transformer_stage_aa_token_alignment_diagnostics\sweep_results.json
```

## 主结果

2 seeds 聚合，随机 top-1 为 12.5%。

| Model | Top-1 | No image | No patch | No object/spatial/count | Runtime params | Train-only params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_parallel | 19.01% | 12.89% | 13.93% | 18.75% | 1.44M | 0 |
| supervised_no_teacher | **24.61%** | 12.24% | 24.09% | 12.63% | 1.44M | 0.22M |
| supervised_teacher | 19.92% | 12.89% | 16.80% | 14.06% | 1.44M | 0.68M |
| token_aligned_teacher | 22.79% | 14.06% | 16.80% | 21.09% | 1.44M | 1.30M |

token 级对齐没有超过 `supervised_no_teacher`，但超过本轮同 seed 控制下的 `supervised_teacher`。更关键的是，`token_aligned_teacher` 关闭 object/spatial/count 后只从 22.79% 降到 21.09%，说明最终 scorer 仍主要走 patch/reasoning 路径，没有充分依赖被对齐后的功能专家。

## 训练头诊断

| Metric | supervised_no_teacher | supervised_teacher | token_aligned_teacher |
| --- | ---: | ---: | ---: |
| object positive cell accuracy | 57.47% | 45.16% | **61.28%** |
| object positive color accuracy | 26.07% | 20.53% | **38.64%** |
| spatial occupied color accuracy | 75.10% | 71.06% | **83.11%** |
| spatial occupied shape accuracy | 49.61% | 35.91% | **74.76%** |
| count positive pair accuracy | **82.42%** | 68.71% | 73.72% |
| teacher bus cell info accuracy | - | 94.89% | **98.49%** |
| teacher bus scene exact | - | 42.97% | **78.52%** |

token 级对齐明显提高了 object/spatial 的可读语义，尤其 color/shape；teacher bus 也更完整。它不是“没有学到”，而是学到的信息没有被最终任务头充分使用。

## 信息丢失

冻结模型后重新训练 probe，避免只看训练头。

| Model | Semantic occupied color | Semantic occupied shape | Count positive diag | Best answer reconstruction |
| --- | ---: | ---: | ---: | ---: |
| baseline_parallel | 23.30% | 26.26% | 67.60% | 16.93% |
| supervised_no_teacher | 38.48% | 34.21% | 73.54% | **19.53%** |
| supervised_teacher | 50.45% | 38.75% | 68.31% | 17.32% |
| token_aligned_teacher | **65.70%** | **48.13%** | **77.69%** | 17.97% |

结论：token 级对齐进一步降低视觉语义信息丢失，尤其 occupied color/shape 和 count positive diag。但 answer reconstruction 没有同步提高，说明对齐后的语义信息还没有稳定转换成任务答案信息。

## 潜变量互读

互读口径：在 source latent 上训练 reader，再直接评估 target latent。offdiag 越高，说明不同 expert 越像同一种语言。

| Model | Semantic diag | Semantic offdiag | Semantic gap | Count positive diag | Count positive offdiag |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_parallel | 56.22% | 43.55% | 12.66% | 67.60% | **47.36%** |
| supervised_no_teacher | 54.87% | 34.05% | 20.82% | 73.54% | 25.48% |
| supervised_teacher | 58.44% | 32.32% | 26.12% | 68.31% | 26.04% |
| token_aligned_teacher | **59.27%** | 37.04% | 22.23% | **77.69%** | 31.86% |

与 `supervised_teacher` 相比，token 级对齐确实改善了互读：

- semantic offdiag：32.32% -> 37.04%。
- count positive offdiag：26.04% -> 31.86%。
- count positive diag：68.31% -> 77.69%。

但它没有把 gap 消掉，且仍低于 baseline 的 count offdiag。baseline 的 offdiag 较高不代表更好，因为 baseline 的功能专家没有明确分工，关闭 object/spatial/count 几乎不影响任务。Stage AA 的问题是“专家更懂语义，但 scorer 没把这些专家当主路径”。

## Token 对齐本身

`token_aligned_teacher` 的训练期 token 对齐指标：

| Metric | Value |
| --- | ---: |
| teacher same-position retrieval top-1 | 14.28% |
| cross-expert same-position retrieval top-1 | 26.81% |
| teacher cosine | 46.81% |
| cross-expert cosine | 73.74% |

随机同位置检索大约是 1 / (batch * cells) = 1 / 1024 = 0.10%。因此 token 对齐本身显著高于随机，说明同位置 latent 被拉近了。但 teacher retrieval top-1 仍只有 14.28%，说明对齐还远不到“统一语义坐标系”。

## 结论

1. **token 级对齐有效，但只解决了一部分。** 它显著降低 object/spatial/count 的信息丢失，并让跨 expert transfer 从 `supervised_teacher` 的低位恢复一部分。
2. **还没有证明更优。** 最终 top-1 为 22.79%，低于 `supervised_no_teacher` 的 24.61%；训练成本也更高。
3. **瓶颈从“信息不存在”转向“信息没被用上”。** object/spatial/count 的训练头语义更强，但消融掉点很小，说明最终 scorer/route 没有把对齐后的专家作为主证据。
4. **下一步不应继续只加 align loss。** 更应该改最终 aggregator/scorer，让答案候选显式 cross-attend 到 object/spatial/count 的 aligned cell tokens，或者加入 expert usage supervision，否则对齐信息停留在辅助头里。

## 产物

- `experiments/omni_transformer_stage_aa_token_alignment_diagnostics.py`
- `artifacts/omni_transformer_stage_aa_token_alignment_diagnostics/sweep_results.json`
- `artifacts/omni_transformer_stage_aa_token_alignment_diagnostics/sweep_runs/`
