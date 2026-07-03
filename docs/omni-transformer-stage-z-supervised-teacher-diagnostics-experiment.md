# Stage Z：显式监督、训练期教师与潜变量互读诊断

## 目的

Stage Y 证明输入专家应并行直读外部图像，但 object/spatial/count 专家没有自然分工。Stage Z 在 Stage Y `parallel_direct` 拓扑上加入训练期监督和教师：

- `object_slots`：slot objectness/color/shape/cell 显式监督。
- `spatial_tokens`：4x4 semantic grid 的 occupancy/color/shape 显式监督。
- `count_tokens`：color-shape count table 显式监督，并对非零对象 pair 加权。
- `supervised_teacher`：训练期 patch-derived semantic teacher bus，teacher 接受 grid supervision；object/spatial/count token 通过 distillation 对齐 teacher cell tokens。

teacher 只在训练期和诊断中使用，推理路径仍然是 input experts 直出 latent，再进入 latent reasoner 和输出 scorer。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_z_supervised_teacher_diagnostics.py --sweep --seeds 20260701,20260702 --modes baseline_parallel,supervised_no_teacher,supervised_teacher --train-size 1024 --val-size 256 --test-size 384 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --output-tokens 8 --reasoning-layers 1 --candidate-count 8 --train-steps 320 --eval-every 160 --semantic-probe-steps 55 --count-probe-steps 55 --answer-probe-steps 55 --output-dir artifacts\omni_transformer_stage_z_supervised_teacher_diagnostics\sweep_runs --aggregate artifacts\omni_transformer_stage_z_supervised_teacher_diagnostics\sweep_results.json
```

## 主结果

2 seeds 聚合，随机 top-1 为 12.5%。

| Model | Top-1 | No image | No patch | No object/spatial/count | Runtime params | Train-only params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_parallel | 20.31% | 15.36% | 14.32% | 19.92% | 1.44M | 0 |
| supervised_no_teacher | 22.66% | 12.24% | 22.66% | 12.24% | 1.44M | 0.22M |
| supervised_teacher | **26.30%** | 16.15% | 20.96% | 20.31% | 1.44M | 0.68M |

显式监督解决了 Stage Y 的关键问题：`supervised_no_teacher` 关闭 object/spatial/count 后从 22.66% 掉到 12.24%，说明功能专家开始真正承载任务信息。`baseline_parallel` 关闭这些专家几乎不掉点，仍是 patch 主导。

teacher 进一步提高最终任务准确率到 26.30%，但关闭 object/spatial/count 后仍有 20.31%，说明 teacher 让 patch/teacher 路径更强，功能专家重要性低于 `supervised_no_teacher`。

## 训练头诊断

训练期监督头能读出专家内部的部分语义：

| Metric | supervised_no_teacher | supervised_teacher |
| --- | ---: | ---: |
| object positive cell accuracy | 59.69% | **62.60%** |
| object positive color accuracy | 27.92% | 28.70% |
| spatial occupied color accuracy | 76.15% | **79.73%** |
| spatial cell info accuracy | **55.17%** | 53.68% |
| count positive pair accuracy | **83.46%** | 82.36% |
| teacher bus cell info accuracy | - | **98.61%** |
| teacher bus scene exact | - | **80.73%** |

这说明 teacher 本身很强，能从 patch tokens 还原接近完整场景。但 student latent 没有完全继承 teacher 的语义，特别是 object slots 的 color 仍很弱。

## 信息丢失

冻结模型后重新训练 probe，避免只看训练头。

| Model | Semantic occupied color | Semantic occupied shape | Best answer reconstruction |
| --- | ---: | ---: | ---: |
| baseline_parallel | 19.27% | 26.39% | 12.76% |
| supervised_no_teacher | 38.75% | 35.83% | **21.22%** |
| supervised_teacher | **52.90%** | **40.56%** | 19.27% |

结论：

- 显式监督显著降低语义信息丢失。
- teacher 进一步降低视觉语义丢失，尤其 occupied color。
- 但 answer 信息最强的是 `supervised_no_teacher`，说明 teacher 提高了最终 scorer 表现，却没有把答案相关信息更好地压进可 probe 的 shared latent。

`cell_info_accuracy` 不能单独看，因为空 cell baseline 约 59%。本报告更重视 occupied color/shape、answer reconstruction 和消融。

## 潜变量互读

互读口径：在 source latent 上训练 reader，再直接评估 target latent。offdiag 越高，说明不同 expert 越像同一种语言。

| Model | Semantic diag | Semantic offdiag | Semantic gap | Count positive diag | Count positive offdiag |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_parallel | 58.04% | 36.87% | 21.18% | 61.70% | **37.86%** |
| supervised_no_teacher | 54.09% | 36.41% | 17.69% | **72.90%** | 27.58% |
| supervised_teacher | **59.21%** | **37.42%** | 21.79% | 67.69% | 26.83% |

显式监督和 teacher 没有真正解决“潜变量语言互通”。它们让各自 expert 更会做本职任务，但跨 expert transfer 没明显提高。count 的 positive offdiag 反而下降，说明专家更专门化，也更不互读。

## 结论

1. **显式监督必要。** 它让 object/spatial/count 从“可关掉”变成真正影响任务结果的输入专家。
2. **teacher 有效但不充分。** teacher bus 自身几乎完整理解场景，并把任务 top-1 推到 26.30%，但 student latent 只部分继承。
3. **信息丢失下降。** occupied color/shape 和 answer reconstruction 均高于 baseline。
4. **互读仍未解决。** 专家之间不是同一种 latent 语言；显式监督强化了专业能力，但没有自动产生 shared semantics。
5. **下一步应把 teacher 蒸馏从 scene/cell level 改成 object/cell 对齐。** 例如同一对象在 patch/object/spatial/count 中的 token 级 contrastive，或者 DETR-style set matching 后对齐对象 token，而不是只用全局 cell bus MSE。

## 产物

- `experiments/omni_transformer_stage_z_supervised_teacher_diagnostics.py`
- `artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/sweep_results.json`
- `artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/sweep_runs/`
