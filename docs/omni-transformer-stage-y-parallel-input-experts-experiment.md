# Stage Y：并行直读输入专家架构修正实验

## 目的

Stage U/W/X 暴露了一个架构偏差：`object_slots`、`spatial_tokens`、`count_tokens` 不是并行读取外部图像，而是在 `patch_tokens` 后继续串联变换；`fusion_tokens` 又读取所有前级 token。这会把“外部信息转潜变量专家”“潜变量推理”“潜变量输出”混在一起，链路过长，信息损失和专家语言不通都更难定位。

Stage Y 只修正这个拓扑问题，继续使用 Stage U 的多物体视觉问答数据和候选答案 scorer，比较三种结构：

- `serial_chain`：Stage U 旧结构，作为串联 baseline。
- `parallel_direct`：`patch_vision`、`object_vision`、`spatial_vision`、`count_vision` 都直接读取图像，`prompt_text` 直接读取文本，然后交给 latent reasoner。
- `merged_direct`：一个更宽的视觉输入专家直接读取图像，吐出 object/spatial/count token slices，然后交给 latent reasoner。

运行时口径被切回三类：

1. 外部信息转 latent：图像输入专家与文本输入专家。
2. latent 推理：`LatentReasoner` 聚合 input expert tokens。
3. latent 输出：answer scorer 从 latent context 评分候选答案。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_y_parallel_input_experts.py --sweep --seeds 20260701,20260702 --variants serial_chain,parallel_direct,merged_direct --train-size 1024 --val-size 256 --test-size 384 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --output-tokens 8 --reasoning-layers 1 --candidate-count 8 --train-steps 260 --eval-every 130 --probe-steps 70 --output-dir artifacts\omni_transformer_stage_y_parallel_input_experts\sweep_runs --aggregate artifacts\omni_transformer_stage_y_parallel_input_experts\sweep_results.json
```

## 主结果

2 seeds 聚合，随机 top-1 为 12.5%。

| Model | Top-1 | MRR | No image | No patch | No object/spatial/count | Params | ms/example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serial_chain | 11.07% | 33.36% | 11.07% | 11.85% | 11.07% | 1.15M | 0.259 |
| parallel_direct | **18.62%** | **39.93%** | 12.37% | 13.02% | **18.62%** | 1.44M | 0.296 |
| merged_direct | 14.32% | 36.40% | 14.19% | 14.19% | 14.58% | 0.94M | 0.249 |

按任务族看，`parallel_direct` 全部高于 `serial_chain`：

| Family | serial_chain | parallel_direct | merged_direct |
| --- | ---: | ---: | ---: |
| cell attribute | 11.46% | **17.71%** | 9.90% |
| spatial relation | 8.85% | **20.31%** | 16.15% |
| counting | 13.02% | **17.19%** | 18.23% |
| spatial filter | 10.94% | **19.27%** | 13.02% |

## 关键解释

Stage Y 支持这个判断：串联输入专家是错误拓扑。把视觉专家改成并行直读图像后，`parallel_direct` 从 `serial_chain` 的 11.07% 提到 18.62%，并且 no-image/no-patch 会掉到接近随机，说明它确实开始使用图像。

但 Stage Y 没有证明 object/spatial/count 专家已经有效分工。`parallel_direct` 的 `no_object_spatial_count` 与完整模型相同，说明当前收益主要来自 `patch_vision + latent reasoner`，不是来自 object/spatial/count 三个功能专家。也就是说，拓扑修正减少了路径问题，但功能专家自然分化仍然失败。

Answer reconstruction 也只小幅高于随机 6.25%：

| Model | wide latent answer probe | raw expert concat answer probe |
| --- | ---: | ---: |
| serial_chain | 8.98% | 8.72% |
| parallel_direct | **10.03%** | **9.51%** |
| merged_direct | 8.98% | 8.72% |

这说明并行直读让 scorer 更容易用到一点视觉信号，但 latent 中可线性读出的答案信息仍然很弱，远不到“信息充分保真”。

## 结论

1. **架构修正成立。** 输入专家应直接读取外部信息，之后才进入 latent 推理；Stage U 的串联专家不符合目标架构。
2. **并行直读优于旧串联。** 在同任务、同 scorer、同训练规模下，`parallel_direct` 明显高于 `serial_chain`。
3. **合并专家没有明显优势。** `merged_direct` 参数更少、速度略快，但准确率只接近随机略上，不如并行直读。
4. **功能专家仍未学会分工。** object/spatial/count 消融不掉点，说明这些专家还没有独立承载对应语义。
5. **下一步不应继续加长链路。** 应保留并行直读拓扑，再给 object/spatial/count 专家显式监督，例如 object/cell-level set loss、slot matching、cell-level contrastive 或 teacher distillation。

## 产物

- `experiments/omni_transformer_stage_y_parallel_input_experts.py`
- `artifacts/omni_transformer_stage_y_parallel_input_experts/sweep_results.json`
- `artifacts/omni_transformer_stage_y_parallel_input_experts/sweep_runs/`
