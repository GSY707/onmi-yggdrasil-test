# Stage AB：文本 latent 对齐、latent-to-answer 专家与 MoE 推理实验

## 目的

Stage AA 说明 token 级对齐能降低信息丢失，但最终 scorer 没充分使用功能专家。本轮按新的假设做三项结构尝试：

- 对齐 `prompt -> latent` 与 `answer -> latent`，让问题文本和答案文本进入同一文本潜空间。
- 新增推理期 `latent -> answer` 专家，用 `wide_latent` 生成 answer latent，并对候选答案单独打分。
- 将潜变量推理从单体 Transformer reasoner 换成 routed MoE reasoner，测试 MoE 推理是否更适合专家聚合。

推理路径中，`text_latent_aligned` 的最终分数为：

```text
final_score = cross_attention_answer_score + 0.35 * latent_answer_score
```

`moe_text_latent_aligned` 在此基础上把 `LatentReasoner` 替换为 4-expert MoE Transformer reasoner。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ab_text_moe_alignment.py --sweep --seeds 20260701,20260702 --modes baseline_parallel,supervised_no_teacher,token_aligned_teacher,text_latent_aligned,moe_text_latent_aligned --train-size 1024 --val-size 256 --test-size 384 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --output-tokens 8 --reasoning-layers 1 --candidate-count 8 --train-steps 320 --eval-every 160 --semantic-probe-steps 55 --count-probe-steps 55 --answer-probe-steps 55 --token-teacher-align-weight 0.08 --token-cross-align-weight 0.04 --token-alignment-temperature 0.07 --text-latent-tokens 8 --text-align-temperature 0.07 --prompt-answer-align-weight 0.08 --latent-answer-align-weight 0.12 --latent-answer-rank-weight 0.20 --latent-answer-class-weight 0.20 --latent-answer-score-weight 0.35 --moe-experts 4 --moe-balance-weight 0.02 --output-dir artifacts\omni_transformer_stage_ab_text_moe_alignment\sweep_runs --aggregate artifacts\omni_transformer_stage_ab_text_moe_alignment\sweep_results.json
```

## 主结果

2 seeds 聚合，随机 top-1 为 12.5%。

| Model | Top-1 | No image | No patch | No object/spatial/count | Runtime params | Train-only params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_parallel | 19.79% | 15.36% | 13.93% | 19.01% | 1.44M | 0 |
| supervised_no_teacher | 19.79% | 14.45% | 20.18% | 13.93% | 1.44M | 0.22M |
| token_aligned_teacher | 21.09% | 12.76% | 19.27% | 18.49% | 1.44M | 1.30M |
| text_latent_aligned | **23.83%** | 14.45% | 19.01% | **23.44%** | 1.95M | 1.30M |
| moe_text_latent_aligned | 20.57% | 13.02% | 15.89% | 19.53% | 2.30M | 1.30M |

文本 latent 对齐 + latent-to-answer runtime scorer 是正向的：`text_latent_aligned` 高于 `token_aligned_teacher` 和 baseline。但它关闭 object/spatial/count 后几乎不掉点，说明收益主要来自 patch/text/answer 对齐路径，而不是功能专家被最终 scorer 更好地使用。

MoE 版本没有提升。它比 `text_latent_aligned` 低 3.25pp，说明当前朴素 MoE reasoner 只增加参数和路由复杂度，没有形成有效专家分工。

## 文本 latent 与 latent-to-answer

| Metric | text_latent_aligned | moe_text_latent_aligned |
| --- | ---: | ---: |
| prompt-answer retrieval top-1 | 3.78% | 3.13% |
| latent-answer retrieval top-1 | 4.04% | 3.26% |
| latent-answer candidate top-1 | **15.36%** | 13.02% |
| latent-answer class accuracy | 14.84% | **15.23%** |
| latent-answer cosine | 95.42% | 97.76% |

batch 内 retrieval 随机约为 1.56%，候选 top-1 随机为 12.5%。因此 text/answer latent 对齐有弱正信号，但还很不充分。

`latent-answer cosine` 很高而 retrieval 很低，说明存在 collapse 风险：latent 和 answer 可能被整体拉近，但样本间区分度不足。latent-to-answer 专家目前只能提供轻微候选区分能力，不足以独立承担输出。

## 信息丢失与互读

| Model | Semantic color | Semantic shape | Count positive diag | Count positive offdiag | Best answer reconstruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_parallel | 23.14% | 25.99% | 67.74% | **44.91%** | 18.23% |
| supervised_no_teacher | 34.38% | 30.91% | 75.66% | 26.19% | 18.23% |
| token_aligned_teacher | 64.15% | 46.88% | 62.50% | 29.39% | 17.32% |
| text_latent_aligned | **64.74%** | **47.57%** | **77.83%** | 33.02% | 16.80% |
| moe_text_latent_aligned | 60.30% | 42.50% | 68.45% | 29.69% | **20.05%** |

`text_latent_aligned` 在语义信息保留上仍然强，且 count positive diag 最高。但 answer reconstruction 没有同步提升，说明答案信息更多体现在 runtime scorer 的交互中，而不是 frozen latent probe 可以单独读出。

## MoE 诊断

`moe_text_latent_aligned` 的 MoE gate：

| Metric | Value |
| --- | ---: |
| gate entropy | 1.37 |
| gate max weight | 29.45% |

4 expert 的均匀 entropy 为 `log(4)=1.386`。当前 gate 接近均匀分配，说明 MoE 没有学出明显的任务/专家路由。这个结果不能否定 MoE 方向，但说明“只把 reasoner 换成 MoE + balance loss”不够。

## 结论

1. **对齐 prompt/answer 文本 latent 是正向尝试。** `text_latent_aligned` 达到 23.83%，高于同轮 `token_aligned_teacher` 的 21.09%。
2. **latent-to-answer runtime 专家有弱正信号，但还不能独立输出。** 单独候选 top-1 只有 15.36%，略高于 12.5% 随机。
3. **文本 latent 仍未形成高质量可分离空间。** retrieval 只有 3%-4%，且 cosine 很高，提示 collapse。
4. **朴素 MoE 推理没有成功。** gate 近似均匀，最终 top-1 低于非 MoE 文本对齐模型。
5. **下一步应强化输出端和路由监督。** 更合理的下一版不是继续加全局 align loss，而是让 latent-to-answer 专家使用 autoregressive/contrastive decoder，并给 MoE router 明确的任务族或证据类型监督。

## 产物

- `experiments/omni_transformer_stage_ab_text_moe_alignment.py`
- `artifacts/omni_transformer_stage_ab_text_moe_alignment/sweep_results.json`
- `artifacts/omni_transformer_stage_ab_text_moe_alignment/sweep_runs/`
