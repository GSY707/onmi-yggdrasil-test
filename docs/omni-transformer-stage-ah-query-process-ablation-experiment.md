# Stage AH：Query alignment 与 process supervision 逐项消融实验

## 目的

Stage AG 说明 teacher-forced multi-step query trace 在四任务上能打开正确读取后的上限，但 relation-only 仍失败。这里按可验证思路逐个测试：

1. CLIP-style query-object contrastive alignment：把 reasoner query 与 evidence pair token 做检索式对齐。
2. Process supervision：显式监督 left/right row/col 与 delta row/col。
3. 二者组合。
4. Detached-key query alignment：固定 evidence key 梯度，只训练 query tower，避免破坏 reader。

## 新增实现

`experiments/omni_transformer_stage_ac_latent_reasoning.py` 新增：

- `--query-alignment-mode none|logits`
- `--query-alignment-weight`
- `--query-alignment-detach-keys`
- `--process-supervision-weight`
- `--contrastive-temperature`
- relation trace 指标：`left/right_pair_retrieval`、`left/right_row`、`left/right_col`、`delta_row`、`delta_col`

默认值保持关闭，不影响 Stage AC-AG 的旧命令口径。

## 运行命令

### Query alignment only

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing none --query-alignment-mode logits --query-alignment-weight 1.0 --process-supervision-weight 0.0 --output-dir artifacts\omni_transformer_stage_ah_query_process_ablation\relation_query_alignment_runs --aggregate artifacts\omni_transformer_stage_ah_query_process_ablation\relation_query_alignment_results.json
```

### Process supervision only

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing train --query-alignment-mode none --query-alignment-weight 0.0 --process-supervision-weight 1.0 --output-dir artifacts\omni_transformer_stage_ah_query_process_ablation\relation_process_runs --aggregate artifacts\omni_transformer_stage_ah_query_process_ablation\relation_process_results.json
```

### Combined

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing train --query-alignment-mode logits --query-alignment-weight 1.0 --process-supervision-weight 1.0 --output-dir artifacts\omni_transformer_stage_ah_query_process_ablation\relation_combined_runs --aggregate artifacts\omni_transformer_stage_ah_query_process_ablation\relation_combined_results.json
```

### Detached-key query alignment

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing none --query-alignment-mode logits --query-alignment-weight 1.0 --query-alignment-detach-keys --process-supervision-weight 0.0 --output-dir artifacts\omni_transformer_stage_ah_query_process_ablation\relation_query_alignment_detached_runs --aggregate artifacts\omni_transformer_stage_ah_query_process_ablation\relation_query_alignment_detached_results.json
```

### 四任务 process-only

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing train --query-alignment-mode none --query-alignment-weight 0.0 --process-supervision-weight 1.0 --output-dir artifacts\omni_transformer_stage_ah_query_process_ablation\all_process_runs --aggregate artifacts\omni_transformer_stage_ah_query_process_ablation\all_process_results.json
```

## Relation-only 结果

| 设置 | full | no evidence | shuffled | teacher-forced queries | left/right query | reader pair row/col | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage AG baseline | 59.77% | 60.55% | 56.25% | 59.77% | 34.77%/44.92% | 97.31%/97.37% | 正确 query 仍不能打开 compare |
| query alignment only | 55.47% | 59.77% | 54.69% | 54.69% | 46.48%/47.27% | 38.72%/34.74% | query 有小幅改善，但破坏 reader |
| process only | 50.00% | 56.64% | 58.59% | 66.80% | 34.38%/36.33% | 100%/99.87% | 修了 teacher-forced compare 上限，但没修自由 query |
| combined | 55.86% | 55.86% | 51.17% | 59.77% | 43.75%/44.14% | 95.51%/93.85% | 两种 loss 互相牵制，没有闭合 |
| detached-key query alignment | 54.30% | 57.03% | 50.00% | 60.55% | 39.06%/41.80% | 98.40%/96.15% | 保住 reader，但 query 改善不足 |

## 四任务 process-only 结果

| 指标 | Stage AG | Stage AH process-only |
| --- | ---: | ---: |
| full answer word | 58.79% | 62.70% |
| no evidence | 16.80% | 16.60% |
| shuffled evidence | 21.29% | 24.41% |
| teacher-forced queries | 83.59% | 84.18% |
| relation family answer | 51.56% | 53.91% |
| reader pair row/col | 96.94%/95.99% | 99.56%/99.63% |

## 结论

1. CLIP-style 思路是对的，但当前“直接用 retrieval logits 替代 pair head”太粗糙：query 从 35%/45% 到 46%/47%，但 reader 表示被拉坏，答案仍靠先验。
2. Detached-key 版本证明问题不是必须破坏 reader：reader 保持约 96%-98%，但 query 只到 39%/42%。这说明固定 evidence tower 可以避免副作用，但 query tower 训练信号仍不足。
3. Process supervision 明确修复了“正确读取后的 relation compare 上限”：relation-only teacher-forced 从 59.77% 到 66.80%，四任务 teacher-forced 从 83.59% 到 84.18%。
4. Process supervision 没修自由 query：relation-only full 只有 50.00%，四任务 left/right query 仍约 35.16%/39.84%。
5. 组合实验没有自动叠加收益，说明 query alignment 与 reader/process loss 的梯度目标会互相牵制。下一步不能只加 loss，需要更结构化的双塔/对象 token 训练。

## 下一步

1. Query alignment 应先独立预训练：冻结 evidence codec 和 pair reader，训练 question-side object query tower；达到 90%+ retrieval 后再接 answer writer。
2. 正负样本应改成同场景 hard negatives + batch negatives，并区分 left object 与 right object role embedding。
3. relation compare 继续加更硬的 truth-table head：由 row/col/delta/op 直接监督 yes/no，不只监督 answer token。
4. scheduled sampling 应从 teacher-forced query 逐步切到 model query，避免训练和评估断层。
