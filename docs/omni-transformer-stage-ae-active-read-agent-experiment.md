# Stage AE：潜变量推理专家主动读取 agent 实验

## 目的

Stage AD 证明只调 latent reasoner、加入 readout/trace 能把四任务 answer word exact 从 40.04% 提到 74.41%。但它仍然是一次性把 evidence latent 放进推理专家，reasoner 需要同时学会看哪里、读什么、怎么绑定值和怎么写回答案 latent。

Stage AE 改成更接近 agent runtime 的主动读取：

1. Reasoner 先读 question latent。
2. Reasoner 产生 query latent/logits。
3. Queryable reader 从冻结 evidence latent 返回局部 observation latent。
4. Reasoner 用 observation 更新状态，写出 answer-token latent。
5. 冻结 answer decoder 输出文本。

文本 codec 和 evidence codec 仍按原方式训练；进入 reasoner 训练阶段后冻结，只训练 `LatentReasoner` 内部的 active query、reader readout 和 answer 写回参数。

## 改动

新增 `--reasoner-variant active_read`：

- `cell query`：用于 `color_at_cell` / `shape_at_cell`。
- `count-pair query`：用于 `count_color_shape`。
- `left/right object-pair query` + `relation-op query`：用于 `relation_yes_no`。
- `pair-object reader`：按 color+shape pair 从 evidence latent 中读对象是否存在、row、col。

新增数据 trace：

- `target_left_pair`
- `target_right_pair`
- `target_relation_op`
- pair-object reader 的 `occupied/single/row/col` 监督表

## 运行命令

1-step cell lookup：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families color_at_cell,shape_at_cell --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant active_read --output-dir artifacts\omni_transformer_stage_ae_active_read\cell_lookup_runs --aggregate artifacts\omni_transformer_stage_ae_active_read\cell_lookup_results.json
```

四任务同场：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant active_read --output-dir artifacts\omni_transformer_stage_ae_active_read\all_active_runs --aggregate artifacts\omni_transformer_stage_ae_active_read\all_active_results.json
```

count-only 与 relation-only 诊断：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families count_color_shape --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant active_read --output-dir artifacts\omni_transformer_stage_ae_active_read\count_only_runs --aggregate artifacts\omni_transformer_stage_ae_active_read\count_only_results.json
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant active_read --output-dir artifacts\omni_transformer_stage_ae_active_read\relation_only_runs --aggregate artifacts\omni_transformer_stage_ae_active_read\relation_only_results.json
```

## 结果

### 1-step cell lookup

| 指标 | 结果 |
| --- | ---: |
| Q/A text codec question/answer exact | 100.00% / 100.00% |
| Evidence codec occupancy/color/shape/count exact | 100.00% / 100.00% / 100.00% / 100.00% |
| Active reasoner full answer word exact | 90.23% |
| Active reasoner no evidence answer word exact | 23.44% |
| Active reasoner shuffled evidence answer word exact | 20.70% |
| `color_at_cell` answer word exact | 89.06% |
| `shape_at_cell` answer word exact | 91.41% |
| trace cell accuracy | 65.62% |
| trace color / shape accuracy | 91.41% / 90.62% |
| reader grid/count/pair exact | 100.00% / 100.00% / 100.00% |

### count-only

| 指标 | 结果 |
| --- | ---: |
| Active reasoner full answer word exact | 83.98% |
| Active reasoner no evidence answer word exact | 12.11% |
| Active reasoner shuffled evidence answer word exact | 8.59% |
| trace count-pair accuracy | 100.00% |
| trace count accuracy | 89.45% |
| reader count table exact | 58.98% |

### relation-only

| 指标 | 结果 |
| --- | ---: |
| Active reasoner full answer word exact | 54.69% |
| Active reasoner no evidence answer word exact | 61.33% |
| Active reasoner shuffled evidence answer word exact | 55.86% |
| trace relation accuracy | 58.98% |
| trace left/right pair accuracy | 34.38% / 28.91% |
| trace relation-op accuracy | 100.00% |
| reader pair row/col accuracy | 100.00% / 99.87% |

### 四任务同场

| 模块 | 指标 | 结果 |
| --- | --- | ---: |
| Q/A text codec | question sequence exact | 100.00% |
| Q/A text codec | answer sequence exact | 100.00% |
| Evidence codec | grid occupancy exact | 100.00% |
| Evidence codec | occupied color accuracy | 100.00% |
| Evidence codec | occupied shape accuracy | 100.00% |
| Evidence codec | count table exact | 91.80% |
| Active reasoner full | answer word exact | 69.73% |
| Active reasoner no evidence | answer word exact | 2.93% |
| Active reasoner shuffled evidence | answer word exact | 29.30% |
| Reader grid occupancy/color/shape | 100.00% / 100.00% / 100.00% |
| Reader pair occupancy/row/col | 99.80% / 100.00% / 99.93% |
| Reader count table exact | 74.80% |

按任务族：

| Family | answer word exact |
| --- | ---: |
| `color_at_cell` | 85.16% |
| `shape_at_cell` | 87.50% |
| `count_color_shape` | 58.59% |
| `relation_yes_no` | 47.66% |

查询/trace：

| Trace | Accuracy |
| --- | ---: |
| operation | 100.00% |
| cell | 62.11% |
| color | 82.81% |
| shape | 89.06% |
| count pair | 100.00% |
| count | 55.47% |
| relation | 51.56% |
| left pair | 38.28% |
| right pair | 32.03% |
| relation op | 100.00% |

另做高 trace 权重诊断：`--reasoner-trace-weight 2.0` 的四任务 full 为 69.34%，没有改善；reader count table 和 pair occupancy 反而下降。因此当前失败不是简单加大 trace loss 可以解决。

## 解释

Stage AE 的结论不是“active_read 比 Stage AD 更强”。四任务 full 从 Stage AD 的 74.41% 降到 69.73%。但 Stage AE 暴露了更接近最终架构的事实：

1. **主动读取能启动。** Cell lookup 达到 90.23%，count-only 达到 83.98%，且 no/shuffled 明显塌缩。
2. **主动读取更依赖证据。** 四任务 no-evidence 只有 2.93%，比 Stage AD 的 28.71% 更像真正的外部证据读取。
3. **reader 不是 relation 的主瓶颈。** relation-only 中 pair reader row/col 几乎 100%，但 left/right pair query 只有 34.38%/28.91%，no-evidence 还高于 full。
4. **真正失败点是 query 生成。** Reasoner 能读 cell、count-pair，也能读对象表；但不能稳定从自然语言 relation 问题中发出左右对象 query。
5. **强监督要更结构化。** 把 trace loss 权重从 1.0 提到 2.0 不够。需要 teacher-forced query、分步 imitation、query contrastive/retrieval、甚至先训练 query policy 再训练 answer 写回。

## 担忧

- 当前 active agent 只有一次读取，不是完整多轮循环；relation 本质上需要至少两次对象读取和一次比较。
- Relation 的 left/right pair query 监督虽然存在，但样本中仍然学不稳，说明问题解析到对象查询这一步需要更强结构。
- Cell query accuracy 仍只有 62%-66%，答案正确可能来自分布式 observation，不是完全可解释的硬查询。
- Count-only 高于四任务 count，说明多任务混训存在干扰；后续需要按 family 分阶段 curriculum。
- 仍是 single seed synthetic 结构化事实表，不证明真实多模态或真实工具环境。

## 下一步

1. 增加 teacher-forced active read：训练早期用真实 query 读取 observation，逐步切到模型 query。
2. 把 query policy 和 answer writer 分阶段训练：先让 query 命中目标，再训练 answer-token latent 写回。
3. 对 query 使用 contrastive/retrieval loss，而不是只用 CE。
4. Relation 改成显式多步：query left object -> observe row/col -> query right object -> observe row/col -> compare relation op -> answer。
5. Count 改成 accumulator 或专门 count-table reader，避免全表 reader exact 低但单点 count 靠目标泄漏过关。

## 产物

- `experiments/omni_transformer_stage_ac_latent_reasoning.py`
- `artifacts/omni_transformer_stage_ae_active_read/cell_lookup_results.json`
- `artifacts/omni_transformer_stage_ae_active_read/cell_lookup_runs/`
- `artifacts/omni_transformer_stage_ae_active_read/count_only_results.json`
- `artifacts/omni_transformer_stage_ae_active_read/count_only_runs/`
- `artifacts/omni_transformer_stage_ae_active_read/relation_only_results.json`
- `artifacts/omni_transformer_stage_ae_active_read/relation_only_runs/`
- `artifacts/omni_transformer_stage_ae_active_read/all_active_results.json`
- `artifacts/omni_transformer_stage_ae_active_read/all_active_runs/`
- `artifacts/omni_transformer_stage_ae_active_read/all_active_trace2_results.json`
- `artifacts/omni_transformer_stage_ae_active_read/all_active_trace2_runs/`
