# Stage AF：MoE 推理专家与分任务阶段训练实验

## 目的

Stage AE 说明主动读取能启动，但 relation 的左右对象 query policy 没学起来。Stage AF 测试两个想法：

1. 在 latent reasoner 内部加入 MoE 推理专家，让不同任务族走不同 answer writer / relation state writer。
2. 用分任务阶段训练引导专家功能分化，避免一个单体 reasoner 同时学 lookup、count、relation。

文本 codec 和 evidence codec 仍按原方式训练；进入 reasoner 阶段后冻结，只训练 latent reasoner。

## 改动

新增 `--reasoner-variant moe_active`：

- 保留 Stage AE 的 queryable reader。
- 新增 4 个 answer writer experts，对应 `color_at_cell`、`shape_at_cell`、`count_color_shape`、`relation_yes_no`。
- 新增 4 个 relation state experts。
- 新增 `moe_gate_head`，监督 gate 对齐任务族。

新增训练参数：

- `--reasoner-training-mode mixed|staged`
- `--reasoner-stage-order`
- `--reasoner-stage-replay-interval`
- `--moe-teacher-forcing`

`--moe-teacher-forcing` 只在训练时把 answer writer 的梯度送到目标专家；推理评估时仍使用模型自己的 gate。

## 运行命令

四任务 staged：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant moe_active --reasoner-training-mode staged --reasoner-stage-order color_at_cell,shape_at_cell,count_color_shape,relation_yes_no --moe-teacher-forcing --output-dir artifacts\omni_transformer_stage_af_moe_staged\all_staged_runs --aggregate artifacts\omni_transformer_stage_af_moe_staged\all_staged_results.json
```

四任务 mixed：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant moe_active --reasoner-training-mode mixed --moe-teacher-forcing --output-dir artifacts\omni_transformer_stage_af_moe_staged\all_mixed_runs --aggregate artifacts\omni_transformer_stage_af_moe_staged\all_mixed_results.json
```

四任务 staged + replay：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant moe_active --reasoner-training-mode staged --reasoner-stage-order color_at_cell,shape_at_cell,count_color_shape,relation_yes_no --reasoner-stage-replay-interval 4 --moe-teacher-forcing --output-dir artifacts\omni_transformer_stage_af_moe_staged\all_staged_replay4_runs --aggregate artifacts\omni_transformer_stage_af_moe_staged\all_staged_replay4_results.json
```

relation-only MoE：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant moe_active --reasoner-training-mode mixed --moe-teacher-forcing --output-dir artifacts\omni_transformer_stage_af_moe_staged\relation_only_moe_runs --aggregate artifacts\omni_transformer_stage_af_moe_staged\relation_only_moe_results.json
```

## 结果

### 四任务 staged

| 指标 | 结果 |
| --- | ---: |
| full answer word exact | 13.09% |
| no evidence answer word exact | 11.52% |
| shuffled evidence answer word exact | 12.30% |
| MoE gate accuracy | 25.00% |
| MoE gate entropy | 0.108 |
| color/shape/count/relation answer exact | 0.00% / 0.00% / 0.00% / 52.34% |

单向 staged 训练出现明显灾难性遗忘：最后阶段训练的是 relation，前三个任务几乎归零；gate 熵很低但准确率只有 25%，说明 gate 塌到了单一专家附近。

### 四任务 mixed

| 指标 | 结果 |
| --- | ---: |
| full answer word exact | 69.34% |
| no evidence answer word exact | 26.56% |
| shuffled evidence answer word exact | 26.56% |
| MoE gate accuracy | 100.00% |
| MoE gate entropy | 0.034 |
| color/shape/count/relation answer exact | 87.50% / 91.41% / 56.25% / 42.19% |

MoE gate 可以被监督到 100%，但整体没有超过 Stage AE active-read 的 69.73%，也没有超过 Stage AD readout 的 74.41%。

### 四任务 staged + replay

| 指标 | 结果 |
| --- | ---: |
| full answer word exact | 64.45% |
| no evidence answer word exact | 24.80% |
| shuffled evidence answer word exact | 28.52% |
| MoE gate accuracy | 100.00% |
| MoE gate entropy | 0.204 |
| color/shape/count/relation answer exact | 80.47% / 86.72% / 48.44% / 42.19% |

每 4 步插入 mixed replay 可以修复 gate 和大部分遗忘，但仍低于 mixed MoE，也低于 Stage AE/AD。

### relation-only MoE

| 指标 | 结果 |
| --- | ---: |
| full answer word exact | 53.12% |
| no evidence answer word exact | 61.33% |
| shuffled evidence answer word exact | 57.03% |
| MoE gate accuracy | 100.00% |
| left/right pair query accuracy | 35.94% / 30.08% |
| relation-op query accuracy | 100.00% |
| pair reader row/col accuracy | 100.00% / 99.94% |

专门 relation expert 也没有解决 relation：reader 能读对象坐标，但 reasoner 仍不能从问题稳定生成左右对象 query。

## 解释

Stage AF 的主要结论是负向但有用：

1. **MoE gate 不是瓶颈。** mixed 和 staged+replay 中 gate 都能到 100%。
2. **单向分阶段训练会灾难性遗忘。** 不加 replay 时，前三个任务几乎归零。
3. **replay 能救遗忘，但不能超过混合训练。** staged+replay 从 13.09% 回到 64.45%，但仍低于 mixed 的 69.34%。
4. **专家分化没有自动修复 query policy。** Relation-only MoE 仍然 left/right pair query 只有 35.94% / 30.08%。
5. **MoE 当前更像路由正确的多头写回器，不是流程控制专家。** 它能按任务族分专家，但没有学会“先生成对象查询，再比较坐标”的流程。

## 担忧

- 当前 staged 只是顺序训练，不含正则化、冻结、蒸馏或充分 replay；它验证了 naive staged 会坏，不代表所有 curriculum 都不行。
- MoE teacher forcing 让专家收到目标任务梯度，但评估时 gate 自己路由；这适合测专家分化，但还不是完整自主路由训练。
- Relation 的失败仍可能需要流程控制专家或显式多步 teacher-forced trace，而不是只加 MoE。
- Single seed synthetic 结果，只能作为方向筛查。

## 下一步

1. 若继续 staged，应采用 replay buffer / distillation / EWC 类防遗忘机制，而不是单向阶段训练。
2. Relation 应先做 teacher-forced multi-step trace：left query、left observation、right query、right observation、compare、answer。
3. Query policy 应单独预训练到高命中率，再接 answer writer；当前把 query 和 answer 一起训会让 relation 走答案先验。
4. MoE 专家分化可以保留，但暂时不应作为修复 relation 的核心手段；核心仍是 query policy 和流程控制。

## 产物

- `experiments/omni_transformer_stage_ac_latent_reasoning.py`
- `artifacts/omni_transformer_stage_af_moe_staged/all_staged_results.json`
- `artifacts/omni_transformer_stage_af_moe_staged/all_staged_runs/`
- `artifacts/omni_transformer_stage_af_moe_staged/all_mixed_results.json`
- `artifacts/omni_transformer_stage_af_moe_staged/all_mixed_runs/`
- `artifacts/omni_transformer_stage_af_moe_staged/all_staged_replay4_results.json`
- `artifacts/omni_transformer_stage_af_moe_staged/all_staged_replay4_runs/`
- `artifacts/omni_transformer_stage_af_moe_staged/relation_only_moe_results.json`
- `artifacts/omni_transformer_stage_af_moe_staged/relation_only_moe_runs/`
