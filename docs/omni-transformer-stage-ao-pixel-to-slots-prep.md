# Stage AO：像素到 cell/count slots 训练前准备

## 目标

Stage AO 接在 Stage AM/AN/AQ 之后，把输入从结构化 evidence 推到合成像素图像。核心问题不是重新证明 unified bus，而是验证：

1. pixel image expert 能否从图像写出一等 `cell slots`、`count slots`、`pair slots`。
2. 这些 slots 能否接回现有统一 bus、answer-token writer 与 relation truth-table 过程状态。
3. count 任务失败时能否明确归因到像素专家 / count slots，而不是笼统归因到 reasoner。

## 已完成的训练前准备

1. 新增 `experiments/omni_transformer_stage_ao_pixel_to_slots.py`。
2. 复用 Stage AC 的四任务数据生成：
   - `color_at_cell`
   - `shape_at_cell`
   - `count_color_shape`
   - `relation_yes_no`
3. 新增 64x64 合成像素渲染：
   - 4x4 grid。
   - 每个 object 用颜色 + 形状符号画入对应 cell。
   - 训练输入只走 `images`，不把结构化 `evidence` 送进模型。
4. 新增 pixel-to-slots 前端：
   - cell patch encoder。
   - cell occupancy/color/shape heads。
   - pair slots 由 cell feature + position latent 聚合得到。
   - count slots 由 cell-level soft pair counts 写入。
5. 复用 Stage AK/AQ 后端：
   - text query retrieval。
   - decoded position compare。
   - answer-token writer。
   - relation delta/truth-table process state。
   - `zero_image` 消融。
6. 聚合 JSON 已输出 AO 主门禁字段：
   - `full_pixel_cell_occupancy_exact`
   - `full_pixel_cell_color_accuracy`
   - `full_pixel_cell_shape_accuracy`
   - `full_pixel_count_table_exact`
   - `full_count_table_exact`
   - `full_model_answer_sequence_exact`
   - `full_answer_count_color_shape_sequence_exact`
   - `full_answer_relation_yes_no_sequence_exact`
   - `no_image_model_answer_sequence_exact`

## Smoke 与短 probe

### Smoke

已完成 2 step smoke：

- 聚合结果：`artifacts/omni_transformer_stage_ao_pixel_to_slots/smoke_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_ao_pixel_to_slots/smoke_runs/seed20260701.json`

smoke 只证明脚本、训练循环、评估、`zero_image` 消融和聚合 JSON 能跑通，不代表训练效果。

### 短 probe

已完成 80 step 单 seed probe：

- 聚合结果：`artifacts/omni_transformer_stage_ao_pixel_to_slots/probe_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_ao_pixel_to_slots/probe_runs/seed20260701.json`

关键结果：

| 指标 | 80 step probe |
| --- | ---: |
| `full_pixel_cell_color_accuracy` | 100.00% |
| `full_pixel_cell_shape_accuracy` | 100.00% |
| `full_cell_occupancy_exact` | 100.00% |
| `full_count_table_exact` | 10.94% |
| `full_model_answer_sequence_exact` | 31.25% |
| `full_answer_count_color_shape_sequence_exact` | 18.75% |
| `full_answer_relation_yes_no_sequence_exact` | 68.75% |
| `no_image_model_answer_sequence_exact` | 21.88% |

结论：像素颜色/形状与 cell slots 有明显学习信号；count table 仍弱，正式训练必须把 count 指标当主门禁。不能只因为总 answer 有提升就判定 AO 通过。

## 正式训练门禁

正式 3 seed 训练完成后，主要看这些指标：

| 指标 | 通过门槛 |
| --- | ---: |
| `full_pixel_cell_occupancy_exact` 或 `full_cell_occupancy_exact` | >= 95% |
| `full_pixel_cell_color_accuracy` | >= 95% |
| `full_pixel_cell_shape_accuracy` | >= 95% |
| `full_pixel_count_table_exact` 或 `full_count_table_exact` | >= 95% |
| `full_answer_count_color_shape_sequence_exact` | >= 90% |
| `full_model_answer_sequence_exact` | >= 85% |
| `no_image_model_answer_sequence_exact` | 明显低于 full |

如果 color/shape 高但 count 低，本阶段应优先修 pixel counting expert 或 count-slot 写入，不应该回退到结构化 evidence，也不应该把失败归因到 answer writer。

## 正式训练结果

正式 3 seed 已完成：

- 聚合结果：`artifacts/omni_transformer_stage_ao_pixel_to_slots/formal_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_ao_pixel_to_slots/formal_runs/seed20260701.json`、`seed20260702.json`、`seed20260703.json`

| 指标 | 3 seed 平均 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| `full_pixel_cell_occupancy_exact` | 100.00% | >= 95% | 通过 |
| `full_pixel_cell_color_accuracy` | 100.00% | >= 95% | 通过 |
| `full_pixel_cell_shape_accuracy` | 100.00% | >= 95% | 通过 |
| `full_pixel_count_table_exact` | 81.45% | >= 95% | 未通过，前端 soft count 诊断仍弱 |
| `full_count_table_exact` | 100.00% | >= 95% | 通过 |
| `full_answer_count_color_shape_sequence_exact` | 100.00% | >= 90% | 通过 |
| `full_model_count_value_count_color_shape_accuracy` | 99.74% | 诊断项 | 基本闭合 |
| `full_model_answer_sequence_exact` | 100.00% | >= 85% | 通过 |
| `full_answer_relation_yes_no_sequence_exact` | 100.00% | 诊断项 | 通过 |
| `no_image_model_answer_sequence_exact` | 26.82% | 明显低于 full | 通过 |

按 seed 看，`full_pixel_count_table_exact` 分别为 75.78%、75.59%、92.97%；但 `full_count_table_exact` 三个 seed 都是 100%，`count_color_shape` answer-token sequence exact 三个 seed 也都是 100%。

结论：Stage AO 的下游 count slots、count answer 和全任务 answer 已闭合；没有证据表明 answer writer 或 unified bus 挂在 count。真正未过门槛的是 `pixel_count_table_exact` 这个前端辅助诊断，它来自 pixel 前端的 soft count 分布，不等同于最终 count slots 经过 learned `count_head` 后的 count table。后续如果继续推进 AO，应该修 pixel count auxiliary head / soft count 诊断一致性，而不是回退到结构化 evidence 或重做 answer writer。

## 正式训练开始命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ao_pixel_to_slots.py --sweep --seeds 20260701,20260702,20260703 --train-size 2048 --val-size 512 --test-size 512 --batch-size 256 --image-size 64 --d-model 96 --layers 2 --heads 4 --steps 2400 --eval-every 600 --answer-len 6 --token-loss-weight 1.0 --process-loss-weight 1.0 --truth-table-state-weight 4.0 --pixel-loss-weight 1.0 --selected-count-loss-weight 0.25 --output-dir artifacts\omni_transformer_stage_ao_pixel_to_slots\formal_runs --aggregate artifacts\omni_transformer_stage_ao_pixel_to_slots\formal_results.json
```

如果 8GB 显存紧张，先把 `--batch-size` 降到 `128`，不要先降 `--steps`、`--d-model` 或删除 count 门禁。batch 变小只影响吞吐，不改变本轮架构问题。

## 训练中检查

1. 控制台是否每个 seed 都输出 `test` 和 `cost`。
2. `formal_runs/seed*.json` 是否逐个落盘。
3. `formal_results.json` 的 `summary.full_pixel_cell_*` 是否先闭合。
4. `summary.full_count_table_exact` 是否跟上 cell/color/shape。
5. `summary.no_image_model_answer_sequence_exact` 是否明显低于 `summary.full_model_answer_sequence_exact`。

## 担忧和不确定点

1. 当前图片是低熵合成图，不证明真实图像检测能力。
2. 80 step probe 里 count table 仍弱，正式训练有可能首先暴露 count slots 写入不足。
3. `zero_image` 下 relation 仍可能保留 yes/no 二分类先验，因此 no-image gap 要结合 count/color/shape 任务一起看。
4. 当前 AO 没有 Stage AJ 那样的 checkpoint/resume。正式命令单 seed 预计不长；如果后续扩大图像尺寸、步数或模型，应先补 checkpoint/resume。
5. pixel 前端目前按 4x4 cell patch 直切，适合验证“像素到 slots”的闭环，不是通用 detector/segmentation 架构。
