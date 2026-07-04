# Stage AM：历史最高难度统一 bus 实验

## 目的

Stage AK/AL 只验证了 relation-only：先把 object/pair latent bus 练硬，再接 compare 和 answer writer。Stage AM 把难度拉回此前四任务最高设置：

- `color_at_cell`
- `shape_at_cell`
- `count_color_shape`
- `relation_yes_no`

目标不是做 MoE，而是验证统一潜空间在 cell、count、pair/relation 三类 slot 都成型后，是否能支撑多任务答案输出。

## 关键实现变化

`experiments/omni_transformer_stage_ak_unified_latent_bus.py` 现在包含三类一等 slot：

- pair/object slots：保存 color-shape pair 的对象位置事实，用于 relation。
- cell slots：保存 grid cell 的 occupancy/color/shape，用于 cell lookup。
- count slots：由 count 输入专家把每个 color-shape pair 的计数写入 latent slot，用于 count lookup。

训练循环也做了吞吐修正：

- train/val/test tensor 一次性常驻 GPU。
- CUDA 下启用 AMP。
- 正式 run 使用 `batch-size=256`，避免小 batch 让 GPU 低功率空等。

## 运行命令

### 最终 all-task

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701 --families color_at_cell,shape_at_cell,count_color_shape,relation_yes_no --train-size 2048 --val-size 512 --test-size 512 --batch-size 256 --d-model 96 --layers 2 --heads 4 --steps 2400 --eval-every 600 --output-dir artifacts\omni_transformer_stage_am_hardest_unified_bus\position_compare_all_task_runs --aggregate artifacts\omni_transformer_stage_am_hardest_unified_bus\position_compare_all_task_results.json
```

### Count-only 诊断

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701 --families count_color_shape --train-size 2048 --val-size 512 --test-size 512 --batch-size 256 --d-model 96 --layers 2 --heads 4 --steps 1600 --eval-every 400 --output-dir artifacts\omni_transformer_stage_am_hardest_unified_bus\count_expert_only_runs --aggregate artifacts\omni_transformer_stage_am_hardest_unified_bus\count_expert_only_results.json
```

## 结果

最终 all-task：

| 指标 | full | no evidence |
| --- | ---: | ---: |
| model answer accuracy | 98.24% | 27.54% |
| color answer accuracy | 100.00% | 25.00% |
| shape answer accuracy | 100.00% | 20.31% |
| count answer accuracy | 100.00% | 12.50% |
| relation answer accuracy | 92.97% | 52.34% |
| count table exact | 100.00% | 0.00% |
| selected count value | 100.00% | 12.50% |
| cell retrieval | 100.00% | 100.00% |
| count-pair retrieval | 100.00% | 100.00% |
| left/right pair retrieval | 100.00% / 100.00% | 100.00% / 100.00% |
| teacher/model compare | 93.75% / 93.75% | 49.22% / 49.22% |

count-only 诊断中，count table、selected count value 和 count answer 全部达到 100%；no-evidence count answer 为 9.57%。

## 失败对照

这轮最有价值的发现来自失败对照：

1. 只把 count head 接在 pair slots 上时，all-task count answer 只有 42.19%。
2. 独立 count slots 但仍用 softmax attention 读取 objects 时，count answer 提到约 54%-56%，但 count family 内的 selected count 仍很差。
3. 给 selected count 直接加 loss 没有解决问题，反而破坏 count table。
4. 非归一化 additive/sigmoid 聚合也失败，count table exact 几乎归零。
5. 切成一等 count 输入专家后，count-only 和 all-task 的 count 都闭合。

## 结论

1. 用户提醒是正确的：没有 first-class `cell/count slots` 时，历史最高难度任务会在 count 上挂。
2. cell slot 用 attention 结构可以闭合；count slot 不能靠普通 attention 自然学出来，需要一个明确的 count 输入专家或等价的计数归纳偏置。
3. relation 在 all-task 中一开始只有约 80%，不是 query 失败，而是 raw pair slot compare 被多任务表征干扰。把 compare context 切到 decoded row/col position state 后，relation answer 提到 92.97%。
4. 当前 Stage AM 证明的是合成结构化 evidence 上的统一 bus 闭合，不证明真实视觉计数或真实语言生成。真实图像里还需要 detector/segmentation/counting expert 来产生同等质量的 count slots。

## 下一步

1. 把 answer writer 从分类头升级回 Stage AC 的 answer-token latent writer。
2. 把 count 输入专家从当前结构化 evidence 版本替换为可训练的 object detector/counting expert，再测真实或更高熵视觉任务。
3. relation 还没有回到 Stage AL 的 99.61%，下一步可加入 relation truth-table/delta slots 或更硬的 process supervision。
4. 做多 seed；当前 Stage AM 仍是单 seed 方向验证。
