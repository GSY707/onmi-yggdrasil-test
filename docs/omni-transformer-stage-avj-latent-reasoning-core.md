# Stage AV-J：潜空间状态推理核心训练准备

日期：2026-07-06

## 目标

Stage AV-J 接续“先打通潜空间思考”的路线。它不训练像素图，也不追整图 latent bank，而是先验证：

```text
source scene record + text operation
-> latent workspace
-> active read / process state
-> target scene record + answer tokens
```

本阶段相比 Stage AK/AN/AQ 多了一步：之前主要是从结构化 evidence 读值并回答；AV-J 要求模型在潜空间中改写世界状态，输出完整 `target_record`。因此它更接近后续图像编辑前的核心推理门。

## 任务族

脚本：

```powershell
experiments\omni_transformer_stage_avj_latent_reasoning_core.py
```

当前包含三类 record 级状态变换：

| family | 要求 | 主要测试点 |
| --- | --- | --- |
| `conditional_recolor` | 读取两个对象，判断 relation 条件，再按条件改写其中一个对象颜色 | relation -> condition -> edit slot/color |
| `same_row_move` | 读取 anchor，找到同 row 且列距离最近的对象，移动到目标 cell | active read、对象绑定、position update |
| `count_delete_or_add` | 统计某 color/shape pair，若数量大于 1 删除最右对象，否则新增对象 | count -> branch -> delete/add target record |

每条样本都有 teacher trace：

- `read_a`
- `read_b`
- `edit_slot`
- `condition`
- `edit_action`
- `edit_color` / `edit_row` / `edit_col`

这些 trace 只作为训练监督和诊断指标，推理期不作为输入。

## 模型和训练

模型组件：

- source record object encoder；
- Transformer latent workspace；
- prompt/text operation encoder；
- 三个 query head：`read_a`、`read_b`、`edit_slot`；
- process state head：condition/action/edit field；
- target record decoder；
- answer-token decoder。

默认 schedule：

| 阶段 | 默认 steps | 目标 |
| --- | ---: | --- |
| `codec` | 200 | source record -> latent slots -> source record reconstruction |
| `operation` | 400 | operation text -> read/edit slot query |
| `process` | 600 | read slots + operation -> process state + target record |
| `joint` | 300 | target record + answer tokens 短程整体调试 |

评估会同时报告：

- `full_*`
- `no_source_*`
- `no_operation_*`
- `no_process_*`

主指标是 `target_record_exact`、`answer_sequence_exact`、`read_*_accuracy`、`condition_accuracy` 和 `edit_action_accuracy`。如果 answer 高但 target record 低，仍判失败。

## 已验证

CPU smoke：

```powershell
artifacts\omni_transformer_stage_avj_latent_reasoning_core\cpu_smoke_result.json
```

- 8 step，四阶段各 2 step。
- 验证 dataset、stage schedule、loss、eval variants、checkpoint 写入和结果 JSON。
- 该 smoke 不提供能力结论。

CPU resume smoke：

```powershell
artifacts\omni_transformer_stage_avj_latent_reasoning_core\cpu_resume_smoke_result.json
```

- 从 `cpu_smoke_checkpoints/latest.pt` 恢复后继续到 9 step。
- 验证 checkpoint/resume 链路可用。

CUDA capacity smoke：

```powershell
artifacts\omni_transformer_stage_avj_latent_reasoning_core\cuda_capacity_smoke_result.json
```

配置：

| 项 | 数值 |
| --- | ---: |
| train / val / test / heldout | 512 / 128 / 128 / 128 |
| d_model / layers / heads | 192 / 3 / 4 |
| batch size | 128 |
| steps | 8 |
| 参数量 | 4,863,087 |
| device | CUDA |

CUDA smoke 只证明 AMP、GPU resident data、checkpoint 和 eval 变体可跑，不证明能力。

70M 档 CUDA capacity：

```powershell
artifacts\omni_transformer_stage_avj_latent_reasoning_core\capacity_70m_batch256_2step_result.json
```

配置：

| 项 | 数值 |
| --- | ---: |
| d_model / layers / heads | 480 / 8 / 8 |
| batch size | 256 |
| steps | 2 |
| 参数量 | 71,694,351 |
| peak CUDA allocated | 4,629.29 MB |
| elapsed | 84.31 sec |

注意：`d_model=768/layers=10/heads=12` 在 AV-J 当前三套 Transformer 结构下不是 70M，而是约 225.8M；batch512 的 20-step capacity 在本轮 180 秒工具窗口内未完成，未产出 JSON。因此正式 70M 命令已改为 `d_model=480/layers=8/heads=8`。

单元测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_stage_avj_latent_reasoning_core.py
```

结果：2 passed。覆盖数据中确实存在 record edit，以及模型 forward/loss/metrics schema。

## 建议大型训练命令

大型训练由用户启动：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avj_latent_reasoning_core.py `
  --output artifacts\omni_transformer_stage_avj_latent_reasoning_core\train_70m_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avj_latent_reasoning_core\train_70m_checkpoints `
  --resume `
  --train-size 200000 `
  --val-size 4096 `
  --test-size 4096 `
  --heldout-size 4096 `
  --batch-size 256 `
  --eval-batch-size 512 `
  --d-model 480 `
  --layers 8 `
  --heads 8 `
  --codec-steps 2000 `
  --operation-steps 4000 `
  --process-steps 8000 `
  --joint-steps 3000 `
  --eval-every 1000 `
  --save-every 1000
```

如果显存或功耗不理想，先把 `--batch-size` 降到 128。当前已做 71.7M batch256 2-step capacity；如果你想在正式长训前再看更长窗口，可以跑 20 step capacity：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avj_latent_reasoning_core.py `
  --output artifacts\omni_transformer_stage_avj_latent_reasoning_core\capacity_70m_batch256_20step_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avj_latent_reasoning_core\capacity_70m_batch256_20step_checkpoints `
  --train-size 4096 `
  --val-size 512 `
  --test-size 512 `
  --heldout-size 512 `
  --batch-size 256 `
  --eval-batch-size 512 `
  --d-model 480 `
  --layers 8 `
  --heads 8 `
  --codec-steps 5 `
  --operation-steps 5 `
  --process-steps 5 `
  --joint-steps 5 `
  --eval-every 10 `
  --save-every 20
```

## 通过标准

正式结果至少要同时满足：

- `test.full_source_record_exact` 高；
- `test.full_target_record_exact` 高；
- `test.full_answer_sequence_exact` 高；
- `test.full_read_a_accuracy`、`test.full_read_b_accuracy`、`test.full_edit_slot_accuracy` 高；
- `test.full_condition_accuracy`、`test.full_edit_action_accuracy` 高；
- `no_source`、`no_operation`、`no_process` 相比 full 明显下降；
- heldout 不明显崩塌。

必须判失败：

- answer sequence exact 高，但 target record exact 低；
- full 与 no_source/no_operation/no_process 接近；
- trace 指标高但 target record 不高；
- 只在 train/val 高，heldout 大幅下降。

## 担忧

1. 当前 AV-J 是 record/text 核心，不证明图像 grounding；图像外设应在本阶段通过后再接。
2. 当前 `no_process` 是禁用 process state 的结构消融，不等价于“完全没有训练期 trace”。它用于检查 target decoder 是否依赖过程态。
3. 三个任务族仍是受控合成任务，正式训练后如果过快饱和，应扩大 heldout 组合、增加多步 chain 和 hard negative。
4. 70M batch256 2-step 已确认可启动，但 20-step capacity 仍建议由用户本地跑，以观察更稳定的功耗和吞吐。

## 70M 长训结果

用户本地已完成：

```powershell
artifacts\omni_transformer_stage_avj_latent_reasoning_core\train_70m_result.json
artifacts\omni_transformer_stage_avj_latent_reasoning_core\train_70m_summary.json
```

成本：

| 项 | 数值 |
| --- | ---: |
| 参数量 | 71,694,351 |
| train / val / test / heldout | 200,000 / 4,096 / 4,096 / 4,096 |
| batch size | 256 |
| schedule | 2,000 codec + 4,000 operation + 8,000 process + 3,000 joint |
| elapsed | 2,982.64 sec |
| peak CUDA allocated | 5,007.75 MB |

最终主指标：

| split | source record exact | target record exact | answer sequence exact | read_a | read_b | edit_slot | condition | edit_action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 99.85% | 67.92% | 98.73% | 92.46% | 72.63% | 71.48% | 98.97% | 99.76% |
| test | 99.78% | 67.77% | 98.78% | 91.46% | 72.22% | 71.19% | 98.51% | 99.66% |
| heldout | 99.95% | 67.19% | 98.97% | 91.31% | 72.56% | 71.17% | 98.97% | 99.71% |

消融：

| split | no_source target | no_source answer | no_operation target | no_operation answer | no_process target | no_process answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 8.62% | 86.13% | 0.00% | 20.39% | 11.74% | 80.32% |
| heldout | 8.98% | 85.23% | 0.00% | 19.02% | 12.06% | 79.74% |

分任务：

| split | conditional_recolor target / answer | same_row_move target / answer | count_delete_or_add target / answer |
| --- | ---: | ---: | ---: |
| test | 91.00% / 97.00% | 62.12% / 100.00% | 50.18% / 99.34% |
| heldout | 90.34% / 97.51% | 61.09% / 100.00% | 50.12% / 99.41% |

结论：**本轮不通过 AV-J 主门槛。**

正信号：

1. source record codec 已基本闭合，test source record exact 为 99.78%。
2. operation / process 的粗粒度判断很强，condition 与 edit_action 都接近 99%。
3. no_operation target record exact 为 0，说明文本 operation 对 target record 有强因果作用。
4. heldout 与 test 接近，没有明显只记训练集。

失败点：

1. target record exact 只有约 67%-68%，没有完成“潜空间状态改写”。
2. answer sequence exact 接近 99%，但 no_source/no_process answer 仍有 80%-86%，说明 answer head 很大程度走任务模板/动作标签捷径。
3. read_b 与 edit_slot 只有约 72%/71%，与 target record exact 的上限接近；当前主要瓶颈是对象选择和编辑槽绑定。
4. `same_row_move` 约 61%-62%、`count_delete_or_add` 约 50%，明显低于 `conditional_recolor` 的约 90%，说明多步绑定和 count-conditioned add/delete 是主失败点。

下一步不应继续只拉长同一配置。应先修 AV-J-B：

- 把 answer loss 后移或降权，避免 answer 先走模板捷径。
- target record decoder 增加 per-slot edit delta / copy-vs-update gate，而不是让每个 target slot 从全局 process state 自己猜。
- 对 `read_b`、`edit_slot` 加 hard negative / contrastive slot loss，尤其 same-row nearest 和 count delete/add。
- 把 `count_delete_or_add` 拆出 explicit count slot、selected delete slot、free insert cell 三个过程态，再让 target record 使用这些过程态。
- 报告字段级 target accuracy，当前只有 record exact，定位还不够细。
