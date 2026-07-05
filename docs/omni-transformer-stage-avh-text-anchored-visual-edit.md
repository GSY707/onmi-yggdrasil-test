# Stage AV-H：文本锚定视觉编辑数据集与训练准备

## 目标

Stage AV-H 替代 AV-F/AV-G 的高熵无锚点 external 互译任务。新任务不再要求 70M 模型凭空学习一套我们看不懂的高熵 token 语言，而是把文本描述、编辑指令、目标对象状态和图像输出放进同一条训练链：

```text
source text description
+ source image
+ edit instruction
-> target text description
-> target object record
-> target image
```

它和 Stage AI/AJ/AP 的区别是：Stage AI/AJ/AP 主要证明图像 output expert 能 copy/edit/generate；AV-H 要求文本成为统一语义锚点，目标 object record 成为中间门禁，图像编辑不能绕过文本/record。

## 数据集

新增生成器：

```powershell
experiments\omni_transformer_stage_avh_text_anchored_visual_edit_dataset.py
```

每条样本包含：

- `source_text`
- `edit_text`
- `target_text`
- `source_record`
- `target_record`
- `edit_type`
- `edit_object`
- `edit_value`

图像不作为 `.pt` 像素 shard 保存，而是训练时由 record 渲染：

- source image = render(`source_record`)
- target image = render(`target_record`)

这样仍然训练图像编辑输出专家，但不会把 10M 数据集变成大体积像素仓库。

### 10M Manifest

已生成：

```powershell
artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_dataset\dataset_10m\manifest.json
```

规模：

| 项 | 数值 |
| --- | ---: |
| total examples | 21,900 |
| train examples | 18,000 |
| val/test/heldout examples | 各 1,300 |
| tokens per example | 458 |
| total tokens | 10,030,200 |
| unique train tokens | 8,244,000 |
| 落盘大小 | 约 14.4 MB |

训练 split 中 edit 类型均衡：

- recolor：6,000
- reshape：6,000
- move：6,000

训练 split 中对象数大致均衡：

- 1 object：5,890
- 2 objects：6,058
- 3 objects：6,052

样例 PNG：

```powershell
artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_dataset\dataset_10m\samples\
```

## 训练入口

新增脚本：

```powershell
experiments\omni_transformer_stage_avh_text_anchored_visual_edit_training.py
```

阶段顺序：

| 阶段 | 默认 steps | 目标 |
| --- | ---: | --- |
| `text_latent` | 800 | source text + edit instruction -> target text / target record |
| `image_ground` | 1000 | source image -> source record |
| `edit_reason` | 1600 | source record/text + edit instruction -> target record / target text |
| `image_output` | 1800 | source image + target record -> target image patches |
| `joint_debug` | 800 | 短程整体接口调试 |

默认配置已经切到接近 70M 的档位：

- `d_model=704`
- `heads=11`
- `layers=10`
- `latent_tokens=12`
- `optimizer=adafactor`
- `batch_size=128`
- `micro_batch_size=64`
- 参数量：68,416,419

性能修复要点：

- 普通训练 step 不再把 GPU indices 转成 CPU `case_ids`。
- 普通训练 step 不再对 loss 分量做 `.item()`；只在 eval/log 时同步。
- record 渲染缓存 palette/centers/meshgrid，并移除 `bool(active.any())` 这类 GPU->CPU 同步点。
- 每个 stage 只解码和计算所需输出头，避免 text/image grounding 阶段也跑 image head。
- 默认使用 `Adafactor` 和 micro-batch 梯度累积，把 effective batch 128 拆成 2 个 micro batch 64，降低激活峰值。

## 已验证

已完成：

| 验证 | 结果 |
| --- | --- |
| dataset smoke | 896 examples、410,368 tokens |
| training CPU smoke | 5 个 stage 各 1 step，通过 manifest IO、record/text/image loss、checkpoint 和 samples 输出 |
| 10M loader smoke | 读取完整 10M manifest，通过 tiny CPU 训练入口 |
| 68M CUDA capacity smoke | `batch_size=128`，1 step 通过，peak CUDA allocated 6,537.45 MB |
| 68M CUDA perf-fix smoke | effective `batch_size=128`、`micro_batch_size=64`，10 step 通过，peak CUDA allocated 4,893.83 MB，训练窗口 1.52 step/s |
| non-leaky eval CPU smoke | 通过；训练脚本后续会同时报告 text_latent、image_ground、edit_reason、image_output、joint_teacher_record 和 joint_no_target_record |

性能修复后的推荐 smoke：

```powershell
artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_training\dataset10m_68m_batch128_micro64_perf_fix_smoke_result.json
```

## 建议长训命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avh_text_anchored_visual_edit_training.py `
  --dataset-manifest artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_dataset\dataset_10m\manifest.json `
  --output artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_training\train_10m_68m_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_training\train_10m_68m_checkpoints `
  --resume `
  --batch-size 128 `
  --micro-batch-size 64 `
  --d-model 704 `
  --heads 11 `
  --layers 10 `
  --latent-tokens 12 `
  --optimizer adafactor `
  --text-steps 800 `
  --image-ground-steps 1000 `
  --edit-reason-steps 1600 `
  --image-output-steps 1800 `
  --joint-steps 800 `
  --eval-every 500 `
  --save-every 500 `
  --eval-batch-size 64 `
  --sample-count 8
```

不要再用单次 `batch_size=128` 反传作为默认长训路径；修复前该路径在 10-step perf smoke 中 peak CUDA allocated 到 9GB 以上。`micro_batch_size=96` 已测可跑，但 peak CUDA allocated 约 6,696.90 MB 且训练窗口慢于 micro 64；除非实测功耗明显不够，否则默认用 micro 64。

## 10M 长训结果

用户本地已完成：

```powershell
artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_training\train_10m_68m_result.json
```

成本：

| 项 | 数值 |
| --- | ---: |
| 参数量 | 68,416,419 |
| steps | 6,000 |
| processed tokens | 351,744,000 |
| elapsed | 3,502.67 sec |
| peak CUDA allocated | 4,894.82 MB |

原始 result JSON 的主指标来自旧版 `joint_debug` 评估；该模式把 `target_record` 作为输入，因此 `target_record_exact` 有 teacher-forcing / copy 泄漏，不能作为端到端通过证据。

原始旧口径指标：

| split | target record exact | source record exact | target text exact | target image MSE |
| --- | ---: | ---: | ---: | ---: |
| test | 95.77% | 39.46% | 1.31% | 0.02619 |
| heldout | 92.69% | 25.08% | 0.00% | 0.03255 |

已补 stage-specific 诊断：

```powershell
artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_training\train_10m_68m_stage_mode_diagnostic.json
```

关键非泄漏指标：

| mode | test | heldout | 判断 |
| --- | ---: | ---: | --- |
| `text_latent` target record exact | 19.46% | 2.69% | 未闭合 |
| `image_ground` source record exact | 1.69% | 0.00% | 未闭合 |
| `edit_reason` target record exact | 14.00% | 3.46% | 未闭合 |
| `image_output` target image MSE | 0.04986 | 0.06156 | 有下降，但不能证明图像编辑 |
| `joint_no_target_record` target record exact | 19.38% | 1.46% | 未闭合 |
| `joint_no_target_record` source record exact | 30.62% | 5.92% | 未闭合 |

样例 contact sheet：

```powershell
artifacts\omni_transformer_stage_avh_text_anchored_visual_edit_training\samples\contact_sheet.png
```

样例显示预测图不是空图，但明显是模糊/纹理化输出，不是清晰 object scene。`target_image_mse` 会被大面积背景稀释，不能单独作为图像输出闭合证据。

结论：AV-H 10M 这次不通过。正信号是模型可以在 teacher-forced `target_record` 条件下把 record 复制/利用起来，并把 image MSE 降低；负信号是文本/图像到 target record 的非泄漏路径没有闭合，source image grounding 基本失败，target text 长序列 exact 也没有学起来。

训练脚本已在本次结果之后修正评估口径：未来 `val_target_record_exact` / `test_target_record_exact` 等兼容字段会映射到 `joint_no_target_record`，并额外保留 `joint_teacher_record` 指标用于区分 teacher-forced 上限。

## 通过门槛

不能只看 target image MSE。

必须同时看：

- `target_text_exact`
- `target_record_exact`
- `source_record_exact`
- `target_image_mse`
- samples 中 source/target/pred PNG

后续正式训练还应补：

- wrong-instruction drop
- wrong-source-image drop
- no-target-record drop
- edit type 分项指标

## 担忧

1. 当前 text 是合成描述，不是自然语言大语料潜空间；它只是显式语义锚点。
2. 图像由 object record 渲染，仍是受控合成视觉，不是真实图片。
3. 目标图像输出现在用 patch MSE；正式结论前还需要加 scene/object parser 指标，否则可能被背景或平滑输出误导。
4. 这个任务更符合架构，但不保证 68M 一次长训通过；如果 target record exact 先过、image MSE 不过，瓶颈在 output expert；如果 source record exact 不过，瓶颈在视觉 grounding。
