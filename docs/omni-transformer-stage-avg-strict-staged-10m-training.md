# Stage AV-G：严格分阶段 10M 训练准备

## 目标

Stage AV-G 是 AV-F 10M 数据集上的严格架构训练入口。它不触碰正在跑的 AV-E joint baseline，不复用 `train_10m_70m_checkpoints`，而是使用独立脚本、独立输出和独立 checkpoint。

本阶段按当前架构设定训练：

```text
text latent base
-> external codec
-> bidirectional external/latent translation
-> latent reason / answer head
-> short joint debug
```

这替代 AV-E 当前命令里的“一步 joint multi-loss”。AV-E joint baseline 可以作为负/对照结果保留，但后续正式路线应以 AV-G 这种 stage schedule 为准。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_avg_strict_staged_10m_training.py
```

默认读取：

```powershell
artifacts\omni_transformer_stage_avf_10m_bidirectional_dataset\dataset_10m\manifest.json
```

默认输出：

```powershell
artifacts\omni_transformer_stage_avg_strict_staged_10m_training\train_10m_70m_strict_result.json
artifacts\omni_transformer_stage_avg_strict_staged_10m_training\train_10m_70m_strict_checkpoints\
```

这些路径与正在跑的 AV-E joint baseline 分离。

## Stage Schedule

默认 6000 step：

| 阶段 | steps | 目标 | 主要 loss |
| --- | ---: | --- | --- |
| `text_latent_base` | 400 | 先用 text-operation positions 建初始文本/操作潜空间 | source/target text-position reconstruction |
| `external_codec` | 1200 | 外部表征和 latent 互译的 codec 基础 | source reconstruction、target reconstruction |
| `bidirectional_translate` | 2200 | source latent -> target external，target latent -> source external | source->target、target->source、latent alignment，少量 recon replay |
| `latent_reason` | 1400 | target latent 支撑 answer head | answer loss，少量 codec/translation replay |
| `joint_debug` | 800 | 最后短程整体调试 | AV-E joint loss |

前面阶段刻意不练太满，给后续 translation/reason/joint 留训练空间。

## 已验证

为了不影响正在跑的 CUDA 长训，本轮只做 CPU smoke：

- `strict_cpu_smoke_result.json`：5 个阶段各 1 step，验证 stage 切换、loss、manifest IO、checkpoint、samples 输出。
- `strict_cpu_resume_smoke_result.json`：从同一 checkpoint resume 后继续 1 step，验证 `latest.pt` 恢复链。
- 两次 smoke 都写入 `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/`，不触碰 AV-E joint baseline 的输出目录。

没有在本轮跑 70M CUDA capacity smoke，原因是用户正在跑 AV-E 10M 训练；避免抢 GPU 或污染功耗判断。

## 建议启动命令

等当前 AV-E joint baseline 跑完或你确认可以占用 GPU 后，再启动：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avg_strict_staged_10m_training.py `
  --dataset-manifest artifacts\omni_transformer_stage_avf_10m_bidirectional_dataset\dataset_10m\manifest.json `
  --output artifacts\omni_transformer_stage_avg_strict_staged_10m_training\train_10m_70m_strict_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avg_strict_staged_10m_training\train_10m_70m_strict_checkpoints `
  --resume `
  --batch-size 512 `
  --d-model 768 `
  --heads 12 `
  --layers 10 `
  --latent-tokens 8 `
  --text-steps 400 `
  --codec-steps 1200 `
  --translate-steps 2200 `
  --reason-steps 1400 `
  --joint-steps 800 `
  --eval-every 250 `
  --save-every 250 `
  --eval-batch-size 128 `
  --sample-count 8
```

如果 batch 512 不稳，先降到 384 或 256。AV-E 的 70M capacity smoke 已证明同一模型档位 batch 256/512 可启动，但 AV-G 本轮没有重复占用 GPU 验证。

## 10M Strict Staged 结果

2026-07-05 用户本地完成 AV-G strict staged 长训：

- 结果：`artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_result.json`
- checkpoint：`artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_checkpoints/`
- 数据：AV-F 10M manifest，184,000 examples，10,120,000 pair tokens。
- 配置：75,970,770 参数，`batch_size=256`，总 `steps=6000`。
- schedule：400 text -> 1200 codec -> 2200 translation -> 1400 reason -> 800 joint。
- 成本：elapsed 2,757.88 sec，peak CUDA allocated 5,621.35 MB，processed pair tokens 84,480,000。

最终指标：

| 指标 | test | heldout |
| --- | ---: | ---: |
| source reconstruction token accuracy | 73.62% | 73.68% |
| target reconstruction token accuracy | 73.60% | 73.62% |
| source-to-target token accuracy | 72.49% | 72.54% |
| target-to-source token accuracy | 72.33% | 72.43% |
| source reconstruction exact | 0.00% | 0.00% |
| target reconstruction exact | 0.00% | 0.00% |
| source-to-target exact | 0.00% | 0.00% |
| target-to-source exact | 0.00% | 0.00% |
| answer accuracy | 77.85% | 78.16% |

结论：AV-G 严格分阶段也不通过。它避免了 AV-E 那种 97% answer shortcut，但没有把 external codec/translation exact 拉起来。

阶段曲线要点：

- `external_codec` 阶段在约 step 1250 已把 source/target reconstruction token accuracy 推到约 73.7%，之后基本平台化。
- `bidirectional_translate` 阶段把 source->target / target->source token accuracy 从约 15% 拉到约 72.8%，但 sequence exact 仍为 0。
- `latent_reason` 和 `joint_debug` 可以提高 answer，但会扰动 codec/translation，仍不能产生完整 external sequence。

按位置诊断显示：

- task、zone/op/op_arg、tool、memory 等结构化或低熵位置接近 99%-100%。
- image zone color positions 仍只有约 31%-34%。
- text/filler variable positions 约 27%-32%。
- 每条 sequence 通常错 6-9 个 token，说明失败集中在可变字段绑定，而不是输出格式字段。

判断：严格 schedule 本身没错，但当前 `decode(latent)` 结构不足。它把 latent 平均池化后加 position query 解码，缺少 per-position 对 latent slots 的 cross-attention/slot binding；对于图像 zone 和 variable text positions，会退化成频率/局部模板预测。

## 通过判断

这条路线不能只看 answer accuracy。

必须同时看：

- `val_source_recon_exact`
- `val_target_recon_exact`
- `val_source_to_target_exact`
- `val_target_to_source_exact`
- `val_answer_accuracy`
- test/heldout 的同名指标

如果 answer 先高、四个 exact 仍接近 0，判为 answer 捷径，不算互译闭合。

## 边界和担忧

1. `text_latent_base` 目前只使用 AV-F compact external 中的 text-operation positions，不是真正的大规模自然语言文本潜空间。
2. 当前仍是 compact token external，不是真实像素、真实文件、真实 action 输出专家。
3. AV-G 严格满足当前训练顺序，但数据分布仍只是 AV-D 1B-first 的 10M 低熵裁剪版本。
4. 如果 AV-E joint baseline 失败、AV-G 也失败，优先怀疑 decoder/latent edit/数据熵，不直接判架构不可行。
5. 现在已经触发第 4 点：下一步应先做 external codec decoder 修复实验，而不是继续扩大同一 AV-G 配置。
