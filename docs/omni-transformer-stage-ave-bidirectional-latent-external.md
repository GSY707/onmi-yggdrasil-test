# Stage AV-E：潜变量与外部表征双向互译

## 目标

Stage AV-E 修正 AV-D/AV-B 的一个核心遗漏：训练目标不能只做识别，也不能只做生成/修改，而应该直接验证 **外部表征 ↔ 潜变量** 的双向互译。

本阶段的样本同时包含：

- source external
- target external
- operation
- answer

训练链路是：

```text
source external -> source latent -> source external
source external -> source latent -> target latent -> target external
target external -> target latent -> target external
target external -> target latent -> source latent -> source external
target latent -> answer
```

这比 AV-B 多了一步：不是只把外部输入翻译到 latent 后回答，而是要求 latent 经过一次编辑/变换后还能回到外部表征，并且 target external 还能重新编码回 latent。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_ave_bidirectional_latent_external.py
```

外部表征目前是 tokenized compact external state：

- image zones
- text operation
- tool slots
- memory slots

操作类型：

| op | 作用 |
| --- | --- |
| 0 | recolor target zone |
| 1 | rotate image zones |
| 2 | repair/update memory at target zone |
| 3 | update tool slot from visual fact and op arg |

训练 loss 同时包含：

- source reconstruction
- target reconstruction
- source-to-target external translation
- target-to-source external translation
- answer classification
- source-to-target latent alignment
- target-to-source latent alignment

## Smoke

小模型 smoke：

- 结果：`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/smoke_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/smoke_runs/seed20260705.json`
- 参数量：589,010
- 结论：脚本、双向 loss、heldout 评估、样例 PNG/JSON 输出链路能跑。

40 step smoke 没有能力结论。一个重要观察是：latent cosine 很快升高到约 0.98，但 token 互译仍只有约 3%-7%，说明不能把 latent cosine 当作互译成功证据。

## Probe

稍长 probe：

- 结果：`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/probe_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/probe_runs/seed20260705.json`
- 配置：`train_size=1024`、`val_size=256`、`test_size=256`、`d_model=192`、`layers=3`、`steps=300`
- 参数量：1,722,258

结果：

| 指标 | test | heldout |
| --- | ---: | ---: |
| source reconstruction token accuracy | 30.22% | 30.82% |
| target reconstruction token accuracy | 30.15% | 30.64% |
| source-to-target token accuracy | 28.40% | 29.64% |
| target-to-source token accuracy | 28.24% | 29.64% |
| source reconstruction exact | 0.00% | 0.00% |
| target reconstruction exact | 0.00% | 0.00% |
| source-to-target exact | 0.00% | 0.00% |
| target-to-source exact | 0.00% | 0.00% |
| answer accuracy | 6.64% | 5.86% |
| edit latent cosine | 68.67% | 68.87% |
| inverse latent cosine | 68.44% | 68.76% |

## 结论

Stage AV-E 当前不是通过结果，但方向比 AV-B 更贴近架构目标。

正信号：

1. 双向 token 互译在 300 step 内从接近 0 提到约 28%-30%，说明任务不是完全不可学。
2. heldout token accuracy 与 test 接近，短 probe 没出现只记 train 的强迹象。
3. 这条任务直接绑定了 source/target external 和 latent edit，比单向识别更符合“互译”目标。

负信号：

1. sequence exact 仍为 0，说明完整外部表征重建还没闭合。
2. answer accuracy 仍接近随机，说明 target latent 还不能支撑最终推理。
3. latent cosine 不可靠；smoke 中 cosine 很高但 token accuracy 很低。
4. 当前 decoder 仍是简化 token decoder，不是真实图像/文件/动作输出专家。

## 10M manifest 训练准备

2026-07-05 已把 AV-E 训练脚本接入 AV-F 10M 级 bidirectional dataset manifest。

新增能力：

- `--dataset-manifest`：从 `.pt` shard 读取 `source_tokens`、`target_tokens`、`answers`。
- `--gpu-resident-data`：10M 级数据较小，可把 train/val tensor 常驻 GPU，减少 CPU batch 搬运。
- `--checkpoint-dir`、`--resume`、`--save-every`：长训支持 `latest.pt`/`best.pt`。
- batched eval：避免完整 8k split 一次性前向。
- 结果 JSON 记录 dataset source、manifest scale、unique train pair tokens 和 processed pair tokens。

已生成数据：

- AV-F smoke manifest：`artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/smoke_dataset/manifest.json`
- AV-F 10M manifest：`artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m/manifest.json`
- 10M total pair tokens：10,120,000
- train unique pair tokens：8,800,000

已验证：

| 验证 | 结果 |
| --- | --- |
| smoke manifest CUDA 20 step | `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/manifest_smoke_gpu_result.json`，peak CUDA allocated 74.35 MB |
| 10M manifest loader CUDA 2 step | `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/dataset10m_loader_smoke_gpu_result.json`，完整 train 分片可读 |
| 70M batch 256 capacity smoke | `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/dataset10m_70m_capacity_smoke_result.json`，75,970,770 params，peak CUDA allocated 4,753.32 MB |
| 70M batch 512 capacity smoke | `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/dataset10m_70m_batch512_capacity_smoke_result.json`，75,970,770 params，peak CUDA allocated 8,923.00 MB |

AV-E 的 10M 长训命令是 joint multi-loss baseline，不是严格分阶段训练。它可以作为一次尝试和对照保留；后续严格架构路线已经拆到 AV-G。

## 10M Joint Baseline 结果

2026-07-05 用户本地完成 AV-E joint baseline 长训：

- 结果：`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_result.json`
- checkpoint：`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_checkpoints/`
- 数据：AV-F 10M manifest，184,000 examples，10,120,000 pair tokens。
- 配置：75,970,770 参数，`batch_size=256`，`steps=6000`，processed pair tokens 84,480,000。
- 成本：elapsed 3,005.44 sec，peak CUDA allocated 5,628.84 MB。

最终指标：

| 指标 | test | heldout |
| --- | ---: | ---: |
| source reconstruction token accuracy | 73.78% | 73.84% |
| target reconstruction token accuracy | 73.74% | 73.85% |
| source-to-target token accuracy | 73.48% | 73.61% |
| target-to-source token accuracy | 73.46% | 73.55% |
| source reconstruction exact | 0.00% | 0.00% |
| target reconstruction exact | 0.00% | 0.00% |
| source-to-target exact | 0.00% | 0.00% |
| target-to-source exact | 0.00% | 0.00% |
| answer accuracy | 97.25% | 97.16% |

结论：AV-E joint baseline 不通过。answer head 学得很高，但外部表征完整互译为 0 exact，这是典型 answer 捷径；不能作为架构闭合证据。

按位置诊断显示：

- `TASK_TOKEN`、zone/op/op_arg、tool slots、memory slots 大多接近 99%-100%。
- image zone color positions 只有约 31%-34%。
- text/filler variable positions 多数只有约 30%-32%。
- 每条序列通常错 6-9 个 token，因此 27-token sequence exact 全为 0。

这说明训练不是单纯数据不够，而是当前 latent decoder/slot binding 无法可靠携带每个可变字段。继续给同一 joint loss 加 step，优先只会继续强化 answer shortcut。

## 下一步

AV-D 的正式数据设计需要把 AV-E 作为核心任务族，而不是把互译当附加 loss。

下一步实现应包括：

1. 保留 AV-E joint baseline 作为负对照：answer 高但互译 exact 为 0。
2. 启动或分析 AV-G 严格分阶段训练。
3. 下一轮应优先修 external codec/decoder，而不是继续拉 AV-E joint step。
4. 训练报告必须同时看 source recon、target recon、source->target、target->source、answer，不允许只看 answer。
5. 如果 answer 先升高但 sequence exact 不升高，判为读头捷径，不算互译闭合。

## 边界和担忧

1. 当前 external 是 tokenized compact state，不是真实像素/文件/动作空间。
2. AV-F 10M 解决了小数据重复问题，但仍不是完整 1B 高熵母分布。
3. 如果后续只把 answer accuracy 拉高，但互译 exact 不高，仍不能算 AV-E 通过。
4. batch 512 已能启动但显存接近边界，长训中要监控 `nvidia-smi` 的功耗、显存和 GPU util。
