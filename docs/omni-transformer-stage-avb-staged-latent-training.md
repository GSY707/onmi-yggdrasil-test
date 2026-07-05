# Stage AV-B：分阶段潜空间训练

## 目标

Stage AV v0 的 70M 端到端训练证明了本机能跑 70M 从零集成模型，但没有学出视觉 grounding。Stage AV-B 按新的训练顺序切方案：

1. 先用文本/结构化 token 建潜空间基座。
2. 分别训练各模态到潜空间的翻译。
3. 再训练潜空间推理。
4. 最后短程 joint tuning，只修接口错位。

这个阶段不继续把所有输入端到端硬塞给 answer loss，也不在前置 translator 上练到饱和，给后续 reasoner 和 joint tuning 留训练空间。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_avb_staged_latent_training.py
```

训练阶段：

| 阶段 | 目标 | 默认短训作用 |
| --- | --- | --- |
| `text_base` | task/text/state 的基础潜空间 | 建立可读 token/state 表示 |
| `image_translate` | image zone color translation | 让视觉 token 先能翻译进公共表示 |
| `tool_memory_translate` | tool issue 与 memory target zone | 建工具/history 与 memory/state translator |
| `latent_reason` | answer head/reasoner | 在已有 translator 上训练潜空间推理 |
| `joint` | 短程联合调试 | 修接口错位，不重训全部能力 |

## 训练性能优化

AV-B 针对长训做了以下优化：

1. train/val/test tensor 一次性搬到 GPU。
2. batch 用 GPU 上的 `torch.randint` 直接索引，不在每步 CPU 构造 batch。
3. 默认 AMP。
4. `torch.set_float32_matmul_precision("high")`。
5. eval 默认只看最多 512 条，避免评估拖慢训练。
6. 支持 `--checkpoint-dir` 和 `--resume`，每次 eval 写 `latest.pt`。
7. 支持 `--compile-model`，但 Windows 上需自行比较是否真的更快。

## Smoke

小模型 smoke：

- 结果：`artifacts/omni_transformer_stage_avb_staged_latent_training/smoke_result.json`
- 参数量：420,015
- 峰值 CUDA allocated：36.48 MB
- 结论：阶段切换、checkpoint、样例输出、JSON 输出链路正常。

70M smoke：

- 结果：`artifacts/omni_transformer_stage_avb_staged_latent_training/large_smoke_result.json`
- 参数量：71,025,455
- 峰值 CUDA allocated：1,505.32 MB
- 结论：70M AV-B 路径能跑。

70M batch 128 smoke：

- 结果：`artifacts/omni_transformer_stage_avb_staged_latent_training/large_batch128_smoke_result.json`
- 峰值 CUDA allocated：2,183.70 MB

70M batch 256 smoke：

- 结果：`artifacts/omni_transformer_stage_avb_staged_latent_training/large_batch256_smoke_result.json`
- 峰值 CUDA allocated：3,094.87 MB

当前 8GB 显存下，batch 256 仍有余量。若长训时 GPU 功耗达不到 60W，应优先提高 batch size 到 384 或 512；如果 OOM，再退回 256。

## 建议长训命令

推荐先用 batch 256 启动：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avb_staged_latent_training.py --train-size 8192 --val-size 1024 --test-size 1024 --batch-size 256 --d-model 768 --heads 12 --layers 10 --latent-tokens 8 --text-steps 600 --image-steps 900 --tool-memory-steps 900 --reason-steps 2400 --joint-steps 600 --eval-every 200 --sample-count 12 --checkpoint-dir artifacts\omni_transformer_stage_avb_staged_latent_training\long_checkpoints --output artifacts\omni_transformer_stage_avb_staged_latent_training\long_result.json
```

恢复续跑：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avb_staged_latent_training.py --train-size 8192 --val-size 1024 --test-size 1024 --batch-size 256 --d-model 768 --heads 12 --layers 10 --latent-tokens 8 --text-steps 600 --image-steps 900 --tool-memory-steps 900 --reason-steps 2400 --joint-steps 600 --eval-every 200 --sample-count 12 --checkpoint-dir artifacts\omni_transformer_stage_avb_staged_latent_training\long_checkpoints --resume --output artifacts\omni_transformer_stage_avb_staged_latent_training\long_result.json
```

训练时监控功耗：

```powershell
nvidia-smi -l 2
```

如果功耗长期低于 60W：

1. 先把 `--batch-size 256` 提到 `384`。
2. 仍低再试 `512`。
3. 如果 OOM，退回上一个可跑 batch。
4. 不要先加训练步数；功耗低说明吞吐没吃满，步数只会让低效训练变长。

## Long Probe 结果

用户本地已完成一次推荐长训：

- 结果：`artifacts/omni_transformer_stage_avb_staged_latent_training/long_result.json`
- 参数量：71,025,455
- 训练配置：`train_size=8192`、`val_size=1024`、`test_size=1024`、`batch_size=256`
- 阶段步数：`text=600`、`image=900`、`tool_memory=900`、`reason=2400`、`joint=600`
- elapsed：1,240.24 秒
- 峰值 CUDA allocated：3,100.77 MB

最终测试结果：

| 指标 | long probe |
| --- | ---: |
| overall answer accuracy | 74.61% |
| cross-expert visual accuracy | 28.95% |
| file-audit tools accuracy | 95.01% |
| memory exploration accuracy | 100.00% |
| task probe accuracy | 99.90% |
| image color probe accuracy | 50.59% |
| no-image answer accuracy | 73.93% |
| no-text answer accuracy | 74.12% |
| no-tool-history answer accuracy | 14.16% |
| no-memory answer accuracy | 75.29% |
| shuffled-image answer accuracy | 74.61% |

## Long Probe 判断

这次 long probe 不通过，但很有诊断价值。

正信号：

1. 70M 分阶段训练能稳定跑完，本机显存不是当前硬阻塞。
2. `file_audit_tools` 达到 95.01%，且 `no-tool-history` 降到 14.16%，说明工具历史路径确实被使用。
3. `memory_exploration` 达到 100.00%。

负信号：

1. `cross-expert visual` 只有 28.95%，接近 8 值/多类任务的弱先验水平。
2. `no-image` 和 `shuffled-image` 几乎不降，说明最终答案没有有效依赖视觉输入。
3. `no-memory` 反而略高于 full，说明 memory 消融没有形成有效因果检验。
4. `image_color_accuracy` 在 `image_translate` 阶段曾达到 100%，但后续阶段掉回约 25%-50%，说明分阶段训练发生了遗忘。
5. unique train tokens 约为 `8192 * 63 = 516,096`，但训练按 batch/step 重采样约 `5400 * 256 * 63 = 87,091,200` tokens；这不是 87M unique tokens，而是反复吃很小的数据集。

结论：AV-B long probe 证明了“训练流程能跑、工具和 memory 子任务能学”，但没有证明从零集成架构闭合。下一步应优先接 AV-C 的 sharded dataset pipeline，用 10M/100M 级 unique token 数据重新训练；同时加入阶段冻结/蒸馏或 replay，避免 image translator 在后续阶段遗忘。

## 边界和担忧

1. AV-B 当前完成了 smoke 和一次 long probe，但 long probe 未通过。
2. batch smoke 只能证明显存可承受，不证明功耗稳定在 60W 以上；功耗要以长训时 `nvidia-smi -l 2` 为准。
3. 早期 translator 不应练到过饱和，否则后续 reasoner/joint tuning 空间会变小。
4. 如果长训后视觉仍不闭合，下一步应先单独提高 image translator/readout，而不是继续端到端加步数。
5. 小数据重复训练会制造虚假的 loss 下降；后续报告必须区分 processed tokens 和 unique dataset tokens。
