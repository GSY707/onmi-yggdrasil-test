# Stage AV-C：大规模合成数据流水线

## 背景

Stage AV-B 的训练速度过快，暴露出一个根本问题：当前数据集只有几千到几万样本，测试集也很小。70M 级模型在这种数据上训练，容易变成“大炮打蚊子”：

- loss 下降不代表架构闭合。
- 不同步骤/阶段 loss 差异大，更多反映任务熵和 batch 组成。
- 小测试集容易误判。
- 即使跑很多 step，也可能是在重复低熵样本。

因此下一步不应继续直接长训，而应先准备可扩展数据流水线。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_avc_dataset_pipeline.py
```

脚本生成 sharded `.pt` 数据集：

- `train/`
- `val/`
- `test/`
- `heldout/`
- `manifest.json`

每个 shard 包含：

- `tokens`: `int32`，形状 `[N, 63]`
- `answers`: `int16`
- `tasks`: `int16`

任务族：

| 任务 | 目的 |
| --- | --- |
| `visual_rule` | 视觉 zone + 文本 rule |
| `tool_audit` | 工具证据审计 |
| `memory_route` | memory/state 路由 |
| `cross_modal` | 视觉、工具、状态混合 |

## Smoke

已生成小规模 smoke 数据：

- manifest：`artifacts/omni_transformer_stage_avc_dataset_pipeline/smoke_dataset/manifest.json`
- train examples：1,024
- val/test/heldout examples：各 128
- total examples：1,408
- seq tokens：63
- total tokens：88,704
- shard size：256

检查结果：

```text
tokens:  (256, 63), int32
answers: (256,), int16
tasks:   (256,), int16
```

## 规模估算

当前每条样本 63 tokens。

| 目标 tokens | 约需 examples |
| ---: | ---: |
| 10M | 158,730 |
| 100M | 1,587,301 |
| 1B | 15,873,015 |

因此，如果按 GPT 网页端建议的 1B token 级别，本项目不能继续用几千条数据做结论。应该逐步生成：

1. 10M token 数据集，用来验证训练/读取/评估流水线。
2. 100M token 数据集，用来做正式本机长训前的中型证据。
3. 1B token 数据集，只在存储、时间和训练策略确认后再生成。

## 建议生成命令

先生成 10M token 级别：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avc_dataset_pipeline.py --train-examples 160000 --val-examples 10000 --test-examples 10000 --heldout-examples 10000 --shard-size 10000 --output-dir artifacts\omni_transformer_stage_avc_dataset_pipeline\dataset_10m
```

100M token 级别：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avc_dataset_pipeline.py --train-examples 1600000 --val-examples 50000 --test-examples 50000 --heldout-examples 50000 --shard-size 20000 --output-dir artifacts\omni_transformer_stage_avc_dataset_pipeline\dataset_100m
```

不建议现在直接生成 1B token。1B token 需要约 15.9M examples，应先让训练脚本接入 shard 流式读取，并确认吞吐、功耗、验证频率和 checkpoint 策略。

## 下一步

AV-C 只完成了数据生成流水线 smoke。下一步应做：

1. 给 AV-B 训练脚本增加 `--dataset-manifest`，从 shard 读取而不是在线构造小数据。
2. 使用 GPU 预取或 shard 常驻缓存，避免 CPU/磁盘拖慢。
3. 训练日志改成按 token 计数，而不是只按 step。
4. 验证集至少 10k 起步，heldout split 必须参与报告。
5. 先跑 10M token 训练，再判断是否扩大到 100M。

## 边界和担忧

1. 当前仍是合成数据，不是真实互联网语料或真实多模态数据。
2. 1B token 是训练规模口径，不自动代表任务难度足够。
3. 数据集太简单时，1B token 也可能只是重复模板；后续需要加入更多组合、噪声、held-out schema 和 hard negatives。
4. 训练吞吐必须按 tokens/sec 和 GPU 功耗报告，否则不能判断长训是否有效。
