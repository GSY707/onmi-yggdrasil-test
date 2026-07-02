# 真实像素输入 Stage C 主动探测实验报告

## 目标

Stage B 说明：未见风格会让视觉解析失败。但如果每张地图里的视觉符号与地形含义是随机绑定的，仅靠图像本身就更不够了。用户指出的关键点是：模型应该允许先尝试、测试地图是什么意思，再执行正式路径。

本轮 Stage C 就验证这个机制：

- 图像仍是 `64x64 RGB` 的 `8x8` 地图。
- 每张地图都会随机生成 `视觉符号 -> 地形语义` 的绑定。
- 同一个红色/蓝色/条纹块，在不同地图里可能代表 normal、right、left、reverse 中任意一种。
- 无探测模型只能猜符号含义。
- 主动探测模型可以对某个符号所在格子试走一次，通过观察转移方向反推该符号含义。

代码里把“试走并观察转移”的反馈压缩成临时 legend：`visual_symbol -> terrain_semantics`。这不是固定标签泄漏，而是对环境交互反馈的可控模拟。

## Baseline

| 组别 | 含义 |
| --- | --- |
| `oracle_semantic` | 直接使用真实语义地形，上界 |
| `passive_symbol_identity` | 解析视觉符号，但不探测，错误假设 symbol id 等于 terrain id |
| `cnn_semantic_no_probe` | CNN 直接从像素预测地形语义，不允许探测 |
| `rule_symbol_probe_1` | 规则解析视觉符号，只探测 1 个符号 |
| `rule_symbol_probe_2` | 规则解析视觉符号，探测 2 个符号 |
| `rule_symbol_probe_4` | 规则解析视觉符号，探测 4 个符号，覆盖完整 legend |
| `cnn_symbol_probe_4` | CNN 解析视觉符号，探测 4 个符号 |
| `latent_symbol_probe_4` | 图像 encoder 输出 latent 符号，解码后用 4 次探测绑定语义 |
| `image_probe_direct` | 端到端模型拿到完整 probe legend，直接预测坐标 |

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\visual_multimodal_stage_c.py --sweep --train-move-count 8 --eval-move-counts 8,16 --seeds 20260701,20260702,20260703 --aggregate artifacts\visual_multimodal_stage_c\sweep_results.json --output-dir artifacts\visual_multimodal_stage_c\sweep_runs
```

结果文件：

- `artifacts/visual_multimodal_stage_c/sweep_results.json`
- `artifacts/visual_multimodal_stage_c/sweep_runs/`
- `artifacts/visual_multimodal_stage_c/sweep_runs/samples/`

## 结果

final exact：

| 测试 move | oracle | passive no probe | CNN semantic no probe | probe 1 | probe 2 | probe 4 rule | probe 4 CNN symbol | probe 4 latent symbol | image direct + probes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 9.47% | 4.26% | 13.77% | 22.53% | 100.00% | 100.00% | 100.00% | 1.96% |
| 16 | 100.00% | 7.42% | 3.78% | 9.02% | 15.95% | 100.00% | 100.00% | 100.00% | 1.82% |
| overall | 100.00% | 8.45% | 4.02% | 11.39% | 19.24% | 100.00% | 100.00% | 100.00% | 1.89% |

semantic cell exact：

| 测试 move | passive no probe | CNN semantic no probe | probe 1 | probe 2 | probe 4 rule/CNN/latent |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 25.10% | 24.51% | 43.61% | 62.53% | 100.00% |
| 16 | 24.86% | 24.41% | 43.38% | 62.39% | 100.00% |

成本：

| 组别 | 平均训练秒数 | 训练步数 | 参数量 |
| --- | ---: | ---: | ---: |
| `cnn_semantic_no_probe` | 10.81 | 700 | 191,964 |
| `cnn_symbol_probe_4` | 17.30 | 700 | 191,964 |
| `latent_symbol_probe_4` | 12.21 | 1,000 | 198,332 |
| `image_probe_direct` | 62.70 | 900 | 1,545,336 |
| 规则 probe 路线 | 0.00 | 0 | 0 |

## 解释

这轮实验直接支持用户的判断：

1. 当视觉符号和地形语义在每张地图里随机重绑定时，不探测的路线基本只能猜。`passive_symbol_identity` 的 semantic cell 约 25%，`cnn_semantic_no_probe` 也约 24%-25%。
2. 探测预算会逐步提高语义恢复率。1 次探测约 43% semantic cell，2 次探测约 62%，完整 4 次探测达到 100%。
3. `rule_symbol_probe_4`、`cnn_symbol_probe_4`、`latent_symbol_probe_4` 都是 100%，说明关键不是某个 CNN 或 latent 魔法，而是“先识别视觉符号，再通过环境反馈绑定语义”。
4. `image_probe_direct` 即使拿到完整 probe legend，也只有 1.89% final，说明把探测信息直接塞给端到端坐标模型并不足够；仍需要结构化使用探测结果。

## 结论

Stage C 把问题推进了一步：真实多模态输入不只是“看图识别块”，还需要主动试验来确定图像符号在当前环境里的语义。这个结果对多模态潜空间架构是正向的，但不是“latent 更优”的证明。

更准确的结论是：

- 无探测时，随机绑定地图不可解，只能接近猜测。
- 有足够探测时，视觉符号路线可以恢复到 oracle。
- latent 路线能参与这个流程，但当前任务里 rule/CNN 符号路线同样能做到 100%。
- Stage D 已补跑，见 `docs/visual-multimodal-stage-d-experiment.md`。Stage D 把地图改成局部可见多帧观察，并验证短期 legend 记忆可以把探测成本从路径长度压到符号种类数。
