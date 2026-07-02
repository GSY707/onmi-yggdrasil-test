# 真实像素输入 Stage A/B 实验报告

## 目标

本轮把异构输入从结构化地形 tensor 推进到真实像素图像。任务仍保持可归因：文本给起点和动作序列，图像给 `8x8` 地形地图，答案是最终坐标与中间坐标。

本轮只覆盖：

- Stage A：可控像素地图，同训练风格测试。
- Stage B：未见颜色/纹理风格测试。

它还不是自然图像、视频或多帧观察实验。

## 图像输入

每个样本会渲染成 `64x64 RGB` 图像，每个格子是 `8x8` 像素 tile。地形类别由颜色和纹理共同表达。

正式 sweep 样例：

- `artifacts/visual_multimodal_stage_ab/sweep_runs/samples/seed20260701/train_style1.png`
- `artifacts/visual_multimodal_stage_ab/sweep_runs/samples/seed20260701/stage_a_same_style_style0.png`
- `artifacts/visual_multimodal_stage_ab/sweep_runs/samples/seed20260701/stage_b_unseen_style_style2.png`

## Baseline

| 组别 | 含义 |
| --- | --- |
| `oracle_visual_parser` | 知道每个样本真实风格 palette 的视觉 parser，上界 |
| `train_palette_parser` | 只知道训练风格 palette 的非学习 parser，测试规则 parser 脆弱性 |
| `cnn_to_terrain_table` | CNN 从像素预测每格地形，再 rollout |
| `cnn_to_latent_codec` | CNN 从像素预测每格 latent，再用冻结 terrain codec 解码并 rollout |
| `image_to_steps_direct` | 端到端模型直接从图像 + 动作序列预测每步坐标 |

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\visual_multimodal_stage_ab.py --sweep --train-move-count 8 --eval-move-counts 8,16 --seeds 20260701,20260702,20260703 --aggregate artifacts\visual_multimodal_stage_ab\sweep_results.json --output-dir artifacts\visual_multimodal_stage_ab\sweep_runs
```

运行设备：

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch：`2.11.0+cu128`
- CUDA：`12.8`

## Stage A 结果

同训练风格测试，final exact：

| 测试 move | oracle parser | train parser | CNN terrain | CNN latent | image direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 100.00% | 100.00% | 100.00% | 2.02% |
| 16 | 100.00% | 100.00% | 100.00% | 100.00% | 1.73% |

同训练风格测试，cell exact：

| 测试 move | oracle parser | train parser | CNN terrain | CNN latent | image direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| 16 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

解释：Stage A 证明像素输入链路能闭合。CNN-to-terrain 和 CNN-to-latent 都能从可控图像恢复地形表并完成推理；直接端到端坐标预测仍然失败。

## Stage B 结果

未见风格测试，final exact：

| 测试 move | oracle parser | train parser | CNN terrain | CNN latent | image direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 6.28% | 8.82% | 15.01% | 2.38% |
| 16 | 100.00% | 3.19% | 4.62% | 7.49% | 1.20% |

未见风格测试，cell exact：

| 测试 move | oracle parser | train parser | CNN terrain | CNN latent | image direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 25.05% | 37.24% | 42.93% | 0.00% |
| 16 | 100.00% | 24.27% | 36.50% | 42.06% | 0.00% |

解释：Stage B 暴露了真实像素阶段的关键问题。训练风格 parser 在未见 palette 上接近失效，CNN 模型也没有学到足够稳定的风格不变表示。`cnn_to_latent_codec` 比 `cnn_to_terrain_table` 略高，但仍远低于 oracle，不能说 latent 路线已经解决风格泛化。

## 成本

| 组别 | 平均训练秒数 | 训练步数 | 参数量 |
| --- | ---: | ---: | ---: |
| `oracle_visual_parser` | 0.00 | 0 | 0 |
| `train_palette_parser` | 0.00 | 0 | 0 |
| `cnn_to_terrain_table` | 9.08 | 700 | 191,964 |
| `cnn_to_latent_codec` | 10.96 | 1,000 | 198,332 |
| `image_to_steps_direct` | 13.69 | 1,000 | 1,542,264 |

## 结论

Stage A 证明了真实像素输入版本的链路可行：图像可以被解析成地形中间表示，再和文本动作一起完成推理。

Stage B 没有证明 latent 更好。它只显示了一个早期迹象：latent codec 路线在未见风格下高于直接 terrain CNN，但差距不大，绝对准确率仍很低。当前更强的结论是：像素阶段的核心难点已经从“能不能接入异构输入”转为“能不能学到跨风格稳定视觉表示”。

Stage C 已补跑，见 `docs/visual-multimodal-stage-c-experiment.md`。Stage C 采用每张地图随机绑定视觉符号与地形语义，并允许主动探测；结果显示无探测接近猜测，完整探测可以恢复到 100%。
