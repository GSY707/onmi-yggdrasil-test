# Stage AV-I：整图 latent token 容量诊断

## 目标

Stage AV-I 专门回答一个问题：AV-H 里 `latent_tokens=12` 是否足够承载一张 64x64 合成图像。

它不测试文本编辑，也不测试端到端 AV-H。它只测试：

```text
whole image -> latent token bank -> whole image reconstruction
```

当前版本使用透明背景，而不是把黑色背景当成前景或与前景求交：

- AV-H record 先渲染成整图；
- RGB 全黑背景转换成 `alpha=0`；
- RGB loss 只在 `alpha=1` 的对象像素上计算；
- alpha loss 在全图计算，背景只监督透明，不参与内容误差；
- greedy prefix 选择的是最小透明整图重建 loss 的 token，不再使用前景交集。

## 记忆树式前缀 curriculum

脚本：

```powershell
experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py
```

每个 latent token 解码为整图 logit residual。按 token 顺序累加 residual 后得到 prefix reconstruction：

```text
prefix 1     = token 1
prefix 2     = token 1 + token 2
prefix 4     = token 1 + token 2 + token 3 + token 4
...
prefix N     = token 1..N
```

训练不再一股脑同时训练所有长度，而是按短到长解锁 active token：

```text
12 tokens: 1 -> 2 -> 4 -> 8 -> 12
24 tokens: 1 -> 2 -> 4 -> 8 -> 16 -> 24
32 tokens: 1 -> 2 -> 4 -> 8 -> 16 -> 32
```

这个设计符合“前序 token 承担更多像素/主结构，后续 token 补充残差”的记忆树直觉。当前 loss 仍会监督中间 prefix，但只对已解锁 token 给梯度。

2026-07-05 后，脚本已把 codec/expert 宽度和 latent token 宽度拆开：

- `encoder_width`：图像 encoder 输出宽度，只影响写入 latent 前的 codec 能力；
- `latent_dim`：每个 latent token 的真实瓶颈宽度；
- `decoder_width`：每个 latent token 到整图 residual 的输出专家宽度。

旧 `--d-model` 仍可作为 legacy 宽度使用；新容量诊断应显式报告 `latent_dim`，否则 token 数会被单 token 宽度掩盖。

同日又加入 compact prefix guidance，用来约束前序 token 不要大面积涂满整图：

```text
compact_prefix_loss = 0.5 * occupied_alpha_area + 0.5 * (1 - target_alpha_coverage)
```

其中 `occupied_alpha_area` 是预测 alpha 在全图上的软面积，`target_alpha_coverage` 是预测 alpha 对非透明目标区域的软覆盖。这个 loss 加在所有未达到最终 token 数的 prefix 上，包括 curriculum 第一段的 `active_tokens=1`；最终完整 prefix 仍以整图还原为主。

随后又把 `target_alpha_coverage` 改成 precision coverage：

```text
target_alpha_coverage = mean((pred_alpha ** n) on target pixels)
```

`n` 默认从 1 线性增到 4。训练初期接近线性奖励，先让模型找到目标区域；训练后期半透明/低置信输出的奖励被压低，只有接近 1 的精确覆盖能拿到高奖励。

## 已验证链路

CUDA smoke：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\transparent_curriculum_cuda_smoke_v2_result.json
```

- latent tokens：4
- d_model：64
- steps：4
- batch size：16
- 结果：脚本、透明 RGBA、curriculum、CUDA AMP、greedy eval、样例输出链路通过。

长 probe：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_transparent_curriculum_12_24_32_3000step_result.json
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_transparent_curriculum_12_24_32_3000step_summary.json
```

运行命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py `
  --output artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_transparent_curriculum_12_24_32_3000step_result.json `
  --latent-tokens-list 12,24,32 `
  --d-model 192 `
  --steps 3000 `
  --batch-size 128 `
  --train-limit 0 `
  --val-limit 1300 `
  --test-limit 1300 `
  --heldout-limit 1300 `
  --eval-batch-size 128 `
  --greedy-eval-limit 512 `
  --sample-count 4 `
  --device cuda
```

配置：

| 项 | 数值 |
| --- | ---: |
| unique train images | 36,000 |
| val/test/heldout images | 2,600 / 2,600 / 2,600 |
| latent tokens | 12, 24, 32 |
| d_model | 192 |
| steps | 每档 3,000 |
| batch size | 128 |
| total elapsed | 370.39 sec |
| peak CUDA allocated | 1,194.43 MB |

## 3000-step 结果

ordered final prefix：

| latent tokens | params | test transparent loss | test object RGB MSE | test alpha IoU | test bg alpha mean | heldout transparent loss | heldout alpha IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 14,697,424 | 0.002505 | 0.002271 | 0.9972 | 0.001038 | 0.003498 | 0.9953 |
| 24 | 15,144,400 | 0.002855 | 0.002432 | 0.9960 | 0.001297 | 0.004021 | 0.9943 |
| 32 | 15,442,384 | 0.003068 | 0.002634 | 0.9960 | 0.001254 | 0.004227 | 0.9935 |

greedy final prefix：

| latent tokens | test transparent loss | test object RGB MSE | test alpha IoU | heldout transparent loss | heldout alpha IoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 0.002679 | 0.002383 | 0.9968 | 0.003371 | 0.9961 |
| 24 | 0.002972 | 0.002497 | 0.9958 | 0.003846 | 0.9950 |
| 32 | 0.003237 | 0.002785 | 0.9954 | 0.004213 | 0.9942 |

12-token ordered prefix 曲线：

| prefix | test transparent loss | test alpha IoU | test object RGB MSE |
| ---: | ---: | ---: | ---: |
| 1 | 0.002812 | 0.9961 | 0.002377 |
| 2 | 0.002511 | 0.9972 | 0.002262 |
| 4 | 0.002508 | 0.9972 | 0.002263 |
| 8 | 0.002512 | 0.9971 | 0.002273 |
| 12 | 0.002505 | 0.9972 | 0.002271 |

24-token ordered prefix 曲线：

| prefix | test transparent loss | test alpha IoU | test object RGB MSE |
| ---: | ---: | ---: | ---: |
| 1 | 0.003135 | 0.9950 | 0.002506 |
| 2 | 0.002852 | 0.9960 | 0.002423 |
| 4 | 0.002841 | 0.9959 | 0.002415 |
| 8 | 0.002848 | 0.9959 | 0.002424 |
| 16 | 0.002889 | 0.9959 | 0.002463 |
| 24 | 0.002855 | 0.9960 | 0.002432 |

32-token ordered prefix 曲线：

| prefix | test transparent loss | test alpha IoU | test object RGB MSE |
| ---: | ---: | ---: | ---: |
| 1 | 0.003545 | 0.9938 | 0.002651 |
| 2 | 0.003092 | 0.9959 | 0.002598 |
| 4 | 0.003065 | 0.9960 | 0.002592 |
| 8 | 0.003073 | 0.9959 | 0.002611 |
| 16 | 0.003043 | 0.9960 | 0.002592 |
| 32 | 0.003068 | 0.9960 | 0.002634 |

样例 contact sheet：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_transparent_curriculum_12_24_32_3000step_result_latent_12_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_transparent_curriculum_12_24_32_3000step_result_latent_24_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_transparent_curriculum_12_24_32_3000step_result_latent_32_samples\ordered_contact_sheet_checker.png
```

## 2 维 token 极限 probe

用户要求做一个极端测试：把每个 latent token 压到 2 维，从 1 token 开始测试容量曲线。这个 probe 保持 codec/expert 有足够宽度，只把真正通过 latent bank 的信息压窄：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py `
  --output artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result.json `
  --latent-tokens-list 1,2,4,8,16,32,64 `
  --latent-dim 2 `
  --encoder-width 192 `
  --decoder-width 768 `
  --steps 3000 `
  --batch-size 128 `
  --train-limit 0 `
  --val-limit 1300 `
  --test-limit 1300 `
  --heldout-limit 1300 `
  --eval-batch-size 128 `
  --greedy-eval-limit 512 `
  --sample-count 4 `
  --device cuda
```

结果：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result.json
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_summary.json
```

配置与成本：

| 项 | 数值 |
| --- | ---: |
| latent_dim | 2 |
| encoder_width / decoder_width | 192 / 768 |
| latent tokens | 1, 2, 4, 8, 16, 32, 64 |
| steps | 每档 3,000 |
| unique train images | 36,000 |
| elapsed | 738.39 sec |
| peak CUDA allocated | 2,083.25 MB |

ordered final prefix：

| latent tokens | latent scalars | test transparent loss | test object RGB MSE | test alpha IoU | heldout transparent loss | heldout alpha IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.150044 | 0.083061 | 0.6213 | 0.194199 | 0.5862 |
| 2 | 4 | 0.135932 | 0.082993 | 0.6763 | 0.177724 | 0.6447 |
| 4 | 8 | 0.090036 | 0.058034 | 0.7699 | 0.119252 | 0.7526 |
| 8 | 16 | 0.071326 | 0.040791 | 0.7778 | 0.096636 | 0.7564 |
| 16 | 32 | 0.067619 | 0.039484 | 0.7934 | 0.088435 | 0.7841 |
| 32 | 64 | 0.064081 | 0.037163 | 0.8166 | 0.083439 | 0.8020 |
| 64 | 128 | 0.081616 | 0.054663 | 0.8127 | 0.104430 | 0.7961 |

样例：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result_latent_1_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result_latent_8_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result_latent_32_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result_latent_64_samples\ordered_contact_sheet_checker.png
```

判断：

- 2 维 token 下，token 数从 1 到 32 有真实容量增益，说明“从 1 开始测”是有效诊断。
- 即使到 32 token、总 latent scalars=64，也远不如 `latent_dim=192, latent_tokens=12` 的宽 token 结果。
- 64 token 没有继续提升，反而退化。主要怀疑是 curriculum 段数更多导致每个 active prefix 段训练步数不足，且长 token bank 的 residual 分工更难。
- 样图有明显 ghost/object 混叠，尤其是多对象图。2 维 token 不是隐藏地完成了重建。

## compact + 7000-step probe

为排除 64 token 因 curriculum 分段太短而退化，并执行“面积最少 / 加入覆盖最大各 50%”的新训练要求，新增 compact guidance 后把每档训练拉到 7000 step：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py `
  --output artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result.json `
  --latent-tokens-list 1,2,4,8,16,32,64 `
  --latent-dim 2 `
  --encoder-width 192 `
  --decoder-width 768 `
  --steps 7000 `
  --batch-size 128 `
  --train-limit 0 `
  --val-limit 1300 `
  --test-limit 1300 `
  --heldout-limit 1300 `
  --eval-batch-size 128 `
  --greedy-eval-limit 512 `
  --sample-count 4 `
  --device cuda
```

结果：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result.json
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json
```

配置与成本：

| 项 | 数值 |
| --- | ---: |
| latent_dim | 2 |
| encoder_width / decoder_width | 192 / 768 |
| latent tokens | 1, 2, 4, 8, 16, 32, 64 |
| compact area / coverage weight | 0.5 / 0.5 |
| compact prefix weight | 0.35 |
| steps | 每档 7,000 |
| unique train images | 36,000 |
| elapsed | 1,950.62 sec |
| peak CUDA allocated | 2,082.53 MB |

ordered final prefix：

| latent tokens | latent scalars | test loss | test object RGB MSE | test alpha IoU | soft area | soft coverage | heldout loss | heldout alpha IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.121690 | 0.072209 | 0.6649 | 0.1597 | 0.8199 | 0.165076 | 0.6352 |
| 2 | 4 | 0.089483 | 0.067226 | 0.7909 | 0.1490 | 0.9091 | 0.127052 | 0.7573 |
| 4 | 8 | 0.080677 | 0.056217 | 0.7980 | 0.1423 | 0.9045 | 0.109042 | 0.7823 |
| 8 | 16 | 0.052217 | 0.032978 | 0.8590 | 0.1394 | 0.9398 | 0.071103 | 0.8315 |
| 16 | 32 | 0.045261 | 0.030985 | 0.8922 | 0.1376 | 0.9593 | 0.060576 | 0.8707 |
| 32 | 64 | 0.044031 | 0.029897 | 0.9091 | 0.1320 | 0.9530 | 0.058630 | 0.8904 |
| 64 | 128 | 0.047084 | 0.032949 | 0.9074 | 0.1364 | 0.9744 | 0.061334 | 0.8885 |

样例：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_8_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_32_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples\ordered_contact_sheet_checker.png
```

判断：

- compact + 7000 step 明显改善 2 维 token 曲线。32-token test alpha IoU 从无 compact 3000-step 的 0.8166 升到 0.9091，test loss 从 0.064081 降到 0.044031。
- 64 token 不再像无 compact 3000-step 那样明显退化，但仍没有超过 32 token；当前最佳是 32 token。
- 样图中大面积 ghost/object 混叠明显减少，但多对象组合仍有颜色/位置错配，不能算 lossless。
- 这轮仍没有触发“该加参数量”的条件：token 增加到 32 仍有收益，64 的问题更像 residual routing / token 分工，而不是参数不足。

## precision compact probe

为测试“减少不精确奖励，增加精确奖励”的想法，compact coverage 从线性 `x` 改成动态 `x^n`，`n` 从 1 线性增到 4：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py `
  --output artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json `
  --latent-tokens-list 1,2,4,8,16,32,64 `
  --latent-dim 2 `
  --encoder-width 192 `
  --decoder-width 768 `
  --steps 7000 `
  --batch-size 128 `
  --train-limit 0 `
  --val-limit 1300 `
  --test-limit 1300 `
  --heldout-limit 1300 `
  --eval-batch-size 128 `
  --greedy-eval-limit 512 `
  --sample-count 4 `
  --compact-precision-power-start 1 `
  --compact-precision-power-end 4 `
  --device cuda
```

结果：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json
```

配置与成本：

| 项 | 数值 |
| --- | ---: |
| latent_dim | 2 |
| encoder_width / decoder_width | 192 / 768 |
| latent tokens | 1, 2, 4, 8, 16, 32, 64 |
| precision power schedule | 1 -> 4 |
| steps | 每档 7,000 |
| unique train images | 36,000 |
| elapsed | 1,909.85 sec |
| peak CUDA allocated | 2,082.53 MB |

ordered final prefix：

| latent tokens | latent scalars | test loss | test object RGB MSE | test alpha IoU | soft area | soft coverage | precision coverage | heldout loss | heldout alpha IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.122681 | 0.072500 | 0.6629 | 0.1603 | 0.8194 | 0.6591 | 0.165853 | 0.6332 |
| 2 | 4 | 0.098955 | 0.072198 | 0.7761 | 0.1511 | 0.9050 | 0.7998 | 0.137683 | 0.7399 |
| 4 | 8 | 0.071013 | 0.052160 | 0.8413 | 0.1409 | 0.9321 | 0.8564 | 0.098644 | 0.8165 |
| 8 | 16 | 0.050214 | 0.033860 | 0.8712 | 0.1393 | 0.9530 | 0.8967 | 0.068573 | 0.8490 |
| 16 | 32 | 0.042794 | 0.030811 | 0.9117 | 0.1325 | 0.9578 | 0.9107 | 0.058801 | 0.8905 |
| 32 | 64 | 0.047576 | 0.033727 | 0.9100 | 0.1312 | 0.9550 | 0.9110 | 0.063287 | 0.8884 |
| 64 | 128 | 0.051138 | 0.035238 | 0.9038 | 0.1351 | 0.9691 | 0.9402 | 0.065510 | 0.8884 |

样例：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_16_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_32_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples\ordered_contact_sheet_checker.png
```

判断：

- 动态 `x^n` 对 4/8/16 token 有正收益。16 token test loss 从线性 compact 的 0.045261 降到 0.042794，alpha IoU 从 0.8922 升到 0.9117。
- 但 32/64 token 没有超过线性 compact。32-token test loss 从 0.044031 变差到 0.047576，64-token 从 0.047084 变差到 0.051138。
- 当前最佳从线性 compact 的 32 token，变成 precision compact 的 16 token；说明 precision schedule 让较短 token 更精确，但可能过早/过强地限制了长 token bank 的残差协作。
- 这个结果支持用户的方向，但不支持直接固定 `n_end=4` 作为最终策略。下一步应 sweep `n_end=2/3/4`，或把 n 的增长绑定到 coverage/alpha IoU 达标，而不是只按 step 线性增长。

## segment precision compact probe

用户进一步修正 schedule：`n` 不应该在整个训练过程只从小到大扫一次，而应该在每个 active-token curriculum 段内都重新经历完整变化。同时尝试 `n=0.5 -> 5`，且变化不是线性，而是两端快、中间慢。

实现后，每个 curriculum 段内部都独立运行：

```text
segment_progress: 0 -> 1
shaped_progress: fast at both ends, slow in the middle
n: 0.5 -> 5
```

例如 64-token、7000-step 时，每段 1000 step：

```text
step 1       active=1   n=0.5
step 1000    active=1   n=5
step 1001    active=2   n=0.5
step 2000    active=2   n=5
...
step 6001    active=64  n=0.5
step 7000    active=64  n=5
```

命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py `
  --output artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json `
  --latent-tokens-list 1,2,4,8,16,32,64 `
  --latent-dim 2 `
  --encoder-width 192 `
  --decoder-width 768 `
  --steps 7000 `
  --batch-size 128 `
  --train-limit 0 `
  --val-limit 1300 `
  --test-limit 1300 `
  --heldout-limit 1300 `
  --eval-batch-size 128 `
  --greedy-eval-limit 512 `
  --sample-count 4 `
  --compact-precision-power-start 0.5 `
  --compact-precision-power-end 5 `
  --compact-precision-curve-power 0.5 `
  --device cuda
```

结果：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json
```

配置与成本：

| 项 | 数值 |
| --- | ---: |
| latent_dim | 2 |
| encoder_width / decoder_width | 192 / 768 |
| latent tokens | 1, 2, 4, 8, 16, 32, 64 |
| precision power schedule | 每段 0.5 -> 5 |
| curve power | 0.5 |
| steps | 每档 7,000 |
| unique train images | 36,000 |
| elapsed | 2,382.84 sec |
| peak CUDA allocated | 2,081.74 MB |

ordered final prefix：

| latent tokens | latent scalars | test loss | test object RGB MSE | test alpha IoU | soft area | soft coverage | precision coverage | heldout loss | heldout alpha IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.122875 | 0.072743 | 0.6590 | 0.1601 | 0.8193 | 0.6374 | 0.166396 | 0.6321 |
| 2 | 4 | 0.093236 | 0.070987 | 0.7914 | 0.1482 | 0.9125 | 0.7955 | 0.130584 | 0.7541 |
| 4 | 8 | 0.064294 | 0.044914 | 0.8493 | 0.1374 | 0.9230 | 0.8227 | 0.090326 | 0.8200 |
| 8 | 16 | 0.053011 | 0.035543 | 0.8766 | 0.1346 | 0.9359 | 0.8522 | 0.073223 | 0.8540 |
| 16 | 32 | 0.041362 | 0.028971 | 0.9087 | 0.1352 | 0.9662 | 0.9186 | 0.055129 | 0.8917 |
| 32 | 64 | 0.037481 | 0.024237 | 0.9127 | 0.1342 | 0.9665 | 0.9230 | 0.051181 | 0.8948 |
| 64 | 128 | 0.048738 | 0.035855 | 0.9221 | 0.1295 | 0.9633 | 0.9270 | 0.064252 | 0.9103 |

样例：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_16_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_32_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples\ordered_contact_sheet_checker.png
```

判断：

- 分段重置 + 非线性 `0.5 -> 5` 是目前最好的 compact schedule。32 token test loss 从线性 compact 的 0.044031、全程 precision 的 0.047576 降到 0.037481。
- 64 token 的 alpha IoU 最高，test 0.9221、heldout 0.9103；但 object RGB MSE 和综合 loss 比 32 token 差，说明 mask/覆盖更全，但颜色/对象属性仍错。
- 当前最佳口径分裂：若看综合重建 loss，32 token 最好；若只看 alpha IoU，64 token 最好。图像还原任务应优先看综合重建和对象字段，因此当前不应把 64 判成通过。
- 这轮仍没有触发加参数量条件。token 增加仍有结构性影响，主要问题是长 token 的对象/颜色 residual 分工，而不是模型容量已耗尽。

## correct-pixel residual smoke

用户提出把 compact prefix guidance 的终点从覆盖率改成“正确像素数量”，错误像素作为惩罚，并继续测试 residual/token 分工。脚本已切到 `schema_version=5`：

- `compact_prefix_loss` 不再用旧的 `area / coverage` 作为训练目标，而是最大化前景正确像素，惩罚背景误涂、前景颜色错误和漏前景；
- 前景正确像素使用 `target_alpha * pred_alpha^n * exp(-rgb_mse / tau)`，其中 `n` 仍在每个 active-token 段内按 `0.5 -> 5` 变化，`tau` 默认按 `0.08 -> 0.02` 变严；
- 相邻 prefix 增加 residual routing loss：保留前序已经正确的像素，限制正确区域外的 delta，并把后续 token 的修正压力集中到前序仍错误的像素；
- 评估新增 soft/hard correct pixel、wrong background、wrong color、missed foreground 指标，避免只看 alpha IoU。

CPU smoke：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py `
  --output artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\correct_pixel_residual_cpu_smoke_result.json `
  --latent-tokens-list 1,2 `
  --d-model 16 `
  --latent-dim 2 `
  --encoder-width 16 `
  --decoder-width 32 `
  --steps 2 `
  --batch-size 4 `
  --train-limit 16 `
  --val-limit 8 `
  --test-limit 8 `
  --heldout-limit 8 `
  --eval-batch-size 4 `
  --greedy-eval-limit 4 `
  --sample-count 1 `
  --device cpu `
  --no-amp `
  --no-gpu-resident-data
```

smoke 结果：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\correct_pixel_residual_cpu_smoke_result.json
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\correct_pixel_residual_cpu_smoke_result_latent_2_samples\samples.json
```

结果只证明链路可跑，不作为模型能力证据：

| 项 | 数值 |
| --- | ---: |
| schema_version | 5 |
| latent tokens | 1, 2 |
| steps | 2 |
| elapsed | 6.18 sec |
| peak CUDA allocated | 0 MB |

建议的正式比较命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avi_whole_image_latent_capacity.py `
  --output artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result.json `
  --latent-tokens-list 1,2,4,8,16,32,64 `
  --latent-dim 2 `
  --encoder-width 192 `
  --decoder-width 768 `
  --steps 7000 `
  --batch-size 128 `
  --eval-batch-size 128 `
  --greedy-eval-limit 512 `
  --sample-count 4 `
  --compact-precision-power-start 0.5 `
  --compact-precision-power-end 5 `
  --compact-precision-curve-power 0.5 `
  --correct-pixel-color-tau-start 0.08 `
  --correct-pixel-color-tau-end 0.02 `
  --correct-pixel-color-tau-curve-power 0.5 `
  --residual-prefix-weight 0.12 `
  --residual-weight-start 0.15 `
  --residual-weight-end 1.0 `
  --device cuda
```

判断：

- 这次 smoke 已验证新 loss、history、评估指标、ordered/greedy 样例输出都能跑通。
- 正式判断仍要等全量 CUDA probe。核心看 64 token 是否不再出现“alpha IoU 上升但 object RGB / correct pixel 变差”的分裂。
- 如果 64 token 的 hard/soft correct pixel 仍低于 32，但 alpha IoU 继续更高，说明 residual/token 分工仍没解决对象/颜色绑定。

## correct-pixel residual formal probe

训练已按上一节正式命令完成：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result.json
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_summary.json
```

配置与成本：

| 项 | 数值 |
| --- | ---: |
| schema_version | 5 |
| latent_dim | 2 |
| encoder_width / decoder_width | 192 / 768 |
| latent tokens | 1, 2, 4, 8, 16, 32, 64 |
| precision power schedule | 每段 0.5 -> 5 |
| color tau schedule | 每段 0.08 -> 0.02 |
| residual weight schedule | 每段 0.15 -> 1.0 |
| steps | 每档 7,000 |
| unique train images | 36,000 |
| elapsed | 1,951.78 sec |
| peak CUDA allocated | 2,083.41 MB |

ordered final prefix：

| latent tokens | test loss | object RGB MSE | alpha IoU | soft correct pixel | hard correct pixel | soft wrong bg | soft wrong color | soft missed fg | heldout loss | heldout soft correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.122681 | 0.072500 | 0.6629 | 0.1869 | 0.1280 | 0.0578 | 0.4508 | 0.1806 | 0.165853 | 0.1233 |
| 2 | 0.093867 | 0.060837 | 0.7622 | 0.3455 | 0.3335 | 0.0402 | 0.3769 | 0.1197 | 0.130519 | 0.2680 |
| 4 | 0.077118 | 0.046518 | 0.7966 | 0.3870 | 0.3545 | 0.0331 | 0.3746 | 0.1062 | 0.106397 | 0.3230 |
| 8 | 0.069165 | 0.042022 | 0.8192 | 0.4096 | 0.3306 | 0.0287 | 0.3987 | 0.0836 | 0.094587 | 0.3617 |
| 16 | 0.057707 | 0.030764 | 0.8444 | 0.4757 | 0.4139 | 0.0249 | 0.3427 | 0.0788 | 0.077176 | 0.4249 |
| 32 | 0.058714 | 0.035651 | 0.8441 | 0.4604 | 0.3437 | 0.0248 | 0.4044 | 0.0568 | 0.077955 | 0.4251 |
| 64 | 0.081500 | 0.050253 | 0.8665 | 0.3534 | 0.2455 | 0.0137 | 0.4885 | 0.0884 | 0.108579 | 0.3327 |

与上一轮 segment precision compact 的公共指标对比：

| latent tokens | loss delta | object RGB delta | alpha IoU delta | precision coverage delta |
| ---: | ---: | ---: | ---: | ---: |
| 16 | +0.016345 | +0.001793 | -0.064264 | -0.100308 |
| 32 | +0.021232 | +0.011414 | -0.068617 | -0.058192 |
| 64 | +0.032763 | +0.014398 | -0.055590 | -0.085169 |

样例：

```powershell
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result_latent_16_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result_latent_32_samples\ordered_contact_sheet_checker.png
artifacts\omni_transformer_stage_avi_whole_image_latent_capacity\probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples\ordered_contact_sheet_checker.png
```

判断：

- 本轮不通过。`schema_version=5` 的 correct-pixel residual 权重组合没有修复长 token 分工，反而使 16/32/64 在公共指标上全部低于上一轮 segment precision compact。
- 当前最佳从上一轮 32 token 退到本轮 16 token：16 token 的 test loss、object RGB MSE、soft/hard correct pixel 和 soft wrong color 都是本轮最好。
- 64 token 仍只在 alpha IoU 上最高，但综合 loss、object RGB、soft/hard correct pixel、wrong color 都明显差于 16/32。它没有证明“更多 token 学会了残差分工”，更像后续 token 在继续压背景和修 mask，同时放大颜色/对象错误。
- 视觉样例确认了这个判断：32/64 输出更紧、更少背景误涂，但目标对象经常缺失、收缩或错色；这说明 correct-pixel/wrong-color loss 当前权重和 schedule 过硬，容易把模型推向保守/错色局部解。

## 判断

宽 token 结果改变了 1000-step 旧判断。透明背景后，背景误涂基本被压住，`latent_dim=192, latent_tokens=12` 已经可以在当前 AV-H 渲染图上重建主要颜色、位置和形状。样图中 1 个 token 已经画出主体，2 个 token 之后多数是边缘和颜色微调。

所以在宽 token 设置下，当前不能说“12 token 不够”。更准确的判断是：

- 当前 AV-H record 渲染图的视觉熵仍然偏低；
- `latent_dim=192` 时，1-2 个连续 latent token 已能承载大部分图像结构；
- `latent_dim=192` 的 12/24/32 后续 token 边际收益很小，且 24/32 没有优于 12；
- `latent_dim=2` 时，token 数量变得重要；分段 precision compact 后 32 token 综合最好，64 token alpha IoU 最高，但 1-64 token 仍没有闭合高保真重建。

本轮没有继续加模型参数量。原因是用户设定的顺序是“先 step/训练集足够，再 token 足够，再参数量”。分段 precision compact 后，2 维 token 仍没有证明“token 趋于无效”：32/64 的差异仍明显，长 token 的问题更像 residual routing 和对象/颜色分工，而不是参数不足。`schema_version=5` 的第一版 correct-pixel residual probe 已证明当前权重组合失败，下一步应修 loss/schedule，而不是加参数量。

## 担忧

1. `latent_dim=192` 的单个 latent token 已有 192 个连续数值，对 64x64 的低熵几何图来说容量很大；用 token 数判断“图像描述是否足够”会被 token 宽度掩盖。
2. 新增 residual routing 后，后续 token 已有残差监督，但第一版权重尺度在正式 probe 中失败；它可能过早惩罚 wrong color 和 delta，导致模型宁可收缩/错色，也不稳定复原对象。
3. AV-H 渲染图仍是 record 生成的合成几何图，离真正高熵图像编辑的 lossless latent 还差很远。
4. 分段 `n=0.5->5` 修正了全程 `n=1->4` 对长 token 的伤害，但 64 token 仍没有在综合重建上超过 32；这说明 schedule 不是唯一瓶颈。
5. 只看透明 loss/alpha IoU 仍不够。当前 correct/wrong pixel 指标已经暴露 64 token 的失败，但仍需要 scene parser 或 record inverse 指标，验证颜色、形状、位置是否逐字段 exact。

## 下一步

1. 不沿用当前 schema v5 权重。先回到 segment precision compact 作为基线，把 correct/wrong pixel 保留为指标。
2. 下一版 loss 应弱化或分阶段启用 wrong-color / residual delta：先 mask/coverage 稳定，再逐步加颜色绑定；避免一开始用严格 `tau=0.02` 把半对像素全部推成 wrong color。
3. 加对象/record inverse parser 指标，报告 scene exact、object exact 和 no-template 背景误涂，再决定对象级 loss 是否可用。
4. 再提高图像任务熵：更多对象、遮挡、细纹理、颜色梯度、局部噪声或更高分辨率。
5. 如果 2 维 token 在修正后的 residual 分工、更高熵数据和 record inverse 指标下仍不能复原，并且 token 增加收益趋平，再按用户设定进入参数量 sweep。
