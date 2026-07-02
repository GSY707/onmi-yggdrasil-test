# Stage U：扩展视觉输入、对象专家与输出 Token Sweep

## 目的

Stage T 证明“保留 raw/weighted expert tokens，再追加 summary tokens”的 latent bus 可以修复 Attention Pump 造成的信息丢失，但 L3 位置组合仍未稳定到 90%+，L4 counting 仍接近随机。Stage U 按用户要求继续提高任务熵，并同时测试三件事：

- 扩大视觉输入 expert：从 Stage Q/T 的少量 vision latent 改为保留 8x8 patch grid，即 64 个 raw patch tokens。
- 新增功能专家：在 patch expert 之外加入 object slot、spatial、counting、prompt、fusion experts。
- 扩大输出 token 数：对追加到保真 latent bus 的 output summary tokens 做 8/16/32 sweep。

## 任务设计

Stage U 使用 4 类多物体合成视觉问答，每个样本 4-8 个对象、4x4 位置网格、8 色、4 形状，统一做 8 候选 ranking：

| Family | 问题类型 | 答案空间 |
| --- | --- | --- |
| `u1_cell_attribute` | 指定 row/column cell，回答该对象颜色 | 8 色 |
| `u2_spatial_relation` | 指定参考对象，回答 left/right/above/below 最近对象颜色 | 8 色 |
| `u3_counting` | 统计指定 color+shape 的对象数量 | 0-7 |
| `u4_spatial_filter` | 回答 topmost/bottommost/leftmost/rightmost 某形状对象颜色 | 8 色 |

## 结构对比

| Variant | 结构 |
| --- | --- |
| `patch_wide_latent` | 64 patch tokens + prompt/fusion tokens；raw + route-weighted tokens 保留，再追加 output tokens |
| `object_slot_spatial_wide_latent` | 在 `patch_wide_latent` 基础上加入 object slot、spatial、counting experts；训练时 25% 概率清零 raw patch context，让专家 token 学会承担证据流 |

关键消融：

- `no_image_modality`：输入图像置零，检查是否真的依赖视觉。
- `no_patch_expert`：对象/空间/计数专家先读图像，然后从 latent bus 清零 raw patch tokens，检查专家 token 能否替代 raw patch。
- `no_object_experts` / `no_spatial_expert` / `no_count_expert`：检查新增专家是否有因果贡献。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_u_object_slots.py --sweep --seeds 20260701,20260702 --output-token-counts 8,16,32 --train-size 1536 --val-size 384 --test-size 512 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --fusion-tokens 8 --candidate-count 8 --train-steps 520 --probe-steps 100 --eval-every 130 --patch-context-dropout 0.25 --output-dir artifacts\omni_transformer_stage_u_object_slots\sweep_runs --aggregate artifacts\omni_transformer_stage_u_object_slots\sweep_results.json
```

## 聚合结果

随机 Top-1 为 12.50%。

| Output tokens | Patch full | Patch no image | Patch no patch | Object full | Object no image | Object no patch | Object no object experts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16.80% | 14.75% | 13.38% | 20.80% | 14.84% | 14.06% | 21.09% |
| 16 | 19.04% | 13.96% | 15.23% | 15.33% | 13.77% | 15.72% | 15.72% |
| 32 | 18.85% | 13.18% | 14.75% | 17.68% | 13.48% | 15.23% | 17.97% |

按任务族看，`object_slot_spatial_wide_latent` 只有 output=8 时整体优于 patch-only；output=16/32 不稳定：

| Output tokens | Variant | Cell | Relation | Counting | Spatial filter |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | patch | 15.62% | 15.23% | 19.92% | 16.41% |
| 8 | object | 21.48% | 21.88% | 18.75% | 21.09% |
| 16 | patch | 22.27% | 16.02% | 19.92% | 17.97% |
| 16 | object | 14.06% | 14.45% | 14.06% | 18.75% |
| 32 | patch | 21.09% | 17.58% | 15.23% | 21.48% |
| 32 | object | 16.02% | 17.19% | 20.70% | 16.80% |

成本：

| Variant | Params | Train seconds/run | Prediction ms/example |
| --- | ---: | ---: | ---: |
| patch, output=8 | 808,132 | 21.23 | 0.179 |
| patch, output=16 | 808,900 | 20.55 | 0.191 |
| patch, output=32 | 810,436 | 21.07 | 0.196 |
| object, output=8 | 1,146,823 | 25.24 | 0.250 |
| object, output=16 | 1,147,591 | 26.50 | 0.245 |
| object, output=32 | 1,149,127 | 26.56 | 0.251 |

## 结论

Stage U 是一个负向边界实验，不是正向突破。

1. 扩大视觉输入到 64 patch tokens 后，`patch_wide_latent` 在高熵多物体任务上能稳定高于随机和 no-image，但准确率只有 16.80%-19.04%，远未解决任务。
2. 新增 learned object/spatial/counting experts 没有被证明有效：只有 output=8 的 full top1 达到 20.80%，但 `no_object_experts` 不降反升到 21.09%；output=16/32 反而低于 patch-only。
3. `no_patch_expert` 基本掉回 14%-16% 附近，说明当前 object/spatial/count tokens 不能替代 raw patch tokens；核心视觉证据仍在 raw patch bus 里。
4. 输出 token 数没有单调收益。patch-only 最好是 output=16，object variant 最好是 output=8；所以当前瓶颈不是追加 output summary token 不够，而是对象级归纳偏置和训练监督不足。
5. 成本上，object variant 约多 42% 参数，预测成本从约 0.18-0.20 ms/example 增到约 0.245-0.251 ms/example，但没有稳定换来准确率收益。

## 对架构的影响

Stage U 不否定“多专家 + 保真 latent bus”的方向，但它否定了一个更具体的假设：只用无监督 QueryResampler 形式的 learned object slots，就能自然解决 counting/spatial/relation。

下一步如果继续做对象/空间/计数，应直接切换到更强专家，而不是在当前 learned slot 上打补丁：

- 使用带辅助监督的 object slot expert：occupancy、color、shape、cell/box、count 辅助 loss。
- 或使用 detector/segmentation/grounding 预训练专家，把对象表或对象 token 送入 latent bus。
- 保留 raw patch evidence tokens，先不要在第一层强压缩；token 成本由记忆树/工作树分层摘要处理。
- 对 output tokens 的后续实验应放在专家已经能可靠抽取对象之后，否则 output token sweep 只是在测弱上游的随机训练波动。
