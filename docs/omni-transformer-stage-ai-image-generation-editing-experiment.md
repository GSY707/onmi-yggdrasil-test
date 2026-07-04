# Stage AI：图像生成与源图编辑输出专家实验

## 目的

本阶段首次把验证目标从“理解图像并输出文本/答案”切到“潜变量输出专家反向还原像素材料”。对应 `note.txt` 里的要求：

- 输入/输出专家需要能从统一潜变量反向还原材料。
- 图像修改不能只靠输出猜测，源图中的未修改属性必须被保留。
- 资源上限必须显式可控，长训不得超过 36 小时。

## 新增实现

`experiments/omni_transformer_stage_ai_image_generation_editing.py` 新增三个可对照模型：

| 变体 | 说明 |
| --- | --- |
| `prompt_direct` | 文本 prompt 直接池化到 pixel decoder，作为非 latent 对照。 |
| `latent_output` | 文本 prompt 先经固定数量 latent output tokens，再由输出专家渲染 PNG。 |
| `latent_image_edit` | 源图 tokens + 低熵 edit prompt 经 latent output tokens，再由输出专家生成目标图。 |

任务是合成 64x64 单物体图像，属性为颜色、形状、位置。生成 prompt 完整描述目标；编辑 prompt 只说改哪一项，例如 `change color to blue`，未提到的形状/位置必须从源图保留。

训练损失使用前景加权 MSE + 属性辅助监督。这里不是为了证明真实扩散模型质量，而是避免小物体被背景像素淹没，直接测试“潜变量能否驱动输出专家还原可解析材料”。

## 运行命令

### Smoke

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ai_image_generation_editing.py --output artifacts\omni_transformer_stage_ai_image_generation_editing\smoke_result.json --train-size 256 --val-size 64 --test-size 96 --batch-size 32 --d-model 64 --layers 1 --heads 4 --latent-tokens 6 --direct-steps 80 --latent-steps 100 --edit-steps 120 --eval-every 40 --max-train-seconds 3600
```

### 正式 sweep

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ai_image_generation_editing.py --sweep --seeds 20260701,20260702,20260703 --train-size 4096 --val-size 1024 --test-size 1024 --batch-size 96 --image-size 64 --prompt-len 96 --d-model 128 --layers 2 --heads 4 --latent-tokens 16 --direct-steps 1800 --latent-steps 2400 --edit-steps 3600 --eval-every 600 --foreground-loss-weight 6.0 --max-train-seconds 129600 --output-dir artifacts\omni_transformer_stage_ai_image_generation_editing\sweep_runs --aggregate artifacts\omni_transformer_stage_ai_image_generation_editing\sweep_results.json
```

脚本默认 `--max-train-seconds 129600`，训练循环会检查全局 deadline；如果到时会停止并在 JSON 中写入 `stopped_by_time_budget`。

## 正式 sweep 结果

设备：NVIDIA GeForce RTX 4070 Laptop GPU，Torch 2.11.0+cu128。正式 sweep 跑了 3 个 seed，平均总耗时 250.092 秒，未触发 36 小时时间上限。

| 模型/消融 | pixel MSE | nearest scene exact | 属性头 scene exact | 备注 |
| --- | ---: | ---: | ---: | --- |
| `prompt_direct` | 0.000472 | 100.00% | 100.00% | 低熵 prompt 直接生成已饱和 |
| `latent_output` | 0.000485 | 100.00% | 100.00% | latent output 同样饱和，没有继续拉开差距 |
| `latent_image_edit` | 0.000193 | 100.00% | 100.00% | 源图编辑在该合成任务上饱和 |
| `latent_image_edit_no_source` | 0.012037 | 3.42% | 4.42% | 移除源图后接近失败，说明编辑确实依赖源图 |
| `source_image_no_edit` | 0.013962 | 0.00% | - | 只复制源图不能完成编辑 |

样例输出在：

- `artifacts/omni_transformer_stage_ai_image_generation_editing/samples/result/seed20260701/`
- `artifacts/omni_transformer_stage_ai_image_generation_editing/samples/smoke_result/seed20260701/`
- `artifacts/omni_transformer_stage_ai_image_generation_editing/sweep_runs/*/result.json`

## 结论

1. Stage AI 证明“低熵文本/源图编辑意图 -> latent output tokens -> 像素输出专家”链路能闭合。
2. 正式 sweep 后，主任务已经完全饱和：`prompt_direct`、`latent_output`、`latent_image_edit` 都是 100% scene exact，因此它不再能区分“direct vs latent”或“哪种输出专家更强”。
3. no-source 和 source-no-edit 两个消融仍有价值：no-source 只有 3.42%，source-no-edit 为 0%，说明编辑成功不是简单猜测或复制源图。
4. 这个阶段不能再被当成“图像生成架构已证明”的证据。它只是低熵单物体模板任务，不含自然图像、多对象、遮挡、局部 mask、复杂纹理、扩散采样或审美质量门禁。

## 担忧与下一步

1. 继续在 Stage AI 上加 seed 或延长训练收益很低；用户已经跑完正式 sweep，结果显示一个 seed 足够暴露这类低熵任务是否饱和。
2. 下一步必须把任务改成更接近“吃下整张图 -> latent -> 完全重新画出来”：源图应包含 prompt 不可见的背景/纹理信息，输出必须重绘完整图，而不只是生成模板物体。
3. 仅看全图 pixel MSE 会被大面积背景主导，后续指标必须分 foreground/background，并保留 nearest scene / 属性保真 / 样例 PNG 门禁。
4. 正式 36 小时训练前仍应加 checkpoint、中间 PNG 样例和断点恢复；当前脚本适合快速 GPU 验证和 sweep 聚合。
