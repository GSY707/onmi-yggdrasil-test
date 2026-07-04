# Stage AJ：Transformer 图像输入到 latent 再完整重绘保真实验

## 目的

Stage AI 已经被正式 sweep 跑到饱和，不能继续区分图像输出架构。Stage AJ 改成更接近未来图像编辑的流程：

```text
整张源图 -> patch tokens -> latent tokens -> Transformer patch decoder -> 完整目标图
```

这里的关键不是“生成一个低熵模板物体”，而是测试输入图中 prompt 不描述的细节能否穿过 latent 并在输出端完整重绘。未来图像编辑如果要吃下整张图再完全重新画出来，就必须过这类保真门禁。

## 新增实现

`experiments/omni_transformer_stage_aj_transformer_image_io_fidelity.py` 现在有两组变体：

| 变体 | 说明 |
| --- | --- |
| `transformer_copy` | 固定 latent tokens 基线：源图 patch tokens 经 Transformer encoder 和 QueryResampler 后，由 Transformer patch decoder 重绘源图。 |
| `transformer_edit` | 固定 latent tokens 基线：源图 patch tokens + 低熵 edit prompt 经 QueryResampler 后，由 Transformer patch decoder 重绘目标图。 |
| `memory_tree_copy` | 记忆树式逐层残差 latent：先生成上层 token 解释主要信息，扣掉已解释部分，再生成下一层 token，最后完整重绘源图。 |
| `memory_tree_supervised_copy` | copy-only 辅助监督变体：在 memory-tree latent 上加对象属性和前景 mask 读头，只训练源图完整重绘。 |
| `memory_tree_edit` | 记忆树式逐层残差 latent：源图 + edit prompt 共同进入分层 residual tokenizer，逐层生成 latent 后完整重绘目标图。 |
| `memory_tree_supervised_edit` | edit 辅助监督变体：源图 + edit prompt 进入 memory-tree latent，latent 读头预测目标对象属性和 mask，再完整重绘目标图。 |
| `text_supervised_generate` | 文本生成辅助监督变体：完整目标文本提示进入 memory-tree latent，在规范背景上生成目标对象。 |

消融：

| 消融 | 作用 |
| --- | --- |
| `transformer_edit_no_source` | 移除源图，只看 edit prompt 是否能猜目标图。 |
| `memory_tree_edit_no_source` | 对记忆树式编辑模型移除源图，验证源图信息是否进入分层 latent。 |
| `source_image_no_edit` | 直接复制源图，验证编辑是否不能靠复制完成。 |

图像仍是 64x64，但每个样本加入 prompt 不可见的合成背景纹理。源图和目标图共享背景，edit prompt 只改颜色、形状或位置之一。因此模型必须同时做到：

- 保留背景纹理。
- 根据 edit prompt 修改对象。
- 保留未修改的对象属性。
- 完整重绘输出图，而不是局部 patch。

## 架构口径

Stage AJ 不使用卷积图像 decoder：

- 图像输入：8x8 patchify，线性 patch projection，加位置向量。
- 图像编码：Transformer encoder。
- latent 压缩：固定基线使用 QueryResampler；新默认使用 progressive residual memory-tree latent。
- 文本编辑：字符级 Transformer text encoder。
- 图像输出：learned patch queries cross-attend latent，再经过 Transformer decoder block 和线性 patch head unpatchify。

线性 patch projection/head 是 token 嵌入与反嵌入层，不是卷积生成器。这个口径更接近“整个模型主干用 Transformer”。

`memory_tree_*` 的核心机制：

1. 第 1 层 latent token 读取完整 source token field，写回它能解释的部分。
2. 源 token 减掉已解释部分，得到 residual token field。
3. 第 2 层继续读取 residual，逐层向下，直到最后一层。
4. 训练时同时约束最终图、各层 prefix 图和 token reconstruction loss，使上层 token 尽量解释最大变化，下层 token 补细节。

`memory_tree_supervised_copy` 额外加入 copy-only 辅助监督：

- latent 对象读头预测源图 `color/shape/position`。
- latent mask 读头预测源图前景 mask。
- 训练 loss 保持完整图像重绘为主，同时加入属性 CE 和带前景权重的 mask BCE。
- checkpoint 选择从全图 pixel MSE 改为更重视 foreground MSE 和 scene exact，避免背景纹理主导模型选择。

`memory_tree_supervised_edit/text_supervised_generate` 复用同一套 latent 对象读头：

- edit 监督目标是编辑后的 `target_attrs/target_mask`，同时输出完整目标图并保留源图随机背景。
- edit 的 no-source 消融会把源图 tokens 清零，用来检查模型是否真的依赖源图保留未修改属性和背景。
- generation 使用完整目标文本提示，例如 `green diamond at bottom right`，输出规范背景上的目标对象；不要求从文本恢复随机背景，因为随机背景没有进入文本。

## 运行命令

### 已跑中等单 seed

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_aj_transformer_image_io_fidelity.py --output artifacts\omni_transformer_stage_aj_transformer_image_io_fidelity\result.json --train-size 1024 --val-size 128 --test-size 128 --batch-size 32 --d-model 96 --encoder-layers 2 --decoder-layers 2 --heads 4 --latent-tokens 16 --copy-steps 300 --edit-steps 500 --eval-every 250 --max-train-seconds 7200
```

### Smoke

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_aj_transformer_image_io_fidelity.py --output artifacts\omni_transformer_stage_aj_transformer_image_io_fidelity\memory_tree_smoke_result.json --train-size 64 --val-size 16 --test-size 16 --batch-size 16 --d-model 64 --encoder-layers 1 --decoder-layers 1 --heads 4 --latent-tokens 8 --latent-levels 4 --copy-steps 4 --edit-steps 4 --eval-every 2 --max-train-seconds 600
```

### Copy-only supervised smoke

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_aj_transformer_image_io_fidelity.py --variants memory_tree_supervised_copy --output artifacts\omni_transformer_stage_aj_transformer_image_io_fidelity\supervised_copy_smoke_result.json --train-size 64 --val-size 16 --test-size 16 --batch-size 16 --d-model 64 --encoder-layers 1 --decoder-layers 1 --heads 4 --latent-tokens 8 --latent-levels 4 --copy-steps 4 --edit-steps 0 --eval-every 2 --train-eval-size 16 --max-train-seconds 600
```

### 36 小时内 copy-only supervised 单 seed 建议

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_aj_transformer_image_io_fidelity.py --variants memory_tree_supervised_copy --output artifacts\omni_transformer_stage_aj_transformer_image_io_fidelity\formal_supervised_copy_result.json --train-size 4096 --val-size 512 --test-size 512 --batch-size 128 --image-size 64 --patch-size 8 --d-model 192 --encoder-layers 2 --decoder-layers 3 --heads 6 --latent-tokens 24 --latent-levels 6 --copy-steps 1500 --edit-steps 0 --eval-every 300 --train-eval-size 128 --template-chunk-size 30 --foreground-loss-weight 8.0 --background-loss-weight 2.0 --prefix-loss-weight 0.25 --residual-token-loss-weight 0.02 --object-aux-attr-loss-weight 1.0 --object-aux-mask-loss-weight 0.5 --object-aux-mask-positive-weight 8.0 --torch-num-threads 1 --max-train-seconds 129600
```

### Edit/generation probe

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_aj_transformer_image_io_fidelity.py --variants memory_tree_supervised_edit,text_supervised_generate --output artifacts\omni_transformer_stage_aj_transformer_image_io_fidelity\supervised_edit_generate_probe_result.json --train-size 2048 --val-size 256 --test-size 256 --batch-size 128 --image-size 64 --patch-size 8 --d-model 128 --encoder-layers 2 --decoder-layers 3 --heads 4 --latent-tokens 16 --latent-levels 4 --copy-steps 0 --edit-steps 600 --generate-steps 600 --eval-every 200 --train-eval-size 128 --template-chunk-size 30 --foreground-loss-weight 8.0 --background-loss-weight 2.0 --prefix-loss-weight 0.25 --residual-token-loss-weight 0.02 --object-aux-attr-loss-weight 1.0 --object-aux-mask-loss-weight 0.5 --object-aux-mask-positive-weight 8.0 --torch-num-threads 1 --max-train-seconds 7200
```

默认 seed 是 `20260701`。Stage AI 已经证明这类快速合成任务多 seed 收益很低，因此 Stage AJ 默认用单 seed 做下一步架构门禁。当前正式命令使用 `--batch-size 128`，因为 192 会在本机爆显存；GPU 低功率问题主要来自旧版 CPU/Python 评估瓶颈，已通过批量 GPU template parser、训练期小验证集和 `--torch-num-threads 1` 缓解。

## 中等单 seed fixed baseline 结果

设备：NVIDIA GeForce RTX 4070 Laptop GPU，Torch 2.11.0+cu128。总耗时 142.149 秒，未触发时间上限。

| 模型/消融 | pixel MSE | background MSE | foreground MSE | nearest scene exact | color | shape | position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer_copy` | 0.012409 | 0.002469 | 0.230095 | 0.00% | 19.53% | 24.22% | 21.09% |
| `transformer_edit` | 0.009900 | 0.002343 | 0.182054 | 3.12% | 25.00% | 25.78% | 53.12% |
| `transformer_edit_no_source` | 0.010548 | 0.001973 | 0.205859 | 2.34% | 21.88% | 25.00% | 51.56% |
| `source_image_no_edit` | 0.014125 | 0.005772 | 0.204409 | 0.00% | 69.53% | 68.75% | 61.72% |

样例输出在：

- `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/result/seed20260701/`
- `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/smoke_result/seed20260701/`
- `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/memory_tree_smoke_result/seed20260701/`
- `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/formal_memory_tree_result/seed20260701/`

视觉抽检：`00_target.png` 是带背景纹理的绿色菱形，`00_transformer_edit.png` 主要重绘出背景/模糊块，没有可靠画回目标物体；`00_transformer_copy.png` 也没有稳定复制源图中的绿色方块。

## 正式 memory-tree 单 seed 结果

运行结果：

- JSON：`artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/formal_memory_tree_result.json`
- 样例：`artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/formal_memory_tree_result/seed20260701/`
- 设备：NVIDIA GeForce RTX 4070 Laptop GPU，Torch 2.11.0+cu128。
- 总耗时：1205.781 秒，约 20.1 分钟，未触发 36 小时时间上限。

配置：train 4096、val 512、test 512、batch 128、d_model 192、heads 6、latent tokens 24、latent levels 6、copy 1500 steps、edit 2400 steps。

| 模型/消融 | pixel MSE | background MSE | foreground MSE | nearest scene exact | color | shape | position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_tree_copy` | 0.010963 | 0.001396 | 0.222561 | 0.59% | 17.77% | 23.44% | 17.97% |
| `memory_tree_edit` | 0.007041 | 0.002332 | 0.112250 | 11.91% | 59.57% | 28.71% | 54.30% |
| `memory_tree_edit_no_source` | 0.010940 | 0.002568 | 0.198056 | 3.71% | 22.66% | 27.93% | 46.88% |
| `source_image_no_edit` | 0.014189 | 0.005336 | 0.211882 | 0.00% | 69.92% | 65.62% | 64.45% |

按 edit family 拆分，`memory_tree_edit` 的 scene exact 是：

| edit family | scene exact |
| --- | ---: |
| color_edit | 1.95% |
| shape_edit | 3.98% |
| position_edit | 28.02% |

训练中 `memory_tree_edit` 的小验证集 scene exact 在 1800 step 达到 10.16%，2400 step 回落到 4.69%；最终 test 为 11.91%。这说明信号存在，但还不稳定，不能把它当成已经收敛。

视觉抽检：

- `00_target.png` 是 `green diamond at bottom right`，`00_memory_tree_edit.png` 生成了多个绿色模糊块，没有稳定画成单个右下菱形。
- `03_target.png` 是 `yellow circle at top left`，`03_memory_tree_edit.png` 能把黄色能量搬到左上，但形状和边界仍然发散。
- `memory_tree_copy` 基本只学会背景，源图对象没有被可靠复制。

## 正式 copy-only 辅助监督结果

运行结果：

- JSON：`artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/formal_supervised_copy_result.json`
- 样例：`artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/formal_supervised_copy_result/seed20260701/`
- 设备：NVIDIA GeForce RTX 4070 Laptop GPU，Torch 2.11.0+cu128。
- 总耗时：219.871 秒；模型训练 214.186 秒，未触发 36 小时时间上限。

配置：train 4096、val 512、test 512、batch 128、d_model 192、heads 6、latent tokens 24、latent levels 6、copy 1500 steps、edit 0 steps。

| 模型 | pixel MSE | background MSE | foreground MSE | nearest scene exact | color | shape | position | aux scene | aux mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_tree_supervised_copy` | 0.000325 | 0.000304 | 0.000794 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

训练轨迹：

| step | foreground MSE | nearest scene exact | aux scene | aux mask IoU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.158277 | 0.78% | 0.00% | 4.40% |
| 300 | 0.069198 | 40.62% | 96.09% | 86.05% |
| 600 | 0.010053 | 100.00% | 100.00% | 100.00% |
| 900 | 0.002193 | 100.00% | 100.00% | 100.00% |
| 1200 | 0.001062 | 100.00% | 100.00% | 100.00% |
| 1500 | 0.000778 | 100.00% | 100.00% | 100.00% |

视觉抽检：`00_source.png` 是右下绿色方块，`00_memory_tree_supervised_copy.png` 已经能在同位置重绘绿色方块，背景仍有轻微重绘噪声，但不再是旧模型那种只学背景、丢对象的失败模式。

中等 probe 也保留为快速验证记录：`supervised_copy_probe_result.json` 在 train 2048、test 256、copy 600 steps 下达到 scene exact 100%、foreground MSE 0.005730、aux scene/mask IoU 100%。正式结果把同一结论扩展到更大训练/测试规模和更大模型。

## Edit/generation 辅助监督 probe 结果

运行结果：

- JSON：`artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/supervised_edit_generate_probe_result.json`
- 样例：`artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/supervised_edit_generate_probe_result/seed20260701/`
- 设备：NVIDIA GeForce RTX 4070 Laptop GPU，Torch 2.11.0+cu128。
- 总耗时：187.107 秒；`memory_tree_supervised_edit` 训练 88.333 秒，`text_supervised_generate` 训练 88.054 秒。

配置：train 2048、val 256、test 256、batch 128、d_model 128、heads 4、latent tokens 16、latent levels 4、edit 600 steps、generate 600 steps。

| 模型/消融 | pixel MSE | background MSE | foreground MSE | nearest scene exact | color | shape | position | aux scene | aux mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_tree_supervised_edit` | 0.001267 | 0.000890 | 0.009834 | 99.22% | 100.00% | 99.22% | 100.00% | 100.00% | 99.56% |
| `memory_tree_supervised_edit_no_source` | 0.012344 | 0.002176 | 0.242842 | 5.08% | 29.30% | 36.72% | 49.61% | 6.64% | 39.87% |
| `source_image_no_edit` | 0.014321 | 0.005742 | 0.208804 | 0.00% | 71.09% | 65.62% | 63.28% | - | - |
| `text_supervised_generate` | 0.000312 | 0.000189 | 0.003115 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

按 edit family 拆分，`memory_tree_supervised_edit` 的 scene exact 是：

| edit family | scene exact |
| --- | ---: |
| color_edit | 97.30% |
| shape_edit | 100.00% |
| position_edit | 100.00% |

视觉抽检：

- `00_target.png` 是随机背景上的右下绿色菱形；`00_memory_tree_supervised_edit.png` 在同一随机背景上重绘出目标对象，但边缘仍有轻微糊化。
- `00_memory_tree_supervised_edit_no_source.png` 无法可靠恢复对象和背景，符合 no-source 5.08% 的消融结果。
- `00_text_supervised_generate.png` 能在规范背景上生成右下绿色菱形；这证明文本到 latent 到 patch decoder 的生成链路成立，但不证明随机背景可从文本生成。

未完成：尝试用 d_model 192、train/test 4096/512、edit/generate 各 1500 steps 跑正式单 seed 时，命令超过 30 分钟工具超时且没有写出 JSON；残留进程已停止。因此 edit/generation 目前只按 probe 等级报告，不当作正式长训结论。

## 结论

1. fixed QueryResampler baseline 是明确负结果：`transformer_edit` scene exact 只有 3.12%，`transformer_copy` 为 0%。
2. memory-tree residual latent 在无辅助监督的 edit 上有弱正信号：`memory_tree_edit` scene exact 到 11.91%，比 fixed edit 的 3.12% 高；no-source 只有 3.71%，说明源图确实提供了信息。
3. 无辅助监督 copy 门禁失败：`memory_tree_copy` scene exact 只有 0.59%，foreground MSE 仍有 0.222561。
4. copy-only 辅助监督把正式 copy 门禁打通：`memory_tree_supervised_copy` test scene exact 100%，foreground MSE 0.000794，aux 对象表和 mask 也都是 100%。
5. 这说明当前 copy 失败不是 Transformer patch decoder 完全不能画对象，而是 latent 缺少可操作的对象约束。属性/mask 辅助监督能把对象信息压进 latent，并让完整重绘跟上。
6. edit/generation 在 probe 规模出现强正信号：supervised edit scene exact 99.22%，no-source 只有 5.08%；text supervised generation 在规范背景上达到 100%。
7. 这说明对象辅助监督后的 latent 不只是能 copy，还能被 edit prompt 改写，并能由完整文本提示直接生成对象图；但 edit/generation 还没有正式大模型长训结果。
8. 全图 pixel MSE 继续不可靠：旧 `memory_tree_edit` pixel MSE 只有 0.007041，但视觉上仍会生成模糊物体或多个物体影子。核心指标必须看 foreground/object/scene exact。
9. 旧版训练看起来 CPU-bound 的主因是评估路径：`nearest_attrs_with_background` 曾经对每个样本循环构造 120 个候选图，而且在同一 batch 内重复解析两遍。现在已改为批量 GPU 模板匹配，并且训练中只用 `train_eval_size` 子集做 checkpoint/log。

## 担忧与下一步

1. 全图 MSE 不能作为主指标；后续必须把 foreground/object/scene exact 作为硬门禁。
2. 纯 memory-tree residual latent 没有自然形成可靠对象表；简单继续加 token 未必解决对象化问题。Stage T/U/X 已经说明，纯 resampler 或 slot 结构容易保背景/平均纹理，却不自然形成对象 token。
3. 辅助监督是当前最强正信号；copy 已有正式结果，edit/generation 已有 probe 结果。下一步应把 edit/generation 正式长训补完，并把对象表升级成显式可修改中间表示。
4. 如果坚持“完整重绘”路线，copy 任务应继续作为硬门禁：源图完整重绘必须先过 foreground scene exact，再测试 edit prompt 修改能力。
5. patch size 可以从 8 降到 4 测一次，但必须和当前 supervised copy 对照；否则容易把更多 patch token 带来的成本误判为架构进步。
6. 这个脚本仍没有 checkpoint 和断点恢复；更长训练前应补中间样例、checkpoint、恢复和失败保护。
