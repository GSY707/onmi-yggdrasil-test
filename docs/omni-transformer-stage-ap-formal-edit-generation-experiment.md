# Stage AP：Stage AJ supervised edit/generation 正式长训

## 目的

Stage AJ 的 `memory_tree_supervised_copy` 已在正式规模闭合，但 `memory_tree_supervised_edit` 和 `text_supervised_generate` 此前只停留在 probe。Stage AP 的目标是用 P0 checkpoint/resume 基础设施补完正式长训：

- `memory_tree_supervised_edit`：验证源图 patch tokens -> memory-tree latent -> Transformer patch decoder 能否完整重绘随机背景上的编辑目标。
- `text_supervised_generate`：验证完整目标文本提示 -> memory-tree latent -> Transformer patch decoder 能否在规范背景上生成目标对象。

这仍是低熵合成对象任务，不证明真实照片级生成能力；它只验证对象属性/mask 辅助监督后的 latent 输出专家能否在更大模型、更多样本和更长训练上保持闭合。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_aj_transformer_image_io_fidelity.py --variants memory_tree_supervised_edit,text_supervised_generate --output artifacts\omni_transformer_stage_ap_formal_edit_generate\formal_edit_generate_result.json --checkpoint-dir artifacts\omni_transformer_stage_ap_formal_edit_generate\checkpoints --train-size 4096 --val-size 512 --test-size 512 --batch-size 128 --image-size 64 --patch-size 8 --d-model 192 --encoder-layers 2 --decoder-layers 3 --heads 6 --latent-tokens 24 --latent-levels 6 --copy-steps 0 --edit-steps 1500 --generate-steps 1500 --eval-every 300 --train-eval-size 128 --template-chunk-size 30 --foreground-loss-weight 8.0 --background-loss-weight 2.0 --prefix-loss-weight 0.25 --residual-token-loss-weight 0.02 --object-aux-attr-loss-weight 1.0 --object-aux-mask-loss-weight 0.5 --object-aux-mask-positive-weight 8.0 --torch-num-threads 1 --max-train-seconds 129600 --checkpoint-every 300 --checkpoint-sample-every 300 --checkpoint-sample-count 3 --device cuda
```

运行设备：NVIDIA GeForce RTX 4070 Laptop GPU。总耗时 3039.263 秒，约 50.7 分钟；未触发时间预算或测试注入停止。

## 结果

结果文件：

- `artifacts/omni_transformer_stage_ap_formal_edit_generate/formal_edit_generate_result.json`
- `artifacts/omni_transformer_stage_ap_formal_edit_generate/formal_stdout.log`
- `artifacts/omni_transformer_stage_ap_formal_edit_generate/checkpoints/`
- `artifacts/omni_transformer_stage_ap_formal_edit_generate/samples/formal_edit_generate_result/seed20260701/`

正式 test 指标：

| 模型/消融 | pixel MSE | foreground MSE | scene exact | aux scene | aux mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_tree_supervised_edit` | 0.000836 | 0.005380 | 99.02% | 98.44% | 98.21% |
| `memory_tree_supervised_edit_no_source` | - | - | 4.10% | - | - |
| `source_image_no_edit` | - | - | 0.00% | - | - |
| `text_supervised_generate` | 0.000073 | 0.001011 | 100.00% | 100.00% | 100.00% |

`memory_tree_supervised_edit` 按 edit family 拆分：

| edit family | scene exact |
| --- | ---: |
| color_edit | 97.40% |
| shape_edit | 99.43% |
| position_edit | 100.00% |

`memory_tree_supervised_edit_no_source` 按 edit family 拆分：

| edit family | scene exact |
| --- | ---: |
| color_edit | 4.55% |
| shape_edit | 2.27% |
| position_edit | 5.49% |

训练轨迹：

| step | edit scene | edit foreground MSE | edit aux scene | edit mask IoU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.78% | 0.15916 | 0.00% | 4.90% |
| 300 | 13.28% | 0.14642 | 42.97% | 41.40% |
| 600 | 27.34% | 0.10166 | 53.12% | 54.20% |
| 900 | 96.88% | 0.01193 | 94.53% | 95.00% |
| 1200 | 91.41% | 0.01729 | 95.31% | 96.20% |
| 1500 | 99.22% | 0.00317 | 99.22% | 99.90% |

| step | generate scene | generate foreground MSE | generate aux scene | generate mask IoU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00% | 0.15899 | 0.00% | 4.90% |
| 300 | 100.00% | 0.00746 | 100.00% | 100.00% |
| 600 | 100.00% | 0.00178 | 100.00% | 100.00% |
| 900 | 100.00% | 0.00110 | 100.00% | 100.00% |
| 1200 | 32.03% | 0.06337 | 100.00% | 95.50% |
| 1500 | 100.00% | 0.00132 | 100.00% | 100.00% |

## 结论

1. Stage AP 通过当前 P1 门槛：正式单 seed 下，edit scene exact 99.02%，no-source 只有 4.10%，text generation scene exact 100%。
2. `memory_tree_supervised_edit_no_source` 大幅掉点，说明 edit 成功依赖源图 latent，而不是只靠文字 prompt 猜目标。
3. `source_image_no_edit` 为 0%，说明结果不是简单复制源图。
4. `text_supervised_generate` 在规范背景上 100%，说明文本到 latent 到 patch decoder 的生成链路在这个低熵任务上正式规模闭合。
5. edit/generation 都出现过中途回落，尤其 generation step 1200 从 100% 掉到 32.03%；P0 的 `best.pt` 与训练中样例是必要设施，不应只看最后一个训练 step。

## 边界和担忧

1. 这不是真实图像生成能力证明。任务仍是 64x64 合成单物体、有限颜色/形状/位置和低熵背景。
2. 这是正式单 seed，不是 3 seed 稳定性结论。按计划，长训任务可以先作为正式方向验证；若要做统计结论，应继续跑 3 seeds。
3. 规范背景 generation 不要求从文本恢复随机背景，因为随机背景没有进入文本；这和 edit 任务的源图保真要求不同。
4. 下一步若继续图像路线，应把对象表升级为可修改中间表示，并引入多对象、遮挡、局部 mask、复杂背景或更真实的 detector/segmentation/counting expert。
