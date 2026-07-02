# Stage O：真实预训练专家接入实验

## 目的

Stage N 使用的是 synthetic domain 的 oracle symbolic expert。Stage O 换成真实公开预训练专家，测试：

```text
真实冻结 CLIP image/text experts
-> task adapters / router
-> direct decoder 或 attention pump + latent decoder
-> 文本输出
```

本阶段使用 `openai/clip-vit-base-patch32`：

- `CLIP vision encoder` 作为真实预训练图像专家。
- `CLIP text encoder` 作为真实预训练文本专家。
- spatial/counting/chart adapters 是本地训练的小模块，不是预训练专家。

这不是为了证明 CLIP 是最适合本任务的专家，而是验证真实 pretrained experts 接入 latent bus 后的准确率和成本边界。

## 脚本与结果

脚本：

- `experiments/omni_transformer_stage_o_pretrained_experts.py`

结果：

- `artifacts/omni_transformer_stage_o_pretrained_experts/sweep_results.json`
- `artifacts/omni_transformer_stage_o_pretrained_experts/sweep_runs/`
- `artifacts/omni_transformer_stage_o_pretrained_experts/sweep_runs/*/samples/*/moe_multimodal_grid.png`
- `artifacts/omni_transformer_stage_o_pretrained_experts/sweep_runs/*/samples/*/samples.json`

依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pretrained]"
```

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_o_pretrained_experts.py --sweep --seeds 20260701,20260702,20260703 --train-size 768 --val-size 256 --test-size 256 --batch-size 64 --d-model 96 --layers 2 --heads 4 --classifier-steps 320 --direct-steps 360 --moe-steps 480 --output-dir artifacts\omni_transformer_stage_o_pretrained_experts\sweep_runs --aggregate artifacts\omni_transformer_stage_o_pretrained_experts\sweep_results.json
```

运行设备：

- NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch CUDA
- Frozen CLIP 参数量：151,277,313

## 对比组

| 方法 | 专家 | 训练内容 | 输出 |
| --- | --- | --- | --- |
| CLIP zero-shot | frozen CLIP image/text | 无训练 | candidate answer 打分 |
| CLIP linear classifier | frozen CLIP image/text | 目标分类头 | 模板渲染文本 |
| CLIP direct decoder | frozen CLIP image/text + adapters | direct text decoder | 字符级文本 |
| CLIP MoE latent decoder | frozen CLIP image/text + adapters | router + attention pump + latent decoder | 字符级文本 |

## 正式结果

3 seed 聚合：

| 方法 | exact | trainable params | total params incl. CLIP | 训练秒数 | cached pred ms/ex | with CLIP ms/ex |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP zero-shot | 50.52% | 0 | 151,277,313 | 0.00 | - | - |
| CLIP linear classifier | 68.23% | 207,061 | 151,484,374 | 1.11 | 0.039 | 3.719 |
| CLIP direct decoder | 67.97% | 1,081,840 | 152,359,153 | 6.12 | 1.945 | 5.625 |
| CLIP MoE latent decoder | 68.36% | 1,081,840 | 152,359,153 | 13.72 | 1.900 | 5.580 |

自回归 greedy：

| 方法 | greedy exact |
| --- | ---: |
| CLIP direct decoder | 67.97% |
| CLIP MoE latent decoder | 68.36% |

CLIP MoE latent 按任务：

| 任务 | exact |
| --- | ---: |
| attribute | 100.00% |
| rule | 96.73% |
| counting | 66.67% |
| spatial | 46.41% |
| chart | 31.37% |

## 消融

以下以 CLIP MoE latent decoder 的 teacher-forced exact 为指标：

| 条件 | exact |
| --- | ---: |
| full CLIP MoE latent | 68.36% |
| no latent access | 0.00% |
| no image modality | 44.40% |
| no text modality | 6.12% |
| no function adapters | 35.68% |
| no CLIP vision token | 67.84% |
| no CLIP text token | 60.81% |
| no spatial adapter | 61.20% |
| no counting adapter | 51.43% |
| no chart adapter | 63.80% |

解读：

- `no latent access = 0.00%`，latent bottleneck 没有被绕过。
- `no text modality = 6.12%`，文本问题/规则是主导信号。
- `no image modality = 44.40%`，图像信号有效，但 CLIP 对这个 synthetic 几何任务并不够强。
- 关闭单个 `clip_vision` token 几乎不掉点，是因为 function adapters 仍能读取 image embedding；真正的无图像消融要看 `no image modality`。
- counting、spatial、chart 仍是短板，说明普通 CLIP 不是对象计数/空间关系/图表读取专家。

## 和 Stage M/N 的关系

| 阶段 | 专家类型 | MoE latent exact | 关键成本/边界 |
| --- | --- | ---: | --- |
| Stage M | 从零学习弱专家 | 79.30% | 1,534,149 参数，counting/spatial 弱 |
| Stage N | oracle symbolic strong experts | 100.00% | 规则 baseline 也 100%，神经 MoE 不具性价比优势 |
| Stage O | real frozen CLIP experts | 68.36% | CLIP 参数 151M，真实但不匹配计数/空间/图表任务 |

Stage O 的结果低于 Stage M，不代表预训练专家路线错误。原因是专家能力和任务不匹配：CLIP 是通用图文对齐模型，不是精确几何计数、left-of 判断或柱状图读取模型。

## 结论

成立的部分：

1. 真实预训练 CLIP image/text experts 可以接入本地 latent bus。
2. CLIP zero-shot 已有约 50.52%，训练小头后到约 68%，说明真实 pretrained features 有可用信号。
3. CLIP MoE latent decoder 的输出确实依赖 latent，`no latent access = 0.00%`。
4. cached downstream 推理成本很低，但真实在线预测主要成本来自 CLIP 编码，约 3.68 ms/example。

边界：

1. CLIP MoE latent 没有超过 CLIP classifier/direct decoder，也没有超过 Stage M/N。
2. 总参数量由 frozen CLIP 主导，达到 151M+，远大于前面 toy 模型。
3. 对这个任务，text modality 比 image modality 更关键；这说明模型有利用语言模板捷径的风险。
4. 真实专家必须按任务选择：DocVQA 应使用 OCR/layout/document experts，UI 应使用 DOM/screenshot/layout experts，计数/空间应使用 object detector/spatial relation experts。

## 下一步建议

1. 不要继续在几何 synthetic 任务上硬压 CLIP；它不是合适专家。
2. 下一轮应换任务以匹配真实专家：DocVQA + OCR/layout encoder，或 UI screenshot + DOM extractor。
3. 继续保留成本指标：frozen expert 参数量、adapter 参数量、expert encoding ms/example、cached downstream ms/example、总预测成本。
4. 如果验证最终 AGI/agent 目标，任务应要求多个真实专家互补，例如 screenshot + DOM + OCR + instruction + tool history，且单一专家不能直接完成。
