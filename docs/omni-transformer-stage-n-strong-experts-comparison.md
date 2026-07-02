# Stage N：强专家 MoE 与 baseline 成本对比

## 目的

Stage M 证明了 MoE + attention pump + latent thought 可以闭合，但 counting/spatial 专家较弱，专家分工也不够硬。Stage N 继续同一组任务，把弱学习型功能专家换成更强的结构化专家，再和 baseline 比较：

- 准确率
- 训练成本
- 预测成本
- 参数量

本阶段的强专家是 synthetic domain 的 oracle symbolic/perceptual expert 上限：它们直接产生结构化视觉槽、文本/规则槽、空间关系槽、计数槽和图表槽。它们不是现实世界的预训练 VLM/OCR/Layout 专家，因此结论只能说明“如果专家足够强，latent bus/decoder 的上层链路会怎样”。

## 脚本与结果

脚本：

- `experiments/omni_transformer_stage_n_strong_experts.py`

结果：

- `artifacts/omni_transformer_stage_n_strong_experts/sweep_results.json`
- `artifacts/omni_transformer_stage_n_strong_experts/sweep_runs/`
- `artifacts/omni_transformer_stage_n_strong_experts/sweep_runs/*/samples/*/moe_multimodal_grid.png`
- `artifacts/omni_transformer_stage_n_strong_experts/sweep_runs/*/samples/*/samples.json`

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_n_strong_experts.py --sweep --seeds 20260701,20260702,20260703 --train-size 2048 --val-size 512 --test-size 512 --batch-size 128 --d-model 64 --layers 2 --heads 4 --strong-direct-steps 260 --strong-moe-steps 260 --raw-classifier-steps 400 --output-dir artifacts\omni_transformer_stage_n_strong_experts\sweep_runs --aggregate artifacts\omni_transformer_stage_n_strong_experts\sweep_results.json
```

运行设备：

- NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch CUDA

## 对比组

| 方法 | 输入 | 输出方式 | 作用 |
| --- | --- | --- | --- |
| deterministic strong expert renderer | oracle 强专家结构化槽 | 规则模板渲染 | 上限/规则 baseline |
| strong direct decoder | oracle 强专家 tokens | decoder 直接读专家 tokens | 测强专家 + 文本输出专家 |
| strong MoE latent decoder | oracle 强专家 tokens | attention pump -> latent -> decoder | 测强专家 + latent bottleneck |
| raw image+prompt classifier | 原始像素 + 文本 token | 分类 target 后模板渲染 | 小型端到端 baseline |

和 Stage M 的关键区别：

- Stage M 的功能专家需要从共享 vision/text token 里自己学计数、空间、图表。
- Stage N 的强专家已经输出 task-relevant symbolic slots，decoder 主要学习把结构化语义转成文本。

## 正式结果

3 seed 聚合：

| 方法 | exact | semantic | 参数量 | 训练秒数 | 预测 ms/example |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic strong expert renderer | 100.00% | 100.00% | 0 | 0.00 | 0.016 |
| strong direct decoder | 100.00% | 100.00% | 283,563 | 19.43 | 1.497 |
| strong MoE latent decoder | 100.00% | 100.00% | 283,563 | 22.41 | 1.484 |
| raw image+prompt classifier | 80.92% | 80.92% | 144,405 | 26.93 | 0.439 |

自回归 greedy 生成：

| 方法 | greedy exact |
| --- | ---: |
| strong direct decoder | 100.00% |
| strong MoE latent decoder | 100.00% |

Stage M 对照：

| 方法 | exact | 参数量 | 训练秒数 |
| --- | ---: | ---: | ---: |
| Stage M weak direct routed experts | 75.65% | 1,534,149 | 32.26 |
| Stage M weak MoE latent bottleneck | 79.30% | 1,534,149 | 57.96 |
| Stage N strong MoE latent decoder | 100.00% | 283,563 | 22.41 |

注意：Stage M 和 Stage N 的模型宽度/步数不同，不能当作严格同模型 benchmark；这个表只用于说明方向性差异：强专家让上层 decoder 负担明显变小。

## 强专家消融

以下以 strong MoE latent decoder 的 teacher-forced exact 为指标：

| 条件 | exact |
| --- | ---: |
| full strong MoE latent | 100.00% |
| no latent access | 0.00% |
| no vision expert | 49.22% |
| no text/rule expert | 48.11% |
| no spatial expert | 80.08% |
| no counting expert | 79.88% |
| no chart expert | 80.08% |

解读：

- `no latent access = 0.00%`，说明输出仍然必须通过 latent scratchpad。
- 去掉 spatial/counting/chart 各掉约 20 个百分点，刚好对应 5 类任务中一个任务族失效，说明功能专家分工比 Stage M 更硬。
- vision 和 text/rule 是跨任务基础专家，去掉后掉到约 50%，影响更大。

## 结论

成立的部分：

1. 换成强专家后，Stage M 的 counting/spatial 短板消失，所有任务 exact 都达到 100%。
2. strong MoE latent 和 strong direct 都能自回归生成 100%，说明强专家语义可以稳定通过 latent bottleneck。
3. 相比 Stage M，强专家让上层 trainable decoder 更小、训练更快、结果更稳。
4. 消融显示专家分工已经更硬：去掉单个功能专家会让对应任务族失败。

不成立或不能过度解释的部分：

1. 规则 baseline 也是 100%，而且 0 参数、0 训练、预测约 0.016 ms/example。低熵 synthetic 任务上，神经 MoE 没有工程性价比优势。
2. raw classifier 只有 80.92%，但它参数更少、推理更快；它失败主要说明 raw 小模型还没学好计数/空间，不说明 MoE 天然更优。
3. strong MoE latent 与 strong direct 都是 100%，MoE latent 没有在这个任务上超过 direct，只是证明 bottleneck 不损失能力。
4. 强专家是 oracle 结构化上限，不是真实视觉/语言专家。真实场景的核心成本会转移到专家本身：OCR、layout、object counting、spatial reasoning、tool state extraction。

## 对架构判断的影响

Stage N 支持一个更具体的架构判断：

```text
最终多模态 LLM / agent 不应该让主干 transformer 从零学习所有感知和专用推理。
更合理的路线是：
强专家先把高熵输入变成低熵结构化 slots，
router/attention pump 把专家输出压进 latent bus，
thought/output experts 负责融合、计划和自然语言/动作输出。
```

但 Stage N 也给出一个成本警告：

```text
如果任务本身可以被规则专家直接完成，
再接一个 neural latent decoder 只会增加参数和预测成本。
MoE 的价值应来自开放任务组合、跨专家融合、输出统一和 agent 控制，
不是来自低熵规则任务上的单点准确率。
```

## 下一步建议

1. 继续使用强专家路线，但把 oracle symbolic expert 替换成真实专家：OCR/text/layout encoder、视觉对象检测/计数专家、空间关系专家、DOM/UI state extractor。
2. 下一阶段应测试“专家各自不完整，必须融合后才能答”的任务，避免 deterministic baseline 直接 100%。
3. 成本指标应继续保留：参数量、训练秒数、预测 ms/example、专家前处理成本、输出 token 数。
4. 如果目标是 AGI/agent，不应只追求单步准确率；需要加入长程任务、工具调用、记忆树读写、错误恢复和跨专家冲突消解。
