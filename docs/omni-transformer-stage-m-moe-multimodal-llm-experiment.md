# Stage M：Tiny MoE 多模态 LLM 实验

## 目的

Stage M 修正前一阶段 tiny omni Transformer 的架构偏差：不再把模型视为单一 decoder，而是按白皮书里的“多专家聚合方案”构造一个小型 MoE 多模态 LLM。

本阶段只验证核心链路：

```text
图像输入 -> VisionExpert
文本问题/规则 -> TextRuleExpert
功能专家 -> Spatial / Counting / Chart experts
Router -> 多专家路由权重
Attention Pump -> 定长 latent scratchpad
ThoughtExpert -> 内部 latent 思考
TextOutputExpert -> 自然语言字符级输出
```

这不是大模型训练，也没有使用预训练视觉/语言模型。它验证的是：功能区分的专家输出能否经注意力泵聚合到 latent，再由输出专家生成文本答案。

## 脚本与结果

脚本：

- `experiments/omni_transformer_stage_m_moe_multimodal_llm.py`

结果：

- `artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_results.json`
- `artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_runs/`
- `artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_runs/*/samples/*/moe_multimodal_grid.png`
- `artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_runs/*/samples/*/samples.json`

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_m_moe_multimodal_llm.py --sweep --seeds 20260701,20260702,20260703 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 96 --layers 2 --heads 4 --direct-steps 450 --moe-steps 700 --probe-steps 100 --output-dir artifacts\omni_transformer_stage_m_moe_multimodal_llm\sweep_runs --aggregate artifacts\omni_transformer_stage_m_moe_multimodal_llm\sweep_results.json
```

运行设备：

- NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch CUDA

## 任务构造

5 类合成多模态任务：

| 任务 | 输入 | 输出 |
| --- | --- | --- |
| attribute | 图像中目标形状 + 文本问题 | `the square is red .` |
| counting | 多个彩色形状 + 文本计数问题 | `there are two green squares .` |
| spatial | 两个目标物体 + left-of 文本问题 | `yes .` / `no .` |
| rule | 视觉规则卡片 + 文本规则 | `the circle should inspect .` |
| chart | 柱状图图像 + 固定问题 | `bar c is tallest .` |

每个样本都有多热专家标签，用于监督 router：

- `vision`
- `text_rule`
- `spatial`
- `counting`
- `chart`

对比模型：

| 模型 | 约束 |
| --- | --- |
| direct routed experts | 输出专家直接读取加权专家 tokens |
| moe latent bottleneck | 输出专家只能读取 Attention Pump + ThoughtExpert 后的 latent tokens |

评估指标：

- `answer_exact`：字符级自然语言答案完全匹配。
- `answer_semantic_accuracy`：从输出文本解析颜色、数量、yes/no、动作或最高柱后比对目标类别。
- `token_accuracy`：非 PAD 字符 token 准确率。
- `router_exact`：5 个专家开关完全匹配。
- `generation_metrics`：自回归 greedy 生成，不使用 gold 前缀。

## 正式结果

3 seed 聚合：

| 模型 | teacher-forced exact | semantic | token | router |
| --- | ---: | ---: | ---: | ---: |
| direct routed experts | 75.65% | 76.11% | 98.78% | 100.00% |
| MoE latent bottleneck | 79.30% | 80.01% | 98.96% | 100.00% |

自回归 greedy 生成：

| 模型 | greedy exact | semantic | token |
| --- | ---: | ---: | ---: |
| direct routed experts | 75.65% | 75.98% | 90.31% |
| MoE latent bottleneck | 79.30% | 80.01% | 90.87% |

MoE latent 按任务的 exact：

| 任务 | exact |
| --- | ---: |
| attribute | 100.00% |
| rule | 98.37% |
| chart | 86.93% |
| spatial | 63.40% |
| counting | 47.90% |

latent probe：

| probe target | accuracy |
| --- | ---: |
| task type | 100.00% |
| answer target class | 79.69% |
| route exact | 99.54% |

## 消融结果

以下均以 MoE latent bottleneck 的 teacher-forced exact 为指标：

| 条件 | exact |
| --- | ---: |
| full latent bottleneck | 79.30% |
| no latent access | 0.00% |
| no prompt | 21.16% |
| shuffled image | 34.83% |
| no image | 40.69% |
| wrong route | 68.88% |
| no text_rule expert | 34.90% |
| no vision expert | 54.62% |
| no spatial expert | 72.66% |
| no counting expert | 74.67% |
| no chart expert | 76.04% |

## 结论

Stage M 是正结果，但不是“MoE 已经更优”的强结论。

成立的部分：

1. 功能区分专家、router、attention pump、latent thought、文本输出专家可以在本机 GPU 上端到端训练闭合。
2. `no latent access = 0.00%`，说明答案输出确实依赖 latent scratchpad，没有绕过 bottleneck。
3. `no prompt`、`shuffled image`、`no image` 都显著掉点，说明模型不是单模态捷径。
4. router 能稳定学到专家组合，3 seed `router_exact = 100.00%`。
5. latent 中可 probe 出任务类型、目标类别和路由，说明 latent 不是空载体。
6. 自回归 greedy exact 与 teacher-forced exact 一致，说明不是只靠 gold 前缀才能输出。

边界也很清楚：

1. MoE latent 只比 direct routed experts 高约 3.65 个百分点，且任务是合成低熵任务，不能宣称架构性能更优。
2. counting 只有 47.90%，spatial 只有 63.40%，说明当前功能专家没有把计数和空间推理学成强能力。
3. `wrong route` 仍有 68.88%，单独移除 `spatial/counting/chart` 专家掉点不大，说明功能专家分工还不够硬，模型仍能从共享 vision/text token 旁路恢复部分答案。
4. 语义准确率只略高于 exact，说明 counting 错误主要不是拼写问题，而是真正的数量判断错误。

## 对架构判断的影响

这次实验修正了“忘记 MoE”的问题：Stage M 确认注意力泵聚合多专家输出是可训练的，latent thought + 文本输出也能闭合。

但它同时说明，最终架构不能只写“有多个专家”就算完成。专家必须有更强的功能边界和训练目标，例如：

- 空间专家需要更强位置归纳偏置或显式空间监督。
- 计数专家需要对象级表示或可微计数辅助目标。
- router 消融应让错误功能路由造成更大损失，否则路由只是解释性标签，不是强控制面。
- 真实任务阶段应接入预训练视觉/OCR/text/layout 专家，而不是让 tiny 从零模型同时学习感知、计数、空间和语言输出。

## 未证明边界

- 没有证明 MoE latent 优于 direct baseline；当前差距太小。
- 没有证明开放式自然语言能力；输出仍是短模板字符序列。
- 没有证明真实图像、真实网页、真实文档或真实工具任务。
- 没有使用预训练专家，因此不能外推到实际多模态 LLM 质量。
- 没有验证长期 agent 记忆树、真实工具调用和错误恢复。
