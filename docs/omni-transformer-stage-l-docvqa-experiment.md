# Stage L：DocVQA 真实文档问答实验

## 目的

Stage L 直接使用 DocVQA 数据，把实验从 synthetic DSL 推到真实文档图像与自然语言问题。

本阶段没有尝试从零训练完整 OCR + 生成式答案模型。任务被收敛为更可控的第一步：

```text
真实 DocVQA 文档图像
+ 真实自然语言问题
+ OCR 行候选
-> latent scratchpad
-> 选择包含答案的 OCR 行
```

这仍然直接使用 DocVQA 的真实图像、真实问题、真实答案和真实 OCR 结果，但把开放式答案生成改成 8 个候选 OCR 行的分类任务。

## 数据源

使用 Hugging Face 镜像：

- `pixparse/docvqa-single-page-questions`
- 数据集说明：DocVQA 约 50,000 个问题，覆盖 12,000+ 文档图像
- 官方项目：`https://www.docvqa.org/`

官方 DocVQA 下载入口需要 RRC 登录；本实验使用已转成 parquet 的 Hugging Face 镜像。第一次运行会下载 parquet 分片，之后使用本机 Hugging Face cache。

## 实验脚本与结果

脚本：

- `experiments/omni_transformer_stage_l_docvqa.py`

结果：

- `artifacts/omni_transformer_stage_l_docvqa/sweep_results.json`
- `artifacts/omni_transformer_stage_l_docvqa/sweep_runs/`
- `artifacts/omni_transformer_stage_l_docvqa/sweep_runs/*/samples/*/docvqa_thumbnail_grid.png`
- `artifacts/omni_transformer_stage_l_docvqa/sweep_runs/*/samples/*/sample_candidates.json`

依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[docvqa]"
```

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_l_docvqa.py --sweep --seeds 20260701,20260702,20260703 --train-size 2048 --val-size 512 --test-size 512 --scan-limit 30000 --batch-size 128 --d-model 128 --layers 3 --heads 4 --direct-steps 700 --bottleneck-steps 900 --probe-steps 120
```

## 任务构造

对每个 DocVQA 样本：

1. 读取真实文档图像。
2. 读取真实问题和答案。
3. 从 `ocr_results.lines` 中寻找包含 gold answer 的 OCR 行。
4. 从同一页 OCR 行里采样 7 个 distractor。
5. 打乱得到 8 个 OCR 行候选。
6. 训练模型选择包含答案的候选行。

输入 token stream：

```text
image patches
question tokens
OCR candidate line tokens + bbox/order features
latent scratchpad tokens
answer query token
```

H2 latent bottleneck 约束继续保留：

- latent tokens 可以 attend 图像、问题和 OCR 候选。
- answer query 不能直接 attend 原始输入，只能通过 latent scratchpad 决策。

同时训练 direct baseline：

- direct answer query 可以直接 attend 原始输入。

## 正式结果

3 seed 聚合，指标是 `candidate_line_accuracy`：

| 方法 | accuracy |
| --- | ---: |
| random candidate | 12.50% |
| lexical overlap baseline | 19.21% |
| direct DocVQA transformer | 21.48% |
| latent bottleneck DocVQA transformer | 14.65% |

消融结果：

| 消融 | accuracy |
| --- | ---: |
| full latent bottleneck | 14.65% |
| no image | 11.91% |
| no question | 14.13% |
| no OCR candidates | 12.24% |
| no latent access | 12.57% |

latent probe：

| probe target | accuracy |
| --- | ---: |
| candidate label | 15.95% |
| derived question type | 61.07% |

## 结论

这是一个负结果，但很有价值：

1. 真实 DocVQA 与之前 synthetic 任务的差距很大。
2. tiny direct transformer 只略高于 lexical overlap baseline，说明它捕捉到一点 DocVQA 信号，但非常弱。
3. latent bottleneck 从零训练明显不足，只有 14.65%，接近 random 12.50%，低于 lexical overlap 19.21%。
4. `no_latent_access` 接近 random，说明 bottleneck 约束没有被绕过；问题在于 latent 内部没有学到足够的文档问答表征。
5. latent probe 能较好读出派生 question type，但读不出正确候选行，说明当前 latent 更像任务类型/格式状态，不是答案证据状态。

## 对架构判断的影响

Stage L 不推翻前面 synthetic agent 结论，但它明确给出一个真实数据边界：

- 当前 tiny omni Transformer + 从零 hash token embedding + 低分辨率缩略图，不足以解决真实 DocVQA。
- 真实文档问答需要预训练视觉编码器、OCR/text encoder、layout-aware document encoder，或者直接使用现有 VLM/LLM 作为专家模块。
- latent bottleneck 仍可以作为聚合层测试，但不应该承担“从零学 OCR 和文档语义”的全部压力。

换句话说，下一步不该继续加深这个 tiny 模型，而应该把架构改成：

```text
DocVQA image
-> OCR / layout / visual expert
-> expert embeddings or text spans
-> latent bus
-> LLM-like decoder / answer agent
```

## 未证明边界

- 没有训练开放式自然语言答案生成，只做候选 OCR 行选择。
- 使用了数据集提供的 OCR 结果，没有测试 OCR 工具本身。
- 图像被压到 48x48，只作为真实视觉输入占位，不能代表高分辨率文档视觉理解。
- 模型没有使用预训练语言模型、LayoutLM、Donut、Pix2Struct、Qwen-VL 等文档/视觉预训练能力。
- 该结果不说明最终架构不可行，只说明“tiny 从零模型直接上 DocVQA”不可行。
