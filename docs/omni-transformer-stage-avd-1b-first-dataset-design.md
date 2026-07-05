# Stage AV-D：1B-first 数据集设计

## 目标

Stage AV-D 重新设计 AV 系列数据路线。原则是：

1. 先设计 1B token 级别的母分布。
2. 再从母分布中定向裁剪到 10M token probe。
3. 10M 不是随机缩小版，而是保留关键结构、降低部分熵的可训练子集。
4. 架构理念不变：文本先建潜空间，各模态训练到潜空间互译，先证明外部表征与 latent 双向可达，再训练潜空间推理，最后短程整体调试。

这替代 AV-C 的临时合成流水线。AV-C 只保留为“shard/manifest 输出链路 smoke”，不再作为正式数据设计。

## 为什么要 1B-first

从小数据集直接放大有几个问题：

- 小数据的任务熵和组合空间通常是后补的，容易形成模板捷径。
- 10M token 如果只是小 toy 重复更多次，不能说明 70M 从零模型真的学到架构能力。
- 先做 10M 再想 1B，会导致字段、任务、heldout 和评估口径不断变。
- 训练失败时无法判断是数据太简单、数据太小、架构弱，还是训练阶段错。

1B-first 的目的不是立刻生成 1B token，而是先锁定目标分布和信息瓶颈，再派生小规模数据。

## 1B 母分布

### 样本单位

每条样本是一个 compact episode，而不是单轮分类题。

样本包含：

- `text`: 指令、目标、规则、约束、输出格式。
- `image`: 小图或图像 token，含对象、位置、遮挡、状态标记。
- `tool`: 文件/表格/DOM/API/tool observation token。
- `memory`: 局部地图、历史事实、错误记忆、已验证事实。
- `state`: phase、预算、工具成本、已读/未读 mask。
- `answer`: token sequence 或 action sequence，不只分类 id。
- `target_external`: 经过操作后的目标外部表征，用于训练 latent -> external。
- `trace`: 可选 latent supervision / process label，只训练时可见。

### 任务族

1B 母分布至少包含 6 个任务族：

| 任务族 | 作用 | 关键消融 |
| --- | --- | --- |
| `text_latent_base` | 建文本潜空间、答案 token writer、格式约束 | no-text、shuffled-rule |
| `visual_grounding` | 图像/对象/位置/颜色/关系翻译到潜空间 | no-image、shuffled-image、hard-negative-image |
| `tool_evidence` | HTML/CSV/PDF/DOM/API 证据翻译到潜空间 | no-tool-history、shuffled-tool |
| `memory_navigation` | memory tree/work tree、局部地图、错误记忆修正 | no-memory、corrupt-memory |
| `cross_modal_reasoning` | image + text + tool + memory 组合推理 | 单模态消融、跨模态 hard negative |
| `bidirectional_translation` | source external ↔ latent ↔ target external，验证互译和 latent edit | source/target recon、source->target、target->source |
| `latent_planning` | 多步 action、预算、主动读取、最终回答 | no-history、wrong-cost、budget-shuffle |

### 熵来源

1B 母分布必须有可控熵来源：

- 对象数量：1-12。
- 网格大小：4x4 到 16x16。
- 属性：颜色、形状、尺寸、材质、状态、遮挡。
- 文本表达：模板、同义改写、字段顺序、否定、条件规则。
- 工具 schema：列名、DOM 属性、PDF key、API field。
- memory 状态：unknown、verified、stale、corrupt、conflicting。
- 任务长度：单步、短链、多步、预算约束。
- 输出形式：分类、短文本 token、action sequence、cited report。

### Hard Negatives

每个任务族必须有 hard negatives：

- 同文本不同图像事实。
- 同图像不同文本规则。
- 同工具 schema 不同证据值。
- 同 memory key 但 verified/stale 状态不同。
- 相同最终答案但不同证据链。
- 相同证据链但不同输出格式。

### Split 设计

不能只做 random split。

需要至少 5 种 split：

- `train`
- `val_seen`
- `test_seen`
- `test_composition_heldout`
- `test_schema_heldout`
- `test_hard_negative`

heldout 维度：

- 未见属性组合。
- 未见文本模板。
- 未见工具 schema。
- 未见图像布局。
- 未见 memory corruption pattern。
- 更长 action horizon。

## 1B Token 配比

目标配比：

| 数据块 | token 占比 | 目的 |
| --- | ---: | --- |
| text latent base | 15% | 建统一文本/答案潜空间 |
| modality translation | 30% | image/tool/memory 分别翻译到 latent |
| single-family reasoning | 20% | 单任务族推理，不混太多模态 |
| bidirectional translation | 15% | 外部表征和 latent 的双向互译、target external 解码 |
| cross-modal reasoning | 15% | 真正组合 image/text/tool/memory |
| planning/agent traces | 10% | 多步工具、预算、memory 写入 |
| hard negatives/eval-like | 5% | 抗捷径 |

1B token 不是平均铺满所有任务；要按阶段训练需要保留可抽样的子集标签。

## 10M 定向裁剪

10M token 不是随机采样，而是从 1B 母分布裁剪。

裁剪目标：

1. 保留所有任务族。
2. 保留所有关键消融通道。
3. 保留 hard negative 的最小集合。
4. 降低文本改写数量、schema 变化数量和长 horizon 比例。
5. 降低对象数量上限，但不删除对象/位置/关系。

### 10M 降熵规则

| 熵维度 | 1B 母分布 | 10M 裁剪 |
| --- | --- | --- |
| grid size | 4x4 到 16x16 | 4x4、8x8 |
| object count | 1-12 | 1-6 |
| text templates | 多模板 + 改写 | 每任务 4-8 个模板 |
| tool schema | 多 schema、多字段名 | 每任务 2-3 个 schema |
| memory corruption | 多类型、多步修正 | stale/corrupt/conflict 三类 |
| action horizon | 1-32 | 1-12 |
| output | 分类/短文本/action/report | 短文本/action/report 为主，少量分类 |
| hard negatives | 高覆盖 | 每任务至少 2 类 |

### 10M 配比

| 数据块 | token 占比 | 估算 tokens |
| --- | ---: | ---: |
| text latent base | 20% | 2M |
| modality translation | 35% | 3.5M |
| bidirectional translation | 20% | 2M |
| single-family reasoning | 15% | 1.5M |
| cross-modal reasoning | 10% | 1M |
| planning/agent traces | 7% | 0.7M |
| hard negatives/eval-like | 3% | 0.3M |

这个配比刻意降低整体任务熵，让 70M 模型先学到翻译和推理分工，而不是被全难度 1B 母分布打散。

## 训练阶段映射

AV-B 的训练阶段保留，但数据来源改成 AV-D/AV-C shard。

| 阶段 | 数据块 | 训练目标 |
| --- | --- | --- |
| `text_base` | text latent base | 文本/答案潜空间、格式、短答案 writer |
| `image_translate` | modality translation:image | 图像对象/位置/属性到 latent，并能从 latent 解回外部 token |
| `tool_memory_translate` | modality translation:tool/memory | 工具证据、memory 状态到 latent，并能从 latent 解回外部 token |
| `bidirectional_translate` | bidirectional translation | source external -> latent -> target external；target external -> latent -> source external |
| `latent_reason` | single-family + cross-modal | 潜空间读取、组合、比较、选择 |
| `joint` | cross-modal + planning + hard negatives | 接口对齐和抗捷径 |

前面 translator 阶段不追求最终任务满分，只要求 translator probe 过门槛，并保留后续训练空间。

## 必须报告的指标

训练报告必须同时包含：

- unique tokens。
- processed tokens。
- tokens/sec。
- GPU power 区间。
- 每阶段 loss 和 probe。
- 每任务族 exact。
- source reconstruction、target reconstruction、source->target、target->source。
- 每个关键消融。
- heldout split。
- hard negative split。
- image/tool/memory translator 是否遗忘。

如果只报告 train/val accuracy，不足以判断架构。

## 下一步实现

2026-07-05 已完成第一版落地：

1. AV-C 当前任务分布已降级为历史 shard/manifest smoke，不作为正式训练分布。
2. 新增 AV-F generator：`experiments/omni_transformer_stage_avf_10m_bidirectional_dataset.py`。
3. AV-F 支持 `--scale smoke|10m|100m|1b`，当前已生成 10M 级 manifest。
4. AV-E 训练脚本已接 `--dataset-manifest`，从 shard 读取 source/target external pairs。
5. AV-E 训练日志已记录 unique train pair tokens 与 processed pair tokens。
6. 新增 AV-G 严格分阶段训练入口，按 `text_latent_base -> external_codec -> bidirectional_translate -> latent_reason -> joint_debug` 训练同一个 AV-F 10M manifest。

剩余下一步：

1. AV-E joint baseline 和 AV-G strict staged 已跑完，均未通过；保留为 10M 负结果。
2. 下一步先做 external codec decoder 修复，让 source/target reconstruction sequence exact 先闭合。
3. codec 通过后再回到 bidirectional translation、answer head 和 100M 扩展。
4. 把 compact token external 逐步替换/扩展到真实像素、文件、action 输出专家。

## 当前判断

目前不是“模型不够大”，也不只是“有没有分阶段”。AV-F 10M 已经排除了最小数据规模问题，但 AV-E/AV-G 暴露出 external decoder/slot binding 不足：低熵字段能学到接近 100%，image zone color 和 variable text positions 仍只有约 30%-34%。70M 模型可以继续作为本机主验证模型，但下一步要先修 codec/decoder，再谈扩大到 100M/1B。
