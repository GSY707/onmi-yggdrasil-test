# 更接近多模态架构的连续潜变量管线实验

## 为什么重做

上一轮特殊 `<latent>` token 实验更像“把 visible CoT 压成同一语言模型里的占位 token”。多 seed 后可以看到，这种迁移并不稳定，短任务上甚至有害。

这个结果不能直接否认未来多模态架构，因为白皮书设想的关键不是“离散 token CoT 迁移”，而是：

1. 文本能力先成熟。
2. 文本状态被翻译成潜变量。
3. 潜变量思考与纯文本思考并行对齐。
4. 运行时由潜变量思考，输出时再解码为纯文本。

本实验按这四步重做。

## 实验结构

任务仍使用可自动判分的 8x8 网格路径任务。输入为文本形式的起点和移动序列，目标为最终坐标。

| 阶段 | 验证内容 | 模型部件 |
| --- | --- | --- |
| 1. 纯文本训练 | 文本 prompt -> 可读坐标思考链 + 最终坐标 | `TextThoughtModel` |
| 2. 潜变量翻译 | `X/Y` 文本状态 -> 连续 latent vector -> `X/Y` 文本状态 | `LatentCodec` |
| 3. 潜变量思考 + 纯文本思考并行 | latent recurrent thought 同时对齐 latent codec、逐步文本坐标、最终坐标 | `LatentReasoner` parallel stage |
| 4. 潜变量思考 + 纯文本输出 | 只保留最终文本输出 loss，检查 latent thought 是否仍能支持答案 | `LatentReasoner` output stage |

关键差别：这里的 latent 是连续向量，不是重复的 `<latent>` 离散 token。

## 单次结果

配置：`move_count=10`，`seed=20260701`。

结果文件：`artifacts/multimodal_latent_pipeline/result.json`。

| 阶段 | 指标 | 结果 |
| --- | --- | ---: |
| 纯文本训练 | final exact | 99.61% |
| 纯文本训练 | step exact | 100.00% |
| 潜变量翻译 | reconstruction exact | 100.00% |
| latent + text 并行 | final exact | 92.58% |
| latent + text 并行 | step text exact | 98.21% |
| latent + text 并行 | codec step exact | 75.39% |
| latent thought + text output | final exact | 96.58% |
| latent thought + text output | step text diagnostic | 98.86% |
| latent thought + text output | codec step diagnostic | 67.36% |

单次结论：

- 四个核心步骤都跑通。
- 文本状态到 latent codec 的独立翻译可以达到 100%。
- 并行阶段能同时保留高质量可读文本思考和较强 latent codec 可解码性。
- 最终只训练文本输出后，final exact 仍达到 96.58%。
- codec diagnostic 从 75.39% 降到 67.36%，说明 final-only 阶段会部分偏离翻译锚点。

## 多 seed / 更长任务 sweep

配置：`move_count = 8, 10, 12`，`seed = 20260701, 20260702, 20260703`，共 9 次训练。

结果文件：

- 聚合：`artifacts/multimodal_latent_pipeline/sweep_results.json`
- 明细：`artifacts/multimodal_latent_pipeline/sweep_runs/*.json`

| move 数 | text final | parallel final | parallel step text | parallel codec step | output final | output step text | output codec step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 99.25% | 94.99% | 98.89% | 78.03% | 96.48% | 98.88% | 67.10% |
| 10 | 97.69% | 93.91% | 98.60% | 74.29% | 96.58% | 98.96% | 66.07% |
| 12 | 97.53% | 94.53% | 98.78% | 73.44% | 96.13% | 98.87% | 65.26% |
| overall | 98.16% | 94.48% | 98.75% | 75.25% | 96.40% | 98.90% | 66.14% |

## 判断

这个实验支持更接近白皮书的前几步思路：

1. 纯文本训练作为先导是可行的。
2. 可读文本状态可以被翻译成连续 latent 变量，并可反向解码。
3. 潜变量思考和纯文本思考可以并行训练，最终答案和逐步文本思考都能保持高准确率。
4. 转入“潜变量思考 + 纯文本输出”后，最终文本答案仍稳定，9 次平均 96.40%。

同时要保留限制：

1. 这仍是 toy task，不是真实多模态输入。
2. latent codec diagnostic 只有约 66% 到 75%，说明 latent 不是完全落在翻译器定义的规范空间里。
3. final-only 阶段会让 latent 状态进一步偏离 codec 锚点。
4. 这个实验支持“架构方向值得继续”，但不证明真实图像/视频/动作 latent space 会自然形成。

## 下一步

下一步应该验证真正的“异构外设”：

1. 把输入拆成文本指令 + 非文本数值观测，例如障碍图、局部视野或传感器向量。
2. 让 latent codec 同时翻译文本状态和非文本状态。
3. 增加 URI/记忆树节点引用，测试 latent thought 是否能按需读取高细节状态。
4. 再考虑小型预训练语言模型 LoRA，而不是从零训练小网络。
