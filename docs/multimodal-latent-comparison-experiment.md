# 连续潜变量管线的 6 组对照实验

## 目的

上一轮连续潜变量管线证明了四步链路能跑通：

```text
纯文本训练 -> 潜变量翻译 -> 潜变量/文本并行思考 -> 潜变量思考 + 文本输出
```

但要判断“这个方法是否值得采用”，必须和 baseline 对比。本轮做 6 类对照/消融，检验收益来自哪里。

## 对照组

| 组 | 问题 |
| --- | --- |
| `direct_text_only` | 不生成思考链、不用 latent，直接输出最终答案能否解决？ |
| `visible_text_cot` | 显式文本思考链上限是多少？ |
| `latent_from_scratch_good_codec` | 不经过纯文本先导，直接训练 latent + good codec 能否做到？ |
| `latent_without_codec_alignment` | 去掉 codec/latent 对齐，只保留 step/final 文本监督会怎样？ |
| `latent_with_bad_codec` | 使用随机 frozen bad codec，最终答案是否仍能学会？ |
| `wide_direct_equal_budget` | 更大 direct baseline、同等训练预算能否追上？ |

完整 pipeline 作为对照中的目标方法：

```text
pure text + good codec + parallel latent/text thought + final text output
```

## 实验设置

- `move_count = 10, 12`
- `seed = 20260701, 20260702, 20260703`
- 共 6 次完整 comparison run
- 结果文件：`artifacts/multimodal_latent_comparison/sweep_results.json`
- 明细文件：`artifacts/multimodal_latent_comparison/sweep_runs/*.json`

## 结果

| 组 | overall final exact |
| --- | ---: |
| `direct_text_only` | 8.92% |
| `visible_text_cot` | 99.58% |
| `full_pipeline` | 97.07% |
| `latent_from_scratch_good_codec` | 93.07% |
| `latent_without_codec_alignment` | 100.00% |
| `latent_with_bad_codec` | 98.52% |
| `wide_direct_equal_budget` | 10.38% |

关键差值：

| 对比 | 差值 |
| --- | ---: |
| full pipeline - direct | +88.15 pp |
| full pipeline - visible CoT | -2.51 pp |
| full pipeline - latent scratch good codec | +4.00 pp |
| full pipeline - no codec alignment | -2.93 pp |
| good codec pipeline - bad codec | -1.45 pp |
| full pipeline - wide direct | +86.69 pp |

full pipeline 的 `codec_step_exact` 平均为 61.74%。也就是说，中间 latent 有一部分仍能被 good codec 直接解码，但并不完全落在 codec 规范空间里。

## 结论

本轮对照支持这些结论：

1. **显式或隐式逐步监督是关键。**  
   `direct_text_only` 和 `wide_direct_equal_budget` 都很弱，说明这个任务需要某种逐步状态建模；单纯扩大 direct baseline 没解决组合泛化。

2. **完整 pipeline 有效，但不是最强 final baseline。**  
   `full_pipeline` 平均 97.07%，接近 `visible_text_cot` 的 99.58%，也高于 `latent_from_scratch_good_codec` 的 93.07%。

3. **纯文本先导对 good-codec pipeline 有帮助。**  
   full pipeline 比 latent scratch good codec 高 4.00 pp。

4. **当前 codec alignment 没有带来最终答案收益。**  
   `latent_without_codec_alignment` 达到 100%，`latent_with_bad_codec` 也达到 98.52%。这说明模型可以通过 hidden state 和文本监督完成任务，不需要 good codec 才能拿高 final accuracy。

5. **good codec 的价值目前更像“可解释/可翻译约束”，不是性能提升。**  
   full pipeline 保留了约 61.74% 的 codec 可解码中间状态，但为了这个约束牺牲了一些 final accuracy。

## 对架构思路的影响

这不否认多模态潜变量架构，但会修正路线判断：

- 可继续：文本先导、潜变量翻译、潜变量/文本并行训练、潜变量到文本输出这条工程链路可行。
- 要警惕：如果只看最终答案准确率，模型可能绕过 codec，使用普通 hidden state 解题。
- 必须补强：未来实验需要让 codec 承载文本模型拿不到的信息，例如图像、局部视野、传感器状态或 URI 指向的外部细节。否则 good codec 只是正则化/解释约束，不是必要推理通道。

## 下一步

下一步不要继续在纯文本网格上堆训练，而应加入真正的异构输入：

1. 文本指令只描述目标。
2. 地图、障碍物、局部视野用非文本 tensor 输入。
3. codec 负责把非文本状态翻译到 latent。
4. direct text-only baseline 不允许看到完整非文本信息，只能通过 latent 或主动采样读取。

只有这样才能判断多模态架构里的“外设 -> 潜变量 -> 大脑推理 -> 文本输出”是否真的有不可替代价值。
