# 真实像素输入 Stage D 局部可见与短期记忆实验报告

## 目标

Stage C 证明：当视觉符号和地形语义在每张地图里随机绑定时，必须先探测才能知道当前地图的符号含义。Stage D 进一步验证局部可见和多帧记忆：

- 不再把完整地图作为输入。
- 每一步只看到当前位置的局部视觉块。
- 遇到未知视觉符号时，可以探测一次，观察转移结果并得到该符号的语义。
- 如果有短期 legend 记忆，后续遇到同一符号不需要重复探测。

本轮验证的是控制与记忆机制，不训练新模型。

## Baseline

| 组别 | 含义 |
| --- | --- |
| `local_passive_no_probe` | 每步看当前块，但不探测，错误假设 symbol id 等于 semantic id |
| `local_probe_every_step_no_memory` | 每一步都探测当前位置，但不记住符号语义 |
| `local_legend_memory_budget_0` | 有 legend 结构但探测预算为 0，等价于 no probe |
| `local_legend_memory_budget_1/2/3/4` | 遇到未知符号时探测并写入短期 legend，预算分别为 1/2/3/4 |

样例：

- `artifacts/visual_multimodal_stage_d/sweep_runs/samples/seed20260701/local_frame_strip.png`
- `artifacts/visual_multimodal_stage_d/sweep_runs/samples/seed20260701/debug_hidden_full_map.png`

`debug_hidden_full_map.png` 只是调试图，不作为模型输入；实际输入是局部帧条带里的当前块序列。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\visual_multimodal_stage_d.py --sweep --eval-move-counts 8,16,32 --test-size 4096 --seeds 20260701,20260702,20260703 --aggregate artifacts\visual_multimodal_stage_d\sweep_results.json --output-dir artifacts\visual_multimodal_stage_d\sweep_runs
```

## 结果

overall：

| 策略 | final exact | step exact | 语义决策正确率 | 平均探测 | 平均局部帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `local_passive_no_probe` | 7.82% | 11.39% | 25.04% | 0.00 | 18.67 |
| `local_probe_every_step_no_memory` | 100.00% | 100.00% | 100.00% | 18.67 | 18.67 |
| `local_legend_memory_budget_0` | 7.82% | 11.39% | 25.04% | 0.00 | 18.67 |
| `local_legend_memory_budget_1` | 10.99% | 24.47% | 50.39% | 1.00 | 18.67 |
| `local_legend_memory_budget_2` | 22.20% | 45.83% | 72.75% | 2.00 | 18.67 |
| `local_legend_memory_budget_3` | 51.76% | 73.39% | 90.16% | 2.96 | 18.67 |
| `local_legend_memory_budget_4` | 100.00% | 100.00% | 100.00% | 3.67 | 18.67 |

按路径长度看探测成本：

| move | passive final | 每步探测 final | 每步探测次数 | legend memory final | legend memory 探测次数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8.54% | 100.00% | 8.00 | 100.00% | 3.27 |
| 16 | 7.46% | 100.00% | 16.00 | 100.00% | 3.78 |
| 32 | 7.46% | 100.00% | 32.00 | 100.00% | 3.97 |

## 解释

1. 局部可见本身不是硬阻塞。只要每步能看到当前位置的视觉块，智能体可以在线执行路径。
2. 没有探测时，语义决策正确率约 25%，就是 4 类随机绑定下的猜测水平。
3. 每步探测但不记忆可以达到 100%，但探测成本等于路径长度，`move=32` 需要 32 次探测。
4. 短期 legend 记忆同样达到 100%，但探测成本随符号种类数封顶。`move=32` 平均只需 3.97 次探测。
5. 探测预算不足时，准确率随已知符号数平滑上升：预算 1/2/3 的 final exact 为 10.99%/22.20%/51.76%。

## 结论

Stage D 证明了“尝试 + 短期记忆”不是附属功能，而是随机视觉语义环境里的核心能力。相较于每步重复探测，legend 记忆把探测成本从 `O(路径长度)` 降到 `O(符号种类数)`。

这更接近世界树计划里的记忆树/工作树思想：环境反馈不应该只作为一次性上下文，而应该写入可复用的临时结构。当前实验里的 legend 是极简短期记忆；下一步可以把它扩展为局部地图记忆、错误探测修正、多帧遮挡和带成本的探索策略。
