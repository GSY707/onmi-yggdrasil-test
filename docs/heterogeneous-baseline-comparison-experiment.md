# 异构输入同等信息 baseline 对比报告

## 目标

上一轮异构输入实验证明了 `异构输入 -> latent codec -> 潜变量思考 -> 文本输出` 这条链路可行，但没有证明它更好。本轮专门补同等信息 baseline：baseline 必须能看到同一个地形 tensor，不能再用看不见地形的 text-only 或坏 codec 作为优劣对照。

## 实验设置

- 训练 move 长度：8
- 测试 move 长度：8、16、32
- seeds：`20260701`、`20260702`、`20260703`
- 每个 seed：训练集 4096，验证集 1024，测试集 2048
- 设备：NVIDIA GeForce RTX 4070 Laptop GPU
- 结果文件：`artifacts/heterogeneous_baseline_comparison/sweep_results.json`

## Baseline

| 组别 | 输入信息 | 结构假设 |
| --- | --- | --- |
| `latent_good_codec` | 地形 tensor | 先经 latent codec，再用确定性 rollout |
| `direct_rule_features` | 地形 tensor | 不经 latent，直接用地形类别和确定性 rollout |
| `learned_effective_move` | 地形类别 + 动作 | 学 `terrain_type + move -> effective_move`，再 rollout |
| `learned_transition_table` | 当前坐标 + 当前地形 + 动作 | 学 `coord + terrain_type + move -> next_coord`，再 rollout |
| `raw_step_direct` | 完整地形 tensor + 动作序列 | 端到端 GRU，逐步监督每个中间坐标 |

## 准确率

final exact：

| 测试 move | latent_good_codec | direct_rule_features | learned_effective_move | learned_transition_table | raw_step_direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 100.00% | 100.00% | 100.00% | 4.05% |
| 16 | 100.00% | 100.00% | 100.00% | 100.00% | 2.86% |
| 32 | 100.00% | 100.00% | 100.00% | 100.00% | 3.42% |
| overall | 100.00% | 100.00% | 100.00% | 100.00% | 3.45% |

step exact：

| 测试 move | latent_good_codec | direct_rule_features | learned_effective_move | learned_transition_table | raw_step_direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 100.00% | 100.00% | 100.00% | 10.36% |
| 16 | 100.00% | 100.00% | 100.00% | 100.00% | 6.90% |
| 32 | 100.00% | 100.00% | 100.00% | 100.00% | 5.09% |

## 成本

| 组别 | 平均训练秒数 | 训练步数 | 参数量 |
| --- | ---: | ---: | ---: |
| `latent_good_codec` | 1.66 | 300 | 3,652 |
| `direct_rule_features` | 0.00 | 0 | 0 |
| `learned_effective_move` | 0.09 | 120 | 64 |
| `learned_transition_table` | 0.17 | 180 | 65,536 |
| `raw_step_direct` | 9.42 | 2,000 | 3,249,728 |

## 解释

本轮结果支持你的预估，但要精确表述：

1. 在准确率上，结构化 baseline 没法“高于” latent codec，因为 latent codec 已经是 100%。它们是同分。
2. 在成本上，baseline 明显更好。`direct_rule_features` 不需要训练，`learned_effective_move` 只用 64 个参数和约 0.09 秒训练就达到同样 100%。
3. `learned_effective_move` 和 `learned_transition_table` 在训练 move=8 后，测试 move=16/32 仍是 100%，说明这个任务的关键不是长序列记忆，而是学到局部转移规则。
4. `raw_step_direct` 看到完整地形，并且逐步监督，但 final 只有 3.45%。这说明普通端到端序列模型在这个随机地形组合任务上不会自然学出可泛化算法；它不能证明 latent 路线更好，只证明无结构黑箱 baseline 很弱。

## 结论

这轮 baseline 否定了“latent codec 在该合成任务上更优”的说法。更准确的结论是：

- latent 路线可行。
- 在低熵、可解析、规则清晰的异构输入上，非 latent 结构化 baseline 能以同等准确率、更低成本完成任务。
- 这里真正有价值的不是 latent 本身，而是“把异构输入翻译成可稳定参与推理的中间表示”。这个中间表示在简单任务上可以是直接规则特征；只有当输入变成图像、视频、音频、传感器流等高熵信号时，latent codec 才可能体现必要性。
