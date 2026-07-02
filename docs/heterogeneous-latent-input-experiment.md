# 异构输入潜变量实验报告

## 目标

这轮实验验证一个更贴近多模态架构的关键问题：当任务答案必须依赖非文本输入时，`纯文本训练 -> 异构输入潜变量翻译 -> 潜变量思考 + 文本输出` 这条链路是否能跑通。

这里的 `text_only_no_hetero` 和 `latent_bad_codec` 不是“谁更好”的有效 baseline，而是信息可见性检查：前者看不见地形，后者看到的是被坏 codec 破坏后的地形。因此它们的低准确率只能说明任务确实依赖异构输入，不能证明 latent 路线优于其他同样能看见地形的方法。

本实验仍是合成任务，不证明真实图像、视频或动作外设质量；它只验证架构前几步是否闭合。

## 任务设计

输入被拆成两路：

- 文本路：起点坐标和动作序列。
- 异构路：`8x8` 地形 tensor，每个格子有 4 类地形。

地形会改变动作含义：

- `0`：正常执行动作。
- `1`：动作右转。
- `2`：动作左转。
- `3`：动作反向。

如果模型只读文本动作，不读地形，就无法知道真实路径。正确异构潜变量需要先把地形 tensor 翻译成 latent，再从 latent 解码出地形规则，并在潜变量 rollout 中逐步更新位置，最后输出文本坐标。

## 对照组口径

| 组别 | 含义 |
| --- | --- |
| `text_only_no_hetero` | 只使用文本动作，假设全部地形正常；用于确认没有地形信息时任务不可解 |
| `latent_good_codec` | 地形 tensor -> 训练好的 latent codec -> 潜变量 rollout -> 文本坐标输出 |
| `latent_bad_codec` | 地形 tensor -> 随机冻结坏 codec -> rollout；用于确认错误潜变量不会泄漏答案 |
| `ground_truth_hetero_oracle` | 直接使用真实地形的 oracle 上界 |
| `raw_hetero_direct` | 端到端模型直接看原始地形 tensor 和文本动作；这是弱 baseline，只能说明当前小模型没有学出泛化算法 |

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\heterogeneous_latent_input.py --sweep --move-counts 8,12,16 --seeds 20260701,20260702,20260703 --aggregate artifacts\heterogeneous_latent_input\sweep_results.json --output-dir artifacts\heterogeneous_latent_input\sweep_runs
```

运行设备：

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch：`2.11.0+cu128`
- CUDA：`12.8`

## 结果

聚合结果来自 `move=8/12/16` 与 `seed=20260701/20260702/20260703` 的 9 次 sweep。

| move | runs | text-only | good codec | bad codec | oracle | raw direct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3 | 4.18% | 100.00% | 5.05% | 100.00% | 2.59% |
| 12 | 3 | 2.96% | 100.00% | 3.65% | 100.00% | 2.67% |
| 16 | 3 | 3.35% | 100.00% | 3.21% | 100.00% | 1.73% |
| overall | 9 | 3.50% | 100.00% | 3.97% | 100.00% | 2.33% |

辅助差值，非优劣结论：

| 指标 | overall mean |
| --- | ---: |
| good codec - text-only | +96.50% |
| good codec - bad codec | +96.03% |
| good codec - raw direct | +97.67% |

前两项差值只用于确认“地形信息是否被正确接入”，不能用于证明 latent 路线更优。第三项只能说明这个小型 `raw_hetero_direct` baseline 在当前设置下失败，不能代表所有端到端异构输入模型。

## 解释

这个实验支持当前多模态潜空间架构的前几步：

1. 纯文本输入本身不够，`text_only_no_hetero` 只有 3.50%，说明任务确实依赖异构输入。
2. 潜变量翻译一旦正确，`latent_good_codec` 达到 oracle 的 100.00%，说明异构输入可以通过 latent codec 接入后续思考链。
3. 坏 codec 只有 3.97%，说明结果不是 rollout 规则或数据分布自动泄漏出来的，而依赖正确的潜变量翻译。
4. `raw_hetero_direct` 只有 2.33%，说明当前小型端到端 baseline 在这个组合泛化设置下没有学到稳定算法；这不是 latent 路线更优的充分证据。

因此，这轮比前面的文本-only / 连续潜变量实验更接近多模态思路：输入信息确实来自非文本通道，潜变量不是单纯复刻 visible text chain，而是承载文本无法提供的外部状态。它证明的是“可行”，不是“更好”。

## 仍未证明

- 地形 codec 只处理 4 类离散地形 one-hot，不等价于真实图像、视频、音频或机器人传感器外设。
- `latent_good_codec` 的 100% 来自可完全解码的低熵异构输入和确定性 rollout，不代表真实大模型能自动学出同样干净的 latent space。
- `raw_hetero_direct` 是一个小型 baseline，不代表所有端到端多模态模型都会失败。
- text-only 和 bad-codec 的低准确率主要来自信息不可见或信息被破坏，对优劣判断没有直接意义。
- 本实验验证的是架构链路，不验证真实 provider 的 KV cache、长期记忆树读写、复杂工具调用或在线自我对齐。

## 下一步有效对比

要判断这条路线是否“更好”，需要让 baseline 看到同等异构信息，并尽量匹配数据量、参数量和训练预算：

1. 同输入端到端模型：强化 `raw_hetero_direct`，扩大数据和模型容量，确认是否仍然泛化失败。
2. 同地形特征非 latent baseline：直接给模型地形类别嵌入或解析后的规则表，不经过 latent codec。
3. learned transition baseline：让模型学习 `当前坐标 + 当前地形 + 动作 -> 下一坐标`，再 rollout。
4. 泛化测试：训练短 move，测试更长 move；训练部分地形分布，测试新地形组合。
5. 成本测试：比较 token/显存/训练步数/样本效率，而不只看最终准确率。

这些对比已经在 `docs/heterogeneous-baseline-comparison-experiment.md` 中补跑。结果显示：在当前低熵离散地形任务上，`direct_rule_features`、`learned_effective_move`、`learned_transition_table` 都和 `latent_good_codec` 一样达到 100%，但成本更低；因此本合成任务不能支持 latent 路线更优。

## 结论

这轮可行性证据比上一轮更贴近目标架构：当答案必须依赖非文本输入时，正确的潜变量翻译可以把外部状态接入潜变量思考，并稳定输出文本答案。这个结果不能证明完整多模态智能体成立，也不能证明这条路线比同等信息、同等预算的替代方案更好；它只证明“纯文本训练后扩展到异构潜变量输入”的核心机制不是被文本 CoT 迁移假象支撑的。
