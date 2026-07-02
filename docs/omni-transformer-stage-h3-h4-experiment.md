# Stage H3/H4：Latent 容量曲线与 OOD 边界实验

## 目的

H1/H2 已经证明：统一 token stream 的 tiny omni Transformer 可以把图像 patch、文本 token、遥测 token 写入 latent scratchpad，并从 latent bottleneck 生成 DSL 文本答案。

H3/H4 继续验证两个问题：

1. H3：latent scratchpad 至少需要多少个特殊 latent token。
2. H4：在更接近 LLM 使用方式的 query/counterfactual 设定下，模型能否工作；遇到未见视觉风格和 held-out 组合时边界在哪里。

## 实验脚本与结果

脚本：

- `experiments/omni_transformer_stage_h3_h4.py`

结果：

- `artifacts/omni_transformer_stage_h3_h4/sweep_results.json`
- `artifacts/omni_transformer_stage_h3_h4/sweep_runs/`
- `artifacts/omni_transformer_stage_h3_h4/sweep_runs/*/samples/*/query_panel_grid_train.png`
- `artifacts/omni_transformer_stage_h3_h4/sweep_runs/*/samples/*/query_panel_grid_ood.png`

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_h3_h4.py --sweep --seeds 20260701,20260702,20260703 --train-size 4096 --val-size 1024 --test-size 2048 --batch-size 128 --d-model 128 --layers 3 --heads 4 --latent-counts 0,1,2,4,8,16 --capacity-steps 650 --h4-steps 850 --heldout-steps 850 --probe-steps 180 --h4-latent-tokens 8
```

## H3：Latent Token 容量曲线

H3 固定任务、模型和 H2 latent bottleneck mask，只改变 latent token 数量 `K`：

```text
K = 0, 1, 2, 4, 8, 16
```

聚合结果：

| latent tokens | answer exact | action exact |
| ---: | ---: | ---: |
| 0 | 1.56% | 17.19% |
| 1 | 100.00% | 100.00% |
| 2 | 100.00% | 100.00% |
| 4 | 100.00% | 100.00% |
| 8 | 100.00% | 100.00% |
| 16 | 100.00% | 100.00% |

K=1 的 latent probe：

| probe target | 准确率 |
| --- | ---: |
| visual | 100.00% |
| goal | 100.00% |
| telemetry | 98.96% |

解释：

- `K=0` 时答案 token 不能 attend 原始输入，也没有 latent scratchpad，因此只能学到输出格式和先验分布，answer exact 只有 1.56%。
- `K=1` 已经足够承载本任务的三路因素和动作决策。
- 这说明当前任务信息熵很低，不能据此推断真实 omni 模型只需要 1 个 latent token。
- 但它确实证明了 latent bottleneck 的容量曲线可测，而且 K=0 与 K≥1 有清晰断点。

## H4：Counterfactual Query

H4 把文本输入从固定 goal tokens 扩展为 query tokens：

```text
observed_goal + query_type + target_goal
```

两类 query：

| query | 含义 |
| --- | --- |
| diagnose | 使用当前 observed goal 输出答案 |
| counterfactual | 假设 target goal 生效，输出反事实动作和目标字段 |

输出仍是严格 DSL：

```text
ACTION_<action> VISUAL_<visual> GOAL_<target_goal> TELEMETRY_<telemetry> EOS
```

结果：

| 测试集 | answer exact |
| --- | ---: |
| mixed diagnose/counterfactual | 100.00% |
| counterfactual only | 100.00% |

样例：

```text
query:      counterfactual
target:     ACTION_hold_state VISUAL_blocked_flow GOAL_throughput_first TELEMETRY_heat_ramp EOS
prediction: ACTION_hold_state VISUAL_blocked_flow GOAL_throughput_first TELEMETRY_heat_ramp EOS
```

结论：在 ID 分布内，latent bottleneck tiny omni Transformer 可以处理简单 query 条件，并生成反事实 DSL 答案。

## H4：未见视觉风格

未见视觉风格测试只改变图像渲染风格：

- 颜色 palette 全部更换。
- 主要形状位置和几何布局改变。
- 文本和遥测保持不变。

结果：

| 测试 | answer exact | action exact | visual exact |
| --- | ---: | ---: | ---: |
| unseen visual style | 0.52% | 27.73% | 0.52% |

解释：

- 模型几乎不能识别未见视觉风格，visual exact 接近 0。
- action exact 仍有 27.73%，主要来自 goal/telemetry 及 action 先验，不代表视觉泛化成功。
- 这说明当前模型没有视觉风格不变性；要进入真实图像阶段，必须引入风格增强、更多视觉分布或预训练视觉编码。

## H4：Held-out 三因素组合

Held-out 测试训练时移除一部分 `(visual, target_goal, telemetry)` 目标组合。模型仍然见过每个单独因素，也见过大量其他组合，但没有见过这些特定三因素到 action 的映射。

结果：

| 测试 | answer exact | action exact | visual exact | goal exact | telemetry exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen target triples | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| held-out target triples | 4.17% | 4.17% | 100.00% | 97.92% | 84.90% |

解释：

- Held-out action 基本失败，这是预期结果：当前策略表是随机 lookup，没有可学习的组合规则。
- visual/goal/telemetry 仍大多可读，说明失败主要不是输入解析完全失效。
- 因为输出顺序是 `ACTION -> VISUAL -> GOAL -> TELEMETRY -> EOS`，第一个 action token 错误会通过 autoregressive 前缀影响后续字段，所以 telemetry 字段也会被拖低。
- 这个实验是负证据：如果任务规则本身不可组合泛化，Transformer/latent bottleneck 不能凭空推出未见组合的随机答案。

## 结论

H3/H4 支持以下判断：

1. Latent bottleneck 的容量边界可以被测出来：当前低熵任务 `K=0` 失败，`K>=1` 成功。
2. Tiny omni Transformer 可以处理简单 query/counterfactual 条件，不只是固定问题模板。
3. 当前视觉分布泛化很弱，未见视觉风格几乎全崩。
4. 对随机 lookup 策略表，held-out 组合不会自然泛化；这不是 latent 失败，而是任务没有组合规则可学。

## 对架构判断的影响

正面证据：

- H2/H3 证明特殊 latent token 可以作为内部 scratchpad。
- H4 counterfactual 证明 query 条件可以进入同一 token stream 并改变输出。

边界证据：

- 真实 omni 架构不能只靠小 Transformer 从单一视觉风格中学出风格不变性。
- 若任务目标是组合泛化，训练数据或任务规则必须具有可组合结构；否则模型只能记忆 seen mapping。

## 后续进展

H5/H6 已在 `docs/omni-transformer-stage-h5-h6-experiment.md` 中完成：

1. H5 已加入视觉增强并重新测试 held-out visual style。
2. H5 已把随机策略表替换为可组合规则并重新测试 held-out 组合泛化。
3. H6 已加入 `FULL`、`ACTION_ONLY`、`READ_VISUAL`、`READ_GOAL`、`READ_TELEMETRY` 多输出格式。
