# Stage H：Tiny Omni Transformer H1/H2 实验

## 目的

Stage H 用一个统一的 decoder-only Transformer 替代 Stage E+F 的多 encoder + fusion classifier。目标是验证更接近最终 omni 模型的架构链路：

```text
image patches + text tokens + telemetry tokens
        -> shared d_model token stream
        -> decoder-only Transformer
        -> optional latent scratchpad tokens
        -> text answer tokens
```

本轮只做 H1 和 H2：

- H1 direct omni transformer：答案 token 可以直接 attend 原始多模态输入。
- H2 latent bottleneck：latent token 可以 attend 原始多模态输入；答案 token 不能 attend 原始输入，只能 attend latent scratchpad 和前序答案 token。

这个设置用于验证：特殊 latent token 能不能承载多模态信息，并驱动纯文本输出。

## 实验脚本与结果

脚本：

- `experiments/omni_transformer_stage_h.py`

结果：

- `artifacts/omni_transformer_stage_h/sweep_results.json`
- `artifacts/omni_transformer_stage_h/sweep_runs/`
- `artifacts/omni_transformer_stage_h/sweep_runs/*/samples/*/diagnostic_panel_grid.png`

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_h.py --sweep --seeds 20260701,20260702,20260703 --train-size 4096 --val-size 1024 --test-size 2048 --batch-size 128 --d-model 160 --layers 4 --heads 4 --direct-steps 800 --bottleneck-steps 1000 --probe-steps 250 --latent-tokens 8
```

## 任务与输出

任务沿用 Stage E+F 的三路输入：

| 输入 | token 化方式 |
| --- | --- |
| 图像 | `48x48 RGB` 设备面板切成 `8x8` patch，共 36 个 image patch tokens |
| 文本 | 5 个 goal tokens |
| 遥测 | `12x4` 连续时序，每个 timestep 投影成 telemetry token |

输出不是分类头，而是固定 DSL 文本 token：

```text
ACTION_<action> VISUAL_<visual> GOAL_<goal> TELEMETRY_<telemetry> EOS
```

评估时用 greedy generation，而不是 teacher-forcing exact。`answer_exact` 要求 5 个输出 token 全部正确。

## Attention Mask

H1 direct：

```text
input tokens -> answer tokens
answer token 可以 attend 所有较早 token
```

H2 latent bottleneck：

```text
input tokens -> latent tokens -> answer tokens
answer token 禁止 attend input tokens
answer token 只能 attend latent tokens 和较早 answer tokens
```

因此 H2 如果成功，答案不能绕过 latent scratchpad 直接读取图像、文本或遥测。

## 正式结果

配置：

- seeds：`20260701,20260702,20260703`
- train / val / test：`4096 / 1024 / 2048`
- Transformer：`d_model=160, layers=4, heads=4`
- latent tokens：`8`
- GPU：`NVIDIA GeForce RTX 4070 Laptop GPU`

聚合结果：

| 模型 | answer exact | action exact | format valid |
| --- | ---: | ---: | ---: |
| H1 direct omni transformer | 100.00% | 100.00% | 100.00% |
| H2 latent bottleneck | 100.00% | 100.00% | 100.00% |

样例输出：

```text
target:     ACTION_increase_throughput VISUAL_blocked_flow GOAL_safety_first TELEMETRY_low_voltage EOS
prediction: ACTION_increase_throughput VISUAL_blocked_flow GOAL_safety_first TELEMETRY_low_voltage EOS
```

## H2 Latent Bottleneck 验证

关键消融：

| 干预 | answer exact | action exact |
| --- | ---: | ---: |
| H2 full | 100.00% | 100.00% |
| H2 no latent access | 0.00% | 17.19% |
| H2 zero image | 7.81% | 31.25% |
| H2 zero text | 11.46% | 32.29% |
| H2 zero telemetry | 13.54% | 30.21% |
| H2 shuffle image | 24.43% | - |
| H2 shuffle text | 24.72% | - |
| H2 shuffle telemetry | 24.80% | - |

解释：

- `no latent access` 同时禁止答案看原始输入和 latent，`answer_exact` 变成 0。这证明 H2 的答案确实依赖 latent scratchpad。
- 清零任一路输入后，完整文本答案从 100% 掉到约 8%-14%。
- 打乱任一路输入后，完整文本答案约 24%-25%，接近该字段被随机替换后的水平。
- 被清零的对应字段会掉到 25% 随机水平：`zero image` 下 visual 为 25%，`zero text` 下 goal 为 25%，`zero telemetry` 下 telemetry 为 25%。

## Latent Probe

冻结 H2 模型，用 latent tokens 的 final hidden state 均值训练线性 probe：

| latent probe target | 准确率 |
| --- | ---: |
| visual | 100.00% |
| goal | 100.00% |
| telemetry | 100.00% |

这说明 H2 的 latent scratchpad 中确实保存了三路多模态因素，而不只是保存最终 action。

## 结论

Stage H H1/H2 支持以下判断：

1. 统一 token stream 的 tiny omni transformer 可以处理图像 patch、文本 token、遥测 token，并生成文本答案。
2. H2 的特殊 latent token 可以作为内部 scratchpad 承载多模态信息。
3. 答案端被禁止直接读取原始输入后，模型仍能达到 100%，说明 latent bottleneck 链路闭合。
4. no-latent-access、删模态、打乱模态和 latent probe 都支持“信息经过 latent scratchpad 流动”的解释。

这比 Stage E+F 更接近最终目标：一个统一 Transformer 大脑，而不是多个外部 encoder 后接分类头。

## 未证明边界

- 任务仍是合成低熵诊断任务，不是真实自然图像、真实语言或真实传感器日志。
- 输出 DSL 固定，未验证开放式自然语言生成。
- H2 使用 8 个 latent tokens；还没有做 K=0/2/4/8/16 容量曲线。
- 没有做未见图像风格、held-out 组合、更多 query 类型或 counterfactual 输出。
- 没有证明该结构优于 direct transformer，只证明 latent bottleneck 在该任务上可行。

## 后续进展

H3/H4 已在 `docs/omni-transformer-stage-h3-h4-experiment.md` 中完成：

1. H3 已比较 `K=0/1/2/4/8/16` latent token 容量曲线。
2. H4 已测试 counterfactual query、未见视觉风格和 held-out 三因素组合。
