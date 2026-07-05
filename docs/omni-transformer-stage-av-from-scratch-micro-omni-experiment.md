# Stage AV：70M 从零集成 Micro-Omni

## 目标

Stage AV 回应“不要只拼预训练专家，也不要一直停在很小 toy”的担忧：在本机 RTX 4070 Laptop 8GB 上，训练一个明显更大的、完全从 0 初始化的集成模型，把 AR/AS/AT 的能力压进同一个 Transformer。

本阶段不是训练通用基础模型，而是验证：

1. 本机能否承受 70M 级从零训练。
2. 单一模型能否同时接视觉 token、文本规则、工具历史、memory/state。
3. 多任务训练是否自然闭合视觉 grounding、工具审计和 memory 探索。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_av_from_scratch_micro_omni.py
```

模型结构：

- 图像输入：4 个 zone 的 RGB token。
- 文本输入：task、target/rule/target-color token。
- 工具输入：HTML/CSV/PDF/screenshot status token。
- 记忆输入：4-slot zone memory。
- 状态输入：telemetry、phase、seen zone、last scan。
- Transformer encoder + latent tokens。
- 输出：12 类统一 answer space。

任务族：

| 任务 | 来源 | 目标 |
| --- | --- | --- |
| `cross_expert_visual` | Stage AR 简化版 | 从视觉 zone 和文本 rule 推出答案 |
| `file_audit_tools` | Stage AS 简化版 | 从工具历史判断文件审计结论 |
| `memory_exploration` | Stage AT 简化版 | 从 memory/state 找到目标 zone |

## 规模探测

先跑小 smoke 验证脚本链路：

- 结果：`artifacts/omni_transformer_stage_av_from_scratch_micro_omni/smoke_results.json`
- 参数量：414,867
- 峰值 CUDA allocated：29.95 MB

随后跑 70M 档容量 smoke：

- 结果：`artifacts/omni_transformer_stage_av_from_scratch_micro_omni/larger_smoke_results.json`
- 配置：`d_model=768`、`layers=10`、`heads=12`、`latent_tokens=8`
- 参数量：70,994,707
- 峰值 CUDA allocated：1,375.26 MB

结论：本机 8GB 可以训练 70M 级从零集成模型；显存不是本阶段直接硬阻塞。

## Probe 结果

最终保留的 probe：

- 结果：`artifacts/omni_transformer_stage_av_from_scratch_micro_omni/probe_70m_residual_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_av_from_scratch_micro_omni/probe_70m_residual_runs/seed20260701.json`
- 样例：`artifacts/omni_transformer_stage_av_from_scratch_micro_omni/probe_70m_residual_runs/samples/`

配置：

- `train_size=4096`
- `val_size=768`
- `test_size=768`
- `batch_size=32`
- `d_model=768`
- `layers=10`
- `heads=12`
- `steps=1000`
- 参数量：70,994,707
- 峰值 CUDA allocated：1,810.62 MB
- elapsed：92.38 秒

结果：

| 指标 | probe |
| --- | ---: |
| overall answer accuracy | 64.32% |
| cross-expert visual accuracy | 23.44% |
| file-audit tools accuracy | 69.53% |
| memory exploration accuracy | 100.00% |
| task probe accuracy | 100.00% |
| no-image answer accuracy | 63.93% |
| no-tool-history answer accuracy | 32.68% |
| no-memory answer accuracy | 31.77% |

门槛判断：

| 门槛 | 结果 |
| --- | --- |
| overall >= 85% | 未通过 |
| 每任务 >= 80% | 未通过 |
| 关键消融明显下降 | tool/history 和 memory 通过；image 未通过 |

## 结论

Stage AV 当前是中间负结果，不是正式通过。

正信号：

1. 本机可以训练 70M 级从零模型，显存峰值约 1.8GB allocated。
2. 单模型能稳定学会 task routing，task probe 为 100%。
3. `memory_exploration` 子任务达到 100%，`no_memory` 从 64.32% 掉到 31.77%。
4. `file_audit_tools` 子任务达到 69.53%，`no_tool_history` 从 64.32% 掉到 32.68%。

负信号：

1. `cross_expert_visual` 子任务只有 23.44%，接近 4 类随机。
2. `no_image` 与 `shuffled_image` 基本不降，说明当前 AV 没有学出有效视觉 grounding。
3. 单阶段多任务训练会优先学容易的工具/history/memory 通道，视觉路径没有自然闭合。
4. 把 answer head 从只读 latent 改成 residual evidence readout 后，仍未修好视觉子任务。

## 下一步判断

不要直接进入 AV 正式 3 seed。当前应先修训练路线：

1. 分阶段预训练视觉 zone/color reader。
2. 冻结或半冻结视觉 reader 后，再训练统一 answer head。
3. 加 task-specific probe：分别报告 AR/AS/AT 的 no-image/no-text/no-tool/no-memory。
4. 如果视觉 reader 单独可达 100%，但集成后 AR 仍随机，再定位为 answer head/latent bus 不使用视觉证据。
5. 如果视觉 reader 单独也学不稳，先修图像 token 表示，不继续扩大参数量。

## 边界和担忧

1. 70M 比前面 toy 大很多，但仍不是“大模型”或基础模型。
2. 当前任务仍是合成任务，不能证明真实图像、真实文档或真实 agent 能力。
3. 显存峰值是 PyTorch allocated，不等于完整系统占用；更长序列、更大 batch、更复杂图像会显著提高显存。
4. 单纯扩大参数量没有自动修复视觉 grounding；这和之前 Stage S/T 的经验一致：必须保留证据通道并让 readout 真正使用它。

## 复现命令

70M 容量 smoke：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_av_from_scratch_micro_omni.py --sweep --seeds 20260701 --train-size 96 --val-size 48 --test-size 48 --batch-size 8 --d-model 768 --heads 12 --layers 10 --latent-tokens 8 --steps 2 --eval-every 2 --sample-count 1 --output-dir artifacts\omni_transformer_stage_av_from_scratch_micro_omni\larger_smoke_runs --aggregate artifacts\omni_transformer_stage_av_from_scratch_micro_omni\larger_smoke_results.json
```

70M residual probe：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_av_from_scratch_micro_omni.py --sweep --seeds 20260701 --train-size 4096 --val-size 768 --test-size 768 --batch-size 32 --d-model 768 --heads 12 --layers 10 --latent-tokens 8 --steps 1000 --eval-every 250 --sample-count 8 --output-dir artifacts\omni_transformer_stage_av_from_scratch_micro_omni\probe_70m_residual_runs --aggregate artifacts\omni_transformer_stage_av_from_scratch_micro_omni\probe_70m_residual_results.json
```
