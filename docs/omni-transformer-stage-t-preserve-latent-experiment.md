# Stage T：保信息 Latent 压缩器实验

## 目的

Stage S 证明当前 `Attention Pump -> thought latent` 会丢信息。Stage T 按新的设计原则替换 latent 压缩器：

- 宁愿保留更多 latent token，也不要过早把信息压坏。
- raw expert tokens 必须有保真路径。
- routing 可以作为附加视角，但不能成为唯一通道。
- summary/slot token 可以追加，但不能覆盖原始证据 token。

## 代码与产物

- `experiments/omni_transformer_stage_t_preserve_latent.py`
- `artifacts/omni_transformer_stage_t_preserve_latent/sweep_results.json`
- `artifacts/omni_transformer_stage_t_preserve_latent/sweep_runs/`
- `artifacts/omni_transformer_stage_t_preserve_latent/sweep_runs/*/*/samples/`

## 模型对照

| 名称 | 结构 |
| --- | --- |
| Stage S `cross_pump_latent` | answer cross-attends `Attention Pump -> thought latent` |
| Stage S `cross_no_pump_expert_tokens` | answer 直接 cross-attends weighted expert tokens |
| Stage T `wide_residual_latent` | 保留 raw expert tokens + route-weighted expert tokens，再追加 summary tokens；answer cross-attends 全部 latent tokens |
| Stage T `slot_resampler_latent` | 32 个 resampler slot 从 expert tokens 重采样，再经 thought Transformer；不保留原 token |

Stage T 的 `wide_residual_latent` 默认 token 数为：

- raw expert tokens：4 experts * 8 tokens = 32
- weighted expert tokens：4 experts * 8 tokens = 32
- summary tokens：8
- 总计：72 latent tokens

这不是最终成本优化方案；它是保信息验证方案。后续 token 过多问题应交给记忆树/工作树的分层选择、摘要和上下文调度处理，而不是在第一层 latent 压缩器里过早丢证据。

## 正式命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_t_preserve_latent.py --sweep --seeds 20260701,20260702,20260703 --levels l1_color,l2_object,l3_position,l4_count,l5_relation --train-size 1536 --val-size 384 --test-size 512 --batch-size 64 --image-size 64 --prompt-len 96 --answer-len 96 --d-model 96 --layers 2 --heads 4 --latent-tokens 8 --slot-latent-tokens 32 --summary-tokens 8 --candidate-count 8 --train-steps 600 --probe-steps 160 --eval-every 200 --output-dir artifacts\omni_transformer_stage_t_preserve_latent\sweep_runs --aggregate artifacts\omni_transformer_stage_t_preserve_latent\sweep_results.json
```

正式 3 seed 平均每 seed 约 378.97 秒。`wide_residual_latent` 约 1,376,933 trainable params。

## 结果

Top-1，3 seeds：

| Level | Random | Stage S cross+pump | Stage S no-pump | Stage T wide residual | Stage T 32-slot resampler |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 color | 12.50% | 16.47% | 100.00% | 100.00% | 16.15% |
| L2 object | 12.50% | 9.96% | 94.40% | 98.63% | 13.35% |
| L3 position | 12.50% | 12.30% | 26.95% | 73.76% | 12.70% |
| L4 count | 20.00% | 19.40% | 19.99% | 19.40% | 19.40% |
| L5 relation | 12.50% | 12.96% | 77.67% | 92.06% | 12.04% |

No-image 消融：

| Level | wide residual | wide residual no image |
| --- | ---: | ---: |
| L1 color | 100.00% | 11.59% |
| L2 object | 98.63% | 12.17% |
| L3 position | 73.76% | 12.76% |
| L4 count | 19.40% | 19.40% |
| L5 relation | 92.06% | 12.17% |

## Reconstruction

Exact answer-class reconstruction，3 seeds：

| Level | wide source concat | wide residual latent | 32-slot latent |
| --- | ---: | ---: | ---: |
| L1 color | 100.00% | 100.00% | 20.64% |
| L2 object | 100.00% | 100.00% | 3.12% |
| L3 position | 31.12% | 25.46% | 0.72% |
| L4 count | 19.53% | 20.05% | 19.40% |
| L5 relation | 92.45% | 92.32% | 13.80% |

## 判断

Stage T 支持新的设计原则：保信息 latent 比早期强压缩 latent 更适合当前架构。

关键证据：

1. `wide_residual_latent` 把 L2 从 Stage S cross+pump 的 9.96% 提到 98.63%，把 L5 从 12.96% 提到 92.06%。
2. L3 从 Stage S cross+pump 的 12.30% 提到 73.76%，说明位置组合信息也能通过保真 token 部分保住，但还没有完全解决。
3. `no image` 后 L1/L2/L3/L5 都回到随机附近，说明提升确实来自视觉 token。
4. `slot_resampler_latent` 即使有 32 个 latent slot 也没有修复 L2/L3/L5，说明“token 数变多”不够；必须保留 raw/residual 信息通道。
5. L4 counting 仍然失败，继续证明它需要对象级 slot/counting expert，而不是靠 latent 压缩器修复。

## 架构含义

当前最合理的 latent bus 方向是：

```text
expert tokens
  -> raw evidence tokens retained
  -> route-weighted tokens appended
  -> summary tokens appended
  -> answer / agent head cross-attends all selected latent tokens
```

后续可以在工作树/记忆树层面处理 token 规模：

- 当前 step 只保留任务相关 token。
- 长程上下文写入记忆树为 LOD summary + evidence URI。
- 工作树节点选择需要展开的 token group。
- 只有在证据已经被消费或摘要稳定后，再做降采样。

## 下一步

- 在 `wide_residual_latent` 基础上做 L3 位置增强：显式 position embedding、object slot、patch coordinate token 或 spatial expert。
- 对 L4 counting 增加 object-slot/counting expert。
- 再测真实 LLaVA/DocVQA 时，不应回到 8-token pump；应先用 wide residual latent 或 object-slot latent 做入口。
