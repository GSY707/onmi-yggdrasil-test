# Stage W：四种专家对齐机制与训练期 Common Semantic Bus

## 目的

Stage V 证明了带监督的共享语义 decoder 能提升下游任务和 `no_patch` 消融，但没有解决跨 expert 互读：不同专家仍然像在说不同语言。

Stage W 因此测试四个对齐机制，并按你的约束处理 common semantic bus：

1. 共享 4x4 semantic grid decoder：同一个 decoder 读取 patch/object/spatial/count/fusion/wide token，预测 occupancy、color、shape。
2. object slot 显式监督：object slots 预测 objectness、color、shape、cell。
3. cross-expert contrastive alignment：同一 scene 的不同 expert pooled embedding 拉近，不同 scene 拉远。
4. common semantic bus：只作为训练期 teacher / alignment target / 诊断分支，不接入最终 scorer，推理期仍然直出 latent。

## 模型对比

| Mode | 机制 |
| --- | --- |
| `ranking_only` | 只训练最终 ranking/router |
| `shared_grid_decoder` | 加共享 semantic grid decoder |
| `shared_grid_slot_targets` | 加共享 decoder + slot targets |
| `shared_slot_contrastive` | 加共享 decoder + slot targets + contrastive |
| `all_four_train_only_bus` | 四个机制全开；common bus 只在训练/诊断使用 |

正式结果中的每个 mode 都记录了 `diagnostics.inference_uses_common_bus = false`。也就是说，`all_four_train_only_bus` 的候选答案 scorer 没有读取 common bus；common bus 只对训练 loss 和诊断指标产生影响。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_w_alignment_mechanisms.py --sweep --seeds 20260701,20260702 --train-size 1024 --val-size 256 --test-size 384 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --fusion-tokens 8 --output-tokens 8 --candidate-count 8 --train-steps 360 --eval-every 120 --shared-loss-weight 0.30 --slot-loss-weight 0.20 --contrastive-loss-weight 0.08 --bus-loss-weight 0.30 --bus-align-weight 0.08 --probe-steps 55 --output-dir artifacts\omni_transformer_stage_w_alignment_mechanisms\sweep_runs --aggregate artifacts\omni_transformer_stage_w_alignment_mechanisms\sweep_results.json
```

随机 Top-1 为 12.50%。正式 sweep 为 2 seeds：`20260701,20260702`。

## 结果

主任务与消融：

| Mode | Full Top-1 | No image | No patch | No object experts | Best val |
| --- | ---: | ---: | ---: | ---: | ---: |
| ranking only | 13.41% ± 2.73% | 11.98% | 11.85% | 13.67% | 19.73% |
| shared grid decoder | **22.66% ± 0.26%** | 11.98% | **21.88%** | **21.74%** | **27.54%** |
| shared + slot targets | 19.92% ± 0.91% | 12.50% | 18.75% | 17.06% | 23.44% |
| shared + slot + contrastive | 19.14% ± 3.78% | 14.19% | 17.97% | 15.89% | 24.02% |
| all four, train-only bus | 21.09% ± 2.60% | 11.98% | 19.14% | 19.66% | 24.80% |

信息还原与跨 expert 互读：

| Mode | Diagonal cell info | Offdiag cell info | Language gap |
| --- | ---: | ---: | ---: |
| ranking only | 57.60% | 38.28% | 19.32% |
| shared grid decoder | 55.65% | **39.95%** | **15.71%** |
| shared + slot targets | 55.02% | 37.73% | 17.29% |
| shared + slot + contrastive | 54.92% | 36.31% | 18.61% |
| all four, train-only bus | 54.79% | 34.98% | 19.80% |

slot 与 bus 诊断：

| Mode | Slot exact | Positive cell | Positive color | Positive shape | Common bus cell info |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared + slot targets | 21.57% | 40.03% | 26.83% | 29.79% | n/a |
| shared + slot + contrastive | 17.29% | **45.46%** | 25.20% | 29.21% | n/a |
| all four, train-only bus | 20.52% | 40.91% | **39.00%** | **31.39%** | 58.58% |

成本：

| Mode | Trainable params | Train seconds |
| --- | ---: | ---: |
| ranking only | 1,146,823 | 18.71s |
| shared grid decoder | 1,261,846 | 32.81s |
| shared + slot targets | 1,286,838 | 63.07s |
| shared + slot + contrastive | 1,353,366 | 76.70s |
| all four, train-only bus | 1,467,030 | 84.45s |

## 结论

Stage W 的正结果是：**训练期对齐监督确实有用**。最小的 `shared_grid_decoder` 就把 Full Top-1 从 13.41% 提到 22.66%，`no_patch` 从 11.85% 提到 21.88%，说明非 patch 路径开始携带可用视觉证据。

但四个机制全开不是最好结果。`all_four_train_only_bus` 的 common bus 自身能还原 58.58% cell info，也提升了 slot color/shape 诊断，但最终 Top-1 只有 21.09%，低于 `shared_grid_decoder`。这说明显式 bus 可以当早期 teacher/诊断器，但当前权重下会给主任务带来优化竞争，不能直接说明“bus 越强越好”。

cross-expert contrastive 在这轮预算下不稳定：两个 seed 的 Full Top-1 分别是 22.92% 和 15.36%，平均低于只加 shared decoder。slot targets 也没有继续提升主任务，可能是排序 slot 目标与最终 ranking 目标存在容量/梯度竞争。

最关键的负结果是：**专家统一语言仍未真正解决**。`shared_grid_decoder` 的 language gap 从 19.32% 降到 15.71%，但 offdiag cell info 只到 39.95%；`all_four_train_only_bus` 反而降到 34.98%。这意味着现在还不能说不同 expert 已经能稳定互读含义。

## 架构口径

本轮不把 common semantic bus 作为最终推理结构。更合适的口径是：

- 初期训练：允许显式 common bus 提供语义监督、teacher target、alignment loss 和诊断读数。
- 过渡阶段：把 bus 学到的语义约束蒸馏回 expert token / wide latent / output latent。
- 最终结构：候选 scorer 或输出专家直接读取 latent token，不依赖显式 common bus 作为运行时中间层。

这更贴近白皮书的“直出 latent”方向，也避免把架构退化成一个显式公共语义表流水线。

## 下一步

1. 保留 `shared_grid_decoder` 作为默认低成本对齐 baseline。
2. 对 common bus 做 loss schedule：前期高权重，后期衰减或停止梯度，只用于蒸馏，不参与最终 scorer。
3. slot targets 不应只用排序对象表；下一轮应尝试 set matching / Hungarian matching，避免 slot 顺序监督引入噪声。
4. contrastive 需要更精细：按对象/区域 token 做正负样本，而不是只做 scene-level pooled embedding。
5. 继续把 `inference_uses_common_bus=false` 作为硬约束；如果某个实验让最终 scorer 读取 bus，应单独标记为偏离目标架构的 ablation。
