# Stage V：专家语义空间对齐与信息丢失诊断

## 目的

Stage U 显示：新增 object/spatial/counting experts 后，最终准确率没有稳定提升，`no_object_experts` 消融也不掉。这很像两个问题叠加：

1. 各 expert token 没有统一到同一个潜变量语义空间，互相“语言不通”。
2. 某些 expert token 本身没有保留足够输入信息，存在信息丢失。

Stage V 因此不只看最终答案准确率，而是给 expert token 加语义监督，并直接测两个诊断指标：

- **信息还原率**：从单个 expert 的 token 还原原始输入中 16 个 grid cell 的 occupancy、color、shape。
- **跨 expert 互译能力**：只用 source expert 训练一个 semantic decoder，然后直接拿它读 target expert，不微调；如果能读，说明两个 expert 的 token 语义空间相近。

## 监督目标

每张 Stage U 多物体图像被还原成一个 4x4 语义网格：

- occupancy：该 cell 是否有对象。
- color：none + 8 色。
- shape：none + 4 形状。

诊断指标：

| 指标 | 含义 |
| --- | --- |
| `cell_info_accuracy` | cell 级 occupancy/color/shape 全对的比例 |
| `cell_info_loss` | `1 - cell_info_accuracy`，作为信息丢失率 |
| `scene_exact` | 16 个 cell 全部还原正确 |
| `diagonal_cell_info_accuracy` | probe 在同一个 expert 上训练和测试，衡量单 expert 信息保留 |
| `offdiag_cell_info_accuracy` | probe 在 source expert 训练后直接测试 target expert，衡量互译/共同语言 |
| `language_gap` | diagonal - offdiag，越大越说明 expert 空间不统一 |

## 模型对比

| Model | 训练方式 |
| --- | --- |
| `ranking_only_object_latent` | Stage U object-slot/spatial/counting 结构，只训练 ranking/router |
| `supervised_shared_semantic_latent` | 同结构，但训练时对 patch/object/spatial/count/fusion/wide tokens 同时施加共享 semantic decoder loss |

共享 semantic decoder 是同一个 decoder 读所有 expert tokens；它用于逼迫不同 expert 输出进入同一可读语义空间。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_v_expert_alignment.py --sweep --seeds 20260701,20260702 --train-size 1024 --val-size 256 --test-size 384 --batch-size 64 --image-size 64 --grid-size 4 --prompt-len 112 --answer-len 32 --d-model 96 --layers 2 --heads 4 --patch-grid 8 --object-slots 8 --spatial-tokens 8 --count-tokens 8 --prompt-tokens 8 --fusion-tokens 8 --output-tokens 8 --candidate-count 8 --train-steps 420 --eval-every 140 --semantic-loss-weight 0.35 --probe-steps 70 --output-dir artifacts\omni_transformer_stage_v_expert_alignment\sweep_runs --aggregate artifacts\omni_transformer_stage_v_expert_alignment\sweep_results.json
```

## 结果

随机 Top-1 为 12.50%。

| Model | Top-1 | No image | No patch | No object experts |
| --- | ---: | ---: | ---: | ---: |
| ranking only | 19.92% | 12.76% | 14.06% | 20.57% |
| supervised shared semantic | 23.57% | 12.24% | 21.22% | 20.83% |

监督训练对下游任务有帮助：Top-1 提高 3.65 个百分点，`no_patch` 从 14.06% 提到 21.22%。这说明非 patch expert 确实开始携带更多视觉证据。

信息还原与互译：

| Model | Diagonal cell info | Info loss | Offdiag cell info | Language gap |
| --- | ---: | ---: | ---: | ---: |
| ranking only | 57.20% | 42.80% | 40.69% | 16.50% |
| supervised shared semantic | 63.10% | 36.90% | 40.52% | 22.58% |

单 expert 信息还原率：

| Stage | ranking only | supervised |
| --- | ---: | ---: |
| patch tokens | 55.03% | 79.39% |
| object slots | 59.14% | 55.40% |
| spatial tokens | 59.14% | 53.82% |
| count tokens | 59.14% | 51.85% |
| fusion tokens | 59.14% | 59.46% |
| wide latent | 51.59% | 78.66% |

监督模型内部训练出来的共享 decoder 读取各 stage 的结果：

| Stage | Cell info accuracy |
| --- | ---: |
| patch tokens | 92.02% |
| object slots | 58.95% |
| spatial tokens | 55.26% |
| count tokens | 56.48% |
| fusion tokens | 59.80% |
| wide latent | 92.00% |

## 结论

Stage V 支持你的怀疑，但要分两层说：

1. **带监督训练是有效方向。** 监督后 Top-1、`no_patch`、patch/wide latent 的语义还原都明显改善，说明 raw patch 之外的路径开始承载视觉证据。
2. **当前 expert 还没有形成统一语言。** offdiag transfer 没有提升，40.69% -> 40.52%；language gap 反而变大，是因为 diagonal 变好了但跨 expert 互读没跟上。
3. **信息仍大量丢失。** 最好的 diagonal cell info 也只有 63.10%，scene exact 仍接近 0%-2.5%。这说明当前 token 还不能完整还原输入场景。
4. **object/spatial/count experts 仍是弱项。** 共享 decoder 能把 patch/wide 读到 92%，但 object/spatial/count 只有 55%-59%。这说明监督主要强化了 patch -> wide 路径，没有让对象专家真正成为可靠对象表。

## 下一步

Stage V 说明“共享 decoder 辅助 loss”还不够。下一步应直接做更强的对齐机制：

- 为 object slots 加 slot-level 监督：每个 slot 对应一个对象，预测 objectness、color、shape、cell/box。
- 加 cross-expert contrastive alignment：同一 scene 的 patch/object/spatial/count/fusion 表示拉近，不同 scene 拉远。
- 引入显式 common semantic bus：各 expert 先投影成统一 object-table tokens，再进入 wide latent，而不是让 raw expert tokens 各说各话。
- 对每个 expert 单独报告输入还原率和跨 expert transfer；没有通过这个诊断前，不再只看最终 ranking 准确率。
