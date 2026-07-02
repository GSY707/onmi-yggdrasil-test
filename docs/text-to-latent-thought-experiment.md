# 纯文本思考到特殊 latent token 内部思考验证

## 验证问题

要验证的问题不是“大模型一定会产生真实潜空间智能”，而是更窄的一步：

> 先用纯文本思考链训练模型，再把中间思考替换成特殊 `<latent>` token，只监督最终答案，这条路径在本机小模型上是否能跑通。

## 环境

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU，8GB VRAM。
- 训练环境：本仓库 `.venv`。
- PyTorch：`2.11.0+cu128`。
- CUDA：`12.8`。
- 结果文件：`artifacts/text_to_latent_thought/results.json`。

## 任务设计

使用可自动判分的 8x8 网格路径任务：

```text
输入：起点 X/Y + 6 个移动指令
输出：最终 X/Y
```

四组对照：

| 组 | 训练方式 | 输出形态 |
| --- | --- | --- |
| `direct_scratch` | 从零训练，直接输出答案 | `A X Y <eos>` |
| `visible_text` | 从零训练，输出每一步文本坐标再输出答案 | `T X Y ... A X Y <eos>` |
| `latent_from_text` | 从 `visible_text` 权重继续训练，把 6 步文本思考替换成 6 个 `<latent>` token，只监督答案 | `<latent> x6 + A X Y <eos>` |
| `latent_scratch` | 从零训练 latent 形态 | `<latent> x6 + A X Y <eos>` |

`<latent>` token 不承载可读内容；它只是重复的特殊 token。信息只能存进 transformer hidden state。

## 实测结果

### 单次基线

随机猜测终点准确率为 `1 / 64 = 1.5625%`。

| 组 | 留出集 exact accuracy | parseable rate | 平均生成或插入 token |
| --- | ---: | ---: | ---: |
| `direct_scratch` | 94.63% | 99.80% | 4 |
| `visible_text` | 100.00% | 100.00% | 22 |
| `latent_from_text` | 85.84% | 99.90% | 10 |
| `latent_scratch` | 79.20% | 99.85% | 10 |

训练过程也支持同一结论：

- `visible_text` 在 500 step 后达到 100% val exact。
- `latent_from_text` 在 700 step 后达到 85.55% val exact。
- `latent_scratch` 在 700 step 后达到 80.08% val exact。

## 结论

单次基线显示特殊 latent token 路线在这个 toy task 上具备初步可行性：

1. `<latent>` token 不输出可读思考链，仍能让模型在最终答案上达到 85.84%。
2. 从纯文本思考链迁移到 latent token，比 latent 从零训练更好：85.84% vs 79.20%。
3. latent 形态用 6 个内部 token + 4 个答案 token，明显短于 22 个显式文本思考输出 token。

## 多 seed / 更长 move sweep

为避免单次 seed 偶然性，追加了 `move_count = 6, 8, 10` 与 `seed = 20260701, 20260702, 20260703` 的 9 次训练 sweep。每次配置：

- train：4096
- validation：768
- test：1024
- text/direct：1000 step
- latent/latent scratch：800 step

结果文件：

- 聚合：`artifacts/text_to_latent_thought/sweep_results.json`
- 单次 run：`artifacts/text_to_latent_thought/sweep_runs/*.json`

| move 数 | direct 平均 | visible 平均 | latent_from_text 平均 | latent_scratch 平均 | text->latent 相对 scratch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 95.35% | 100.00% | 89.06% | 92.32% | -3.26 pp |
| 8 | 87.53% | 100.00% | 82.88% | 76.50% | +6.38 pp |
| 10 | 78.22% | 100.00% | 80.73% | 56.18% | +24.54 pp |
| overall | 87.03% | 100.00% | 84.22% | 75.00% | +9.22 pp |

更严格的结论：

1. 特殊 `<latent>` token 机制在 9 次实验中都能明显高于随机，说明“内部 token 承载中间状态”不是偶然单次成功。
2. 纯文本思考预训练对更长任务更有价值：move=8 和 move=10 平均优于 latent scratch，move=10 的平均差距达到 +24.54 pp。
3. move=6 上文本预训练没有稳定收益，latent scratch 反而更高。这说明短任务上直接学 latent 形态已经足够，文本阶段可能不是必要条件。
4. 这个收益不是每个 seed 都稳定出现：move=8 和 move=10 都各有一个 seed 里 latent scratch 反超。因此不能声称“文本思考预训练必然提升 latent”。
5. 显式文本思考仍是上限：所有 move 数都是 100%。当前特殊 latent token 还没有完全压缩掉显式推理链的信息损失。

## 当前判断

这条路线比单次实验时更值得继续，但结论要收窄：

- 可以成立：`先纯文本思考训练，再扩展到特殊 latent token 内部思考` 是可运行机制；在更长组合任务上，文本阶段平均能帮助 latent 训练。
- 不能成立：文本预训练不是稳定必胜技巧；latent token 没有稳定接近 visible text；也没有稳定超过 direct。

但它没有证明更强的结论：

1. `latent_from_text` 没有超过 `direct_scratch`，所以本轮不能声称 latent 思考比直接答案训练更强。
2. 这是小模型和合成任务，不证明大模型会形成可靠潜空间推理。
3. 这不是多模态实验，不证明图像、视频或动作空间可行。
4. 这不是长期任务实验，不证明它能接入 `世界树计划` 的真实 runtime 后稳定工作。

## 下一步判断

这条路线值得继续，但下一步应验证更硬的情况：

1. 训练后逐步减少 visible text 阶段依赖，测试需要多少纯文本思考监督。
2. 延长 latent fine-tune step，看 move=10 的弱 seed 是否能追上，区分训练预算不足和表示能力不足。
3. 改成更强的组合任务，例如带障碍物、条件分支、子目标或多段路径合成。
4. 再接入一个小型预训练语言模型 LoRA，而不是只用从零训练的小 transformer。
