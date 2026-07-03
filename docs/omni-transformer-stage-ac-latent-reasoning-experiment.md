# Stage AC：Q/A 潜空间、外部信息互译与答案 token 潜空间推理实验

## 目的

Stage AB 之后路线切换为四段：

1. 用问题和答案先建立高保真的文本潜空间。
2. 让外部信息翻译到同一个潜空间。
3. 在潜空间里推理，生成答案 token latent。
4. 由答案 token latent 解码成最终文本。

Stage AC 是这个方向的第一轮最小测试。它故意不使用图像前端，而是使用结构化多物体事实表，避免视觉识别错误掩盖核心问题。任务族包括：

- `color_at_cell`：按 row/column 查颜色。
- `shape_at_cell`：按 row/column 查形状。
- `count_color_shape`：按 color+shape 计数。
- `relation_yes_no`：判断两个对象的空间关系。

## 运行命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --output-dir artifacts\omni_transformer_stage_ac_latent_reasoning\sweep_runs --aggregate artifacts\omni_transformer_stage_ac_latent_reasoning\sweep_results.json
```

另做了一个 `color_at_cell` 单任务诊断：Q/A codec 与 evidence codec 均达到 100%，但 latent reasoner 仍只有 37.5% 左右。

## 主结果

单 seed `20260701`，正式四任务测试：

| 模块 | 指标 | 结果 |
| --- | --- | ---: |
| Q/A text codec | question sequence exact | 100.00% |
| Q/A text codec | answer sequence exact | 100.00% |
| Evidence codec | grid occupancy exact | 100.00% |
| Evidence codec | occupied color accuracy | 100.00% |
| Evidence codec | occupied shape accuracy | 100.00% |
| Evidence codec | count table exact | 91.80% |
| Latent reasoner full | answer word exact | 40.04% |
| Latent reasoner no evidence | answer word exact | 27.34% |
| Latent reasoner shuffled evidence | answer word exact | 26.56% |
| Latent reasoner full | answer latent cosine | 94.96% |

按任务族：

| Family | answer word exact |
| --- | ---: |
| `color_at_cell` | 37.50% |
| `shape_at_cell` | 30.47% |
| `count_color_shape` | 29.69% |
| `relation_yes_no` | 62.50% |

## 解释

这轮最重要的结论不是 40.04% 本身，而是三个阶段被拆开后，瓶颈位置变得清楚：

1. **第一步成功。** 问题 token latent 和答案 token latent 都能 100% 还原文本。
2. **第二步基本成功。** 外部事实表 latent 能 100% 还原 occupancy、颜色、形状，count table 也有 91.80% exact。
3. **第三步失败。** 即使前两步高保真，通用 Transformer latent reasoner 仍不能稳定完成最简单的潜空间查询与组合推理。

`full` 比 `no_evidence` 和 `shuffled_evidence` 高，说明 reasoner 不是完全忽略证据；但提升只有约 12.7-13.5 个百分点，远没到“潜空间推理闭合”的标准。

`answer latent cosine` 高达 94.96%，但答案 exact 只有 40.04%，这再次说明 cosine 不能作为成功指标。模型能靠近答案 latent 分布，却没有稳定落到正确答案 token。

## 新发现

Stage AC 支持一个新的判断：**高保真互译能力是必要条件，但不是充分条件。**

我们已经能分别得到：

- `问题/答案 <-> token latent` 的高保真 codec。
- `外部事实表 <-> latent` 的高保真 codec。

但把两者交给普通 latent reasoner，不会自动产生可靠推理。换句话说，第三步需要自己的训练目标和结构约束，不能假设“有了潜空间，推理自然会发生”。

这也解释了 Stage AB 的问题：候选 scorer 和 latent-to-answer scorer 不是足够强的第三步。真正的第三步应当让模型在潜空间里执行 lookup、comparison、counting、relation 等操作，并显式生成 answer-token latent。

## 下一步

下一轮不应继续只增加训练步数。`color_at_cell` 单任务中，Q/A 和 evidence 均为 100%，把 reasoner 从 1500 步拉到 5000 步也没有提升。

更合理的下一步：

1. 给 latent reasoner 加显式操作监督，例如 target cell token、selected object token、count accumulator、relation pair token。
2. 把 evidence latent 设计成可操作结构，而不是只要求 decoder 能读出事实；reasoner 需要知道哪个 token 是 cell、object、count pair。
3. 给 answer-token latent 加离散分离约束；不要只用 MSE/cosine 靠近答案 latent。
4. 加 direct structured baseline，确认任务本身不是训练预算问题。
5. 继续保留 no-evidence 和 shuffled-evidence 作为硬门禁；full 必须远高于二者才算潜空间推理成立。

## 产物

- `experiments/omni_transformer_stage_ac_latent_reasoning.py`
- `artifacts/omni_transformer_stage_ac_latent_reasoning/sweep_results.json`
- `artifacts/omni_transformer_stage_ac_latent_reasoning/sweep_runs/`
