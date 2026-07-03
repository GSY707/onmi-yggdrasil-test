# Stage AD：只调潜变量推理专家的 readout/trace 实验

## 目的

Stage AC 证明了前两步可以高保真：问题/答案 token latent 可 100% 互译，外部事实表 latent 也能高精度还原。但普通 latent reasoner 只有 40.04% answer word exact，说明“互译能力”不是“潜空间推理能力”。

Stage AD 只调整第三步：潜变量推理专家。文本 codec 与 evidence codec 仍先按原目标训练；进入 reasoner 训练阶段后两者冻结，只更新 `LatentReasoner` 参数。

## 改动

新增 `--reasoner-variant readout`：

1. 推理专家从 question latent 产生操作/selector 表示。
2. 推理专家从冻结 evidence latent 中读出 cell/count readout。
3. 推理专家把读到的结果写回 answer-token latent，再交给冻结 answer decoder 输出文本。

新增两个只作用于 reasoner 的监督：

- `--reasoner-trace-weight`：监督 operation、target cell、target color/shape、count pair、count、relation 等中间 trace。
- `--reasoner-reader-weight`：监督 reasoner 自己的 evidence readout 能否从冻结 evidence latent 还原 occupancy/color/shape/count table。

这不是训练新的 evidence codec；reader 监督只训练推理专家内部读头，用来验证推理专家是否会读已经存在的潜空间证据。

## 运行命令

`color_at_cell` 聚焦诊断：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families color_at_cell --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant readout --output-dir artifacts\omni_transformer_stage_ad_reasoner_trace\color_readout_reader_runs --aggregate artifacts\omni_transformer_stage_ad_reasoner_trace\color_readout_reader_results.json
```

四任务同场测试：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant readout --output-dir artifacts\omni_transformer_stage_ad_reasoner_trace\all_readout_reader_runs --aggregate artifacts\omni_transformer_stage_ad_reasoner_trace\all_readout_reader_results.json
```

## 结果

### color_at_cell 聚焦诊断

| 指标 | 结果 |
| --- | ---: |
| Q/A text codec question/answer exact | 100.00% / 100.00% |
| Evidence codec occupancy/color/shape/count exact | 100.00% / 100.00% / 100.00% / 100.00% |
| Latent reasoner full answer word exact | 91.80% |
| Latent reasoner no evidence answer word exact | 29.30% |
| Latent reasoner shuffled evidence answer word exact | 27.34% |
| Reasoner trace color accuracy | 92.97% |
| Reasoner reader occupancy/color/shape/count | 100.00% / 100.00% / 100.00% / 100.00% |

对照上一轮只加普通 trace 的 `color_at_cell`：full answer word exact 为 39.45%，trace cell 已经 100%，但 trace color 只有 35.55%。Stage AD readout 把失败点从“不会把证据值搬到答案 latent”推进到“多数情况下可以读值并写回答案 token latent”。

### 四任务同场测试

| 模块 | 指标 | 结果 |
| --- | --- | ---: |
| Q/A text codec | question sequence exact | 100.00% |
| Q/A text codec | answer sequence exact | 100.00% |
| Evidence codec | grid occupancy exact | 100.00% |
| Evidence codec | occupied color accuracy | 100.00% |
| Evidence codec | occupied shape accuracy | 100.00% |
| Evidence codec | count table exact | 91.80% |
| Latent reasoner full | answer word exact | 74.41% |
| Latent reasoner no evidence | answer word exact | 28.71% |
| Latent reasoner shuffled evidence | answer word exact | 28.71% |
| Reasoner reader | occupancy/color/shape/count table | 100.00% / 100.00% / 100.00% / 75.20% |

按任务族：

| Family | answer word exact |
| --- | ---: |
| `color_at_cell` | 85.16% |
| `shape_at_cell` | 89.06% |
| `count_color_shape` | 65.62% |
| `relation_yes_no` | 57.81% |

trace 指标：

| Trace | Accuracy |
| --- | ---: |
| operation | 100.00% |
| target cell | 62.11% |
| target color | 82.03% |
| target shape | 89.06% |
| count pair | 100.00% |
| count | 64.84% |
| relation | 51.56% |

## 解释

Stage AD 的结论是正向但不完整：

1. **只调 latent reasoner 有明显收益。** 四任务从 Stage AC 的 40.04% 提到 74.41%；`color_at_cell` 从约 39.45% 提到 91.80%。
2. **收益来自证据使用，而不是答案先验。** 四任务 full 为 74.41%，no/shuffled 都只有 28.71%；`color_at_cell` full 为 91.80%，no/shuffled 约 27%-29%。
3. **推理专家需要自己的读证据结构。** 单纯 trace 监督能学会“问哪个格子”，但不能稳定读出格子里的值；加入 readout 后，reasoner reader 能从冻结 evidence latent 还原 occupancy/color/shape。
4. **还没有完成完整潜空间推理。** target cell trace 只有 62.11%，relation trace 只有 51.56%，relation answer 只有 57.81%。当前 readout 更像“可训练的潜空间值读取器”，不是完整的符号式中间推理链。

## 担忧

- target cell accuracy 不高，说明 readout 的可解释 selector 还没有严格对齐到真实 cell index；答案正确可能来自分布式值读取，而不一定是稳定的“先定位再取值”。
- relation 任务基本没被这版 readout 解决；它需要选出两个对象、恢复坐标并执行比较，而不只是读一个 cell 或 count pair。
- count 受两层限制：evidence codec count table 本身是 91.80%，reasoner reader count table 只有 75.20%，最终 count answer 65.62%。
- 这是 single seed synthetic 结构化事实表实验，不证明真实图像、OCR、UI 或开放工具环境。

## 下一步

1. 给 selector 建硬对齐：固定 cell/object token 坐标，或加入 target-cell attention/contrastive 监督，让 trace cell 接近 100%。
2. 给 relation 单独建对象对 readout：按 color+shape 选 left/right object，再读 row/col，再比较 relation。
3. 给 count 建 accumulator/计数 readout，而不是只把 count table 当普通分类头。
4. 稳定后再做多 seed；当前先保留 single seed，因为这一轮目标是验证 reasoner-only 方向是否有正信号。

## 产物

- `experiments/omni_transformer_stage_ac_latent_reasoning.py`
- `artifacts/omni_transformer_stage_ad_reasoner_trace/color_readout_reader_results.json`
- `artifacts/omni_transformer_stage_ad_reasoner_trace/color_readout_reader_runs/`
- `artifacts/omni_transformer_stage_ad_reasoner_trace/all_readout_reader_results.json`
- `artifacts/omni_transformer_stage_ad_reasoner_trace/all_readout_reader_runs/`
