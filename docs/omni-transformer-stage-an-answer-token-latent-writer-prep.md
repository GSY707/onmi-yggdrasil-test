# Stage AN：统一 bus 接 answer-token latent writer 训练前准备

## 目标

Stage AN 复用 Stage AM 的四任务统一 latent bus：

- `color_at_cell`
- `shape_at_cell`
- `count_color_shape`
- `relation_yes_no`

这轮不再把分类 answer head 当作主证据，而是把输出升级为 answer-token writer。核心问题是：`cell/count/pair` 统一 bus 是否能支撑文本 token 级答案输出，而不是只支撑一个分类头。

## 已完成的训练前准备

1. `experiments/omni_transformer_stage_ak_unified_latent_bus.py` 已加入 answer-token decoder。
2. 训练 loss 已同时包含 teacher/model 两条 token writer 路径。
3. 评估 JSON 已输出：
   - `full_model_answer_token_accuracy`
   - `full_model_answer_sequence_exact`
   - `full_model_answer_word_exact`
   - `full_answer_*_sequence_exact`
   - `no_evidence_model_answer_sequence_exact`
   - `no_evidence_model_answer_word_exact`
4. `--answer-len` 与 `--token-loss-weight` 已暴露为 CLI 参数。
5. 已完成 smoke，产物为：
   - `artifacts/omni_transformer_stage_an_answer_token_writer/smoke_runs/seed20260701.json`
   - `artifacts/omni_transformer_stage_an_answer_token_writer/smoke_results.json`

smoke 只验证脚本、指标和聚合 JSON 能运行，不代表训练效果。

## 正式训练结果

正式 3 seed 已完成：

- 聚合结果：`artifacts/omni_transformer_stage_an_answer_token_writer/formal_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_an_answer_token_writer/formal_runs/seed20260701.json`、`seed20260702.json`、`seed20260703.json`

| 指标 | 3 seed 平均 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| `full_model_answer_sequence_exact` | 98.11% | >= 95% | 通过 |
| `full_model_answer_word_exact` | 98.11% | 接近 sequence exact | 通过 |
| `full_model_answer_token_accuracy` | 99.37% | 诊断项 | 正常 |
| `full_answer_relation_yes_no_sequence_exact` | 92.45% | >= 90% | 通过，但仍是最弱任务 |
| `no_evidence_model_answer_sequence_exact` | 27.93% | 明显低于 full | 通过 |
| `full_count_table_exact` | 100.00% | 维持高保真 | 通过 |
| `full_model_compare_accuracy` | 93.23% | relation 不应明显低于 Stage AM | 基本持平 |

按任务族看，`color_at_cell`、`shape_at_cell`、`count_color_shape` 的 answer-token sequence exact 都是 100%；`relation_yes_no` 平均 92.45%，是主瓶颈。`no_evidence` 下整体 answer sequence exact 降到 27.93%，其中 relation no-evidence 仍有 51.82%，符合 yes/no 二分类任务更容易靠题型和先验猜中的风险。

结论：Stage AN 通过原定门槛。当前统一 `cell/count/pair` latent bus 不只支撑分类 answer head，也能支撑 answer-token writer 输出；但 relation 仍没有回到 Stage AL 的 99%+，下一步应继续做 Stage AQ 的 relation delta/truth-table/process supervision，而不是回退到分类头。

## 正式训练门禁

正式 3 seed 训练完成后，主要看这些指标：

| 指标 | 通过门槛 |
| --- | ---: |
| `full_model_answer_sequence_exact` | >= 95% |
| `full_answer_relation_yes_no_sequence_exact` | >= 90% |
| `full_model_answer_word_exact` | 应接近 sequence exact |
| `no_evidence_model_answer_word_exact` | 应明显低于 full，最好接近随机 |
| `full_count_table_exact` | 应维持 Stage AM 的高保真 |
| `full_model_compare_accuracy` | relation 不应低于 Stage AM 太多 |

分类头指标仍保留为诊断项。如果分类头高、token writer 低，这一轮应优先修输出 token 分离、answer-token 对齐或 token writer loss；不要回退到把分类头当主结论。

## 正式训练开始命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701,20260702,20260703 --families color_at_cell,shape_at_cell,count_color_shape,relation_yes_no --train-size 2048 --val-size 512 --test-size 512 --batch-size 256 --d-model 96 --layers 2 --heads 4 --steps 2400 --eval-every 600 --answer-len 6 --token-loss-weight 1.0 --output-dir artifacts\omni_transformer_stage_an_answer_token_writer\formal_runs --aggregate artifacts\omni_transformer_stage_an_answer_token_writer\formal_results.json
```

如果显存紧张，先只把 `--batch-size` 降到 `128`，不要先降模型维度或步数。降 batch 会改变吞吐，不改变本轮架构口径。

## 训练中检查

训练开始后应检查：

1. 控制台是否每个 seed 都写出 `test` 和 `cost`。
2. `formal_runs/seed*.json` 是否逐个落盘。
3. `formal_results.json` 的 `summary.full_model_answer_sequence_exact` 是否接近分类头表现。
4. `summary.no_evidence_model_answer_word_exact` 是否明显低于 `summary.full_model_answer_word_exact`。

## 担忧和不确定点

1. 当前 token writer 是从统一 bus 的 question/cell/count/relation context 生成 token，不是完整 Stage AC 的冻结 text codec。它足以测试 token 输出门禁，但还不是未来正式语言输出专家的完整形态。
2. `answer_word_exact` 与 `sequence_exact` 在正式结果中相同，说明这轮没有被 BOS/EOS/PAD 细节拖垮；但未来多 token 文本答案仍要继续单独看 sequence exact。
3. relation 仍是最弱任务：正式均值 92.45%，最低 seed 为 89.06%，没有完全稳定压过 90%。Stage AQ 仍有必要。
4. no-evidence relation 仍有 51.82%，主要来自 yes/no 二分类先验；后续 relation 任务应加入 harder negatives 或更丰富答案形式，避免二分类随机上限过高。
5. 当前脚本还没有像 Stage AJ 那样的磁盘级 checkpoint/resume。正式命令单 seed 约 4 分钟，暂时可接受；如果后续把 steps 或模型规模继续拉大，应先补 checkpoint/resume 再跑。
