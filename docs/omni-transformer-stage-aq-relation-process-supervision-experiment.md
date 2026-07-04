# Stage AQ：relation delta / truth-table 过程监督实验

## 目的

Stage AN 已证明 Stage AM 的统一 `cell/count/pair` latent bus 可以接 answer-token writer，但 relation 仍是最弱任务：

- Stage AN `full_answer_relation_yes_no_sequence_exact`：92.45%
- Stage AN 最低 seed：89.06%
- Stage AN no-evidence relation：51.82%

Stage AQ 测试一个更具体的问题：relation 是否需要显式过程状态，而不是只靠 raw pair compare 或普通 MLP 自己学比较规则。

## 实现变化

`experiments/omni_transformer_stage_ak_unified_latent_bus.py` 在 Stage AN 基础上加入：

- `delta_row_head` / `delta_col_head`：从 relation state 预测左右对象的 row/col delta。
- `truth_table_head`：保留 learned truth-table head，作为失败对照和辅助监督。
- deterministic truth-table slot：由预测 delta 与 relation op 显式计算 yes/no truth-table。
- `--process-loss-weight`：控制 delta 与 learned truth-table 的过程监督。
- `--process-state-weight`、`--truth-table-state-weight`：控制过程状态和 truth-table slot 注入 relation state 的强度。

关键修正是 v3：把 deterministic truth-table state 的权重提高到 `4.0`。v1/v2 证明，仅让 truth-table 与普通 relation MLP 状态同权，answer writer 仍可能不稳定读取 truth-table。

## 正式命令

最终通过命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701,20260702,20260703 --families color_at_cell,shape_at_cell,count_color_shape,relation_yes_no --train-size 2048 --val-size 512 --test-size 512 --batch-size 256 --d-model 96 --layers 2 --heads 4 --steps 2400 --eval-every 600 --answer-len 6 --token-loss-weight 1.0 --process-loss-weight 1.0 --truth-table-state-weight 4.0 --output-dir artifacts\omni_transformer_stage_aq_relation_process_supervision\formal_runs_v3 --aggregate artifacts\omni_transformer_stage_aq_relation_process_supervision\formal_results_v3.json
```

## 结果

最终 v3 聚合结果：

| 指标 | 3 seed 平均 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| `full_model_answer_sequence_exact` | 99.93% | 诊断项 | 通过 |
| `full_answer_relation_yes_no_sequence_exact` | 99.74% | >= 98% | 通过 |
| `relation_delta_row_accuracy` | 99.74% | process probe 可读 | 通过 |
| `relation_delta_col_accuracy` | 98.70% | process probe 可读 | 通过 |
| `relation_truth_table_accuracy` | 100.00% | process probe 可读 | 通过 |
| `no_evidence_relation_truth_table_accuracy` | 50.52% | 不能同步升高 | 通过 |
| `no_evidence_model_answer_sequence_exact` | 26.69% | 明显低于 full | 通过 |
| `full_count_table_exact` | 100.00% | 不能破坏 count | 通过 |

每个 seed：

| seed | relation sequence exact | delta row | delta col | truth-table | no-evidence relation truth-table |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260701 | 100.00% | 100.00% | 97.66% | 100.00% | 57.03% |
| 20260702 | 100.00% | 100.00% | 100.00% | 100.00% | 48.44% |
| 20260703 | 99.22% | 99.22% | 98.44% | 100.00% | 46.09% |

## 失败/中间结果

v1：只加 learned delta/truth-table head 和 process loss。

- `answer_relation_yes_no_sequence_exact`：89.32%
- `relation_delta_row_accuracy` / `relation_delta_col_accuracy`：99.48% / 98.44%
- `relation_truth_table_accuracy`：89.84%

结论：delta 已经可读，但 learned truth-table head 没有稳定学会比较规则，最终 relation answer 反而低于 Stage AN。

v2：改成 deterministic truth-table slot，但 truth-table state 与普通 relation state 同权。

- `answer_relation_yes_no_sequence_exact`：96.61%
- `relation_truth_table_accuracy`：100.00%

结论：显式 truth-table 已经正确，但 answer writer 没有稳定读取这个 slot；这不是 process probe 不可读，而是输出路径权重不够硬。

v3：将 `--truth-table-state-weight` 提到 `4.0`。

- `answer_relation_yes_no_sequence_exact`：99.74%
- `relation_truth_table_accuracy`：100.00%
- `no_evidence_relation_truth_table_accuracy`：50.52%

结论：relation 需要显式 delta/truth-table 过程状态，而且 truth-table 必须作为强状态进入 answer writer；只靠普通 MLP 或弱注入都不稳定。

## 结论

1. Stage AQ 通过原定门槛，把 relation 从 Stage AN 的 92.45% 拉到 99.74%。
2. relation 的关键不是 pair slot 读不到：left/right query、delta row/col 已经能到 98%-100%。
3. 真正有效的是显式过程状态：delta -> relation op -> deterministic truth-table slot -> answer writer。
4. learned truth-table head 本身不是可靠修复；它在 v1 中只有 89.84%，说明把比较规则交给普通 MLP 仍然不稳。
5. no-evidence relation truth-table 约 50.52%，没有随 full 一起升高，说明这轮提升来自 evidence-dependent delta/truth-table，而不是 yes/no 先验。

## 担忧和下一步

1. 这轮任务仍是结构化 evidence，不证明真实视觉 relation；Stage AO 仍要把 pixel/object expert 接入 `cell/count/pair` slots。
2. deterministic truth-table 是明确结构偏置；这是架构进步还是任务规则硬编码，要在更复杂 relation 任务中继续验证。
3. 当前 yes/no 输出仍有 50% 随机上限；后续 relation 应加入多类 relation、hard negative、或输出具体 delta/position，降低二分类先验解释空间。
4. 如果后续把 AQ 扩到更长训练或更大模型，应先把 Stage AJ 的 checkpoint/resume 口径移植到该脚本。
