# Stage AG：Teacher-forced multi-step query trace 实验

## 目的

Stage AF 说明 MoE gate 可以被监督到 100%，但没有修复 active-read reasoner 的核心失败点：query policy，尤其是 relation 任务中的左右对象查询。

Stage AG 暂停 MoE，改测更直接的训练信号：

- 保留 Stage AE 的 queryable reader。
- 把 relation 拆成多步 trace：query left object -> observe row/col -> query right object -> observe row/col -> compare relation op -> answer。
- 训练时启用 `--query-teacher-forcing train`，用真实 query target 读取 observation，让 answer writer 先学“读到正确信息后怎么写 answer-token latent”。
- 评估时同时报告自由查询 `full` 和诊断上限 `teacher_forced_queries`。

## 新增实现

`experiments/omni_transformer_stage_ac_latent_reasoning.py` 新增：

- `--reasoner-variant trace_multistep`
- `--query-teacher-forcing none|train|always`
- `teacher_forced_queries` 评估分支
- 显式 pair reader row/col observation 到 relation compare 的路径

这次没有继续推进 MoE，因为 Stage AF 已证明 gate/route 不是当前主瓶颈。

## 运行命令

### Smoke

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 64 --val-size 32 --test-size 32 --batch-size 16 --d-model 32 --layers 1 --heads 4 --latent-tokens 8 --text-steps 2 --evidence-steps 2 --reasoner-steps 2 --eval-every 1 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing train --output-dir artifacts\omni_transformer_stage_ag_teacher_forced_trace\smoke_runs --aggregate artifacts\omni_transformer_stage_ag_teacher_forced_trace\smoke_results.json
```

### Relation-only

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --families relation_yes_no --train-size 1024 --val-size 256 --test-size 256 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1000 --reasoner-steps 1500 --eval-every 500 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing train --output-dir artifacts\omni_transformer_stage_ag_teacher_forced_trace\relation_only_runs --aggregate artifacts\omni_transformer_stage_ag_teacher_forced_trace\relation_only_results.json
```

### 四任务

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ac_latent_reasoning.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 64 --layers 1 --heads 4 --latent-tokens 16 --text-steps 500 --evidence-steps 1200 --reasoner-steps 1800 --eval-every 600 --reasoner-trace-weight 1.0 --reasoner-reader-weight 0.5 --reasoner-variant trace_multistep --query-teacher-forcing train --output-dir artifacts\omni_transformer_stage_ag_teacher_forced_trace\all_trace_runs --aggregate artifacts\omni_transformer_stage_ag_teacher_forced_trace\all_trace_results.json
```

## 结果

| 设置 | full answer word | no evidence | shuffled evidence | teacher-forced queries | 关键 trace/reader |
| --- | ---: | ---: | ---: | ---: | --- |
| relation-only | 59.77% | 60.55% | 56.25% | 59.77% | left/right query 34.77%/44.92%，relation op 100%，pair row/col reader 97.31%/97.37% |
| 四任务 | 58.79% | 16.80% | 21.29% | 83.59% | cell 64.06%，count 37.50%，relation 52.34%，left/right query 35.94%/39.84%，pair row/col reader 96.94%/95.99% |

四任务按任务族：

| family | answer word exact |
| --- | ---: |
| color_at_cell | 71.09% |
| shape_at_cell | 70.31% |
| count_color_shape | 42.19% |
| relation_yes_no | 51.56% |

## 结论

1. `teacher_forced_queries` 在四任务上从 58.79% 提到 83.59%，说明“主动读取路径 + answer-token latent 写入”本身不是死路；如果 query 被给对，系统能明显更接近正确答案。
2. 自由查询仍低，left/right pair query 只有 35.94%/39.84%，所以 query policy 仍是主瓶颈之一。
3. relation-only 的 teacher-forced queries 没有打开上限，59.77% 与 full 完全相同，且 no-evidence 60.55% 更高。这是新的负发现：只把 pair query 强制正确还不足以让 relation compare 学会稳定使用 observation。
4. pair reader row/col 已到 95%-97%，relation-only 仍失败，说明问题已经不是“证据 latent 里没有 row/col”，而是中间比较监督和 answer latent 写入仍不够硬。
5. 继续做 MoE 没有意义；下一步应加强分步监督和离散化的中间目标，而不是增加专家路由。

## 担忧

- 当前 teacher forcing 只强制 query selection，不强制最终 compare 的符号规则；relation 任务仍可能退化到 yes/no 先验。
- `teacher_forced_queries` 是诊断上限，不是最终 runtime 能力；真实能力仍应看 `full`。
- 四任务 teacher-forced 大幅改善，可能主要来自 color/shape/count 的正确读取，不能误读成 relation 已解决。
- 当前只有 1 seed，足够做方向筛选，不够做稳定性结论。

## 下一步

1. 对 relation 加强中间监督：left row/col、right row/col、row delta、col delta、relation truth table 或 compare logits。
2. 对 query policy 加 contrastive/retrieval loss，不能只靠 CE。
3. 把 answer writer 分阶段训练：先用 teacher-forced observation 训练写 answer latent，再逐步 scheduled sampling 切回模型 query。
4. 加 direct structured baseline，确认 relation compare 的训练预算和标签分布不是隐藏瓶颈。
