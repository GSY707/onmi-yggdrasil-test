# Stage AL：统一潜空间接入答案输出头实验

## 目的

Stage AK 证明统一 object/pair latent bus 可以把 relation 的对象检索和位置比较打到 99%-100%。Stage AL 继续推进一步：在同一条硬化潜空间上接一个最小 answer writer，验证 selected latent 能否直接输出 yes/no。

这轮仍不做完整文本 decoder，不训练多任务 reasoner，不加入 MoE。目标是看最小链路：

```text
evidence -> unified pair/object bus
question -> left/right query + relation op
selected slots + op -> compare
selected slots + op -> answer writer
```

## 实现

在 `experiments/omni_transformer_stage_ak_unified_latent_bus.py` 中新增：

- `answer_writer`
- `teacher_answer_accuracy`
- `model_answer_accuracy`
- `--answer-loss-weight`

answer writer 和 compare head 读同样的 selected left/right slots 与 relation op，但独立输出 yes/no。这样可以区分“compare 能算对”和“answer writer 能否使用同一 latent”。

## 运行命令

### Smoke

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701 --train-size 64 --val-size 32 --test-size 32 --batch-size 16 --d-model 32 --layers 1 --heads 4 --steps 2 --eval-every 1 --output-dir artifacts\omni_transformer_stage_al_unified_bus_answer\smoke_runs --aggregate artifacts\omni_transformer_stage_al_unified_bus_answer\smoke_results.json
```

### Relation-only answer writer

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 96 --layers 2 --heads 4 --steps 1600 --eval-every 400 --output-dir artifacts\omni_transformer_stage_al_unified_bus_answer\relation_answer_runs --aggregate artifacts\omni_transformer_stage_al_unified_bus_answer\relation_answer_results.json
```

## 结果

| 指标 | full | no evidence |
| --- | ---: | ---: |
| pair occupancy exact | 98.63% | 0.00% |
| pair row accuracy | 100.00% | 25.73% |
| pair col accuracy | 100.00% | 25.12% |
| left pair retrieval | 100.00% | 100.00% |
| right pair retrieval | 100.00% | 100.00% |
| relation op | 100.00% | 100.00% |
| teacher selected compare | 99.61% | 60.94% |
| model selected compare | 99.61% | 60.94% |
| teacher selected answer | 99.61% | 59.96% |
| model selected answer | 99.61% | 60.35% |

## 结论

1. 硬化后的统一潜空间可以接入答案输出头。`model_answer_accuracy=99.61%`，与 `model_compare_accuracy=99.61%` 对齐。
2. 这进一步说明 Stage AE-AH 的问题不是 relation 架构天然不行，也不是 answer writer 天然不能用 latent；关键是先前潜空间没有统一，训练目标在局部正确但整体偏离架构。
3. no-evidence answer 只有 60.35%，说明 full 的 99.61% 不是纯问题侧捷径。问题文本能给出对象名和 relation op，但没有 evidence row/col 时无法可靠判断关系。
4. 当前答案头只是 yes/no 二分类，还不是完整 answer-token decoder；但它已经证明“硬潜空间 -> 检索 -> 比较 -> 输出”的最小链路闭合。

## 下一步

1. 把这个 answer writer 扩展为 Stage AC 的 answer-token latent 写入，而不是二分类头。
2. 增加 color/shape/cell/count slots，做四任务统一 bus。
3. 冻结 unified bus，只训练 answer-token writer，验证问题是否转移到输出层。
4. 多 seed 验证；当前仍是单 seed 方向验证。
