# Stage AK：统一潜空间硬化实验

## 目的

这轮不训练最终 answer-token reasoner，而是先验证“潜空间是否成型”。目标是把之前混在一起的问题拆开：

- object/pair latent bus 是否能保真承载外部对象事实。
- text query 是否能在同一 latent bus 中检索正确对象。
- 选中对象后，relation compare 是否能直接使用这些 latent。
- no-evidence 下是否会退化，避免把问题侧捷径误判成潜空间能力。

## 实现

新增 `experiments/omni_transformer_stage_ak_unified_latent_bus.py`。

核心设计：

- 复用 Stage AC 的合成 scene/question 数据。
- evidence encoder 输出固定 16 个 color-shape pair slots。
- 每个 slot 使用统一 schema token，监督：
  - occupied
  - row
  - col
  - color
  - shape
- question encoder 输出：
  - left object query
  - right object query
  - relation op
- query 直接在 pair slots 上做 retrieval。
- compare head 只读 selected left slot、selected right slot 和 relation op，不接最终文本 answer decoder。

这相当于先训练“共同对象语言”，再谈多步 reasoner。

## 运行命令

### Smoke

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701 --train-size 64 --val-size 32 --test-size 32 --batch-size 16 --d-model 32 --layers 1 --heads 4 --steps 2 --eval-every 1 --output-dir artifacts\omni_transformer_stage_ak_unified_latent_bus\smoke_runs --aggregate artifacts\omni_transformer_stage_ak_unified_latent_bus\smoke_results.json
```

### Relation-only unified bus

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ak_unified_latent_bus.py --sweep --seeds 20260701 --train-size 2048 --val-size 512 --test-size 512 --batch-size 64 --d-model 96 --layers 2 --heads 4 --steps 1600 --eval-every 400 --output-dir artifacts\omni_transformer_stage_ak_unified_latent_bus\relation_runs --aggregate artifacts\omni_transformer_stage_ak_unified_latent_bus\relation_results.json
```

## 结果

| 指标 | full | no evidence |
| --- | ---: | ---: |
| pair occupancy exact | 99.22% | 0.00% |
| pair row accuracy | 100.00% | 25.99% |
| pair col accuracy | 100.00% | 24.67% |
| left pair retrieval | 100.00% | 100.00% |
| right pair retrieval | 100.00% | 100.00% |
| relation op | 100.00% | 100.00% |
| teacher selected compare | 99.61% | 60.55% |
| model selected compare | 99.61% | 60.55% |

## 结论

1. 统一潜空间先行是正确方向。Stage AK 同时把 object slot readout、question-to-object retrieval、relation compare 打到 99%-100%。
2. 这说明 Stage AE-AH 的失败不是“relation 任务本身学不了”，而是之前让 text/evidence/reasoner 各自形成局部 latent 后再硬接，空间没有一开始统一。
3. no-evidence 下 row/col 回到约 25% 随机附近，compare 只有 60.55%，说明 full 的 99.61% 主要来自 evidence slots，而不是纯问题侧先验。
4. no-evidence 下 left/right retrieval 仍为 100%，这是合理的：问题文本本身包含 left/right object 的 color-shape 名称；但没有 evidence row/col，所以 compare 不能闭合。
5. 当前只覆盖 relation-only 的 pair/object bus，没有覆盖 cell lookup、count、最终 answer-token 写入和多轮 agent 控制。

## 对前三类问题的定位

- 潜空间没建起来：Stage AE-AH 很可能属于这一类。Stage AK 把统一 bus 前置后，query 和 compare 立刻闭合。
- 训练没练好：之前的 query alignment/process loss 是在未统一空间上补丁式训练，指导虽然局部正确，但偏离了最终架构。
- 架构错了：当前还不能判错。相反，Stage AK 支持“统一潜空间 + 主动读取 + compare”的架构分解。

## 下一步

1. 把 Stage AK 的 unified pair bus 接回 Stage AC/AE 的 answer-token reasoner，但保持 bus 冻结，先只训练 answer writer。
2. 扩展 unified bus：加入 cell slots、count slots 和 relation op/truth-table slots。
3. 加 scheduled sampling：先 teacher-selected slots，再逐步切到 model retrieval。
4. 多 seed 验证 relation bus 稳定性；当前是单 seed 方向验证。
