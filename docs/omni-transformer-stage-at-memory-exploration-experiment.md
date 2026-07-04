# Stage AT：长程局部记忆和探索成本

## 目标

Stage AT 验证 `memory tree / work tree` 方向最小闭环：在局部地图、遮挡信息、不可重置探测成本和多目标任务下，先主动读取并写入 memory，是否能在同等预算下显著优于不保留长期记忆的策略。

本阶段不再把 memory 当作短离散标签，而是让 agent 在 rollout 中写入并复用局部地图：

1. 4 个遮挡 zone，每个 zone 隐藏一种颜色。
2. episode 要按目标序列收集 4 个颜色目标。
3. `SCAN_ZONE_i` 是不可重置探测成本，正式配置中每次 scan cost 为 3。
4. `WRITE_ZONE_i_color` 把 scan 结果写入 memory。
5. 后续目标只能通过已写 memory 降低重复扫描。
6. 额外评估错误 memory 注入，要求局部再探测后纠正。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_at_memory_exploration.py
```

模型输入是受控状态 token：

- 当前目标和 phase。
- 当前所在位置。
- 4 个 zone 的长期 memory。
- 当前 phase 的 seen mask。
- last scan 结果。
- 局部位置颜色。
- 最近动作历史。

模型输出动作：

- `SCAN_ZONE_0..3`
- `MOVE_ZONE_0..3`
- `WRITE_ZONE_i_color`
- `COLLECT_TARGET`
- `FINAL_REPORT`

正式策略采用“首轮主动建局部地图”：第一阶段先 scan + write 全部遮挡区，后续 4 个目标只读 memory 并移动/收集。这个设计是本轮 probe 后的修正：早期“需要时再 scan”虽然能学会动作，但会在部分目标顺序下把最后一个隐藏区拖到末尾，导致预算边界不稳定。

## 对照

正式 no-memory 对照不是弱模型，而是更强的 oracle no-memory policy：

- 不允许保留跨 phase memory。
- 允许理性使用当前 phase 的 scan 结果。
- 每个目标阶段都要重新探测，避免把失败归因到“模型不会用无记忆状态”。

脚本同时记录 `no_memory_model` 作为诊断项，但正式门槛以 oracle no-memory 为准。

## Probe 调整记录

初版 3 目标任务差距不足，no-memory oracle 在宽预算下也能完成。随后改为 4 目标全收集任务，让 memory 最多扫描 4 次，而 no-memory 平均需要重复扫描 10 次。

第二个问题是预算门槛过紧时，full memory 策略会被目标顺序和延迟 scan 放大失败。最终修正为首轮主动建图，并把 scan cost 固定为 3、rollout budget 固定为 45。

最终 probe：

- full episode success：100.00%
- no-memory oracle episode success：0.00%
- full mean scans：4.00
- no-memory oracle mean scans：10.00
- corrupt-memory correction rate：100.00%

## 正式 3 seed 结果

正式结果：

- 聚合结果：`artifacts/omni_transformer_stage_at_memory_exploration/formal_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_at_memory_exploration/formal_runs/seed20260701.json`、`seed20260702.json`、`seed20260703.json`
- 样例地图和 trace：`artifacts/omni_transformer_stage_at_memory_exploration/formal_runs/sample_traces/`

| 指标 | 3 seed 平均 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| action accuracy | 99.98% | 诊断项 | 通过 |
| episode success | 100.00% | >= 85% | 通过 |
| no-memory oracle episode success | 0.00% | 明显低于 full | 通过 |
| success gap | 100.00 pp | >= 30 pp | 通过 |
| mean scans | 4.00 | 低于 no-memory | 通过 |
| no-memory oracle mean scans | 10.00 | 对照 | 通过 |
| scan reduction | 6.00 | > 0 | 通过 |
| mean cost | 38.00 | 低于 no-memory | 通过 |
| no-memory oracle mean cost | 47.96 | 对照 | 通过 |
| cost reduction | 9.96 | > 0 | 通过 |
| corrupt-memory success | 99.93% | >= 85% | 通过 |
| corrupt-memory correction rate | 99.80% | >= 85% | 通过 |

## 结论

Stage AT 已经证明：在一个受控长程局部记忆任务中，主动读取并写入 memory 能把多目标 episode 的重复探测成本从 10 次 scan 降到 4 次 scan，并在固定预算下从 no-memory oracle 的 0% success 提升到 100% success。

错误 memory 注入后，模型仍能通过局部再探测修正，3 seed 平均 correction rate 为 99.80%。这说明本轮不是只证明“记住正确标签”，而是覆盖了错误记忆检测/修正的最小闭环。

## 边界和担忧

1. 地图是受控抽象 grid/zone，不是真实导航、真实 UI 或物理 embodied 环境。
2. 视觉是生成 PNG 样例和符号化局部状态，不证明真实图像定位或 SLAM。
3. memory 写入是固定 schema 的 4-slot 局部地图，不证明开放式 memory tree 自动扩展。
4. no-memory oracle 是强对照，但仍在同一受控环境里，不代表所有无记忆 agent 都会同样失败。
5. 当前任务要求首轮主动建图，证明的是“探索成本 amortization”；后续如果要更接近真实 agent，应加入动态地图、错误工具返回、分支任务和更长 horizon。

## 正式复现命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_at_memory_exploration.py --sweep --seeds 20260701,20260702,20260703 --train-size 2048 --val-size 512 --test-size 512 --batch-size 128 --d-model 96 --heads 4 --layers 2 --latent-tokens 6 --steps 600 --eval-every 200 --sample-count 10 --output-dir artifacts\omni_transformer_stage_at_memory_exploration\formal_runs --aggregate artifacts\omni_transformer_stage_at_memory_exploration\formal_results.json
```
