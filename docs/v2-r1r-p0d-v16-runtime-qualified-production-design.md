# V2-R1R P0-D v16：稳定运行资格化与新 seed production

日期：2026-08-10

合同版本：`r1r-p0d-v16-runtime-qualified-production`

## 1. 核心判定与本轮目标

v15 的判别机制没有失败：合成 decision-power、14 项 fail-closed fault、3,264 个 exact 对照、tie fallback、single-bank streaming 与只读 replay 均已通过。唯一失败是 Q08 把一次墙钟相对比值 `4.7617x` 与正确性合并，并用 `5.0x` 单点阈值否决整轮。当前只读摸底在同一 192 条 fixture 上得到 15/15 次 fast 胜出、paired speedup 中位数约 `6.67x`、最差约 `6.37x`；完整 v14 规模 G09+G10 为约 159 秒、峰值工作集约 3.47 GB，且所有 cell 的 successes、total 与 prediction hash 保持一致。

因此 v16 不重新设计 G09，也不把 v15 事后改写为 PASS。它另立两个不可倒置的阶段：

1. **Q：runtime qualification**。将 v15 中已封存的判别力与 exact-equivalence 作为只读前置组件证据，并用稳健的成对重复计时、完整旧规模绝对预算、内存上限、封存预测投影一致性和 fault matrix 资格化当前运行路径。
2. **F：fresh-seed production formal**。只有 v16-Q 唯一 sealed assessment 为 PASS，才使用未生成、未预览结果的新 root seed `2026081602` 生成并验收 26,624 条 production 数据。

Q 不接受 v14 的 G09 统计结论；v14 仍是 G09 rejected diagnostic，只在 Q 中充当已封存的真实工作负载和 prediction oracle。F 不允许根据新 seed 的结果修改统计规则、文本模板或 Gate。

## 2. Q：稳定运行资格化

### 2.1 只读输入与证据继承

Q 必须先复算：

- v15 Q root `artifacts/v2-r1r/p0d-v15-g09-qualification-20260810-1/` 的完整 evidence seal，且 seal 文件 SHA-256 固定为 `997166EECC490F7B6AF6E80C01E28E187539BEDA351764597F0D79DE42C1EE38`；
- v15 assessment 必须精确为 Q01–Q07、Q09 true，Q08 false，整体 `FAIL_G09_QUALIFICATION`，且没有创建 v15 F root；
- decision-power、decision fault、registry、synthetic、tie、streaming 与 replay 各自为 true；旧 performance 必须 prediction identity 为 true、fallback 合格，并且只有单次 speedup 未到 `5.0x`；
- v14 root `artifacts/v2-r1r/p0d-v14-full-production-20260809-1/` 的完整 seal 仍可复算。

这些条件只继承通过的窄组件，不把 v15 整体提升为 PASS。任一条件不符，v16-Q fail closed。

### 2.2 固定成对 benchmark

fixture 固定使用 v14 sealed ERE 数据：train 抽样 512 条，每个 heldout split 抽样 32 条，共 192 个 `full_text_char_3_5_nb` prediction；抽样 seed 固定 `2026081601`。模型、analyzer、Laplace smoothing、label order、mask 和 exact fallback 语义不得改变。

先各 warm up 两次，再执行 15 个 paired repetition；偶数轮 exact→fast，奇数轮 fast→exact。计时区间关闭 Python GC，结束后恢复。每轮 exact 与 fast prediction 必须逐项一致。Gate 冻结为：

- 192/192 prediction identity，fast ordinary fallback rate `<= 0.01`；
- 15 轮 paired speedup 中位数 `>= 5.0x`；
- 至少 14/15 轮 fast 严格快于 exact；
- fast duration 的 nearest-rank p95 `<= 0.20s`。

不使用单轮 ratio、最慢轮 ratio 或事后选择的 repetition。相对 Gate 防算法退化，绝对 Gate 防 exact 路径同时变慢后仍伪装成 speedup。

### 2.3 完整旧规模运行资格

在同一正式 Q 进程内加载 v14 的 14,336 条数据，按 production 顺序运行完整 G09 和 G10。计时包括 dataset load、G09、G10，不包括 Python 模块 import；内存使用 Windows `GetProcessMemoryInfo` 的进程生命周期 `PeakWorkingSetSize`。

必须同时满足：

- G09 恰有 98 个 source cell 与 14 个 family aggregate；每个 cell/aggregate 的 successes、total、prediction SHA-256 和 passed 与 v14 sealed report 一致；已知失败仍只能是 ERE composition char 与 CPS horizon char 两格，14 个 aggregate 全通过；
- G10 恰有 24 个 claim cell；balance、successes、total、prediction SHA-256 和 passed 与 v14 sealed report 一致，整体通过；
- 两族三个 analyzer 的 sparse/dense equivalence 全通过；同一时刻 feature bank 数为 1；进度事件精确 12 个且顺序完整；batch prediction 数非零；
- load+G09+G10 总耗时 `<= 240s`；进程峰值工作集 `<= 4.5 GiB`。

完整路径 Gate 使用绝对预算，不要求 v14 的 G09 统计转为 PASS，也不比较失败原因字符串。字符串命名不是统计语义；封存投影才是 oracle。

### 2.4 Fault matrix 与 Q Gate

资格判定器必须用纯数据突变至少覆盖：v15 seal/gate 失真、prediction identity、fallback、paired median、fast win 数、p95、source cell 缺失/hash 改写、aggregate 改写、已知失败集改写、claim cell/hash/balance 改写、完整耗时、峰值内存、feature-bank 与 progress 拓扑。每个突变都必须使对应 Gate 失败；不能依赖登记表名称自动通过。

Q 的唯一 root 固定为：

`artifacts/v2-r1r/p0d-v16-runtime-qualification-20260810-1/`

| Gate | 正式判定 |
| --- | --- |
| Q01 | v15 窄组件证据、v15/v14 seal 与 v15-F 不存在性全部精确 |
| Q02 | 192 条 × 15 轮 prediction identity 与 fallback 条件通过 |
| Q03 | paired median、fast-win 与 fast p95 稳健性能条件通过 |
| Q04 | 完整 v14 G09 的 98/14 投影、已知失败集与 scorer equivalence 一致 |
| Q05 | 完整 v14 G10 的 24-cell、balance 与 prediction 投影一致 |
| Q06 | 完整路径总耗时与峰值工作集在冻结绝对预算内 |
| Q07 | single-bank、batch、12-event streaming 拓扑完整 |
| Q08 | runtime assessor fault matrix 全部命中 |
| Q09 | v14/v15/accepted inputs 只读、source snapshot、assessment 与 evidence seal 可复算 |

机器状态只能是 `PASS_RUNTIME_QUALIFICATION` 或 `FAIL_RUNTIME_QUALIFICATION`。root single-use；失败、异常、中断或不完整立即停止，不得创建 F root。

## 3. F：fresh-seed production formal

### 3.1 冻结数据与新鲜性

F root seed 固定 `2026081602`。preflight 只能使用独立 capacity seed `2026081691`；不得用 `2026081602` 调用 builder、生成样本、计算 fingerprint、渲染文本或预跑统计。代码中声明和测试 seed 常量不算预览结果。

每族固定 train 4,096；validation、composition OOD、length/horizon OOD、entity/distractor OOD、language OOD、causal_pairs 各 1,536。每族 13,312、总计 26,624 records；causal_pairs 为每族 768 对。

唯一 F root 固定为：

`artifacts/v2-r1r/p0d-v16-full-production-20260810-1/`

除版本、root seed、v16-Q 前置引用外，generator、renderer/parser、model-view、simulator、claim、fingerprint、permutation、tokenizer、长度上限和 G09/G10 决策规则继承 v15。不得加入兼容 alias、隐藏 salt、seed literal、split marker 或针对 v14 两个失败 cell 的模板修补。

### 3.2 F Gate

| Gate | 正式判定 |
| --- | --- |
| G01 | v7/v9/v10/v11/v12/v13 accepted seal、v16-Q PASS seal 与声明只读输入 byte identity 通过 |
| G02 | 14 文件、26,624 条、4,096/1,536 配额、v16 schema/version/model-view 与 root-to-record provenance 精确 |
| G03 | source roundtrip、fresh simulator、answer/label、teacher trace 与全部 claim replay 一致 |
| G04 | 全局 semantic/surface overlap 0；1,536 causal pair 单叶、共享表面因素与答案翻转 100% |
| G05 | ERE primitive/query/depth/composition/entity/length 与 3-attribute/6-value 域独立派生 |
| G06 | CPS unique optimum/NONE、suboptimal、hard negative、composition/horizon/distractor 配额独立派生 |
| G07 | 每 split 标签计数差 `<=1`；choice/candidate/action/definition 与 alpha/permutation 正控通过 |
| G08 | language template 分离、marker 禁止、pinned tokenizer 无截断，train/validation p99 `<=900` |
| G09 | 冻结 98/14-cell decision 全部通过；预测完整、fast/exact 与 streaming 资格引用有效 |
| G10 | claim truth、kind/pair 极性平衡与 24-cell simultaneous claim upper 全部通过 |
| G11 | full regeneration、artifact read-only replay、上游只读、source snapshot、进度 ledger 与 evidence seal 一致 |

机器状态只能是 `PASS_P0D_PRODUCTION` 或 `FAIL_P0D_PRODUCTION`。任一 Gate false、超时、异常或中断都结束唯一 F attempt；不得补写、换 seed 或换 suffix 重跑。

## 4. 执行顺序与证据边界

1. 冻结合同时允许在 v14 sealed input 和非正式 capacity seed 上反复做测试；
2. v16 tests/contract guard 全通过后，删除 v15 活动 tests/CLI 引用，保留 v15 artifact 与 source snapshot；
3. 唯一执行一次 v16-Q；Q FAIL 立即停止；
4. Q PASS 才唯一执行一次 v16-F；F 返回后立即停止写 formal artifact；
5. 父任务只读复算 seal、Gate、预测投影与新鲜性，再同步主审、结果、路线和目录索引。

本合同只资格化 production 数据和测量运行面。即使 F PASS，也不授权 P0-M、hidden cache、模型、optimizer、GPU、训练、Pareto、A1.22A、V2-B 或 `integrated-system` 表述。
