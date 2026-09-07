# V2-R1R R1E v12：production entry qualification 冻结设计

日期：2026-08-09

合同版本：`r1r-r1e-v12-entry-qualification`

本阶段只关闭 R1 generator 的两个入口条件：一套语义充分、source-only、可逆且可审计的 ERE/CPS production renderer，以及一套能在真实长度上稳定运行的 source-only shortcut learner。它不生成正式 P0 数据，不启动 P0-M、模型、cache、GPU 或训练，也不产生架构结论。

## 1. 决策与直接切换

v11 已接受的 R0A–R0D 测量链保持只读；其 qualification surface 不再扩写为 production renderer，v10 learner 也不修补。v12 新增独立 `production/` 层，并把活动 CLI/test 直接切换到 v12。v11 fixed root、文档和 source snapshot 永久保留；活动 `tests/v2_r1r_v11/` 只有在 v12 预测试通过、formal 准备就绪后删除，不能保留并行活动合同。

production AST 使用 accepted `common.simulator` 的完整 ERE/CPS schema。renderer 支持 ERE 七种 primitive、两种 predicate、CPS 五种 condition、六种 effect、空集合和嵌套 ERE primitive。production identifier 限定为 ASCII atom `[A-Za-z][A-Za-z0-9_-]*`；这是生成域，不是假装覆盖 simulator 接受的任意自由字符串。

## 2. Renderer 合同

### 2.1 输入与输出

公共 API 只接受：

- `family`：`ERE` 或 `CPS`；
- `program_ast`：accepted simulator AST；
- `label_mapping`：局部标签 `A`–`I` 到语义 choice 的双射；
- `template_id`：两个 train grammar 或一个 language-OOD grammar。

输出是单个自然语言 `source_text`。解析 API 只接受 `source_text`，返回 `family`、完整 AST、choice mapping 和 template family；不得读取 record、hidden AST、answer、teacher、certificate 或文件旁路。

ERE choice 只允许 `value:<atom>`、`true`、`false`；CPS choice 只允许 `candidate:<zero-based-index>`、`none`。正确 choice 由 simulator 输出和 mapping 在 renderer 外计算，renderer 不接收 `answer_index`。

### 2.2 三个 grammar

- `plain_v1`：定义—世界—序列—问题的主动语态；
- `reordered_v1`：世界—定义—问题—序列的重排主动语态；
- `indirect_v1`：独立 section 名称、被动/后置条件表达和不同 opening；只用于 language OOD。

三个 grammar 可以共享 token quoting 与递归 expression grammar，但 section 顺序、opening、condition/effect phrasing 必须由模板表显式区分。不得在文本中写 `train`、`OOD`、split、seed、answer 或正确标签。

### 2.3 语义充分性

对 qualification matrix 的每个 case 和每个适用 grammar，必须同时满足：

1. `parse(render(ast, mapping))` 与 canonical AST/mapping 精确相等；
2. parsed AST 经 fresh simulator 的输出与原 AST 相同；
3. grammar 间 canonical parse 相同而 surface fingerprint 不同；
4. 删除或修改每一种已覆盖语义字段会导致 parse 拒绝或 canonical parse 改变；
5. 追加 `Answer:`、`Correct:`、未知 section、重复 section 或第十个标签时 fail-closed；
6. model view 只含 `example_id/source_text/reasoning_budget/valid_choice_mask`。

qualification matrix 至少覆盖 ERE 七 primitive、两 predicate、literal/argument/neighbor operand、嵌套 `IF`/`FOREACH_LINKED`、attribute/relation query，以及 CPS 五 condition、六 effect、budget、goal、final constraint、empty/non-empty state 和 `NONE` choice。

## 3. 长度与 learner scaling 合同

v12 使用手写 minimal/typical/maximal production envelopes，在三个 grammar 下记录 UTF-8 bytes、normalized characters 和 pinned Qwen tokenizer token count。只要任一有效 source 超过 400 normalized characters，scaling 分支即为强制；不能通过缩短、截断或减少语义字段规避。

新 scalable learner 复用 v10 的 tokenizer、fit counts、Laplace smoothing、mask 和 exact rational 排序语义，但不展开或把巨大 numerator/denominator 转成十进制字符串。每类 score 保存为整数因子—指数的 canonical 表达；比较先使用带保守误差带的 log score，落入误差带时才以整数乘积精确比较，结构完全相同则直接判 tie。报告只包含 prediction、score order、每类 exact rational expression 的定长 SHA-256 digest、used-feature digest/count 和运行摘要。v10 runtime 保持逐字节只读。

scaling qualification 必须满足：

- 在规范化长度 `<=400` 的 overlap fixture 上，word 与 char-3–5 prediction、tie-break 和 mask 与 v10 exact predictor逐条一致；
- 在 `512/1024/2048/4096/8192` character profiles 上，两次运行 prediction 与 compact report 逐字节相同；
- 九标签、部分 valid mask、未知 feature、重复 feature和零命中均有正控；
- public report 不含任意长度 numerator/denominator decimal string，单个 score digest 固定 64 hex；
- 8192-character char profile 在 CPU 上完成；wall time 只记录不作为脆弱 Gate，但 formal 进程总超时为 120 秒；
- v11 `<=400` integration fixture 通过 adapter 后 prediction 与旧接口一致。

## 4. v12 Gate

| Gate | 判定 |
| --- | --- |
| G01 | v7/v9/v10/v11 accepted roots、active common/audit/learner/integration 与 frozen digest 全部只读且匹配 |
| G02 | qualification matrix schema、identifier domain、choice domain 与 accepted simulator validation 全部通过 |
| G03 | 三 grammar 的 AST + choice exact roundtrip 与 fresh simulator output identity 为 100% |
| G04 | source-only/model-view boundary、禁止字段、答案注入和未知/重复结构全部 fail-closed |
| G05 | primitive/condition/effect/query/state/empty/nesting coverage 达到冻结全集；grammar surface 分离成立 |
| G06 | pinned Qwen tokenizer 无截断；长度 census 完整；`>400` 时 scaling branch 已执行而非规避 |
| G07 | scalable word/char learner 在 overlap 上与 v10 prediction/tie/mask 精确等价 |
| G08 | 512–8192 长度、九标签和 v11 adapter 的 compact deterministic report 全部通过 |

formal conjunction 是 G01–G08 全真、所有 negative controls 被杀死、所有 public metrics 均至少由一个 fault 定向杀死、artifact replay/read-only 和 evidence seal 全部通过。机器状态只能是 `PASS_R1E_ENTRY` 或 `FAIL_R1E_ENTRY`。

## 5. Artifact 与执行纪律

唯一 formal root 固定为：

`artifacts/v2-r1r/r1e-v12-entry-qualification-20260809-1/`

formal root 必须包含 attempt、qualification bundle、renderer/scaling report、fault/coverage/metamorphic/replay ledger、assessment、run metadata、source snapshot 和 root evidence seal。formal 命令在 root 已存在时拒绝执行；attempt 必须在任何正式计算前原子创建。

先运行任意次数的临时目录预测试；只有预测试、pytest、compileall、CLI help、prior seal 和 dirty-worktree scope audit 均通过，才运行唯一 fixed-root formal。formal 无论通过或失败均立即停止 v12，不能修改、覆盖或重跑。只有父任务完成主设计层独立验收并接受 `PASS_R1E_ENTRY`，才进入另立的 R1 generator smoke 合同。

## 6. 明确不属于本阶段

- production dataset 或完整 P0-D；
- generator 质量、split/causal pair、label balance 或 shortcut 结论；
- Qwen hidden cache、Boundary、recurrent core、baseline、GPU 或训练；
- P0-M、P1–P3、A1.22A、V2-B 或 V2-C；
- “latent architecture 成立”或“自然语言任务可学”的结论。
