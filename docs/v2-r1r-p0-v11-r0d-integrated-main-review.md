# V2-R1R P0-D v11 R0D-integrated 主设计层验收

日期：2026-08-02  
formal artifact：`artifacts/v2-r1r/p0d-v11-r0d-integrated-20260802-1/`  
机器状态：`PASS_R0D_INTEGRATED`  
主设计层判定：**有限 `R0D integrated measurement-system accepted`**

## 1. 核心判定

v11 的唯一 formal attempt 真实完成，机器 G01–G08 `8/8`、F401–F420 `20/20`、raw metric kill `19/19`、positive metamorphic `4/4`、artifact-internal replay 与 import boundary 全部通过。主设计层独立复算 fixed root、bundle seal、attempt 时序、所有 ledger、public API 正控与三项 registry 外探针，没有发现 v8 式 false negative。因此接受 R0D 在冻结 12-case、规范化 source 不超过 400 characters 的有限 profile 内完成了 R0A–R0C 测量部件组合。

这个结论关闭的是 integrated measurement-system qualification，不是 production P0-D，更不是模型或架构通过。qualification surface 明确不是完整 ERE AST 的充分 renderer；word/char learner 的低准确率只能说明当前有限 surface 没被已测捷径利用，不能说明一个可解 production task 已经无捷径，也不能说明 latent core 会学习到语义机制。

## 2. 独立复算

预测试和 formal 前 guard 结果为：三棵 prior artifact 共 103 个 sealed files 全树复算；active `common/audit/learner` 与 v10 source snapshot逐字节一致；pytest `33/33`、compileall、CLI help、`git diff --check` 与完整 preflight 均通过。preflight 和 formal 都得到 positive `8/8`、fault `20/20`、metric kill `19/19`、metamorphic `4/4`、replay `1/1`。

formal root 精确十项，`attempt.json` 早于 run completion，assessment 为 `PASS_R0D_INTEGRATED` 且 `authorization_created=false`。`integration-bundle/input-seal.json` 对 135 个非自身文件的 map 与逐文件 SHA-256 完全一致；public `audit_integrated_bundle` 对 sealed bundle 重新返回 8 Gate 全真、0 failures。

root `evidence-seal.json` 的 SHA-256 为 `1C4AD436CECA3F96F4286A2E5A62D272782E7EF02DF5A52BBD5C6493A43408CF`。按冻结 runner 口径，它直接列出并命中 142 个非 `evidence-seal.json` 文件；全树另外两个同名文件位于 embedded v10 source snapshot 中。二者都由 bundle input seal 直接哈希，而该 input seal 本身由 root seal 哈希，所以传递覆盖为 144/144。这个“按 basename 排除所有 nested seal”的口径沿用了 v9/v10，实现上容易让独立 reviewer 误判；未来 sealer 应显式写成“只排除当前 root seal”或在 schema 中声明传递覆盖，但不能修改或重跑 v11 artifact。

## 3. 跨组件证据

12 个 integration case 同时经过 accepted simulator、public answer-label binding、typed invariant derivation、toy parser 和真实 learner：

- ERE 为三组 open/closed 单叶 counterfactual，fresh AST 只差 flag mode；typed provenance 6/6，answer flip 3/3。
- CPS 从 accepted rich AST 固定选择 winner、budget-invalid、final-constraint-invalid，并覆盖六种 candidate permutation；四类 composition witness 与 winner equivariance 均为 6/6。
- label 和 correct position 在 ERE/CPS 内各为 `2/2/2`；所有 deterministic heuristic 不超过 `1/2`。
- word CV、char CV 都为 `1/12`；word/char heldout 都为 `0/6`。fit/eval group 严格隔离，learner text逐字节来自 model view。

F409、F411、F416、F417 的多 Gate 失败集合与冻结预期完全一致，说明 surface/semantic/shortcut 间的耦合被显式测量，而不是被强行伪装成单 Gate 正交性。20 个 fault 的 killed metric union 精确覆盖全部 19 项 raw metric。

## 4. Registry 外探针

主设计层在 sealed bundle 的临时副本上使用独立 sealer 运行三项未登记输入；formal bytes 未改：

| 探针 | 结果 | 主要 false Gate |
|---|---|---|
| 在 source 中加入 parser 未声明的 `Answer: L0` | 拒绝 | G03–G08 |
| 在 `model_view` 注入 hidden `answer` 字段 | 拒绝 | 仅 G02 |
| 把规范化 source 延长到 400 characters 以上 | 拒绝 | G02、G08 |

三项均 fail-closed。第二项尤其证明 exact model-view boundary 能隔离 hidden answer；第三项确认长度边界是可执行 Gate，而不是文档注释。

## 5. 重要限制与下一步条件

设计阶段曾让较长 CPS surface 直接经过 v10 char 3–5 gram learner，exact `Fraction` score 在 Python 默认 4300 位整数转字符串保护处失败。v11 没有修改 accepted learner或关闭保护，而是把 qualification profile 收缩到规范化 source `<=400`。这让 R0D 可以诚实关闭有限组合资格，但同时留下两个不能跳过的 production entry condition：

1. production renderer 必须提供语义充分、source-only、可审计的 ERE/CPS 文本；不能沿用当前不充分 qualification surface，也不能用 AST/label adapter 偷渡真值。
2. 如果真实 source 超过 400 characters，必须先另立 statistical learner scaling 合同，采用可封印且数值稳定的 scoring/report 方案，并重新验证与 R0D 的接口；不得把缩短文本当长期解决方案。

因此 v11 通过不自动授权 generator 或训练。合理的下一动作是设计 R1 generator smoke 合同，并先用手写长度/语义充分性 probe 判断是否需要一个独立 scaling qualification。只有新的合同明确 task、renderer、split/pair、learner scaling、Gate 与停止规则后，才能运行 R1。P0-M、模型、cache、GPU 和训练仍未授权。

## 6. 完成与未完成

| 项目 | 状态 |
|---|---|
| v11 direct switch、12-case pack、integration runtime | 已完成 |
| G01–G08、20 fault、19 metric kill、4 metamorphic | 已完成并通过 |
| import boundary、artifact snapshot replay、single-use formal | 已完成并通过 |
| fixed root 与 bundle/root seal 主审 | 已完成；144/144 直接或传递覆盖 |
| registry 外边界探针 | 3/3 拒绝 |
| R0D 有限测量系统资格 | accepted |
| production renderer、learner scaling、R1 generator | 未设计完成、未授权 |
| P0-M、模型、训练、架构结论 | 未运行、未授权 |
