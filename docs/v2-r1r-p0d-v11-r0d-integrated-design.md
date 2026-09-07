# V2-R1R P0-D v11 R0D-integrated 冻结设计

日期：2026-08-02  
阶段目标：验证已接受的 R0A simulator、R0B invariant audit 与 R0C statistical learner 在同一批有限 reference cases 上能否形成可封印的测量链  
证据等级：有限 integrated measurement-system qualification；不是 production data、模型、训练或完整架构证据

## 1. 核心判断

R0D 不再分别证明三个部件“各自能跑”，而是证明同一 case identity 能沿三条彼此受限的路径闭合：手写语义记录由 accepted simulator 产生真值；严格 `model_view` 只暴露 qualification surface；公开 choice 或 candidate index 把 fresh answer 绑定到 label；accepted deterministic heuristics 与真实 grouped learner 只消费同一份 model-view source。任何一步都不得读取 hidden answer、AST、teacher output、certificate、split role 或 fault expected value。

v11 只新增 `integration/` 编排层，不修改、不复制实现 v7 `common/`、v9 `audit/` 或 v10 `learner/` 的算法。它直接复算 v7/v9/v10 三棵 prior formal evidence tree，并在 bundle 内嵌 v9 control bundle 与 v10 qualification bundle后重新运行两套 public audit。不存在旧 CLI alias、compatibility wrapper、hidden label adapter、reference generator、随机采样或训练路径。

R0D 的通过只能说明这套有限测量链在冻结 profile 上可组合、可拒绝已知跨层错误、可搬迁重放。它不能说明 qualification surface 是 production renderer，也不能说明真实任务可学、架构成立或模型会组合泛化。

## 2. 显式接口与已知长度边界

v10 toy parser 无法无损表达完整 ERE AST。v11 因而明确拒绝伪造“自然语言到 AST 的可逆转换器”，并冻结两个不同职责：

- `semantic projection`：只从 v9 hand-authored record 选择固定 AST；ERE 不做变换，CPS 只从 `cps-rich` 固定选择 candidate `0/4/5` 并应用六个显式排列。它不读取 model view 或 label。
- `qualification surface`：是同一 case 的受限模型视图，用来验证公开 answer-label binding、surface heuristic 与 learned shortcut audit。它不是 AST 的充分序列化，也不宣称是 production task text。

`model_view` exact schema 只有 `family`、`source_text`、`valid_choice_mask`。case 外层可保存 simulator source id、expected digest、pair/group/split 与 golden label，但任何 learner row 只能由 `row_id/group_id/text/label/valid_choice_mask` 构造，其中 `text` 必须逐字节来自 model view。

设计探针发现：v10 character 3–5 gram learner 对较长文本使用 exact `Fraction` 后，会在 Python 默认 4300 位整数转字符串保护处失败。R0D 不修改 accepted learner，也不关闭保护；qualification surface 的规范化长度因此冻结为每例不超过 400 characters，并同时运行 word 与 char 路径。这个边界是 R0D 的有效域，不得外推为生产长度能力；生产尺度若继续，需要另立 learner-scaling 合同。

## 3. 冻结 reference pack

`integration-cases.json` 精确包含 12 个 case、6 个 paired groups、3 个 label：

- 6 个 ERE：三组 `open/closed` 单叶 counterfactual。每组 choice mapping 相同，三组循环映射使 ERE label 为 `L0/L1/L2` 各 2 例；fresh AST 差异必须仅为 `/initial_state/attributes/flag/mode`。
- 6 个 CPS：从同一 accepted rich AST 选择 winner、budget-invalid、final-constraint-invalid 三个 candidate，并覆盖全部六种排列。winner position 为 `L0/L1/L2` 各 2 例，winning plan 始终为 `unlock, charge, finish`。
- train 与 heldout 各 3 个完整 group、6 个 row，label 均为 `2/2/2`；同一 pair 不跨 split。grouped CV 使用全部 6 groups，固定 5 folds。

每个 fresh simulator output 的 SHA-256、ERE typed provenance class、CPS 四个 derived composition witness、deterministic heuristic projection，以及 word/char CV 和 heldout report digest都由主设计层冻结。运行时只能比较，不得回写 golden。

## 4. 八 Gate

| Gate | 责任 | 冻结条件 |
|---|---|---|
| G01 | 双向 reference/fault matrix | metric registry、F401–F420 target binding 与非递归 completed-matrix fixture 精确 |
| G02 | 输入与边界完整性 | outer manifest/input seal、三棵 prior evidence、v9/v10 embedded bundle public audit、model-view exact schema 全真 |
| G03 | 语义与 label binding | 12/12 fresh simulator digest、12/12 public answer-to-label binding |
| G04 | split/pair/surface interface | 六组 pair 不拆分；train/heldout group 不重叠；12/12 parser 与 CPS candidate/action/cost projection 精确 |
| G05 | ERE 机制 | 6/6 typed provenance 与 trace binding；3/3 single-leaf open/closed answer flip |
| G06 | CPS 机制 | 6/6 四类 composition witness；6/6 candidate-order equivariant winner/failure profile |
| G07 | 确定性 surface shortcut | ERE/CPS label 和 winner position 平衡；13 类 heuristic golden 精确；任一 accuracy 不超过 `1/2` |
| G08 | learned shortcut | word/char grouped CV 与 train-heldout digest/confusion 精确；任一路径 accuracy 不超过 `1/2`；learner source boundary 全真 |

正控的实际统计值冻结为：word CV `1/12`、char CV `1/12`、word heldout `0/6`、char heldout `0/6`。这些低值只证明当前有限 surface 没有被已测捷径利用，不代表任务难度或架构能力。

## 5. F401–F420 与跨 Gate 因果

20 个 fault group 使用独立 bundle copy。除 F402 故意破坏 input seal 外，其余均由 stdlib-only sealer 重新封印；public audit 的 actual false Gate set 与 killed metric set必须和冻结表完全相等。group 责任沿用 G01、G02×3、G03×2、G04×3、G05×2、G06×3、G07×4、G08×2。

R0D 不要求错误只击中一个 Gate。以下多 Gate 结果是设计的一部分，而不是不正交噪声：

- F409 把一个 visible candidate cost 从 7 改为 6：G04 surface/AST projection 与 G08 fitted-state golden 同时失败。
- F411 把 closed counterfactual坍缩为 open：G05 necessity、G07 shortcut ceiling 与 G08 learned results 同时失败。
- F416 伪造一个 CPS label：G03 binding、G06 equivariance、G07 balance 与 G08 learner result 同时失败。
- F417 让 ERE first-rule literal 直接对应答案：G07 deterministic ceiling 与 G08 learned shortcut 同时失败。

所有 19 个 raw metric 至少被一个 fault 杀死；覆盖率必须为 `19/19`。

## 6. Metamorphic、import 与 artifact replay

四个 positive metamorphic 为 relocation、case-order reversal、JSON formatting 和 CRLF/LF source normalization。四者都必须保持 audit report canonical bytes 完全相同且不修改输入。

`integration/` 只能 import accepted `common/audit/learner` 与标准库，不能 import tests。materializer、sealer、fault matrix 与 metamorphic harness不得 import runtime implementation。formal bundle 必须内嵌完整 source snapshot；隔离 Python 子进程从搬迁后的 artifact snapshot 导入 runtime，输出必须与 live positive report逐字节相同。

## 7. Formal 与 PASS conjunction

- contract version：`r1r-p0d-v11-r0d-integrated`
- fixed formal root：`artifacts/v2-r1r/p0d-v11-r0d-integrated-20260802-1/`
- PASS status：`PASS_R0D_INTEGRATED`
- public API：`audit_integrated_bundle(path)`
- CLI：`audit-integrated`、`preflight-integrated`、`seal-integrated`

只有 G01–G08 `8/8`、fault `20/20`、metric kill `19/19`、metamorphic `4/4`、import boundary、artifact-internal replay、十项 formal root 与递归 evidence seal 全部成立，机器才能输出 PASS。formal root 先创建 `attempt.json` 再做任何 evaluation；root 已存在即 BLOCKED，禁止覆盖或 suffix 重跑。

无论 PASS 或 FAIL，本合同都不创建 production generator、R1–R3、P0-M、模型、cache、GPU 或训练授权。任何 Gate 失败立即停止并分析，不进入后续阶段。
