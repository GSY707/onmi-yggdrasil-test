# V2-R1R P1-NR1：数值/关系测量系统资格合同

日期：2026-08-17  
合同地位：V8L 之后的新路线入口；不是 V8 修补、P1 v9 或 P1 完成证明

## 1. 核心判断与任务边界

V8L 已以 `FAIL_P1_V8L_BOOTSTRAP` 封存。它留下的有效证据是：fixed-anchor 训练通路在 ERE 上出现迁移方向，但 CPS 的 lexical metric teacher 没有预先证明单调性、加法一致性、比较反对称性和 oracle-state decision power；全局 effective rank 又被 ERE 主导，不能排除 CPS 数值坐标的低秩与方向冲突。因此，继续在同一匿名 K=8 加 lexical-anchor 组合上调整 exposure、loss、prompt 或阈值，不构成一条获准路线。

NR1 只回答一个更窄、但对后继架构必需的问题：

> 一个不读取任务标签、答案、model forward 或被测 state 来选择审计目标的 typed numeric/relation measurement system，能否在手工冻结预期、独立 oracle、范围外 holdout、非链式拓扑、正向 metamorphic 和定向 fault registry 下，稳定测量累计、比较、选择与关系组合？

NR1 不训练模型，不判断 shared core 或 mixed/typed core，也不授权 K=1、direct/text-CoT、P2 或 V2-B。PASS 只允许设计 `P1-H1` shared-vs-mixed/typed development comparison；FAIL 必须原样停止，不能重跑、换 seed、删 fault 或降低 Gate。

## 2. 与目标架构的关系

白皮书允许 `S_t=(A_t,H_t)`：连续 `H_t` 承载语义内容，`A_t` 只保存身份、类型、来源 handle、阶段、权限与合法寻址。NR1 资格化的是未来训练/评测所需的 typed measurement surface，不把 exact teacher state 注入部署 forward。

后继 H1 若获准，仍必须满足：source 由 frozen Qwen hidden 与 learned Boundary 读取；handle/type/route 由内容学习，不能使用 task id、oracle span、candidate index 或答案；数值/关系 expert 只能更新连续 payload，不能调用 simulator 或直接输出答案；anonymous shared core 与 mixed/typed core 使用 fresh data、同 teacher、matched active compute 与一致 token/FLOPs 记账；正式答案只能读取最终 latent state。

这一方向与数值归纳偏置、可交换 slot 和连续 latent reasoning 的研究相容，但论文不是本项目的合格证据。参考：Neural Arithmetic Units（<https://arxiv.org/abs/2001.05016>）、Slot Attention（<https://arxiv.org/abs/2006.15055>）、Coconut（<https://arxiv.org/abs/2412.06769>）。

## 3. 直接切换后的活动面

活动实现只有：

- `src/yggdrasil_v2/r1_revalidation/nr1/schema.py`：case/state 自有 handle registry、DAG 与未知 handle fail-closed；
- `reference.py`：Python integer `accumulate` 与逐 source/query BFS 的独立 oracle；
- `measurement.py`：显式迭代累计、scalar compare、fixed-point closure 与 state-derived decision；不得 import reference；
- `faults.py`：只接受冻结 target 的故障注入器，不从被测 state 选择 winner、bridge 或 false query；
- `qualification.py`：fresh case、手工 fixture、拓扑覆盖、metamorphic、metric-to-fault ledger 与 N02–N06；
- `artifacts.py` / `runner.py`：single-use root、Git/source/snapshot identity、进程树、transport、post-action identity 与 evidence seal；
- `tests/v2_r1r_p1_nr1/`：当前唯一活动 P1 预测试；
- `experiments/v2_r1_revalidation.py`：只暴露 `run-p1-nr1`。

旧 `src/yggdrasil_v2/r1_revalidation/p1/*.py` 与 V8L 活动测试已移除，不提供兼容 alias。历史源码只由 V8L 三个 sealed root 的 `source_snapshot/` 与历史文档追溯。

## 4. 独立预期与算法边界

大样本 oracle 与 measurement 使用不同算法族，但这仍不足以排除共享生成假设。因而另冻结 `tests/v2_r1r_p1_nr1/fixtures/nr1_hand_authored_cases.json`：6 个手工 numeric case、6 个手工 relation DAG 以及 11 类 fault target；预期由独立对话按前缀累加与路径展开手算，文件 SHA-256 固定为 `C94D372B7A06448F2031ABCA03C2EEE2538B3C627E4A5CD00E522C3F18EFB57F`。

正式 Gate 同时要求 measurement 与 reference 都逐项匹配这份外部预期。AST 边界审计机械确认两模块互不 import：numeric oracle 调用 `accumulate`，measurement 使用逐步 `AugAssign(Add)`；relation oracle 调用逐源 BFS，measurement 使用不调用 BFS 的 fixed-point `while`。这不是靠源文本关键词猜测实现类别。

## 5. Case 与测量语义

### 5.1 Numeric

每个 case 含 2–9 个 opaque handle 候选。所有轨迹在最终关键更新前由正负成对 delta 构成、前缀中同时出现正负更新；最终更新使唯一 winner 形成。measurement 只能逐步累计 typed scalar state，再由最终 scalar 的严格最小值导出 winner handle。case schema 没有 answer、label、task id、候选位置真值或 oracle state 字段。

两个互斥域为：

- qualification：384 cases，2–5 candidates、3/5 steps、`|delta|` 为 1–32；
- heldout：256 cases，6–9 candidates、7/9/11 steps、`|delta|` 为 64–1024；
- fingerprint 零重合，winner position 按 case/candidate count 轮换，不能使用第一位置捷径。

最终关键更新让 sign-flip 与 drop-last 成为真实 decision fault；off-by-one 只注册为结构/精确值 fault，不伪称必然改变最终选择。

### 5.2 Relation

每个 case 使用 opaque handles 与有向无环 direct edges。qualification 为 384 个 7–8 handle case，heldout 为 256 个 9–16 handle case。每个域都包含 branch、merge、multiple paths、multiple components、irrelevant edges、显式冗余 direct edge 与非 direct transitive closure；query 数量在 4–8 间变化，真值位置在正式 corpus 中形成至少 16 种布局，整体 true rate 必须在 0.35–0.65。

measurement 通过 fixed-point closure 得到可达关系，再对无答案 query pair 输出 decision；独立 oracle 使用逐 source/query BFS。handle 列表顺序、edge 顺序、query 顺序与拓扑顺序解耦。case schema 拒绝 cycle，state 保存自己的 known-handle registry，closure 或 query 出现未知 handle 时 fail closed。

## 6. Metamorphic 与 fault registry

Numeric 正向不变量包括逐步/最终精确值、比较 oracle 与反对称性、共同平移、正比例缩放、候选置换、handle rename。Relation 包括 direct 保留、完整且无额外边的 closure、组合、反对称性、edge/query permutation、handle rename 与添加真实 transitive redundant edge 后 closure 不变。qualification/heldout 的全部注册精确率必须为 1.0。

11 类 fault 为：numeric sign-flip、drop-last、final off-by-one、constant collapse、winner/runner swap、position shortcut；relation drop reachable bridge、reverse pollution、spurious reachable、unknown handle alias、query-position shortcut。

每个正式 fault target 先由独立 `accumulate`/BFS oracle 形成 manifest，再传给 fault 注入器；注入器不得调用 decision 或从 measured closure 选择目标。每类 fault 都预注册必须杀死的 metric。全部 state detection rate 必须为 1.0；全部 required metric kill rate 必须 `>=0.80`；所有 decision-affecting fault 的 decision kill rate 必须 `>=0.80`。unknown handle alias 预期由 schema 拒绝，off-by-one 只要求 exact-value metric 被杀死。

## 7. Gate 与停止语义

- **N01 contract/source/history integrity**：冻结设计/执行 hash、V8L 三根 status/seal/tree hash、预测试、单一 CLI、fixed-root freshness、Git/source/snapshot identity 全通过。
- **N02 schema/reference/fixture independence**：无禁止字段；case DAG、state/query handle fail closed；reference/measurement import 与算法族分离；手工 fixture hash、预期和 fault registry 全匹配。
- **N03 numeric qualification**：两个域的 trace、final、comparison、winner 与全部 numeric metamorphic 为 1.0。
- **N04 relation topology qualification**：两个域的 closure、query、composition、antisymmetry 与全部 relation metamorphic 为 1.0，且非链式拓扑和可变真值布局覆盖达标。
- **N05 adversarial metric/decision power**：11 类 fault detection 为 1.0，required metric 与 decision Gate 达标，target manifest 可复算。
- **N06 holdout/non-leakage**：两个域 fingerprint 零重合，heldout magnitude/horizon/cardinality 严格越界；没有训练、超参选择、audit 拟合或 label 输入。
- **N07 stop/process/artifact integrity**：唯一进程树、固定 transport、preflight/qualification source identity、Git status fingerprint、post-action identity、source snapshot replay、即时 seal verify 与 successor absence 全闭合。

全部通过才是 `PASS_P1_NR1_MEASUREMENT_QUALIFICATION`。PASS 仍设置 `p1_completed=false`、`p2_eligible=false`，只设置 `p1_h1_design_authorized=true`。

## 8. 后继总合同

从当前状态到 P1 完成只允许：`P1-NR1` 独立 measurement qualification → `P1-H1` fresh-data shared-vs-mixed/typed development comparison → `P1-F1` fresh-seed K=8/K=1/direct/text-CoT 完整 falsification 与因果 Gate。H1/F1 必须另立冻结合同、预测试与 single-use roots；NR1 PASS 不能预写任何后继为通过，任一阶段 FAIL 都在该阶段停线。
