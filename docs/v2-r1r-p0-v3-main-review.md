# V2-R1R P0-D v3 主设计层独立验收

日期：2026-08-01  
验收对象：`artifacts/v2-r1r/p0-preflight-v3-20260801-1/`、当前 `r1_revalidation` package、D0 测试和 D2 source snapshot 中的旧 v3 第 17 节合同  
最终判决：**接受执行层的失败证据与停机纪律；拒绝 v3 作为合格 P0-D 合同实现。D3、formal、P0-M 与训练继续禁止。**

## 1. 核心判断

本轮不是“六个阈值差一点”。D2 确实暴露了真实的数据分布问题，但同时暴露了 D0 自验证盲区、审计器聚合错误和合同本身的一处覆盖矛盾。现有六个红灯不能逐项修到绿色后继续 formal，因为若只修已触发项，仍会留下审计器漏检的 CPS 结构、claim 配额和 provenance 问题。

执行 agent 做对了两件事：D0、D1 后只运行一次 D2；D2 失败后没有改代码、重生数据、运行 D3 或创建 formal authorization。主设计层因此接受该 artifact 作为可复现的失败诊断。它没有证明 latent 架构失败，也没有证明 P0-D 已经只剩工程收尾。

## 2. 独立复核证据

主设计层重新运行了 28 项目标测试、`compileall` 和 `git diff --check`，均以 exit code `0` 完成；又用当前实现只读重算全部 5,120 条 D2 记录。除 post-run 文档同步导致的 source hash 漂移外，G05、G06、G07、G08、G11、G13 的机器结果与保存的 audit 一致。验收开始时当前 Python 代码与 artifact snapshot hash 一致，漂移只有 README、目录索引、阶段计划和结果文档；形成主设计层判决后，v3 合同被 v4 直接覆盖，当前 v3 执行命令被删除。artifact 的 `source_snapshot/` 仍保存运行时精确副本。

这同时暴露了 provenance 协议的自失效：audit 把会在运行后强制更新的四份状态文档也当作“当前源码必须保持同 hash”的对象，所以完成必需的结果同步后，fresh audit 会额外使 G03/G15 失败。snapshot 自身仍完整，但现有 verifier 不能把“artifact 内部可复算”与“当前工作树后来发生合法变化”分开。

## 3. 根因分层

### 3.1 D0 没有证明审计器能接受正确世界

D2 snapshot 中旧第 17.2 节要求每个 required metric 在 known-good mini fixture 上产生有限值并被检查，但 `tests/test_v2_r1r_contract.py` 只断言注入故障后目标 Gate 为 false，没有断言未注入故障时 15 Gate 能通过。于是 G07/G08 的 gate 聚合读取不存在的 family-level `passed`，导致它们对任何正式数据都恒为 false，D0 仍然全绿。

合同还同时要求“每个 Gate 至少有一个 fault fixture”，但 F01–F15 映射没有覆盖 G02 与 G04。当前 ledger 为这两项保存空 fixture，validator 只检查全部 fault id 的并集，没有检查每个 Gate 的覆盖，因此把矛盾静默为通过。这部分责任属于主设计合同，不应归咎于数据采样。

### 3.2 审计器低于冻结合同

- G06 只检查少量 template/长度/count/profile 分组，未实现 recipe、answer provenance、首末 candidate 长度、首 action 定义位置及冻结的二阶组合；candidate-position 小组还把正式最小样本数从 20 降为 3。
- G08 只检查 split 级 failure reason 并集，没有检查“每条 5-candidate 至少三类失败、8-candidate 至少四类失败”，也没有检查 valid-suboptimal 同时覆盖等长与更长。
- G13 使用 `>=0.10`，而旧 v3 第 17.7 节冻结的是每类 `0.20–0.30`；因此除 composition 外的多个不合格 CPS split 被误判为通过。
- G15 没有完整验证 Git/diff、依赖版本、named-stream derivation 和冻结 snapshot 内部复算；反而要求当前工作树与运行前状态文档永久一致。

因此保存的 `9/15` 只是当前审计器的输出，不是“合同真实通过九项”。

### 3.3 CPS 生成器仍保留了被禁止的固定候选骨架

当前 CPS 不是从依赖图变换库生成多样 hard negative，而是围绕固定的 optimal、valid-suboptimal、constraint/precondition failure、short failure、long distractor 五个角色拼装。表面 assignment 平衡的是某个语义 candidate 是否排在首位，不是正确 candidate 的实际显示位置；action definition 位置也只是逐样本随机，没有分层配额。

独立逐行复核显示：

- CPS 正确 candidate 位置最大相对偏差在普通/各 OOD split 为 `0.43–0.90`；
- non-NONE 记录满足“至少三类失败”的比例只有 train `0.2075`、validation `0.1878`，distractor 满足“四类失败”的比例为 `0`；
- 所有非 causal split 的 valid-suboptimal 都是与最优计划等长，更长 valid-suboptimal 数为 `0`；
- 每条记录同时具有四类 CPS claim 的比例为 train `0.6602`、validation `0.6758`、composition `0.1953`；
- `goal_only` 在 causal/distractor/horizon 达到 `0.4023/0.3672/0.3047`，是真实 source-only 捷径，不是单纯聚合 bug。

所以 CPS 需要重写 semantic constructor 与 surface allocator，不能只洗牌或调阈值。

### 3.4 ERE 语义构造比机器 Gate 显示得更健康

ERE 的 train、validation、entity OOD、language OOD、length OOD 在独立 structure report 中都满足 spine、五类 provenance 和逐事件必要性；G07 为 false 是聚合实现错误。ERE 的主要真实问题在表面层：choice-mask 配额和 causal pair 的答案字母调度没有确定性均衡。causal pair 的 `0.5/0.6504` 必要性数值不属于旧 v3 第 17.4 节对普通/长度 split 冻结的阈值，不能据此否决 ERE 主任务语义构造。

## 4. 路线判决

下一步不是 D3，也不是修复 v3 后直接重跑 D2。当前 `docs/v2-r1-revalidation-task-design.md` 第 17 节已把该判决具体化为 P0-D v4：先完成独立 R0 audit-reference，使每个 required metric 都有 known-good 有限值和定向 fault，补足 artifact/model-view、pair/language 故障，并让 snapshot replay 只依赖 artifact 内冻结源。

R0 只建立 common/audit/reference，不得实现 generator。只有 R0 经主设计层验收并另行授权 R1 后，才可保留已通过的 simulator/ERE 语义思想，并直接重写 CPS constructor、surface allocator、claim constructor 和测试，不建立 v3 兼容层。生成器必须把逐记录语义约束当作 construction-time invariant，把跨样本位置/label/choice/action 配额作为确定性匹配问题；无法满足时在生成阶段失败。

当前只授权 `docs/v2-r1r-p0d-v4-r0-execution-command.md` 定义的 R0；R0 无论成败都停止。R1/R2 preflight/R3 formal/P0-M 均未授权。当前项目状态仍是 P0-D 未通过；该结论只评价实验基础设施，不更新 V2 latent 架构成立概率。
