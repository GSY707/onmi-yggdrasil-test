# V2-R1R R1G v13 production generator smoke 主设计层验收

日期：2026-08-09

验收对象：`r1r-r1g-v13-generator-smoke`

## 1. 核心判定

v13 通过主设计层有限验收，正式判定为 **`R1G fixed-seed production generator smoke accepted`**。

它证明 accepted simulator、production renderer、scalable shortcut learner 与新 production generator 能在固定 seed `20260809` 下共同构造并审计 1,440 条 ERE/CPS production-shaped 数据；不证明完整 P0-D、模型可学、共享 Boundary/core 有效或架构成立。完整每族 train 4,096、各 heldout/pair 512 的 P0-D，P0-M、Qwen cache、模型和 GPU 训练仍未授权。

## 2. 正式 artifact 与机器事实

唯一 formal root 为：

`artifacts/v2-r1r/r1g-v13-generator-smoke-20260809-1/`

唯一命令返回 `PASS_R1G_SMOKE`，G01–G11 全为 true。根目录精确包含 dataset、九个 JSON 报告/协议文件和 source snapshot；evidence seal 覆盖 62 个非 seal 文件，父任务只读复算 62/62 完全一致。根 seal SHA-256 为：

`A9CF968F15634E7F1F9471CDE5383736C6D0E17DCC8208E871B6C779062AE455`

正式数据包含 14 个 JSONL、每族 720 条、总计 1,440 条和 72 个因果 pair。1,440/1,440 source-only roundtrip、fresh simulator output、answer-label binding、teacher trace 与 16,848 条 claim replay 全部一致；semantic/surface fingerprint 都为 1,440/1,440 唯一。

ERE 独立审计覆盖七种 primitive、两种 query、36 条 `LINK→FOREACH→COPY` 和 36 条 `SWAP→IF→COPY` composition、6/8 entity 配额与 72 条 dependency depth 8 length OOD；720/720 指定反事实均翻转答案。CPS 各 split 的 NONE、唯一最优、valid-suboptimal、precondition/final-constraint/budget/goal hard negative、6–8 步 horizon 和八候选 distractor 均成立。

Qwen pinned tokenizer 的最大 source 长度为 968 tokens；ERE train/validation p99 为 490/494，CPS 为 666/658，没有截断。九标签在所有 split 精确均衡。七个 source-only shortcut 全部低于 `mean random + 0.10`；最窄正式余量为 ERE validation full-text word NB：`0.361111 < 0.370833`。claim 的每个 kind 在每个 split 正负严格平衡，heldout word/char NB 全部精确为 `0.50`。

G11 证明两次独立生成的 dataset bytes、两份完整报告 bytes 和对 fixed artifact 的只读 replay 相同，accepted 输入在运行前后字节不变。assessment 明确 `full_p0_or_training_authorization_created=false`。

## 3. 主审外部探针

父任务在 formal registry 外使用 400 个新 seed case，交叉覆盖 ERE train/composition/length/entity 与 CPS train/composition/horizon/distractor。每个 case 均验证单叶 AST 差异、fresh 答案翻转、原例/反事实 fingerprint 分离、alpha rename 不变、CPS action/candidate permutation 等价、ERE map insertion permutation 等价，以及三种 grammar 的 source-only exact roundtrip。400/400 全部通过；396 个不同 fingerprint 中的四次重复来自同构语义形状，正式生成器会按全局 uniqueness 规则重采样。

另以 `20260810/20260811/20260812` 做三组非 formal shortcut 稳健性诊断。两组全部通过；`20260811` 只有 ERE causal-pair char NB 一格为 `0.375000 > 0.370833`，即 72 条中多命中一条。失败没有在其他 seed、split 或 baseline 重复，符合小样本离散波动而非稳定 surface shortcut。它不推翻 fixed-seed formal，但说明 72 条上对约 98 个相关 baseline cell 逐格设硬阈值会产生多重比较脆弱性。

## 4. 本轮真正修复的机制问题

本轮不是单纯把一个 seed 调到通过。关键修复包括：

1. fingerprint refinement 从不会可靠收敛的 hash equality 改为单调等价类细分，完整生成由数分钟卡死降到约 14–20 秒；
2. CPS alpha-invariant 语义空间由不足 288 种的窄成本组合扩展为足以支持跨 split 全局去重的合法成本空间，没有放松 overlap Gate；
3. 正确标签由与结构共享 index 的模周期改为 split-local 精确均衡、独立洗牌日程，choice 行位置改为分层均衡，消除结构—标签与 last-line shortcut；
4. 反事实证书保存可 fresh replay 的单叶 before/after patch，而非只相信生成器自报的答案 hash；
5. CPS 成本干预后原 P* 仍保持合法次优，composition 的 budget hard negative 独立成立；
6. claim 改为局部交叉配平：同一对象/值/资源/fact 在正负两侧具有相同边际，NONE 不伪造不可达 fact transition。此前 0.63–0.70 的 claim NB 因此回到 0.50。

这些是 production-data 机制不变量，不是对训练器的补丁；后续完整 P0 应保留。

## 5. 接受边界与下一合同

v13 只接受固定 seed 的有限 generator smoke。其 public generator 虽保留非默认 `seed` 参数，但 record provenance 的 `smoke_seed` 仍按冻结合同写固定 `20260809`；因此非默认 seed 只被本主审当作临时诊断，不是已资格化的可发布接口。完整 P0 合同必须让 root seed 在 manifest、record provenance 和 replay 中逐项一致，不能继承这个边界。

下一步不是 P0-M 或训练，而是另立完整 production P0-D 合同。该合同至少应：

- 使用每族 train 4,096、validation/OOD/causal 各 512 的正式规模，并采用不同于 v13 的 root seed；
- 保留 v13 的 renderer、single-leaf replay、alpha/permutation、claim 局部配平和 source-only baseline；
- 把 shortcut 判定从 72 条逐 cell 单阈值改为预注册的较大样本判定，同时报告置信区间、最差 cell 与 family-level 聚合，避免多重比较噪声驱动返工；
- 明确验证任意正式 root seed 的 provenance，而不是只验证默认常量；
- 在新 fixed root 上先完整 preflight、再唯一 formal，失败即停，并由父任务再做未知捷径复核。

只有完整 P0-D 正式数据通过并经主审接受后，才可能另行讨论 P0-M。当前没有模型、训练、质量—成本 Pareto 或 V2 架构成功证据。
