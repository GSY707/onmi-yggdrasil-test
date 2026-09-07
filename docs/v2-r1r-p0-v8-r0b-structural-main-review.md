# V2-R1R P0-D v8 R0B-structural 主设计层验收

日期：2026-08-01  
对象：`artifacts/v2-r1r/p0d-v8-r0b-structural-20260801-1/`、当前 `audit_structural_bundle()`、冻结 v8 合同与父任务 registry 外反例  
证据等级：measurement-system qualification review；不是生产数据、模型、训练或架构证据

## 结论

v8 的唯一 formal attempt 真实完成，机器结果为 `PASS_R0B_STRUCTURAL`。主设计层独立复算 fixed root、43 文件 evidence seal、24 个 frozen input、37 个 raw metric、41 个 fault、4 个 metamorphic control 与执行协议，确认这些机器事实成立；正控的两次调用也确定、只读，缺失 `records.jsonl` 时公共 API 能 fail-closed 返回 JSON 报告。

但是，父任务随后构造的五个 registry 外反例全部被公共 API 错误接受。它们分别违反冻结合同中的 claim 精确基数与极性、ERE provenance witness 的真实指向，以及 language pair 与 AST 的语义绑定，却仍得到 G02–G06 全真。因此本阶段最终判定为 **R0B-structural machine-pass、main-review rejected**。

这不是 v7 simulator 回归，也不是 formal 协议失败。根因位于 R0B 审计定义与实现：fault matrix 证明了“每个现有 metric 至少能被某个登记 fault 杀伤”，但没有证明 metric 已完整表达书面不变量；部分 witness 检查只验证类型、存在性或字符串形状，language pair 又允许用自报 fingerprint、span 和重新计算的表面统计自证语义。

按照停止规则，v8 实现、冻结输入和 formal artifact 不修补、不覆盖、不换 suffix 重跑；R0C、R0D、R1 generator、P0-M、模型、cache、训练与 GPU 均保持未授权。

## 1. 接受的机器与协议事实

| 项目 | 独立复核结果 |
| --- | --- |
| formal 状态 | `PASS_R0B_STRUCTURAL`，exit `0`；2026-08-01 20:29:59 开始，20:32:00 结束 |
| 正控 | G02–G06 `5/5`，37/37 raw metric 全真 |
| 负控 | 41/41 fault 的实际 false Gate set 与冻结 expected 精确一致 |
| metamorphic | 4/4 通过 |
| metric kill ledger | 37/37 metric 被至少一个登记 fault 杀伤，coverage `1.0` |
| artifact | 固定 root 精确八个顶层 entry；43 个递归 seal hash 全部可复算 |
| frozen inputs | 24/24 当前 formal 输入与冻结 manifest 一致；manifest SHA-256 `9A49F40E13F4CC2CE536DA894E0A24418A72D2499A03EB9B7D556D6DC7B1E0FA` |
| 直接切换 | v8 CLI 已直接替换 v7；`tests/v2_r1r_v7/` 已删除；没有 v7 alias 或 generator/model/train 路径 |
| formal 纪律 | root 在求值前原子占用；正式命令只调用一次；`authorization_created=false`；执行层未进入后续阶段 |
| post-formal pytest | 12 项通过，唯一失败是 protocol test 按设计要求 fixed root 尚不存在；root 已由 formal 创建，因此这是预期封存阻断，不是实现回归 |

fixed root 顶层文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `assessment.json` | `C02F905A3CDAEFD2A27336E9DE839874E08BA6F90111E26A0214FEEA04A88AF7` |
| `attempt.json` | `570D67A3CD099B47A5CC189DF259FE26F1F0D74A366CA88FF098DEDFE2249553` |
| `coverage-ledger.json` | `BEC293933A1D749AF4442299599B4444E7A5EFBE8FC16CF203E6CB5B306E736B` |
| `evidence-seal.json` | `C2BB9D6A29F25873A5042EAAEE9AB9FA7A9020FE5B3182E035C0747EBF766C51` |
| `fault-matrix.json` | `2C8195FAD671C2D4775439823E0420DAAAEF5A715F3273532FA35E3CCFFC312E` |
| `run-metadata.json` | `98388FD28E06201F7BD442EF5FFC191ADC5B59C2528D3DE6411EC3DB0A21B47A` |
| `structural-report.json` | `7798D7E0A0D329346EC1EAA643F5E57C1EFE68156AB068610107AA77CC318072` |

## 2. Registry 外公共 API 验收

父任务没有修改 fixed root。每个反例都从 sealed `control-bundle/` 复制到系统临时目录，使用标准库重算数据文件 hash、manifest count 与 `input-seal.json`，再调用同一个公共 `audit_structural_bundle()`。这排除了“只是 seal 被破坏”或“调用了另一条测试路径”的解释。

探针使用 formal 相同的系统 `python`；仓库 `.venv` 会把同一 tokenizer snapshot 加载成慢版 `Qwen2Tokenizer`，因 class 不符而按 G02 fail-closed，所以该次环境不一致的预调用被丢弃、没有计入结果。下表结果在状态文档同步前完成，当时 baseline 的 runtime/Git provenance 仍精确匹配 formal manifest；同步文档后若直接重放，G02 会额外因 dirty diff hash 改变而失败，复现这些定向 false negative 应使用 formal source/diff snapshot，而不能把后续文档漂移混入目标 Gate。

| 新鲜探针 | 冻结合同期望 | 实际结果 |
| --- | --- | --- |
| baseline 连续调用两次并比较输入文件 hash | 两次报告相同、输入不变、全 Gate 真 | 通过 |
| 删除 `records.jsonl`，不修复 seal | fail-closed，返回非通过 JSON | 通过；G02–G06 全假 |
| 给现有合法 claim pair 增加一份新 pair，使 claim 从 18 变为 20、正负从 9/9 变为 10/10 | G03 应拒绝精确基数漂移 | **错误接受；G02–G06 全真** |
| 互换一对 true/false claim 的 `pair_role=positive/negative` | G03 应拒绝 role 与 truth 极性脱钩 | **错误接受；G02–G06 全真** |
| 把 `ere-initial-copy` 的 `origin_path` 从真实初始值改为无关的 `/query` | G05 provenance witness 应失败 | **错误接受；G02–G06 全真** |
| 把 relation witness 的 `neighbor` 改成 `not-an-actual-neighbor` | G05 AST/trace provenance 应失败 | **错误接受；G02–G06 全真** |
| 把 ERE language pair 正文替换为香蕉/石头等无关语句，仅重算 token count、surface fingerprint 并提供存在的 span | G04 language–AST identity 应失败 | **错误接受；G02–G06 全真** |

这些反例不是对未来生产分布提出无限完备要求，而是直接攻击 v8 冻结设计逐字声明的有限不变量：18 条 claim、正负精确 9/9、provenance witness 指向真实 AST/trace 因果链、language pair 与同一 AST 保持语义身份。因此不能把它们降级为 R0C/R0D 的范围外问题。

## 3. 根因

第一，**metric coverage 被误当成 property coverage**。`37/37 killed` 只说明已有 metric 对登记 patch 有响应；当精确 cardinality、role→truth 映射和 witness denotation 根本没有进入 raw metric 时，再完整的 kill ledger 也无法发现遗漏。

第二，**certificate 检查仍有自报成分**。initial-copy 的 `origin_path` 只要求指向非空对象，relation 的 `neighbor` 只要求是字符串；审计器没有从 AST 与 replay trace 重建“哪个初始值经哪些 event 到达 query”的有向证据链。这使 certificate 形状正确但语义错误时仍可通过。

第三，**language audit 只验证表面调度，没有验证语义绑定**。当前逻辑能确认文本不同、span 存在、token/fingerprint 可复算，却不能确认两段文本描述引用的 AST；代码中的 choice mask 和 label mapping 检查还是字段与自身相等的恒真式。若这一层暂时无法独立验证语义，就必须把 Gate 明确降格为 surface schedule，而不能继续命名为 language AST identity。

## 4. 后继合同条件

下一步仍应停留在 R0B，另立新合同直接替换 v8，而不是修补本次 formal 或进入 R0C。新合同至少需要：

1. 把 control schema 的精确 record/claim/pair cardinality、每 kind 一对、`positive↔true`、`negative↔false` 写成原子 raw metric；
2. 用 typed witness verifier 从 AST、event index、primitive operand 与 fresh replay outcome 重建 provenance 链，不再接受“存在路径/字符串类型”作为语义证据；
3. language pair 要么来自冻结、可逆的 AST→template slot renderer 并逐 slot 复核，要么诚实收缩为 surface-only schedule，语义等价留给另一个有独立 oracle 的 Gate；
4. fault 设计从逐 metric 单例改为逐不变量的等价类：缺失、额外、互换、重定向、保持表面统计但破坏语义，并在 formal 前保留一组主审 holdout family。
5. replay provenance 以 artifact 内 source snapshot/seal 为真源；当前 worktree drift 只单独报告，不能让结果文档同步后同一 sealed bundle 自失效。

v7 accepted simulator 继续只读。是否冻结并执行下一份 R0B 合同，需要主设计层另行授权。

## 5. 完成与未完成

已完成：v8 audit/CLI/tests 直接切换、唯一 formal、八项顶层 artifact 封存、41 faults、4 metamorphic、37-metric ledger、hash/protocol 独立复核、baseline/fail-closed 检查与五项 registry 外反例验收。

未完成：R0B 主设计层资格、下一版 typed invariant audit、R0C shortcut learner、R0D integration、R1 generator、P0-M、Qwen hidden Boundary、共享 latent core、公平 baseline、训练/GPU 与任何架构通过结论。
