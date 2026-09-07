# V2-R1R P0-D v6 R0A-strict 主设计层验收

日期：2026-08-01  
对象：`artifacts/v2-r1r/p0d-v6-r0a-strict-20260801-1/`、当前四文件实现、冻结 v6 合同与合同外新鲜黑盒探针  
证据等级：semantic-oracle / API-boundary 验收；不是数据、模型、训练或架构证据

## 结论

v6 的唯一 formal attempt 真实完成，机器结果为 `PASS_R0A_STRICT`。主设计层接受其中的窄事实：14/14 canonical、4/4 prefix、12/12 corruption、245/245 frozen invalid-AST probes、输入不变、deterministic report、import boundary、15 个冻结输入 hash、六文件 artifact 和 evidence seal 全部通过；执行层也遵守了先占用固定 root、只调用一次 formal、结束后停止的协议。

但阶段判定仍为 **R0A-strict machine-pass、main-review rejected**。合同外新鲜探针发现：ERE query 中的 `$neighbor` 和 `$arg:*` 在 attribute query 的 `entity/attribute` 以及 relation query 的 `relation/source/target` 共十个组合里全部被静默接受，没有按书面合同抛出 `AST_VALIDATION|invalid_placeholder|<query pointer>`。因此冻结的 245 项矩阵没有完整测量自己宣称覆盖的 query × placeholder 边界，机器 conjunction 不能升级为阶段通过。

本轮在此停止。v6 实现、测试和 sealed artifact 不修补、不覆盖、不重跑；R0B/R0C/R0D、R1、P0-M、模型、cache、训练和 GPU 继续未授权。

## 1. 接受的机器与协议事实

| 项目 | 独立复核结果 |
| --- | --- |
| formal 状态 | `PASS_R0A_STRICT`，exit `0` |
| 正向语义 | canonical `14/14`；prefix `4/4` |
| 冻结负向语义 | corruption `12/12`；strict matrix `245/245` |
| 完整 conjunction | assessment 的 12 项检查均为 `true` |
| artifact | 固定 root 精确六个 JSON 文件；六项 SHA-256 与 evidence seal 一致 |
| frozen inputs | 15 个文件全部未漂移；manifest hash `59A5F0718511CEFDD22969BFBCE1DE99107D6A4E912EED65917B7EB19773AAC9` |
| 实现边界 | 只有四个获准文件发生切换；`tests/v2_r1r_v5/` 已删除；没有 compatibility、generator、audit、model 或 train 路径 |
| formal 纪律 | root 先被原子占用并写 `attempt.json`；正式命令只调用一次；结束后未修改实现 |
| 父任务复核 | guard 通过、pytest `18 passed`、compileall/help/diff-check 通过，CLI 只含 `preflight-strict` / `seal-strict`；R0B 及后续 artifact 不存在 |

六个 sealed 文件及复算 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `assessment.json` | `3859D06293AB0D100A04A5A9A6C149096D9A60D9D9D8A2DBA4847A9961FE0278` |
| `attempt.json` | `190A759BE3F48CD7EFD7D0448059C5E2223F4C24F7E1A4462F572124A15D27D4` |
| `evidence-seal.json` | `6AF55D61C84B321BFB156663A78157A5FE9B29F95BF176C2AD799F6C6DB15329` |
| `manifest.json` | `86172F65974062C12D0B6532A19163809B1C923ECB095A3DD6310946EF9D10DD` |
| `run-metadata.json` | `4B06635CE9D9CDD433E079B0FE0A983838C863FDD3FA866B23788C5973056050` |
| `strict-report.json` | `185CBDF75BBCFA3E83BA487FF077CC47CA591A9CA2FF5BE50FBFEF11159E22F8` |

这些事实证明执行与封存没有造假，也证明实现通过了冻结矩阵；它们不能证明矩阵完整表达了书面合同。

## 2. 新鲜黑盒反例

主设计层没有复用 frozen matrix case，而是先构造任意 nonce 名称的 ERE/CPS 输入，再加入冻结合同外的 query placeholder 组合。

正控制共七项全部通过：任意 ERE、ERE prefix、ERE 输入不变、任意 CPS、CPS prefix、CPS 输入不变，以及“未使用坏 rule 必须先于非法 prefix 被 eager validation 拒绝”。这排除了公共 API 完全损坏或探针调用错误。

失败反例如下：

| query 位置 | 注入值 | 合同预期 | 实际 |
| --- | --- | --- | --- |
| attribute `/query/entity` | `$neighbor` / `$arg:dst` | `invalid_placeholder` | 返回 `answer=None` |
| attribute `/query/attribute` | `$neighbor` / `$arg:dst` | `invalid_placeholder` | 返回 `answer=None` |
| relation `/query/relation` | `$neighbor` / `$arg:any` | `invalid_placeholder` | 返回 `answer=False` |
| relation `/query/source` | `$neighbor` / `$arg:any` | `invalid_placeholder` | 返回 `answer=False` |
| relation `/query/target` | `$neighbor` / `$arg:any` | `invalid_placeholder` | 返回 `answer=False` |

十个组合均未抛异常，失败率为 `10/10`。书面合同规定 `$arg:name` 必须引用“当前 rule”的已声明 param，而 query 不处于任何 rule；同时 `$neighbor` 只允许出现在 `FOREACH_LINKED.effect` 子树，query 显然在该子树之外。因此不能在 formal 后把这些 token 重新解释为 query 中的普通 literal。

## 3. 根因

问题不在执行层纪律，而在“书面不变量 → 可执行覆盖”的编译过程：

1. 冻结矩阵只有四项 `invalid_placeholder` 探针：`E042`、`E043`、`X057`、`X058`；它们全部修改 rule primitive operand，没有任何一项修改 query operand。
2. matrix 的 `covers=[ere_placeholder, invalid_placeholder]` 只证明分类标签出现过，并不证明 placeholder 在所有受约束语义位置上都被测过；设计文档据此写“每一种 query、placeholder”形成了错误的组合覆盖印象。
3. 当前 `_validate_ere_query` 对 query operand 只调用字符串类型检查，不调用 reserved-placeholder validator；因此冻结实现与冻结矩阵彼此自洽，却共同漏掉书面作用域不变量。

这是测量系统缺口，不是通过多跑 seed 或训练可以解决的问题。v6 的 narrow machine evidence 有效，但 `PASS_R0A_STRICT conjunction` 的完备性主张无效。

## 4. 停止与下一合同边界

v6 已封存，不能在原 root、原合同或原版本上补一个测试后重跑。若继续，必须另立新版本合同，并至少满足：

- 明确声明 `$` 保留 token 的词法域：query 没有 rule-param 环境，所有 `$arg:*` 必须拒绝；`$neighbor` 只在 `FOREACH_LINKED.effect` 的递归子树内有效；未知 `$*` 在所有 operand 位置都拒绝。
- 由 schema 的“语义位置 × reserved token 类别 × API”生成或机械审计覆盖笛卡尔积，不能再用粗粒度 `covers` 标签代替位置覆盖。
- 为 attribute/relation query 的五个 operand 位置冻结 exact error type/code/pointer，并同时保留 arbitrary-name 正控制，防止把普通 nonce 错杀。
- 继续采用 eager whole-AST validation、输入不变、原子 single-attempt root、独立主审新鲜探针和失败即停规则。

新合同尚未设计、冻结或授权。R0B 不得以“机器已 pass”为理由越过本次主审失败。

## 5. 完成与未完成

已完成：四文件直接切换、v5 活跃测试删除、v6 preflight、唯一 formal attempt、六文件封存、父任务 hash/测试/边界复核、合同外新鲜探针与主设计层判决。

未完成且未授权：新的后继合同、任何 v6 修补或重跑、R0B/R0C/R0D、R1 generator、P0-M、Qwen hidden Boundary、共享 latent core、公平 baseline、训练/GPU 与架构结论。
