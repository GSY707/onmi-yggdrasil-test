# V2-R1R P0-D v7 R0A-lattice 主设计层验收

日期：2026-08-01  
对象：`artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1/`、当前公共 simulator、冻结 v7 合同与父任务新鲜黑盒探针  
证据等级：semantic-oracle / API-boundary qualification；不是数据、模型、训练或架构证据

## 结论

v7 的唯一 formal attempt 真实完成，机器结果为 `PASS_R0A_LATTICE`。主设计层独立复算 frozen、implementation、artifact 与 evidence-seal hash，重跑 guard、pytest、compileall、CLI help 和 diff-check，并在不导入 lattice builder、fixture 或测试代码的条件下另写 27 项公共 API 黑盒探针；所有检查均通过。因此本阶段最终判定为 **R0A-lattice accepted**。

这次接受是有限且具体的：它证明当前冻结 ERE/CPS AST schema 的公共 simulator 能复现 14 个手写语义世界、4 个 prefix、12 个 corruption、245 个历史非法 AST，以及完整的 858-cell ERE operand `slot × scope × token` 覆盖格。它不证明生产数据、G02–G08 audit、generator、模型可训练性、共享 latent core 或完整架构成立。

R0A 已达到继续验证所需的充分门槛；不再增加新的 R0A 完美化版本。下一步应直接进入另立合同的 R0B structural audit，只资格化 G02–G06 的 artifact/model-view、语义回放、split/pair、ERE/CPS 结构审计机制。R0C、R0D、R1、P0-M、模型、cache、训练和 GPU 在新合同冻结前仍未授权。

## 1. 接受的机器与协议事实

| 项目 | 独立复核结果 |
| --- | --- |
| formal 状态 | `PASS_R0A_LATTICE`，exit `0` |
| 继承语义 | canonical `14/14`、prefix `4/4`、corruption `12/12`、legacy strict `245/245` |
| operand lattice | invalid `508/508`、acceptance `350/350`、coverage `858/858` |
| coverage ledger | required/observed 均为 `858`；无 missing、unexpected、duplicate、wrong-validity、failure 或 missing sentinel |
| 确定性与输入边界 | 两次 report bytes 一致；成功与失败输入均不变；公共 API、import 和文件边界通过 |
| artifact | 固定 root 精确七个 JSON 文件；manifest 和 evidence seal 全部可复算 |
| frozen inputs | 19 个文件未漂移；manifest SHA-256 `3FF7A1A3CF985A86E4EA9F6A2BA279AB551F434E38BC6C3E14531C8D81DB66F9` |
| 直接切换 | CLI 与 v7 template 字节一致；`tests/v2_r1r_v6/` 已删除；没有旧 alias 或 compatibility |
| formal 纪律 | root 先被原子占用；正式命令只调用一次；执行层在 formal 后只读停止 |
| 父任务复核 | guard 通过；pytest `32 passed`；compileall/help/diff-check 通过；后续 preflight 因 fixed root 已存在而按合同 `BLOCKED` |

七个 sealed 文件及父任务复算 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `assessment.json` | `6A9D78D221D817F831E831038756F9790D639FEC89D7AD96267A1357F4D8D653` |
| `attempt.json` | `1CDB6411075DDF36A3B05EC748EAC6C40F8909043D4534433BD3FE181011CF86` |
| `coverage-ledger.json` | `FBBEA9B5BDB50C6962AB27867ADA054705A2B5DB1DC3CE2D2D39F37C01AE3846` |
| `evidence-seal.json` | `794CB9518F094EE6CF99800AF0B5848D22A7746DAA5E76A84489F3F8201987F0` |
| `manifest.json` | `EBA16F2775CB9761B866EEA778D5093E9C1679835D06F22C52860AB9C09B6406` |
| `run-metadata.json` | `A72A16025C0E2CA7150F922B23A2736B115A23F74D0D8096121EE1E6F023CB44` |
| `strict-report.json` | `4541737284D3A68B582C584B8C25E1A3C02C192E0C40C95426BA07D57E4C7113` |

coverage ledger 的 required/observed cell-set SHA-256 均为 `DBB047B92CDD4523F7881CFA78440BA06D7A29630F57BD0DF391ADC2717C33B5`。这证明 auditor 没有通过少算 required set 获得 `858/858`。

## 2. 父任务新鲜黑盒验收

父任务只导入公共 `simulate_ere` / `evaluate_cps`，没有导入 v7 spec、builder、coverage auditor、fixture 或 pytest helper。共运行 27 项检查：

1. 五个 query operand 分别注入 `$arg:newname`、`$novel_placeholder`、`$arg:`、`$neighbor`，20 个组合都返回 exact `AST_VALIDATION|invalid_placeholder|/query/<field>`，且失败输入不变；
2. 非法 query 与非法 prefix 同时存在时先报告 query，确认 whole-AST eager validation；
3. 不以 `$` 开头的 nonce query literal 正常执行，排除过度拒绝；
4. 两层 `FOREACH_LINKED` 中内层 `$neighbor` 正确遮蔽外层值，外层值又能用于内层 `source`；
5. 内层 relation 名可由外层 `$neighbor` 动态解析，内层 effect 再使用新的邻居值；
6. event argument 中 `$neighbor` / `$arg:*` 保持字面量，只有进入 rule operand 后才按声明参数替换；
7. state identifier/value 与 CPS string 中所有前导 `$` 保持普通数据。

黑盒脚本返回 `PASS_PARENT_BLACK_BOX`，27/27 通过。它覆盖了 frozen query holdout 未使用的新 token/name，并额外验证作用域遮蔽和“只在 ERE operand 中解释 `$`”的系统边界。

## 3. 为什么本轮可以接受

v6 失败的根因是粗粒度标签无法证明组合覆盖；v7 改变了测量表示，而不是追加个别测试。接受依据有三层彼此独立的约束：

- 规范层显式列出 23 个 rule/predicate slot、5 个 query slot、6 个递归作用域等价类和 6 个 token 类；
- 机械层对 858 个 cell 做 exact-set、exact-path、exact-validity 和正负执行，并由独立 coverage auditor 复算 required set；
- 主审层使用未冻结名称与嵌套运行时行为做公共 API 黑盒探针。

这不意味着 validator 对所有未来 schema 已被数学完备证明。它意味着对当前有限 schema 已取得足够、可复算且没有已知反例的资格证据，可以停止继续雕琢 R0A 并把失败风险移交给下一层 audit 机制。

## 4. 下一阶段边界

下一合同必须是 R0B，而不是 v8 R0A。R0B 的目标是验证 G02–G06 审计机制能否：

- 从 sealed bundle 独立复算 artifact、snapshot、provenance、token count 与 exact model view；
- 使用已接受的公共 simulator fresh replay answer、teacher、prefix claim 与预算；
- 识别未声明 overlap、causal pair、language pair、composition 与 fold 破坏；
- 通过实际 ablation/plan replay 判断 ERE 多步必要性和 CPS hard-negative / NONE 结构；
- 对手写正控全绿、对定向负控给出 exact false Gate set，并形成 metric-to-fault kill ledger。

R0B 不生成生产数据，不实现 G07/G08 统计 learner，不进入 generator、P0-M 或训练。为避免 v4 的自洽测量，正负 expected values 必须来自主设计层冻结控制包；执行层只实现 audit 与一次 formal，不得自行发明 expected、fault 或阈值。

## 5. 完成与未完成

已完成：v7 四文件直接切换、v6 活跃测试删除、唯一 formal、七文件封存、全部 hash/测试/CLI/边界复核、27 项合同外黑盒验收，以及 R0A-lattice 最终接受判决。

未完成：R0B 新合同与实现、G02–G06 structural audit、R0C shortcut audit、R0D integration、R1 generator、P0-M、Qwen hidden Boundary、共享 latent core、公平 baseline、训练/GPU 与架构结论。
