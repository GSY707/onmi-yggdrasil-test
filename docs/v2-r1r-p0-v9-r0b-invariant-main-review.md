# V2-R1R P0-D v9 R0B-invariant 主设计层验收

日期：2026-08-02
formal artifact：`artifacts/v2-r1r/p0d-v9-r0b-invariant-20260802-1/`
机器状态：`PASS_R0B_INVARIANT`
主设计层判决：**accepted as finite R0B measurement-system qualification**

## 1. 核心判断

v9 已经修复 v8 的表示层错误：审计结论不再由登记 fault 数量和输入自报 witness 决定，而由固定 qualification profile、exact schema、fresh replay、可逆 grammar 与 AST/trace typed derivation 的 conjunction 决定。formal 后的 registry 外探针没有发现已知 false negative，因此接受该有限测量切面，R0B-invariant 可以关闭。

这个接受不等于 V2-R1R 数据通过、模型通过、训练通过或白皮书架构成立。v9 只证明：在当前有限 ERE/CPS AST schema 和四种可逆 render grammar 内，G02–G06 的资格化审计器能够拒绝已知结构违约，并能对合法等价变换保持不变。

## 2. formal 事实

唯一 formal 于 `2026-08-01T19:47:28.991085+00:00` 开始，于 `2026-08-01T19:47:54.092703+00:00` 结束。fixed root 在求值前占用，没有第二次调用。

formal 结果：

| 项目 | 结果 |
| --- | --- |
| positive Gate | G02–G06 `5/5` |
| adversary lattice | `48/48` |
| transform-family holdout | profile/language/provenance/CPS 各 `1/1` |
| raw metric kill | `45/45` |
| positive metamorphic | `6/6` |
| formal root | 九项顶层 entry 精确 |
| evidence seal | 45 个递归文件复算一致 |
| evidence-seal SHA-256 | `94B80FBD2BB38BC46E95F88E0747D9DB1D8CB279E689F68BB4A43AE7B76D8209` |
| 后续授权 | `authorization_created=false` |

六种合法不变换是 JSON key order、record row order、claim pair row order、全局 alpha rename、source whitespace normalization 和离线 relocation/replay；六项都经公共 API 得到与正控完全相同的 canonical report bytes，且输入保持只读。

## 3. v8 根因是否真正修复

主审没有只重放 lattice，而是从公共 API 构造六项不参与 metric mapping 的新探针：

1. 增加一对完全自洽、并重算 count/hash/seal 的 relation claim，被固定 profile 在 G02/G03 拒绝；
2. 在另一组 condition claim 上互换 positive/negative role、保持 label 与 truth 不变，被 G03 polarity binding 拒绝；
3. 重新注入指向 `/query` 的自报 origin path，被 exact certificate schema 在 G02 拒绝；
4. 用重新计算 token count 和 surface fingerprint 的无意义 CPS 文本替换合法 render，被 G04 reversible grammar/AST binding 拒绝；
5. 将 relation 样本改成由初始 `mid` 值直接给出答案，同时把 FOREACH effect 改写到无关实体；随后重算 teacher digest、semantic fingerprint、ablation answer、manifest 与 seal。G03/G04 仍通过，但 G05 fresh necessity 与 typed provenance 拒绝该捷径；
6. 在 rich/NONE 两个 CPS AST 中同步把真正的 resource precondition 换成 fact shortcut，同时保留资源更新、答案、成本、五类失败原因和唯一 budget leaf 差异，并重算两个 teacher digest、fingerprint、manifest 与 seal。G03/G04 仍通过，但 G06 derived composition witness 拒绝该捷径。

第五、六项最关键：它们排除了“只是 teacher/hash 没更新所以失败”的解释。v9 的新增价值来自独立语义重建，而不是更多外围一致性补丁。

## 4. 为什么现在认为设计可行

设计可行不来自 `48` 或 `45` 这两个数字，而来自不变量所在的位置已经改变：

- cardinality、role、schema 和 snapshot profile 由 runtime 固定，输入不能自我扩容；
- claim truth、causal flip 与 language projection 都从 fresh AST semantics 复算；
- ERE query payload 的 origin、IF branch、relation edge、neighbor 和 effect 由第二套 typed dataflow 重建；
- CPS 四类 composition 由 action topology、candidate plan、budget 和 fresh outcome 联合推导；
- adversary case 只用于检验这些判定器，不再充当判定器的语义真源。

因此，新增未登记样本只要破坏同一原子不变量，就不需要先登记 fault id 才能被拒绝。这正是 v8 缺失而 v9 已经形成的能力。

## 5. 剩余边界

R0B-invariant 仍是有意收缩的有限证明：

1. profile 固定为 11 records、18 claims、2 causal pairs 和 2 language pairs；它没有证明未来生产分布的覆盖率；
2. language 只接受四种可逆 grammar，不审计自由自然语言。生产 renderer 仍需在 R0D 单独资格化；
3. typed derivation 与 accepted simulator 共享同一书面语义，虽然实现独立，仍不能排除规范本身同时错误；
4. semantic fingerprint 仍是有限 schema 的 name-invariant 图摘要，不是通用程序等价证明；
5. 没有 generator、真实 learner、Qwen Boundary/core、cache、训练、GPU 或 matched Pareto 证据；
6. R0C 只能另立合同验证真实 G07/G08 learner measurement，不能把本结果外推为 R0C/R0D 或 P0-M 授权。

## 6. 完成与未完成

已完成：v9 设计、runtime audit、CLI/tests 直接切换、旧 v8 active test source 删除、冻结 guard、preflight、唯一 formal、seal 复算和 registry 外主审。

未完成：R0C、R0D、R1 generator、P0-M、模型、训练、GPU 实验与完整架构验证；这些阶段没有被本验收创建授权。
