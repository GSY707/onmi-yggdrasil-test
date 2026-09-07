# V2-R1R P0-D v5 R0A 主设计层验收

日期：2026-08-01  
验收对象：`artifacts/v2-r1r/p0d-v5-r0a-20260801-1/`、当前公共 simulator、冻结 fixture/validator/runner/tests 与执行记录

## 结论

执行层产出的机器状态是 `PASS_R0A`，其 artifact 内部完整性与冻结样例结果可以复现；主设计层不接受它作为 R0A 阶段通过。项目状态记为 **R0A machine-pass、main-review rejected**，R0B 继续未授权。

拒绝原因不是 14 个手写世界的语义值错误，而是公共 API 没有满足同一份冻结合同中的 fail-closed 语法要求。合同明确规定未知 primitive/condition/effect、参数缺失和格式错误必须抛出明确异常；主审黑盒探针证明当前 simulator 会静默接受这些输入。冻结资格测试与 sealed assessment 没有覆盖该要求，因此机器 conjunction 比书面合同窄。

本轮不提供数据、训练或 latent 架构证据，也不改变 A1.20D 仅为机制正控制、V2-A 尚未通过的状态。

为保持 artifact 所记录的 execution-time SHA-256 可复算，`docs/v2-r1-revalidation-task-design.md` 与 v5 执行命令维持冻结字节，不在文件内回写本结论；当前状态与权限由本文、结果文档、下一阶段计划、README 和目录索引共同覆盖。

## 可接受的机器证据

主设计层独立复核确认：

- `contract_guard` 重新运行通过，冻结文件为 `11` 个；
- `pytest -q tests/v2_r1r_v5` 为 `10 passed`，compileall、唯一 CLI help 与 `git diff --check` 退出码均为 `0`；
- sealed root 只有 `manifest.json`、`oracle-report.json`、`assessment.json`、`evidence-seal.json`、`run-metadata.json` 五个文件；
- machine assessment 为 14/14 canonical、4/4 prefix、12/12 negative，coverage、连续两次 report bytes、import boundary 与 frozen validation 全真；
- evidence seal 的 manifest/oracle-report/assessment 三项 SHA-256 均可按当前 artifact bytes 复算；当前 simulator 与 CLI hash 也分别等于 manifest 记录；
- v4 audit/tests 已直接删除，CLI 与冻结 template 逐字节一致；没有 R0B、generator、数据、模型、训练或 authorization 产物。

因此，这个 artifact 可以继续作为“当前实现逐字段复现冻结 14 个世界，且 validator 能发现 12 个指定 corruption”的正诊断，不应删除或覆盖。

## 阻断项：公共 API 没有 fail closed

主审从冻结正例复制 AST，只删除必要字段或加入未执行的坏定义；没有修改 fixture、expected、实现或 sealed artifact。下列六项合同期望均为抛异常，实际均正常返回：

| 探针 | 合同期望 | 实际 |
| --- | --- | --- |
| 已执行 ERE `attribute_equals` predicate 缺 `value` | 明确异常 | 静默取 `None` 并选择 false 分支 |
| 未执行 ERE rule 含未知 primitive | 明确异常 | 未遍历该 primitive，正常返回 |
| 未执行 ERE rule 的 `SET` 缺 `value` | 明确异常 | 未遍历该 primitive，正常返回 |
| 已执行 CPS `attribute_equals` precondition 缺 `value` | 明确异常 | 静默视为 precondition false |
| 未使用 CPS action 的 `add_fact` effect 缺 `fact` | 明确异常 | 未校验 effect 参数，正常返回 |
| 未使用 CPS action 的 `fact_true` condition 缺 `fact` | 明确异常 | 未校验 condition 参数，正常返回 |

根因是当前实现把“AST 合法性验证”与“当前路径执行”混在一起：`_ere_rules` 只验证 rule 容器，不递归验证所有 primitive；`_cps_actions` 只验证 condition/effect 的 kind，不验证完整字段；`attribute_equals` 用 `get("value")`，使缺字段退化成普通 false。这样一来，非法但未执行的定义可以进入系统，已执行的缺值条件也会被解释为业务语义，而不是 schema 错误。

这不是附加完美主义。R0B 将依赖 R0A simulator 生成回放真值；若 oracle 对非法 AST 静默降级，后续 audit 可能再次把生成器或 schema 错误误写成合法负例，重现 v3/v4 的自洽测量问题。

## 执行协议偏差

执行层在正确 formal 调用前，曾把 execution hash 手误写为错误值并调用同一 CLI。runner 在创建 output root 前 fail-closed，未产生 artifact、未暴露语义结果、未修改实现；随后使用正确参数生成了当前 artifact。因此当前五文件的内容与 hash 没有被这次手误污染，反适应防线的实质目的也没有被突破。

但从字面合同看，“sealed command 只运行一次”并未完全遵守，不能再宣称这是无偏差的单次调用。下一合同必须把 hash 从人工复制改为由冻结 manifest/launcher 读取，并把“只读 preflight”与“原子创建 attempt record 后的 formal run”做成两个不可混淆的命令；runner 还应拒绝任何非精确 sealed root。

## 下一步边界

v5 R0A 到此停止，不修补、不删除、不覆盖或换 suffix 重跑；R0B/R0C/R0D、R1、P0-M、模型和训练仍未授权。

若继续，应另立一个新的 R0A-strict 合同，而不是给当前 artifact 打补丁。新合同至少需要：

1. 在实现前冻结可执行的 malformed-AST matrix，逐类覆盖全部 primitive/condition/effect、缺字段、未知 kind、非法 prefix/index、未使用坏定义和重复定义；每项冻结异常类别与定位；
2. 公共 API 先对整棵 AST 做 eager validation，再开始任何状态执行；执行器不得兼任 schema 猜测器；
3. 对“rules 是 mapping，无法表达重复 key”的矛盾作一次明确设计：改为可检测重复的 declaration list，或删除不可执行的重复-rule 要求，不能继续把它留在 prose；
4. formal launcher 从冻结清单读取 hash，强制精确 output root，并在运行前原子写入 attempt provenance；
5. 当前实现必须先在新矩阵上形成预期红灯，修复后才允许新的、不同版本的 sealed root。

在这些条件冻结并由主设计层另行授权前，不进入 R0B。
