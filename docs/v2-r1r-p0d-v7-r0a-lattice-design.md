# V2-R1R P0-D v7 R0A-lattice 冻结设计

日期：2026-08-01  
阶段目标：把 ERE operand 的保留 token 作用域从粗粒度标签改成可机械证明的有限覆盖格，并保留 v6 全部窄证据  
证据等级：semantic-oracle / API-boundary qualification；不是数据、模型、训练或架构证据

## 1. 核心判断

v6 的机器结果没有造假：14/14 canonical、4/4 prefix、12/12 corruption、245/245 frozen invalid probes 与封存协议都成立。失败发生在测量系统的完备性声明：四条 primitive placeholder probe 被粗粒度标签汇总成“query、placeholder 均已覆盖”，但 query × placeholder 的组合从未执行。

v7 不在 v6 matrix 尾部追加十条补丁，也不扩大到 R0B。它直接切换覆盖方法：

1. v6 oracle 与 245 项 matrix 以 SHA-256 精确 byte-copy 进入 v7，只作为不可回退的语义/回归证据；
2. 独立声明所有 ERE operand slot、递归作用域等价类与 token 类；
3. 机械展开完整有限笛卡尔积；
4. coverage auditor 检查每个 cell 恰好出现一次，不能信任 `covers` 标签；
5. 另用不导入 lattice builder 的十项 query holdout 防止生成器与审计器共同漏项；
6. formal 仍是固定 root、先占用、只调用一次，之后无论结果如何停止。

阶段名使用 **R0A-lattice**，而不是声称任意未来 schema 的无限“完全验证”。它只对本文件冻结的有限 AST schema、六种作用域等价类和六类 token 做精确 conjunction。

## 2. 继承证据与直接切换

v7 继承但不修改：

- `oracle-spec-v6-bytecopy.json`：SHA-256 `66066999FD0FEEDA6702C696A60E6DB2DDE3C1C4C8A45EBE9E9E20D96885A0E1`；语义 payload 仍为 14 canonical、4 prefix、12 corruption，expected 全部由主设计层手写；
- `legacy-invalid-ast-matrix-v6-bytecopy.json`：SHA-256 `E78204FB28A6771547A0F259227BE8F3B8056CEC118400FB3DD7C3C301D5F4ED`；245 项全部必须继续精确通过；
- 三个公共 API、合法输入输出语义、eager whole-AST validation、输入不变、JSON-only 结果和 exact error message。

v7 执行时直接删除 `tests/v2_r1r_v6/` 活跃旧合同，不保留 pytest ignore、CLI alias、wrapper 或 compatibility mode。v6 文档与 sealed artifact 继续保留历史证据。当前 runtime 最终仍只允许：

```text
src/yggdrasil_v2/r1_revalidation/
  __init__.py
  common/
    __init__.py
    simulator.py
experiments/
  v2_r1_revalidation.py
```

不得新增 runtime helper、schema module、audit、generator、model、cache 或 train 路径。

## 3. 规范性 token 语义

operand 定义为本合同列出的 ERE primitive、predicate 与 query string field。只有这些位置把首字符 `$` 解释为保留语法：

- 普通不以 `$` 开头的非空字符串永远是 literal；
- `$arg:<name>` 只在 rule 环境中有效，而且 `<name>` 必须是当前 rule 的已声明 param；
- `$neighbor` 只在至少一个 `FOREACH_LINKED.effect` 祖先之下有效；
- `$arg:`、引用未声明参数的 `$arg:*` 和其他 `$*` 永远非法；
- query 没有 rule-param 环境，也没有 FOREACH effect 祖先，因此任何 `$` 开头的 query operand 都非法。

本规则不扩展到 state literal、event argument value、identifier 或 CPS string；这些继续是普通数据。这样既修复 query 漏洞，也不把新限制误加到不解析 placeholder 的字段。

所有非法 operand 必须精确抛出：

```text
ValueError("AST_VALIDATION|invalid_placeholder|<exact_json_pointer>")
```

成功与失败调用都不得修改输入。

## 4. 覆盖格

### 4.1 slot 轴

rule/predicate 共 23 个 slot：

- `SET`：target、attribute、value；
- `COPY`：source、target、attribute；
- `SWAP`：left、right、attribute；
- `LINK` / `UNLINK`：各 relation、source、target；
- attribute predicate：entity、attribute、value；
- relation predicate：relation、source、target；
- `FOREACH_LINKED` 自身：relation、source。

query 共 5 个 slot：attribute query 的 entity/attribute，以及 relation query 的 relation/source/target。

### 4.2 作用域轴

递归 AST 深度无限，不能伪称枚举每一条路径。v7 按 validator 的有限状态进行等价类划分，并让每个 rule slot 穿过六种上下文：

| context | 是否在 FOREACH effect 下 | 是否实际执行 |
| --- | --- | --- |
| `rule_direct` | 否 | 是 |
| `rule_if_then` | 否 | 是，true branch |
| `rule_if_else` | 否 | 是，false branch |
| `foreach_direct` | 是 | 是 |
| `foreach_if_then` | 是 | 是，true branch |
| `foreach_if_else` | 是 | 是，false branch |

这同时测量 direct validation、IF then/else 的 scope 透传，以及跨越 FOREACH effect 边界后的 `neighbor_allowed=false→true`。嵌套 FOREACH 自身的 relation/source 也作为 23 个 slot 的一部分进入 true-scope，测量 `true→true`。

query 使用独立 `query` context，`has_rule_environment=false`、`neighbor_allowed=false`。

### 4.3 token 轴

六类 token：

| token class | 代表值 | 合法条件 |
| --- | --- | --- |
| `plain_nonce` | 按 entity/attribute/value/relation 类别给出的普通 nonce | 始终合法 |
| `declared_arg` | `$arg:bound` | 仅 rule 环境合法 |
| `neighbor` | `$neighbor` | 仅 true neighbor scope 合法 |
| `undeclared_arg` | `$arg:missing` | 永远非法 |
| `empty_arg` | `$arg:` | 永远非法 |
| `unknown_dollar` | `$mystery` | 永远非法 |

### 4.4 精确规模

机械展开：

```text
23 rule slots × 6 contexts × 6 token classes = 828
5 query slots × 1 context × 6 token classes = 30
total = 858 cells
```

其中 `350` 个必须被接受并真正执行，`508` 个必须按 exact type/code/path 拒绝；query 为 `30` 个 cell，其中 `25` 个非法。所有 valid witness 都使用语义安全的 state、relation、attribute 与 event argument，不能以“没有抛 AST 错误但随后 runtime 崩溃”冒充接受。

## 5. 覆盖审计器与正负控制

`operand-lattice.json` 是规范性轴定义；`lattice_builder.py` 只把轴展开为具体 AST，不导入 runtime；`coverage_auditor.py` 不调用 runtime，只独立复算 required cell set、validity、counts 与 sentinel。

设计层测试另外硬编码 23 个 rule slot 和 5 个 query slot 的完整 field/category 映射，避免 spec 自己删一行后仍自洽。coverage auditor 必须拒绝：

- 任意 missing cell；
- 任意 duplicate cell；
- 任意 unexpected/invented cell；
- 被篡改的 expected validity；
- 缺少 query sentinel；
- 用 coarse coverage tag 代替具体 cell。

`test_independent_query_holdouts.py` 不导入 lattice builder，以任意 nonce state 独立测试五个 query operand 对 `$neighbor` / `$arg:outside` 的十项 exact error。这是冻结的第二实现路径；formal 后父任务还必须再构造不同 token/name 的新鲜探针。

## 6. 公共实现要求

公共 API 与 `__all__` 不变：

```python
simulate_ere(ast: Mapping[str, Any], prefix: int | None = None) -> dict[str, Any]
evaluate_cps(ast: Mapping[str, Any]) -> dict[str, Any]
replay_cps_prefix(ast: Mapping[str, Any], candidate_index: int, prefix: int) -> dict[str, Any]

__all__ = ["evaluate_cps", "replay_cps_prefix", "simulate_ere"]
```

当前 v6 已通过全部非 query lattice cell。v7 所需语义切换必须在 `_validate_ere_query` 中把五个 query operand 当作无参数、无 neighbor scope 的 ERE operand 验证，而不是普通 `_string`；不得读取 fixture/spec/cell id，不得按 query kind、token 值或 probe id 特判。

实现只能使用标准库，且不得含 fixture/matrix/spec/test/case/control/probe/cell id token。CLI 必须与 v7 frozen template 字节一致，只提供：

```text
python experiments/v2_r1_revalidation.py preflight-lattice
python experiments/v2_r1_revalidation.py seal-lattice
```

## 7. 执行前红灯

v7 设计层自检 `12 passed`。在 sealed v6 实现和 v6 CLI 上，完整 v7 pytest 为 `15 failed, 17 passed`：

- independent query holdout `10/10` 未抛异常；
- generated lattice 的 508 个 invalid cell 中恰有 query `25` 项未抛异常；
- valid lattice `350/350` 通过；
- v6 legacy strict `245/245`、14/4/12 语义均保持；
- 其余失败来自 CLI 尚未直接切换和 `tests/v2_r1r_v6/` 尚未删除。

因此 v7 不是对当前实现恒绿的合同；同时红灯只指向已知作用域缺口与直接切换边界。

## 8. formal protocol 与 artifact

`preflight-lattice` 可反复运行，但固定 formal root 必须保持不存在。`seal-lattice` 没有 output、hash、fixture、manifest 或 suffix 参数，固定 root 为：

```text
artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1
```

formal 必须先以 `mkdir(exist_ok=False)` 占用 root 并写 `attempt.json`，然后才运行任何语义或 lattice check。root 一旦存在，后续调用全部 `BLOCKED`。

成功 artifact 精确包含七个文件：

```text
attempt.json
manifest.json
strict-report.json
coverage-ledger.json
assessment.json
run-metadata.json
evidence-seal.json
```

evidence seal 覆盖前六个文件。

## 9. PASS_R0A_LATTICE conjunction

只有以下同时成立才能得到机器 `PASS_R0A_LATTICE`：

1. frozen manifest 及其全部文件 hash 一致；
2. canonical `14/14`、prefix `4/4`、corruption `12/12`；
3. v6 legacy strict `245/245`；
4. lattice invalid `508/508` exact type/code/path；
5. lattice acceptance `350/350`，实际执行成功且 JSON-only；
6. lattice coverage `858/858`、无 missing/unexpected/duplicate、counts 与 sentinels exact；
7. 连续两次完整 report bytes 一致；
8. 所有成功/失败输入均不变；
9. runtime file/import/public API/CLI boundary exact，v4/v5/v6 活跃测试不存在；
10. formal root、attempt-before-evaluation 与七文件集合 exact；
11. evidence seal 前六文件 hash 全部可复算；
12. `authorization_created=false`，没有 R0B 或后续 artifact。

机器 PASS 仍不等于阶段最终通过。父任务必须复算所有 hash、重跑测试，并使用未冻结的新名字/token/context 做主设计层黑盒验收。

## 10. 停止规则与证据边界

预测试失败只允许修改四个 runtime/CLI 文件；需要改 spec、builder、auditor、tests、expected counts 或 formal path 才能通过时状态为 `BLOCKED`，返回主设计层重立合同。

`seal-lattice` 一旦调用，无论成功、失败、异常或中断都立即停止。不得修改实现、删除/覆盖 artifact、换 suffix 或重跑。机器 PASS 也不自动授权 R0B；R0B/R0C/R0D、R1、P0-M、模型、cache、训练和 GPU 全部等待父任务另行决策。

