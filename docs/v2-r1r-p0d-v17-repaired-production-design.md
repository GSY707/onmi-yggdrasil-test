# V2-R1R P0-D v17 可见域与路径语义修复合同

## 1. 判定目标

v16 已接受 runtime measurement 窄组件，但 fresh-seed production 因 G05/G07 失败而拒绝。v17 只回答两个问题：

1. ERE pattern-specific override 之后，声明的六个值是否仍全部存在于可见初态，而不是依赖随机背景碰巧补回；
2. alpha-renaming 是否与 semantic fingerprint 使用同一套 AST path 语义，使动态符号可以与 schema/operator literal 同名。

本合同不改变 simulator、renderer、G09/G10 scorer、streaming implementation、P0 Gate 或阈值。v16 formal artifact 永久保留为 rejected diagnostic；v16 runtime qualification 永久保留为 accepted narrow component。

## 2. 直接切换

活动版本直接切换为：

```text
schema_version = yggdrasil.v2-r1r.p0d-v17.record.v1
generator_version = r1r-p0d-v17-visible-domain-alpha-repair
repair qualification seed = 2026081701
capacity seed = 2026081791
fresh formal seed = 2026081702
formal train = 4096 / family
formal heldout = 1536 / split / family
```

v17 不保留 v16 CLI/test compatibility alias。v16 的 source snapshot 只用于证明 runtime scorer 受保护表面未改变。

## 3. 修复设计

### 3.1 ERE visible-domain postcondition

`if_copy` 在写入 branch control value 后，必须在不被该 branch、query 或 counterfactual 使用的第二实体 control attribute 上恢复另一个 witness value。所有 ERE builder 返回前统一执行 postcondition：

```text
declared choice_values 恰有 6 个不同值
base initial-state visible values == declared choice_values
counterfactual initial-state visible values == declared choice_values
```

失败必须立即抛错，不能通过重试、隐藏 salt、缩小 choice domain 或修改 metadata 掩盖。

### 3.2 path-aware alpha semantics

fingerprint 与 alpha transform 共享两条路径规则：

- `rules`、initial-state dynamic maps 与 event arguments 的键是 domain symbol，即使拼写为 `rules`、`value` 或 `SET`；
- 只有 `op`/`kind` 字段中的冻结 grammar value 与 `$neighbor` 是 reserved literal；同一拼写出现在实体、属性、资源、fact、rule、action、parameter 或 candidate 位置时仍是可重命名 symbol。

不允许用 nonce 黑名单规避冲突。

## 4. Repair qualification

唯一 root：

```text
artifacts/v2-r1r/p0d-v17-repair-qualification-20260810-1
```

必须 conjunctively 通过：

| Gate | 要求 |
| --- | --- |
| R01 | v16 runtime qualification assessment 与 evidence seal 完整、Q01–Q09 全 true |
| R02 | runtime qualification、G09 decision、NB scorer、scalable learner 文件字节不变；generator/audit 中被 runtime 使用的函数 AST hash 不变 |
| R03 | 50,000 个 `if_copy` sweep 全部满足六值可见域与反事实翻转；新 capacity seed 下 13,312 个正式形状 ERE fingerprint 唯一且全部满足 postcondition |
| R04 | 全部 schema/reserved 拼写 collision matrix alpha-invariant；ordered plan 负控仍改变 fingerprint |
| R05 | v16 的 307 条 G05 失败全部可按原 record seed 重建为合格；`cps-validation-0777` 精确回归修复且旧 transform 仍能复现失败 |
| R06 | 删除 visible witness 与恢复 path-insensitive transform 两个 fault 都必须被杀死 |
| R07 | qualification 独立复算一致、accepted input 只读、source snapshot 完整且 seal 可复验 |

任一 R Gate 失败，v17 不得创建 fresh formal。

## 5. Fresh formal P0-D

唯一 root：

```text
artifacts/v2-r1r/p0d-v17-full-production-20260810-1
```

使用 seed `2026081702` 生成 26,624 records，完整运行既有 G01–G10，再以全量字节重生、artifact replay 与 watched-input read-only 形成 G11。Gate、阈值、cell topology、Wilson decision、model-view 禁止字段和 causal pair 合同均不改变。

正式 root 只能创建一次。任何 Gate 失败都保留该 root，不修补、不覆盖、不换 suffix；后继必须升 generator version 并消费另一个未使用 seed。

## 6. 结果边界

v17 formal PASS 只表示 production P0-D 数据与 verifier 关闭。父任务完成独立 main review 后，才可进入另立 P0-M 合同。P0-M 只验证 overfit/loss/cache/throughput 通路；P1 才验证跨任务、OOD、因果干预与架构完整性。v17 不产生架构成立结论。
