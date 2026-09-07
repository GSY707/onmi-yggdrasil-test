# V2-R1R P0-D v17 主设计层验收

## 1. 最终判决

v17 的 repair qualification 与 fresh production formal 均通过主审，判定为：

```text
P0-D production data + verifier accepted
P0-M eligible
architecture not evaluated
```

这不是对 v16 的事后改判。v16 Q 继续只作为 accepted runtime component，v16 F 继续作为 rejected diagnostic；v17 使用新的 generator version、repair root、fresh seed 和 full formal root。

## 2. 证据身份

| 层 | Root | 结果 | Seal SHA-256 |
| --- | --- | --- | --- |
| repair qualification | `artifacts/v2-r1r/p0d-v17-repair-qualification-20260810-1/` | R01–R07 全 true | `6A31B0218F2B46979714A1D8F296BBE085E8FE9F0FF6E9B10476904C3885D103` |
| fresh production | `artifacts/v2-r1r/p0d-v17-full-production-20260810-1/` | G01–G11 全 true | `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D` |

qualification wall time `102.13s`；formal wall time `1958.12s`。正式数据 seed `2026081702`，共 `26,624` records、约 `460.17 MB` JSONL payload；全量 regeneration 字节一致，固定 artifact 的第二轮 audit 报告一致，watched inputs 只读。

## 3. 两个 v16 根因已被机制性关闭

ERE 修复同时具备 builder 后置不变量、50,000-seed `if_copy` sweep、13,312-fingerprint 正式形状 capacity、v16 307 条失败记录逐 seed 回归和 visible-witness fault kill。正式 G05 failure list 为空。因此成功原因不是换 seed，而是 pattern override 后的六值可见域被代码强制成立。

alpha 修复让 fingerprint 与 transform 共享 path-aware grammar/domain 区分。58 个 schema/reserved literal collision 全部 alpha-invariant，旧 path-insensitive transform 对 58/58 正控均被杀死；`cps-validation-0777` 恢复原 fingerprint `090C730635B1CA146653592A828284C8ED65DD0D30D098AC5E39CF7A4931B35F`，同时 ordered-plan 负控仍敏感。正式 G07 failure list 为空。

R02 还证明 v16 已接受的 runtime/G09/G10 受保护文件或函数 AST hash 未变化，故本轮没有借修复统计器来降低既有 Gate。

## 4. 其余正式证据

G09/G10 均通过。最紧 source cell 是 `ERE/validation/full_text_char_3_5_nb`：accuracy `0.29948`、Wilson upper `0.33916`、threshold `0.37083`，仍有 `0.03167` 余量。claim cell 最紧 upper 为 `0.51086 < 0.60`。全部 simulator/answer/teacher replay、pair single-leaf/flip、结构覆盖、长度/tokenizer、model-view 禁止字段与 provenance Gate 均为 true。

这些结果足以关闭 P0-D，但不能证明 Qwen Boundary、K=8 recurrent core、claim supervision、direct/text-CoT SFT 或 GPU 训练通路可用。

## 5. 后继边界

下一步进入独立 P0-M 合同，只运行：ERE K=8 overfit64、CPS K=8 overfit64、ERE+CPS shared K=8 overfit128、direct/text-CoT 各 overfit64，以及 100-step throughput benchmark。smoke 失败首先归因 loss、mask、cache、数值或吞吐并修复；不得扩大数据掩盖。

P0-M 即使全部通过也只建立“训练通路可用”，不建立架构泛化结论。父任务必须停在 P1 前；P1 需要另一个数据 seed、fresh model seed 和跨任务/OOD/因果合同。
