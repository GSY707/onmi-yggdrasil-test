# V2-R1R P0-D v16 主设计层验收

日期：2026-08-10

验收对象：

- `artifacts/v2-r1r/p0d-v16-runtime-qualification-20260810-1/`
- `artifacts/v2-r1r/p0d-v16-full-production-20260810-1/`
- `docs/v2-r1r-p0d-v16-runtime-qualified-production-design.md`
- `docs/v2-r1r-p0d-v16-runtime-qualified-production-execution-command.md`

## 1. 最终判决

v16 的两个阶段必须分开判定：

1. **runtime qualification accepted**。唯一 Q 的 Q01–Q09 全 true，稳定证明当前 scorer/audit 路径在冻结机器与工作负载上具备 exact identity、稳健速度、完整旧规模预算、内存上限和 fault sensitivity。v15 的单点墙钟悖论在新合同下被正确解决，没有通过降低统计 Gate 或事后接受 v15。
2. **fresh-seed P0-D rejected**。唯一 F 的 G05、G07 false，其余 G01–G04、G06、G08–G11 true，最终状态 `FAIL_P0D_PRODUCTION`。G05 是 307 条 ERE 的真实 visible-domain 不变量破坏；G07 是一条 path-insensitive alpha-renamer 误报。G07 即使按主审归因为测量 bug，也不能消除 G05，因此 v16 不能关闭 P0-D。

正式结论是：**v16 runtime measurement component accepted；v16 full production machine-fail、main-review rejected**。不得把 G09/G10 通过、26,624 条成功生成或 G07 误报写成近似 P0-D PASS；P0-M、模型、cache、GPU 与训练继续禁止。

## 2. Q：运行资格化证据

唯一 Q root 完整形成 11 个顶层 entry。父任务复算 58 个 root-seal entry 全部一致，evidence-seal SHA-256 为：

`EFBF833D5E55321012C98E6FF67C6B19E509687360E61588D4E34063B958513E`

source snapshot 为 57 个文件，重算摘要 `6CFD5EAEFE887EF0474D5937BA81223AFB90A44B8E65BE7BF2C9764461C97755`，与 run metadata 一致。Q assessment SHA-256 为 `3DC177F91595638582A8E0F2055DB7690D5999312772CD795116591BB6B45B8C`。

关键运行结果：

| 项目 | 正式结果 | Gate |
| --- | ---: | --- |
| paired predictions | 192 × 15，逐轮 exact identity | PASS |
| ordinary exact fallback | 0 | PASS |
| paired speedup median | 6.7282x | `>=5.0x` |
| paired speedup minimum | 6.1559x | 诊断值 |
| fast wins | 15/15 | `>=14/15` |
| fast p95 | 0.1119 s | `<=0.20 s` |
| 完整旧规模 | 14,336 records、98 source cells、14 aggregates、24 claim cells | 全部投影一致 |
| load+G09+G10 | 170.81 s | `<=240 s` |
| peak working set | 3,457,564,672 bytes | `<=4.5 GiB` |
| streaming | single bank、43,008 batch predictions、12/12 events | PASS |
| runtime faults | 18/18 | PASS |

Q 对 v14 的两格 G09 失败只要求原样复现，没有将其改成通过；v15 assessment 也保持整体 FAIL。Q 的意义严格限于当前运行/测量组件，不是数据、模型或架构正证据。

## 3. F：完整 formal 事实

F 首次将 root seed `2026081602` 送入 builder。生成 ERE/CPS 各 13,312、总计 26,624 records，包含 1,536 causal pairs 与 311,636 claims。artifact 共约 461 MB；正式运行约 2,102 秒。父任务复算 75 个 root-seal entry 全部一致，evidence-seal SHA-256 为：

`2CE3CBCDB720FA777C1E357026209108D3B57C444E751FD2689ED9C6B6AB74E7`

F source snapshot 59 文件的摘要为 `C1E105F7587D5E106CF966E0D9B14E71428E1ECF4963F575287C7144C353B8A8`，与 run metadata 一致。assessment SHA-256 为 `894FFDB047BB6C4FDA897A91383875D5CFA05FF756226A429EBE3C23FABF871A`。full regeneration byte-identical、artifact report replay identical、accepted inputs read-only，G11 true。

机器 Gate：

| Gate | 结果 | 主审解释 |
| --- | --- | --- |
| G01–G04 | true | 上游/Q 引用、schema/provenance、26,624 replay、全局唯一与 1,536 causal pairs 均通过 |
| G05 | **false** | 307 条 ERE 未保持声明六值全部出现在初态 |
| G06 | true | CPS optimum/NONE、suboptimal、hard negative 与 OOD 结构通过 |
| G07 | **false** | 一条 schema-name collision 触发 path-insensitive alpha renamer 误报；标签配平及其余 permutation 通过 |
| G08 | true | 最大 995 tokens，train/validation p99 558/559/661/661，无截断 |
| G09 | true | 98/98 source cells、14/14 aggregates 全部通过 |
| G10 | true | 24/24 claim cells 与 balance 全部通过 |
| G11 | true | regeneration、artifact replay、只读输入、snapshot 与 seal 通过 |

G09 的最紧 source cell 仍有余量：`ERE/validation/full_text_word_nb` accuracy `0.30013`、upper `0.33983`、threshold `0.37083`；最紧 family aggregate 为 `ERE/full_text_char_3_5_nb`，upper `0.28368 < 0.31042`。这说明 v14 的 shortcut failure 没有在新 seed 重现，但不覆盖 G05/G07 conjunction。

## 4. G05 根因：后置覆盖破坏 visible six-value witness

v15 为完整规模语义容量加入了三 attribute、六 value 的可见状态结构。`_state()` 先在第二 attribute 的两个 entity 上放置 `values[2:4]` witness，再在第三 attribute 上放置 `values[4]、values[5]、values[0]、values[1]`，理论上使六值全部进入初态。

`if_copy` builder 随后又把第一个 entity 的第二 attribute 强制写成 control value `values[2]`。当原先随机 witness 排列把 `values[3]` 放在这个位置、另一个位置放 `values[2]`，且随机背景没有再次出现 `values[3]` 时，后置覆盖会把六值域压成五值域。事件、query、answer、claim 和 causal certificate 仍然可重放，所以 G03/G04 通过；但 `generation_metadata.choice_values` 声明六值，初态实际只含五值，违反 G05 冻结合同。

父任务对 sealed dataset 全量只读复算得到 307/13,312 条 ERE 失败，全部属于 `if_copy`：

| split | 失败 records |
| --- | ---: |
| train | 138 |
| validation | 53 |
| language_ood | 54 |
| entity_ood | 10 |
| causal_pairs | 52（26 对） |
| 合计 | 307（约 2.31%） |

formal report 的通用 `_metric` 只保留前 50 条 failure message，因此 artifact 的 failure list 不能直接给出总数；主审计数来自对 sealed 13,312 条 ERE 的同一 domain predicate 全量复算。每条都只缺一个声明 value，没有额外 value。

这是生成器 postcondition 缺失，不是 seed 运气应被接受。新 seed 只是把单样例测试未覆盖的状态覆盖顺序暴露出来。

## 5. G07 根因：alpha test transformer 不理解动态路径

唯一 G07 failure 是 `cps-validation-0777`。该样本合法生成了资源名 `rules`。正式 `semantic_fingerprint()` 按路径识别 `initial_state.resources` 的键为动态符号；但 `_alpha_rename()` 只检查字符串是否属于全局 `SCHEMA_KEYS`，因此把动态定义键 `rules` 留在原处，同时把 action 中对该资源的引用改成 `z0`，人为制造了定义/引用断裂。

主审用独立 path-aware rename 对同一 sealed AST 复算：原 fingerprint 与重命名 fingerprint 均为 `090C730635B1CA146653592A828284C8ED65DD0D30D098AC5E39CF7A4931B35F`。这证明原程序的 alpha 语义没有改变，失败来自审计变换器，而不是 dataset fingerprint 或 simulator。

这条误报仍有工程意义：现有测试只用普通随机名字验证 alpha，不包含动态 symbol 与 schema literal 冲突。未来应让 renamer 按 AST path 区分 schema field 与 dynamic key，而不是简单把 `rules` 加入 nonce 黑名单；后者会缩小任务域并掩盖 parser/fingerprint 对同名符号的真实鲁棒性。

## 6. 为什么 preflight 没有提前捕获

v16 的容量测试证明了非正式 seed 下 13,312 个 ERE semantic fingerprint 可唯一生成，但没有对每个生成结果复用完整 G05 postcondition。另一个 domain 测试只检查单个固定 seed，恰好没有触发 witness 被覆盖后缺值。alpha 测试同样只覆盖每族一个普通样本，没有 adversarial schema-name collision matrix。

因此本轮 preflight 对“容量”证明过窄，对“每条记录域不变量”和“符号命名空间碰撞”证明不足。正式 F 不是无效浪费：它正确发现了两个分布规模才显现的缺口，并且没有被 G09 性能问题再次遮蔽。

## 7. 后继边界

本轮必须停在 v16，不修补、不覆盖、不换 suffix/seed 重跑。若另立 v17，合理的最小范围是：

1. 在所有 pattern-specific override 之后建立并验证 ERE visible-domain postcondition，确保六个声明 value 都存在于不影响答案的可见状态位；不能加入隐藏 salt。
2. 将 alpha rename 改成与 fingerprint 相同的 path-aware dynamic-key 语义，并加入 schema/reserved literal collision 正控与负控。
3. 用独立非正式 seeds 对所有 ERE pattern/split 做分布级 postcondition probe；capacity、domain、causal necessity 必须同时检查，而不是只查 fingerprint uniqueness。
4. 保持 v16-Q 的 scorer/runtime 核心字节不变时，可把其作为 accepted 窄组件引用；任何 scorer、G09/G10 或 streaming 改动都必须重新资格化。
5. 通过新的 repair qualification 后只能使用另一个未生成的新 seed formal；seed `2026081602` 已消耗，永久只作 rejected diagnostic。

这些是后继设计建议，不构成 v17 执行授权。当前 P0-D 未关闭，P0-M 及之后所有模型/训练阶段保持停止。
