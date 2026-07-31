# V2-R1R P0-D 数据与接口结果

日期：2026-08-01  
证据等级：rejected formal diagnostic（非模型、非训练证据；不得用于 P0-M）

## 结论

generator v1 的生成与旧机器审计已完成，旧 assessment 给出 `passed=true`；但 2026-08-01 主设计层独立复核发现 ERE/CPS 都有无需目标计算即可满分的结构化捷径，且存在 CPS prefix claim 错标、composition/language split 不成立、真实 tokenizer Gate 证明方法错误和 certificate path 未核对。因此该机器通过已被否决，V2-R1R 的正式状态仍是 **P0-D 失败**。完整证据见 `docs/v2-r1r-p0-main-review.md`，修订合同见 `docs/v2-r1-revalidation-task-design.md` 第 16 节。

执行在旧 P0-D 终点停止，未创建 P0-M/model/cache/train 代码，未启动 GPU 训练，也未启动 OPS。这一停止是正确的；下一步只能实现并全量重跑 generator v2，不能进入训练。

## 规范与 provenance

- 唯一设计规范：`docs/v2-r1-revalidation-task-design.md`
- 设计规范 SHA-256：`8e90c4ac9d3413cb959904f43a8a5618c940b8c66a24a804fa5d5609a378c796`
- generator version：`r1r-p0-generator-v1`
- formal seed：`20260801`
- formal manifest：`artifacts/v2-r1r/p0-v1/manifest.json`
- formal assessment：`artifacts/v2-r1r/p0-v1/p0-assessment.json`
- formal audit SHA-256（assessment 中）：`57d0759b3938ea036f7c4b0a768d9daf6e82fb059773fa50118c434dea73403c`

每条 JSONL 记录保留 generator version、episode seed、输入/设计文档 hash、环境、命令、开始/结束时间；manifest 另外记录所有数据文件 SHA-256。正式生成和审计均使用项目 `.venv\Scripts\python.exe`。

## 数据规模

| family | train | validation | OOD splits | causal_pairs |
| --- | ---: | ---: | --- | ---: |
| ERE | 4096 | 512 | composition/length/entity/language 各 512 | 512 条 = 256 对 |
| CPS | 4096 | 512 | composition/horizon/distractor/language 各 512 | 512 条 = 256 对 |

smoke 也按合同完成：每族 train 64、validation 与各 OOD 32、causal_pairs 32 条；smoke 只作 generator/audit 连通性证据，formal Gate 以完整规模结果为准。

## P0 十项硬 Gate

formal `audit.json` 的十项 conjunction 全部为 `true`：

| # | Gate | formal 值 |
| ---: | --- | --- |
| 1 | simulator 重放、answer verifier、teacher trace verifier 100% | `true` |
| 2 | 未声明跨 split semantic/surface overlap 为 0 | `true` |
| 3 | train 与 512 split 的 label/choice balance | `true` |
| 4 | 必要事件、条件、资源或 action 至少 95%，且因果 certificate 有效 | `true` |
| 5 | causal pair 单点修改与答案翻转 100% | `true` |
| 6 | ERE dependency depth、CPS hard-negative/valid-suboptimal coverage | `true` |
| 7 | majority、position、length、last mention、question/full-text unigram NB | `true` |
| 8 | claim positive/negative balance 与 claim-only heuristic 上限 | `true` |
| 9 | token limit、schema、hash、seed、version 与 provenance integrity | `true` |
| 10 | `model_view` forbidden-field audit | `true` |

旧机器总判定：`artifacts/v2-r1r/p0-v1/p0-assessment.json::passed = true`。主设计层判定：`rejected`。这里保留旧布尔值是为了准确记录审计器当时的行为，不代表当前路线 Gate 通过。

## smoke 与 formal 命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_v2_r1r_data.py tests\test_v2_r1r_audit.py
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py generate-p0 --seed 20260801 --output artifacts\v2-r1r\p0-smoke-v1 --smoke --overwrite-smoke
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py audit-p0 --input artifacts\v2-r1r\p0-smoke-v1
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py generate-p0 --seed 20260801 --output artifacts\v2-r1r\p0-v1
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py audit-p0 --input artifacts\v2-r1r\p0-v1
```

结果：目标 pytest `10 passed`；最终 smoke audit `passed=true`；最终 formal audit `passed=true`。最终收尾的 compileall、目标 pytest 与 `git diff --check` 均通过。

## 实现与修复记录

本轮实现是独立的 `src/yggdrasil_v2/r1_revalidation/` package，包含 schema/view、nonce/label symbols、typed simulator、ERE/CPS causal generator、自然语言 renderer 和 audit；CLI 只有 `generate-p0` 与 `audit-p0`。没有触碰旧 `reasoning_medium` 代码。

执行中发现并修复了三类真实工程问题：ERE `SWAP` 路径缺少 COPY 规则；episode seed 重用导致 canonical 去重退化；随机/确定性 label permutation 会同时造成 finite-sample imbalance 或 unigram leakage。最终实现使用 index 派生的可复现 episode seed、每 block salted label cycle，以及独立的 active-mask/answer-position schedule；formal audit 重新生成并通过，而不是沿用失败运行。

失败运行没有覆盖或删除，已保留为可追溯诊断目录：`artifacts/v2-r1r/p0-v1-failed-heuristics/`、`artifacts/v2-r1r/p0-v1-failed-label-balance/`、`artifacts/v2-r1r/p0-v1-failed-answer-balance/`。

## 未完成与路线判决

未完成项包括 frozen-Qwen hidden cache、Boundary、K=1/K=8 latent core、direct/text-CoT baseline、P0-M overfit、P1 OOD/causal intervention、P2 matched Pareto、P3 fresh seeds 和成本账本。这些阶段未被本轮启动，不能从 P0-D 结果推断架构通过。

当前路线判决是“P0-D v1 无效，按第 16 节实现并重跑 P0-D v2”；P0-M、OPS、A1.22A 与 V2-B 全部保持停止。`artifacts/v2-r1r/p0-v1/` 只作为 rejected diagnostic 保留，不得缓存或训练。
