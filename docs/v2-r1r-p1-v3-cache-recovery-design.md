# V2-R1R P1 v3：immutable cache recovery 与 P1 恢复

日期：2026-08-11

状态：**冻结的新合同；P1 v2 cache 保持 sealed FAIL；只允许以独立 recovery 证据授权只读复用；任一正式 Gate 失败立即停止；P1 assessment PASS 后才允许另立 P2。**

## 1. 核心判定与修订边界

P1 v2 已正式接受 4,096-heldout decision power 与 fresh 65,536-record data。cache 的 28,672 source entries、191,144 claim entries、132+26 shards 已完整生成；同一 full-content audit 在关闭 progress 输出后通过。唯一正式 cache 仍因 progress/stdout `OSError [Errno 22]` 为 sealed FAIL，不能被重判或改写。

P1 v3 只解决两个工程事实：

1. science ledger 必须先可靠落盘，console telemetry 失败不得改变计算结果；
2. 在全新 root 中以两次独立 full audit 和 immutable-input checks 证明 v2 banks 可由训练只读复用。

本轮不修改 generator、数据、模型子集、Qwen revision、Boundary/core、K、loss、训练预算、seeds、行为/因果 Gate 或 P2 条件；不重新生成 power、data 或 84.2 GB cache。v2 failed root 永久保持失败证据。

## 2. 固定输入、训练语义与 roots

固定 source contract 为 `r1r-p1-v2`。模型选择器仍使用 v2 domain separator，模型/数据顺序 seeds 仍为：K=8 `2026082011`、K=1 `2026082012`、direct `2026082013`、text-CoT `2026082014`、data order `2026082021`。训练规格与 P1 v2 完全相同。

固定只读输入：

```text
artifacts/v2-r1r/p1-v2-preflight-20260810-1
artifacts/v2-r1r/p1-v2-g09-power-20260810-1
artifacts/v2-r1r/p1-v2-data-20260810-1
artifacts/v2-r1r/p1-v2-cache-20260810-1
```

其中 cache 的固定 SHA-256 为：`result.json=1568B67C...9E25`、`evidence-seal.json=55C00FDF...E315`、`manifest.json=0FF152DB...7255`、`source-index.json=D15B8AFF...FCA6`、`claim-index.json=7CB4F176...15E`。活动实现保存完整值并在 preflight/recovery 中机器比较；省略号只用于本文排版。

固定新 roots：

```text
artifacts/v2-r1r/p1-v3-preflight-20260811-1
artifacts/v2-r1r/p1-v3-telemetry-qualification-20260811-1
artifacts/v2-r1r/p1-v3-cache-recovery-20260811-1
artifacts/v2-r1r/p1-v3-k8-20260811-1
artifacts/v2-r1r/p1-v3-k1-20260811-1
artifacts/v2-r1r/p1-v3-direct-20260811-1
artifacts/v2-r1r/p1-v3-text-cot-20260811-1
artifacts/v2-r1r/p1-v3-assessment-20260811-1
```

每个新 root single-use、拒绝覆盖。preflight 封存活动 source identity，并验证全部新 roots absence、v2 power/data PASS、v2 cache 精确失败状态和固定文件 hashes。

## 3. T：telemetry decision-power qualification

正式 qualification 不读取 84.2 GB payload，也不授权训练。它必须通过以下合取 Gate：

| Gate | 判定 |
| --- | --- |
| T01 | progress ledger 使用同目录临时文件、flush/fsync 与 atomic replace，且 ledger 写入先于 console |
| T02 | console 抛出 `BrokenPipeError` 时 event 已持久化、调用不失败、后续 console 被禁用 |
| T03 | console 抛出 `OSError(EINVAL)` 时同样 fail-open，仅 telemetry state 记录错误 |
| T04 | ledger/artifact persistence 抛出 `OSError` 时必须向上传播，保持 science fail-closed |
| T05 | final-status console 抛出 `BrokenPipeError/OSError` 时不得覆盖已持久化 result/seal |
| T06 | 正常路径 event index 连续、ledger canonical replay 相等、无临时文件残留 |

T01–T06 全 true 才允许 recovery。这里的 fail-open 只适用于 console；正式 artifact、ledger、checkpoint、manifest 或 seal 写失败仍必须失败。

## 4. R：immutable cache recovery

recovery 对 v2 cache 只读执行，输出只写入新的 v3 recovery root：

| Gate | 判定 |
| --- | --- |
| R01 | v2 cache 231-file evidence seal 全量有效，固定 metadata hashes 全匹配 |
| R02 | v2 result/failure 精确为 cache-stage `OSError [Errno 22]` 且 `passed=false` |
| R03 | v2 power/data 仍为 sealed PASS，dataset/model-subset identity 与 cache manifest 一致 |
| R04 | full-content audit A：source `28672/132/17181347`、claim `191144/26/3382066` 全 entry 通过 |
| R05 | full-content audit B 独立重放并达到同一全覆盖 |
| R06 | audit A/B canonical bytes 完全一致，`failures=[]`、`passed=true` |
| R07 | audit 前后 source root 文件集合、size/mtime fingerprint 与固定 metadata hashes 不变 |
| R08 | recovery result 明确记录 `source_cache_formal_passed=false`、`training_cache_authorized=true`，不得重判 v2 |
| R09 | seal 前的 result、两份 audit、telemetry report 与 source-snapshot 输入完整；`run_stage` 随后封存 root，下游只接受有效 seal |

R01–R09 全 true 时，`training_cache_authorized=true` 只表示训练 runner 可以只读打开 v2 banks；它不把 v2 cache result 改为 PASS，也不形成任何架构结论。

## 5. 恢复后的 P1 顺序

唯一顺序为：implementation tests → v3 preflight → telemetry qualification → cache recovery → K=8 → K=1 → direct → text-CoT → P1 assessment。

- 训练 runner 必须同时要求 v3 recovery sealed PASS、v2 cache 固定 metadata hashes 和 v3 active-source identity；
- K=8 仍先于所有对照，模型、预算、Gate 与 P1 v2 完全相同；K=8 FAIL 立即停止，不运行 K=1/baselines；
- assessment 使用 v2 power/data、v3 recovery 和 v3 训练 roots，且必须明确 v2 cache formal 仍为 FAIL；
- P1 assessment 全 Gate PASS 后，父任务另立 P2 matched-Pareto 合同；活动 P1 CLI 不暴露 P2。

## 6. 证据与停止规则

- smoke/pytest 只证明实现；T formal 只证明 telemetry decision power；R formal 只证明 immutable cache 可复用；K=8 及后序才产生模型证据；
- 任一新 formal 阶段 FAIL：封存该 root，立即停止，不修补、不覆盖、不重跑、不继续后序；
- v2 roots 禁止写入、移动、重 seal 或追加文件；
- 不得把 recovery PASS 写成 P1、P2 或架构 PASS。
