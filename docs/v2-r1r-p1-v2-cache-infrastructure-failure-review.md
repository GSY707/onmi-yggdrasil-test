# V2-R1R P1 v2 cache 基础设施失败复核

日期：2026-08-11

证据地位：P1 v2 唯一正式 preflight、G09 power、fresh-data 与 cache roots 的主设计层复核。本文不修改已冻结合同，不重判正式 FAIL，也不授权 K=8、后续对照或 P2。

## 1. 核心判决

P1 v2 成功解决了 P1 v1 的测量功效问题：正式 Q01–Q09 全通过，fresh seed 的 65,536-record data formal 为 G01–G11 全 true。原失败敏感单元 `ERE/validation/full_text_char_3_5_nb` 在 4,096 条 validation 上 accuracy 为 `0.29638672`、simultaneous upper 为 `0.32033553`，明显低于 `chance+0.10=0.37087402`。

P1 v2 随后在 cache 阶段正式失败，但失败不来自数据或 hidden 内容。132 个 source shards 和 26 个 claim shards 已全部生成，manifest/index 均落盘；外层执行工具在四小时等待上限退出后，存活的正式子进程继续运行，并在 cache 内容审计的 progress callback 中返回 `OSError: [Errno 22] Invalid argument`。正式 root 因而正确写成 `passed=false` 并封存。K=8、K=1、direct、text-CoT、P1 assessment 与 P2 均未启动。

## 2. 正式通过的前序证据

| 阶段 | 正式结果 | seal SHA-256 |
| --- | --- | --- |
| preflight | PASS；冻结 source identity、上游 seals、CUDA、磁盘和 roots absence 全通过 | `F247C6EB7EE8C9991D400B97F77E04DD3C5D76030569694A6416F594A94E7443` |
| G09 power | Q01–Q09 全 true；100,000 trials、8192/4096 topology | `102699D8A7801B9FB1632D541EE2B7BF81948D8BF494DDE50D0D1E36E9777218` |
| fresh data | G01–G11 全 true；65,536 records；两次生成、只读 replay、watched-source 与固定模型子集全一致 | `47FC5B56D21F1A83288877B1EA5CA40458A59C523394A880092560A8713409F6` |

模型子集固定为 12 个非 train cells 各 1,024 条，共 12,288 heldout sources；selection SHA-256 为 `9A6317109D67A5B059BA65902BD21F3F5FCF2DB6FCBC5EDFBC83749DCE503389`。两次完整 dataset tree SHA-256 均为 `8A5BDCE03C09F902A0A106F107B9FF285CE2CBB058FE258988B74171E81F2342`。这证明 4,096 只扩大统计测量，不扩大后续模型 heldout 成本。

## 3. cache 正式执行事实

正式 cache root 为 `artifacts/v2-r1r/p1-v2-cache-20260810-1/`。生成阶段完成：

| bank | entries | shards | tokens |
| --- | ---: | ---: | ---: |
| source | 28,672 | 132 | 17,181,347 |
| claim | 191,144 | 26 | 3,382,066 |

cache payload 约 84.2 GB。source 编码能持续达到约 50–100% GPU utilization、约 50–100 W；claim 路径则因同一个 source-longest benchmark 选出 batch `4`，大量短 claim 被 kernel-launch/Python 开销主导，常见功率约 15–30 W，每个 shard 约 5.3–7.2 分钟。它是明确的吞吐工程债，但没有产生 OOM、截断、缺 shard 或内容错误。

外层命令在 `14,404s` 后以 exit `124` 停止等待，但没有杀死子进程。子进程完成 claim 26/26、写出 manifest/index，并进入 `cache:audit:source:start`。随后正式 result/failure 均记录：

```json
{"error":"[Errno 22] Invalid argument","error_type":"OSError","passed":false,"stage":"cache"}
```

失败 root 的 231-file evidence seal 已重新全量验证，seal SHA-256 为 `55C00FDF43757F5E4B958FA0E6C672AC033B3181D480859D7F314F731083E315`。所有后续 roots 均不存在。

## 4. 停止后的只读根因闭合

封存后使用同一 `audit_p1_cache(..., verify_content=True)` 对同一 artifact 做只读重放，只把 progress callback 替换为无输出函数。结果为：source 28,672/132/17,181,347 全内容通过，claim 191,144/26/3,382,066 全内容通过，`failures=[]`、`passed=true`。

因此可以排除 dataset identity、manifest、index schema、禁止字段、shard hash/shape、offset、FP16 finite、逐 entry hidden hash、容量和 batch qualification。正式 `EINVAL` 被定位在 progress callback，而不是 cache 内容审计本体。该 callback 只做 stdout `print` 与 ledger 写入；结合外层等待刚刚超时、子进程失去稳定输出通道的时间顺序，stdout/telemetry 通道失效是最强解释。

这项 post-stop PASS 是诊断证据，不覆盖正式 `passed=false`，也不能让现有 runner 的 `require_sealed_pass(cache)` 放行。

## 5. 架构影响

当前仍没有 P1 行为或架构结果。已经确认的是：

- 4096-heldout 的测量修复成立，fresh data 可进入模型路径；
- Qwen source/claim hidden cache 已物理生成且内容审计可通过；
- P1 v2 被基础设施 telemetry 失败截断在 K=8 之前，因此没有检验 shared recurrent core；
- 不能据此上调或下调架构成立概率，也不能进入 P2。

## 6. 推荐后继：另立 P1 v3 cache-recovery

不应重生成数据，也不应重做约四小时 cache。下一合同应把 P1 v2 failed cache root 作为 immutable input：

1. 固定 P1 v2 preflight/power/data/cache 的 result、manifest、index 与 evidence-seal hashes；旧 root 不修改。
2. 先资格化 stdout 断开：science ledger 必须先原子写文件，console telemetry 只能 best-effort，`BrokenPipeError/OSError` 不得使计算结果失败；真实 artifact 写失败仍必须 fail-closed。
3. 在全新小 root 中对 sealed cache 做两次独立 read-only full audit，要求 canonical report 相等、source/claim 全 entry 覆盖、旧 root seal 有效。
4. 可加入固定 hash 选择的 source/claim spot replay，用 pinned Qwen revision 重算 hidden，作为比原合同更强但小成本的 provenance 正控。
5. recovery assessment PASS 后，训练 runner 只读使用旧 packed banks，并以 recovery seal 代替旧 cache `passed`；先 K=8，后序与 P1 v2 Gate/seed/预算不变。
6. P2 前单独修 cache 工程：source 与 claim 分开 benchmark，候选按 token bucket 扩到适合短 claim 的 batch，并报告 tokens/s、launch 数与功率；该优化不得回写本次 P1 cache。

## 7. 完成与未完成

已完成：P1 v2 合同与预测试、正式 preflight、G09 power、65,536-record fresh data、完整 source/claim cache 生成、失败 root 封存、seal 全量复核与 post-stop 内容审计。

未完成且未运行：正式 cache PASS、K=8、K=1、direct、text-CoT、P1 assessment、P2 matched-Pareto、多 seed/P3 与架构通过判决。
