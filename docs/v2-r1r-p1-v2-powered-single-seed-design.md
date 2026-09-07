# V2-R1R P1 v2：4096-heldout powered single-seed falsification

日期：2026-08-10

状态：**冻结的新合同设计；P1 v1 不重判；P1 v2 任一正式 Gate 失败即停止；只有 P1 assessment PASS 才允许另立并启动 P2。**

## 1. 核心判定与修订边界

P1 v1 的唯一 data formal 在 `ERE/validation/full_text_char_3_5_nb` 失败：point accuracy `0.33203125` 低于 `chance+0.10=0.37099609`，但 1,024 条样本的 98-cell Bonferroni-Wilson upper 为 `0.38190468`。停止后的只读复算表明，若真实 accuracy 保持不变，heldout=2,048 的单次通过概率只有约 `0.638`，heldout=4,096 提高到约 `0.974`。

v15 的 1,536-heldout qualification 只预注册接受局部 `+0.04` 和拒绝局部 `+0.10`；P1 v1 把 heldout 改成 1,024、train 改成 8,192，却没有重新资格化该功效。P1 v2 因此修复 **measurement power 与 model-evaluation cost 的耦合**，不修改 generator 语义、G09 ceiling、98/14-cell topology、模型、loss、训练预算或行为 Gate。

P1 v1 的 preflight/data roots 保持只读失败证据。P1 v2 使用全新 power/data/model/order seeds 和全新 roots，不追加、覆盖或重跑 v1。

## 2. 固定 seeds、规模与 roots

| 项 | 固定值 |
| --- | --- |
| power seed | `2026082001` |
| fresh generator seed | `2026082002` |
| K=8 / K=1 seed | `2026082011` / `2026082012` |
| direct / text-CoT seed | `2026082013` / `2026082014` |
| shared data-order seed | `2026082021` |
| train / full heldout | 每族 `8192 / 4096` |
| model-evaluation heldout | 每族每 split 固定 `1024` |

固定 roots：

```text
artifacts/v2-r1r/p1-v2-preflight-20260810-1
artifacts/v2-r1r/p1-v2-g09-power-20260810-1
artifacts/v2-r1r/p1-v2-data-20260810-1
artifacts/v2-r1r/p1-v2-cache-20260810-1
artifacts/v2-r1r/p1-v2-k8-20260810-1
artifacts/v2-r1r/p1-v2-k1-20260810-1
artifacts/v2-r1r/p1-v2-direct-20260810-1
artifacts/v2-r1r/p1-v2-text-cot-20260810-1
artifacts/v2-r1r/p1-v2-assessment-20260810-1
```

每个 root single-use、拒绝覆盖。preflight 必须证明所有后续 roots 不存在；所有正式阶段必须验证 preflight 的 active-source identity、前序 PASS seal 与 P1 v1 失败证据 hash。

## 3. Q：4096-heldout G09 decision-power qualification

Q 使用生产 G09 的同一个 `evaluate_g09_counts`、同一个 one-sided Wilson、Bonferroni `0.05/98`、cell ceiling `+0.10`、aggregate `0.05/14` 与 ceiling `+0.05`。固定 NumPy PCG64、100,000 trials、train=8,192、heldout=4,096，同时模拟完整 98-cell 矩阵。

| Gate | 判定 |
| --- | --- |
| Q01 | 98 source cells、14 aggregates、七 baseline、train/heldout counts 与默认决策参数精确 |
| Q02 | global null 的 accept one-sided 95% Wilson lower `>=0.995` |
| Q03 | ERE/CPS 单格 `+0.04` 的 accept lower 均 `>=0.995` |
| Q04 | ERE/CPS 单格 `+0.06` 的 accept lower 均 `>=0.950` |
| Q05 | ERE/CPS 单格 `+0.10` 的 reject lower 均 `>=0.995` |
| Q06 | ERE/CPS 六 heldout char-NB diffuse `+0.05` 的 reject lower 均 `>=0.995` |
| Q07 | 向量化判定与公共 scalar API 逐 trial parity 全等 |
| Q08 | 缺 cell、非法 count/random、false complete 与三种参数 mutation 的 14 项 fault 全 kill |
| Q09 | P1 v1 sealed counts 经同一公共 API 仍精确失败于原 G09 cell，证明没有削弱测量器 |

Q01–Q09 全 true 才允许 fresh data。Q 不读取新 data seed，不生成任务数据，也不授权 cache/training。

## 4. D：完整数据与固定模型子集

继续使用 accepted `r1r-p0d-v17-visible-domain-alpha-repair` generator。每族 train `8,192`；validation、四个 OOD、causal-pairs 各 `4,096` records。每族共 `32,768`、两族共 `65,536`；causal records 共 `8,192`，即 `4,096` 完整 pairs。

G01–G10 对全部 65,536 records 运行；G09 的所有 heldout cells 使用完整 4,096。G11 要求第二次全量生成逐 bytes 相等、artifact read-only replay 相等、watched inputs 不变、model-eval subset 重算逐 bytes 相等。

模型路径不缓存全部 4,096。data formal 在不知道 label、answer、source 或 metric 的选择器中，只对 `contract-version + family + split + example_id` 做 domain-separated SHA-256 排序；普通 split 取最小 1,024 个 hash。causal split 只对 `pair_id` 排序，取 512 个 pair 并保留两端，共 1,024 records。选择器固定报告输入字段、salt、全部 id、selection hash、pair 完整性和计数；不得按 answer、mask、token length、pattern、G09 prediction 或模型表现选择。

train 全部进入 cache/training；每个非 train split 只有上述固定 1,024 条进入 Qwen cache和模型 Gate。因此 source cache 总量仍为 28,672 records，与 P1 v1 计划规模相同。full audit bank 不进入模型、teacher 或 checkpoint selection。

只有 G01–G11、subset audit 与 evidence seal 全通过，才允许 cache。

## 5. C/K/B：模型与训练合同不变

cache、K=8、K=1、direct、text-CoT 沿用 P1 v1 已冻结的实质合同：

- `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`；latent 路径冻结 Qwen，packed FP16 mmap final hidden；
- shared token-wise `2048→512` Boundary、两层八头 recurrent core、K=8/K=1、T<=24、九类 latent-only readout；
- update 1–2,400，batch 16，统一 data-order ledger；answer/claim pair/ranking/owner contrast 全程保留，checkpoint selection 只从 update>=1,200 的双任务 validation 选择；
- K=8 仍先于所有对照，validation 每族 `>=0.85`、每个 OOD `>=0.75`、causal flip `>=0.80`、middle intervention drop `>=0.40`、T1 hard-OOD drop 每族 `>=0.15`、aux strip delta `<=0.01`，完整性 Gate 全通过；
- K=1 使用完全相同的训练/评测合同；direct/text-CoT 使用同数据、同 episode ledger、top-4 rank-8 LoRA、2,400 updates 与 greedy answer；
- baseline 允许在固定候选 batch 中以最长输入做 GPU 安全/吞吐资格化；只能改变 batching，不得改变 prompt、token、greedy decode 或答案判定。

P1 v2 不在线测 Pareto。最终 PASS 仍是单 seed cross-task behavior/causal 证据，不是多 seed 稳定性。

## 6. 顺序、停止与 P2 条件

唯一顺序：implementation tests → preflight → Q → D → cache → K=8 → K=1 → direct → text-CoT → P1 assessment。

- 任一阶段 FAIL：封存当前 root，立即停止，不修补、不换 seed、不继续后序；
- K=8 FAIL：不得运行 K=1 或文本基线；
- P1 assessment 只有 P101–P108 全 true 才返回 `PASS_P1_SINGLE_SEED`、`p2_eligible=true`；
- 用户已授权：若 P1 PASS，父任务随后另立完整 P2 matched-Pareto 合同并直接切换/启动；P1 CLI 自身不包含 P2，不能在没有新设计、Gate、root 和 source snapshot 的情况下越级；
- P2、P3、A1.22A、V2-B/V2-C 在 P1 PASS 前均停止。

