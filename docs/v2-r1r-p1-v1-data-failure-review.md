# V2-R1R P1 v1 fresh-data Gate 失败复核

日期：2026-08-10

证据地位：P1 v1 唯一正式 data root 的主设计层复核。本文解释为什么合同在 G09 后停止，以及下一份合同应修什么；它不修改已经冻结的 P1 v1 设计、Gate 或 artifact，也不构成模型或架构结果。

## 1. 核心判决

P1 已经启动，但 **P1 的模型验证尚未开始**。唯一 fresh-data formal 在 G09 被拒绝，`cache_authorized=false`；因此没有建立 cache，没有运行 K=8、K=1、direct、text-CoT、因果干预或 P1 assessment。

失败应归类为 **full-scale generator surface/label coupling**：P0-D v17 在 train=4096 时通过的 ERE 字符 source-only 检查，在 P1 的 train=8192 下获得了足够统计功效，能够从 validation 正文和合法答案 mask 中学到局部 label 相关性。该结果拒绝这批 P1 输入，但没有检验、更没有否定 shared Boundary/recurrent core。

## 2. 已完成的执行事实

P1 v1 直接切换了活动 CLI、`p1/` package 和预测试，不保留 P0-M 兼容入口。实现检查为 P1 目标测试 `11 passed`、全仓 `105 passed`，`compileall` 与 `git diff --check` 通过。正式 preflight root 为 `artifacts/v2-r1r/p1-v1-preflight-20260810-1/`，项目 Python、CUDA、8 GiB 级 GPU、磁盘、固定前提 hash 和所有 formal roots 不存在检查均为真；seal SHA-256 为 `941A985A475F1E08DD793C05F02C37B99E4625C2895B5204FE0E0B553156263D`。

唯一 data formal 使用 generator seed `2026081901`，每个 family 为 train `8192`、validation 与各 OOD/causal split `1024`，共生成 `28,672` records。固定 root 为 `artifacts/v2-r1r/p1-v1-data-20260810-1/`，最终 `FAIL_P1_DATA`；seal SHA-256 为 `51D57EB3083A5BDF25C42C87A40B3C9BE642B57DEE95B1D2C0D159DA0478B94B`。

G01–G08、G10、G11 全部为真，只有 G09 为假。全量第二次生成与原 dataset 逐 bytes 相等，从 artifact snapshot 的只读重放也相等，watched inputs 未改变；两次 tree SHA-256 均为 `CDA9B2D95504F047F2754F54B6724FBD2FDD8B677E4D95829D62AB305F119011`。因此这不是一次非确定性生成、wrapper 或封印失败。

## 3. 唯一失败单元

失败单元是 `ERE/validation/full_text_char_3_5_nb`：

| 量 | 值 |
| --- | ---: |
| validation 样本数 | 1,024 |
| accuracy | 0.33203125 |
| mean random chance | 0.27099609375 |
| point excess | +0.06103515625 |
| chance + 0.10 ceiling | 0.37099609375 |
| 98-cell Bonferroni-Wilson upper | 0.38190468400647426 |

判据要求 simultaneous upper 不高于 ceiling；这里高出约 `0.01091`，所以必须失败。其余 97 个 source cells 都通过。ERE character-NB 的六 split family aggregate 也通过：accuracy `0.27815755`、upper `0.29379010`、ceiling `0.31041667`。aggregate 不能覆盖已预注册的局部失败。

## 4. 停止后的只读定位

以下诊断在 formal root 封存后只读进行，不属于合同 Gate，也没有改写 artifact。

同一 P1 validation 上，该字符分类器随 ERE train 子集规模的 accuracy 为：`1024 → 0.29004`、`2048 → 0.28711`、`4096 → 0.30762`、`8192 → 0.33203`。作为参照，v17 的 train=4096/validation=1536 同类 cell 为 accuracy `0.29948`、upper `0.33916`、ceiling `0.37083`，当时合法通过。扩大训练集正在暴露可学习信号；但 point accuracy 仍低于合同 ceiling `0.03896`，v1 的正式失败来自 1,024 样本下 simultaneous upper 的不确定性，不能仅凭该 FAIL 判断信号已经超过允许效应。

按 ERE validation pattern 分解，`relation_query` accuracy 为 `0.60465`（86 条、二选一），其余 `copy_chain/existing_foreach_copy/if_copy/swap_copy` 为 `0.26471–0.32941`。去掉 `CHOICES` 图例但保留真实 valid-choice mask 后，accuracy 仍为 `0.32031`；完整正文加真实 mask 为 `0.33203`，只用 mask 的 categorical baseline 为 `0.27441`。改成全九标签 mask 后，完整正文和去图例正文分别降到 `0.15332/0.13867`。

这些消融把问题定位到：图例只贡献约 1.17 个百分点，合法 mask 本身接近 chance；主要信号来自 **在合法 mask 条件下，正文表面与局部答案 label 的弱相关性**。当前证据还不能把唯一根因锁定到某一行 RNG 代码，也不能把正确修复直接等同于删除该信号；必须先回答现有 `+0.10` Gate 在正式 train 规模下是否有足够判别功效。

## 5. 机制解释与影响

当前 generator 的确留下局部弱相关性；P0-D 的 4,096 train 不能自动外推到 P1 的 8,192 train。但 v15 的 1,536-heldout power qualification只保证以至少 95% 概率接受局部 `+0.04`、以至少 99.5% 概率拒绝 `+0.10`。P1 v1 未重新资格化就把 heldout 降为 1,024；当前 point excess `+0.06104` 落入原 qualification 的灰区。

若真实 accuracy 保持 `0.33203`，同一公式在 heldout 1,536/2,048/3,072/4,096 的 upper 约为 `0.37255/0.36702/0.36049/0.35662`，单次通过概率约为 `0.447/0.638/0.886/0.974`。所以 2,048 仍不适合作为 single-use formal；4,096 能把当前允许级弱信号与 `+0.10` 禁止级信号稳定分开。

影响边界不变：P1 v1 输入被拒绝，行为实验未运行，不是模型失败。更准确的后继优先级是先修 measurement power；只有在 4,096-heldout 下 point accuracy 或 simultaneous upper 仍失败，才把 generator assignment/presentation repair 提升为必要条件。

## 6. 推荐后继：P1 v2 powered measurement repair

下一份合同保持 generator、模型、训练预算、K=8-first、因果 Gate、K=1 与 matched baselines 不变：

1. 用 train=8,192、heldout=4,096 重新资格化 null、局部 `+0.04/+0.06/+0.10` 与 diffuse `+0.05` 的 98/14-cell decision power。
2. 将本次失败 root 作为正回归：同一公共 API 必须继续精确拒绝 v1，证明没有削弱审计器；不得降低 ceiling、alpha、cell 数或聚合范围。
3. 用未预览的新 seed 生成每族 32,768、总 65,536 records，G09 使用完整 4,096 heldout。
4. 通过与 label/source/metric 无关的固定 hash，每个 heldout 预选 1,024 条进入 Qwen cache/model Gate；统计增样本不扩大模型成本。
5. 只有新 data formal 仍失败，才另立 generator assignment/presentation repair；若通过则继续同一 P1 的 K=8-first 路径。

完整冻结后继见 `docs/v2-r1r-p1-v2-powered-single-seed-design.md`。只有 P1 v2 assessment PASS 才授权另立 P2；A1.22A、V2-B/V2-C 仍停止。

## 7. 完成与未完成

已完成：P1 v1 合同冻结、活动实现直接切换、预测试与全仓回归、正式 preflight、唯一 fresh-data formal、封印/全量重生成/只读 replay，以及 G09 的只读定位。

未完成且按合同未运行：cache、K=8、K=1、direct、text-CoT、P1 assessment、P2 与任何架构通过判决。
