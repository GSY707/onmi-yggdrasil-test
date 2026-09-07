# V2-R1R P1 v6R selection-replay 恢复合同

日期：2026-08-11

证据目标：恢复 v6 assessment 因 JSON key normalization 产生的单一形式 false negative；不重训、不重建 cache、不修改 v6 artifact，也不完成 P1。

## 1. 恢复原则

v6 五个 roots 和原 `FAIL_P1_V6_CTW` 永久保留。v6R 不是给失败结果换后缀重跑，而是一个新的测量合同：它固定五个 evidence-seal 文件 SHA-256、原 assessment 的唯一失败形状、原 source snapshot，以及 query-cache 的持久化身份。

selection 的正式等价关系定义为 canonical JSON bytes 相等，并要求文件内 `selection_witness_sha256` 同时等于 recorded 与 independently replayed selection 的 canonical preimage SHA-256。Python 字典在 JSON 写入前后的键类型不属于持久语义；example、query、label、split、pair、count 或顺序的任何变化都属于语义变化，必须失败。

## 2. 固定输入与独立性

只读输入为 v6 的 preflight、witness-audit、query-cache、mechanism-compare 和 assessment roots。每个 root 必须同时通过自身 tree seal 和本合同固定的 evidence-seal 文件哈希。replay 所调用的原 `src/` Python closure 必须与 sealed v6 assessment source snapshot 逐文件一致；v6R 新增的验收器、合同和 CLI 不得改变 witness selection 实现。

恢复阶段重新从 sealed witness catalog 与原 production rows 派生 selection，重新扫描两份 query mmap 的 18,899 entries/209,391 tokens，并验证以下三类负控均被 canonical bytes 与 preimage hash 拒绝：query ID 改写、label 顺序翻转、reasoning-budget count 改写。

## 3. 阶段与 Gate

顺序固定为 `preflight → selection-recovery-assessment`。

Preflight 要求：三份合同/复核文档 hash 固定；v6 五个 roots 的 seal 与 pinned evidence hash 全部成立；原 assessment 只有 W604 false，其余八个 Gate true；原 replay source closure 与当前活动旧源逐文件一致；恢复预测试通过；recovery assessment、integrated P1 和 P2 roots 不存在。

Recovery assessment 的 conjunction 为：

1. `R601`：v6 五 roots pinned 且 sealed；
2. `R602`：原 assessment 原样保持仅 W604 失败；
3. `R603`：selection replay closure 与 sealed v6 source 一致；
4. `R604`：recorded/replayed canonical bytes 与 JSON round-trip 相等；
5. `R605`：raw Python 不等仅来自预登记的六个 integer count keys；
6. `R606`：records、partition、selection preimage hash 完整重放；
7. `R607`：query content `18,899/18,899` 全量重审通过；
8. `R608`：三种真实篡改负控全部被拒绝；
9. `R609`：sealed mechanism compare 仍是 static FAIL、temporal 全 Gate PASS、预注册选择 temporal；
10. `R610`：integrated P1/P2 roots 不存在。

任一 Gate 失败即 sealed FAIL 并停止。不得调 Gate、修 v6 文件、重建 selection/query cache、重训 comparator、启动 integrated P1 或 P2。

## 4. 通过含义

`PASS_P1_V6R_SELECTION_RECOVERY` 只证明 v6 的 W604 是表示层 false negative，并恢复 `temporal_witness` 的训练机制资格。它授权父任务另立 integrated fresh-seed P1 合同，把 answer objective、full 8,192/族、K=8/K=1、direct/text-CoT、K01–K09 和成本纳入同一比较；它不把 v6 原 root 改成 PASS，也不提供完整 P1、跨 seed、Pareto、P2 或白皮书架构成立证据。
