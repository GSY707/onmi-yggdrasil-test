# V2-R1R P0-M v3 M04 失败复核

## 正式结果

v3 的 cache qualification、ERE M02 与 CPS M03 均 sealed PASS。ERE/CPS 单任务在 update 1200 均为 answer `1.000`，stripped reload prediction 一致。唯一 joint root `artifacts/v2-r1r/p0m-v3-joint-k8-overfit128-20260810-1` 在 2400 updates 后为 ERE `1.000`、CPS `1.000`、claim `0.627688`，因此 M04 FAIL，并按顺序停止；baseline、throughput 与 assessment 未运行。

该结果接受 shared K=8 core 对 128 个混合 episodes 的小样本答案容量，但不接受 claim 机制。1800 updates 前 claim 最高约 `.64`；1800 后合同把 claim weight 归零，继续到 2400 不可能修复 probe，排除“训练更久”解释。

## 机制诊断

从失败 checkpoint 固定 Boundary/core，抽取相同 1488 claims 的 `H_t` 与 claim embeddings：

- 原 attention probe 单独重训 800 steps 仍为 `.500`；state-only 控制也为 `.500`。
- claim-only MLP 为 `.863`，显示固定 overfit 集存在记忆能力，但未解释正式失败。
- 保留 `q⊙pooled` 与 `|q−pooled|` 的 interaction probe 在 800 steps 达 `1.000`；owner shuffle 后为 `.659`，下降 `.341`。
- 在同一正负 pair 上加入相对排序损失后，400 steps 已达 `.929`，800 steps 为 `1.000`；owner shuffle drop `.359`。

根因是原 probe 的 claim query 只影响 slot softmax 权重，最终分类器只看到 pooled state，丢失绝对 claim–state compatibility；它不是 core 容量不足，也不是任务混合失败。

## v4 修复

v4 不改 Boundary、recurrent core 或 answer head，只修改生命周期内可物理剥离的 claim probe：query-attention 后输出 `q⊙pooled` 与 `|q−pooled|` interaction，并在已有一正一负 pair 上使用 CE + ranking loss。M04 额外要求 owner-shuffle accuracy drop ≥`.15`，防止用更强 probe 形成 claim-only 记忆捷径。
