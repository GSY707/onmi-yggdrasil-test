# V2-R1R P1 v4-LQ 失败复盘

日期：2026-08-11

## 1. 判决

P1 v4-LQ 在 `scale512` 正确停止。B128 证明训练链能够启动；S512 证明当前目标能把 512/族的训练 episode 和其 claim 完整记住，却没有形成跨 episode 的状态算法。该结果不否定 K=8 latent core，但否定“owner-contrast + 小集合逐级 Gate”这条学习路径。

正式证据为只读 sealed roots：preflight `PASS`，B128 `PASS`，S512 `FAIL_P1_LQ_SCALE512`；后续 S2048、F8192、assessment roots 均不存在。S512 在 4,800 updates、平均每 episode 75 次暴露后，训练答案 ERE/CPS 均为 `1.0`，训练 claim 为 `0.962890625`，但 validation 仅为 `0.234375/0.1884765625`。

## 2. 根因

v4 的 owner-contrast 把任意其他 episode 的 state 当作 positive claim 的负 owner。不同 episode 使用独有 nonce 名称，且“别的 episode 中该 claim 的真假”没有定义；因此这个目标奖励的是 source–claim 身份匹配，而不是状态真值。训练 Gate 又只在见过的 claim 上测 owner-shuffle drop，恰好把这种身份记忆当成成功。

父任务对 sealed S512 checkpoint 做了只读、非正式 Gate 的反事实诊断。见过的 128/族 episode 上，答案为 `1.0/1.0`、claim 为 `0.939453125/0.96484375`；从同一 train 分布、但未进入 S512 梯度的 128/族 episode 上，答案降为 `0.3359375/0.1953125`，claim 降为 `0.498046875/0.505859375`。zero-state 与 zero-claim 均约为 `0.5`。因此失败不是 claim-only 词面泄漏，而是 source 与 claim 的联合 episode 记忆。

## 3. 对 v5 的约束

v5 必须把两个长期不变量写进训练合同：辅助目标只能使用真值有定义的同 episode 反事实；辅助机制必须在从未参与梯度的 episode 上验收。owner-contrast 训练项和 owner-shuffle 成功 Gate 必须从活动实现删除，不能降权、保留兼容入口或改名继续存在。

v5 不修改部署模型结构，也不降低正式 P1 K01–K09。它先用 7,168/族优化、1,024/族永久隔离的 train-distribution audit 验证 paired claim 与答案的跨 episode 迁移；通过后才把全部 8,192/族并入训练并运行原正式 K=8 Gate。

