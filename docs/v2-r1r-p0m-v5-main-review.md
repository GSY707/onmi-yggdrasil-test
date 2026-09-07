# V2-R1R P0-M v5 主设计层验收

## 1. 最终判决

P0-M v5 的八项 Gate 全部通过，主设计层接受其有限结论：

```text
P0-M training-path smoke accepted
P1 eligible
P1 not started
architecture generalization not evaluated
```

这轮证明的是：在 P0-D v17 的固定小样本选择上，冻结 Qwen hidden cache、tokenwise Boundary、单一共享 K=8 recurrent core、可剥离 claim 辅助头、direct SFT 与 text-CoT SFT 都能稳定进入可训练状态。它不证明 OOD、组合泛化、跨 seed 稳定性、因果必要性、K 容量优势或质量—成本 Pareto。

## 2. 正式证据

| Gate | 结果 | 核心证据 |
| --- | --- | --- |
| M01 cache | PASS | immutable cache 内容、模型/revision、无禁字段/截断和吞吐资格均通过 |
| M02 ERE K=8 | PASS | update 1200；answer `1.0`；claim `0.99740`；owner-shuffle drop `0.36589`；剥离后预测逐字节相同 |
| M03 CPS K=8 | PASS | update 1200；answer `1.0`；claim `0.99861`；owner-shuffle drop `0.34583`；剥离后预测逐字节相同 |
| M04 joint K=8 | PASS | ERE/CPS answer `1.0/1.0`；claim `0.90659`；owner-shuffle `0.53427`、drop `0.37231`；共享参数唯一 |
| M05 direct SFT | PASS | 同 64 episodes；greedy answer `1.0`，ERE/CPS 各 `1.0`，64/64 可解析 |
| M06 text-CoT SFT | PASS | 同 64 episodes；greedy answer `1.0`，ERE/CPS 各 `1.0`，64/64 可解析 |
| M07 CUDA throughput | PASS | 100 个 measured optimizer steps；`71.706 examples/s`；median `0.21274s`、p95 `0.36923s`；峰值约 `1.04 GiB`，采样 utilization/power 最高 `60%/43.18W`；finite、无 OOM/fallback |
| M08 provenance/seal | PASS | 八个固定 root、seed、source snapshot、同 episode、公平 LoRA、no-truncation 与 conjunction 均可重放 |

最终 assessment 为 `PASS_P0M`，`p1_eligible=true`、`p1_started=false`。八个 evidence-seal 文件的 SHA-256 分别为：

| Root | Seal SHA-256 |
| --- | --- |
| `p0m-v5-cache-qualification-20260810-1` | `F5FEF16EE4D1AFC3BDBB7779E681A806B289DF9B31781B995C6712567134FF9D` |
| `p0m-v5-ere-k8-overfit64-20260810-1` | `DAB1DABE1818C0761188D03A388AC5553D207C918728C3FE756DE0E8AA8F4E35` |
| `p0m-v5-cps-k8-overfit64-20260810-1` | `5EC4B0D7222C25041D61DA7F77B85FB1C85E9110CA26104487A79AC011BD390F` |
| `p0m-v5-joint-k8-overfit128-20260810-1` | `91921CD18A245D34CB1DF3CBEE94706FA11C01E3B1735513E788AD7A2AEB341B` |
| `p0m-v5-direct-overfit64-20260810-1` | `E3573F6843A4DB2326C66D1AB01C1AE5898CE7A75DF829A734D608A0112B1AAE` |
| `p0m-v5-text-cot-overfit64-20260810-1` | `2F2508DE93A2781BF279C4202FF1F913A9168251F9ECD85A4368B71D43A1D618` |
| `p0m-v5-throughput-20260810-1` | `30BD34FF6C11069E3FF854E7F7D5E9ED4D3A350817A5E7A731C741162E897684` |
| `p0m-v5-assessment-20260810-1` | `17F45EAC8A3DF129B236688D3D1E2B22BD9B634D4098F903142CA42E7FEEF09E` |

## 3. 失败—修复闭环的有效结论

v1 的失败来自错误地用最小 batch latency 选 cache batch；v2 在训练前暴露 BF16 mask overflow 与 baseline 全序列 logits 的工程浪费；v3 证明原 claim probe 只通过注意力权重消费 query，分类器仍只看到 pooled state，因而无法表达 state–claim compatibility；v4 加入 interaction probe 与 pair ranking 后仍让 joint claim 在 1800 update 后失去监督，并允许 claim shortcut。v5 没有降低 Gate，而是把正确 owner 与同族错误 owner 的相对兼容性写入训练目标，并让 claim 梯度贯穿正式训练窗口，最终同时取得高 claim accuracy 和大幅 owner-shuffle drop。

这说明此前的“训练目标悖论”不是一次性调参问题：如果内部状态要承担可验证的语义职责，训练合同必须持续提供与该职责一致的正负约束。owner contrast、ranking、辅助头剥离和剥离后复载验证应作为后续架构生命周期机制，而不是只在本轮使用的临时补丁。

## 4. 对架构的有限含义

正面证据是：同一个不含任务分支、算子 embedding、oracle slot 或答案旁路的 K=8 core，能够在混合 ERE/CPS 小样本上同时拟合答案；claim 辅助头物理删除后答案不变，说明部署答案路径没有依赖 probe。这排除了“当前实现根本无法训练”这一低层失败模式，也表明混合 core 值得进入真正的泛化验证。

但本轮没有建立“完整架构成立”。所有模型只在单一固定 selection 上做 overfit；direct 与 text-CoT 也都达到 `1.0`，因此没有质量优势，吞吐数字也尚未形成 matched online Pareto。owner contrast 使用 episode owner 身份，仍可能学习 episode 区分而非可迁移的语义转移。P1 必须用 fresh data/model seeds、heldout 组合、因果干预和 K 对照区分记忆、任务识别与真实共享推理。

## 5. 后继边界

P0-D 与 P0-M 至此关闭，项目获得另立 P1 合同的资格。当前任务严格停止在 P1 前；没有创建或运行任何 P1 数据、训练或 artifact。P1 不得复用本轮固定 selection 作为正式测试，也不得把 P0-M 的 overfit 成功当作 Gate 阈值的先验保证。
