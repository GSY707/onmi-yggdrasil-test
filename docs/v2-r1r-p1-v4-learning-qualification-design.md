# V2-R1R P1 v4 学习路径资格合同

日期：2026-08-11

证据目标：P1-LQ，single-seed、嵌套数据规模、可停止的训练路径资格；不是 P1 完成证明，也不是 P2 Pareto 证据。

## 1. 判决问题

P1 v3 已经证明数据、只读 hidden cache、GPU 训练、checkpoint、probe 剥离和完整评测链可运行，但 K=8 在训练 claim 诊断与 heldout 上都接近随机。它只给 16,384 个训练 episode 平均 `2.34375` 次暴露，前 400 个所谓机制预热更新折算到每条 episode 仅 `0.1953125` 次；因此该阶段没有资格区分“架构不可学”和“训练从未进入学习区”。

P1 v4 不改模型结构，不降低 P1 K01–K09，也不恢复 oracle span mask、显式三寄存器或任务专用分支。它只回答一个更窄的问题：用可复现的小规模机制启动和逐级扩展课程，当前匿名 K-slot recurrent reasoner 能否从 memorization fixture 进入 full-scale heldout、OOD 与因果学习区。

## 2. 固定输入和不可变边界

数据、model-eval subset 与 hidden cache 继续只读复用 P1 v2；cache 的 formal 状态仍为 FAIL。训练授权只来自 P1 v3 recovery 的 sealed `PASS_P1_CACHE_RECOVERY`，不得修改或重封 P1 v2/v3 artifact。

活动实现直接切换为 `r1r-p1-v4-lq`。P1 v3 源码只存在于 sealed source snapshot 和历史文档中；活动 CLI、schema、roots 与测试不得保留 v3 兼容入口。

模型固定为 P1 v3 的 K=8 配置：2048→512 tokenwise Boundary、8 个匿名 slot、2 层共享 recurrent block、最多 24 步、latent-only answer readout、训练期可剥离 claim probe。禁止 task embedding、任务专用参数、teacher/AST 输入、oracle span/role、显式寄存器和输入到答案旁路。

## 3. 训练路径

四个阶段使用同一模型参数连续扩展，但每阶段重置 AdamW optimizer 和学习率日程，避免小集合的旧动量支配扩大后的分布。学习率使用 5% warmup、cosine decay 和 `0.1` floor；不会像 v3 一样在阶段末降到零。bootstrap 的前 400 update 使用 answer/claim 权重 `0.25/1.0`，之后以及所有扩展阶段始终使用 `1.0/0.5`。claim 与 owner-contrast 监督贯穿全程，部署 checkpoint 物理移除 probe。

每族训练集合由与标签、pattern、reasoning budget 平衡的确定性次序取前缀形成，因此 `64 ⊂ 512 ⊂ 2048 ⊂ 8192`。选择只读取 train metadata，不读取 validation/OOD、模型预测或 Gate 结果。

| 阶段 | 每族 unique episode | 最大 update | 最早判定 | eval 间隔 | 最大单 episode 暴露 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B128 | 64 | 2400 | 1200 | 100 | 300× |
| S512 | 512 | 4800 | 1200 | 200 | 75× |
| S2048 | 2048 | 8192 | 2048 | 256 | 32× |
| F8192 | 8192 | 16384 | 4096 | 512 | 16× |

每个阶段预先生成最大预算 schedule，但 result 与 ledger 记录实际完成的前缀。达到全部阶段 Gate 后可提前停止；未达到则运行至最大预算并以失败封存。后续阶段只允许读取前一阶段 sealed PASS 的 full continuation checkpoint；不得从失败 checkpoint 继续。

## 4. 学习 Gate

训练 answer 诊断在当前嵌套 selection 上按族评测，最多各 512 条；claim 诊断固定各 64 条、共 512 个 paired claims，并同时测 owner-shuffled accuracy。heldout 使用 P1 v2 预先冻结、每族 1024 条的 validation subset。

| 阶段 | train answer 每族 | claim accuracy | owner-shuffle drop | validation 每族 |
| --- | ---: | ---: | ---: | ---: |
| B128 | ≥0.90 | ≥0.90 | ≥0.15 | 仅报告 |
| S512 | ≥0.85 | ≥0.80 | ≥0.10 | ≥0.35 |
| S2048 | ≥0.80 | ≥0.75 | ≥0.10 | ≥0.55 |
| F8192 preliminary | ≥0.80 | ≥0.75 | ≥0.10 | ≥0.85 |

B128 是 P0-M 能力的活动链正控制；它失败意味着训练实现或 fresh selection 上的机制启动未复现，必须停止。S512/S2048 失败分别否定当前扩展课程在该规模的资格，不得跳到更大规模补救。F8192 preliminary 失败时不得运行昂贵的 OOD/causal evaluation。

F8192 preliminary 通过后，使用物理剥离 probe 的 checkpoint 运行原 P1 K01–K09，阈值保持：validation 每族 `0.85`、八个 OOD cell 各 `0.75`、causal pair flip 每族 `0.80`、zero/shuffled middle 平均 drop `0.40`、T1 hard-cell drop 每族 `0.15`、probe stripping delta `≤0.01`。K08 只修复 v3 已确认的 batch-size 写死：检查 rank、末维、Boundary 调用次数及当前实际 batch 一致性，不改变架构要求。

## 5. 阶段判决和停止纪律

执行顺序固定为 `preflight → B128 → S512 → S2048 → F8192 → assessment`。任一阶段 non-PASS 后立即停止，不创建后序 root。每个 root 只能创建一次；异常、Gate 失败和完整证据都必须 seal，不得删除、覆盖、续写或在同一 root 重启。

P1-LQ assessment 只有在四段 sealed PASS、selection 嵌套、continuation hash 链闭合、F8192 K01–K09 全 true、源 cache 仍 formal FAIL 时才可 `PASS_P1_LQ`。它只授权另立 fresh-seed 的完整 P1 合同；不授权把 LQ checkpoint 当成 P1 formal，不授权 P2。

若 P1-LQ PASS，下一合同必须冻结本轮实际停止 update、相同规模课程和新的模型/data-order seed，并从头运行 K=8。K=8 PASS 后，K=1、direct、text-CoT 需要相同 episode schedule/暴露合同，最后才形成 P1 assessment。若完整 P1 任一 Gate 失败，仍不得进入 P2。

## 6. 可复现性和证据

每个阶段必须保存 contract/selection manifest、最大与实际 episode ledger、progress、train/claim/validation diagnostics、continuation checkpoint hash、部署 checkpoint hash（仅 F8192）、吞吐/显存、源码快照、run metadata 和 evidence seal。preflight 固定活动源码 identity；preflight 后任何受监视源码或合同变化都使后序阶段拒绝运行。

本合同的成功含义是：训练目标悖论可以用不依赖任务脚手架的通用“机制启动→扩大数据分布→持续联合监督”路径解决，并在 full-scale K=8 上重现原正式 Gate。它仍不证明多 seed 稳定、K=8 优于 K=1、latent 优于文本基线或完整白皮书成立。
