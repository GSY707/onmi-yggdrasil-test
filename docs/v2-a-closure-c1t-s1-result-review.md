# V2-A Closure C1T S1 Overfit32 结果复盘

更新日期：2026-09-01

## 1. 核心判决

C1T S1 已按 fresh identity 完成一次且仅一次 preflight 和一次且仅一次正式运行。preflight 完整通过，正式训练也按冻结 schedule 到达唯一 `fixed_4000` endpoint；但科学终态为 `FAIL_V2_A_C1T_S1_QUALIFICATION`，`authorizes=nothing`。

这不是运行故障。封存、固定端点、记账、置换等变、no-core necessity 与 source/cache integrity 均通过；失败来自注册的学习 Gate：完整答案只有 `22/32`，8 个完整 factorial group 只有 1 个全对，并且逐记录“双支持对象都构成因果贡献者”的比例远低于门槛。因此同一 identity 不得重跑、延长训练、换 seed、挑 checkpoint 或降低阈值，S2、S3 和 single-seed formal 保持 `NOT_RUN`。

## 2. 身份、单用执行与封存

正式 identity 为 `V2-A-CLOSURE-C1T-CPW-S1-OVERFIT32-20260901-1`，输出 root 为 `artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1/`。冻结 source identity 为 `07E8707EF30A3B78D93C9FD0C1AD0572D6F947BC8945DEB5A6A477085E65900E`；前置 S0 result/seal/cache 分别固定为 `831BC5C2B7DE284F99CAB7E27F4A3BF3DB33FB71B4BE928B1D9A62B22A41D6A8`、`E4D9B3DC63D2409BD947500517B9382B42EAB8D04DB5827D0421A29A6F5FA0EE` 和 `44232C7DBA87421E33DA4BD6E54F8A2792A5B720C45920FD4D573837E67A4723`。

唯一 preflight 在 disposable model 上执行 32 个 optimizer steps，正式 optimizer/model writes 保持 `0/0`；P101–P105 全部通过。其 result/seal SHA256 为 `3E48FD4C2E97558D14DB03B99B806133FADF598A6ACA5244E3E2568388E1C646` / `3E78499C59B61384D4ECB778F04CD7C46C1656ADB5CD2BBD098422D986A8BF62`。preflight 没有消费正式 root。

正式运行随后从 fresh model seed `2026090102` 开始，以 order seed `2026090111` 执行固定 4,000 updates。正式记账为 optimizer steps `4000`、disposable steps `0`、model writes `1`；这一写入只对应 `fixed_4000.pt`，没有 intermediate checkpoint、optimizer state 或 checkpoint selection。训练 wall time 为 `1019.621305 s`，step median/p95 为 `0.298004/0.329823 s`，峰值显存记账为 `385,010,176 bytes`。

正式 result/seal/endpoint SHA256 分别为：

- `9B37620F3F1CADE4740955F7849EACEA19C83DF208ADC3EA8F76AA5B74BC1DFF`
- `89C530396FE9DD478EFFF941776722EA391B5D13FB2CD10C0D5D5B008FA99BCF`
- `E7E66F00BF4AE29CB7229D4E012630530F49DFB0E2EDA7302E783D9ED31896F9`

正式 seal 已完成 `43/43` replay；注册 schedule identity 为 `E42F56A4C3994BFE48DBE4CE6F28901E171B074C237FD94E0D7A837B69ADBA7F`。

## 3. Gate 结果

| Gate | 结果 | 判定依据 |
| --- | --- | --- |
| R101 predecessor/source/cache/schedule identity | PASS | S0、source、card cache 与冻结 schedule 身份一致。 |
| R102 fixed endpoint and accounting | PASS | 正好 4,000 formal updates，只写一个固定 endpoint，无选择性 checkpoint。 |
| R103 full answer and factorial exact | **FAIL** | overall `22/32 = 0.6875`；完整 factorial group 仅 `1/8` 全对。 |
| R104 no-core necessity | PASS | 两族 no-core accuracy 与 margin-drop 均过门，8 组 no-core logits 组内 max delta 均为 `0.0`。 |
| R105 two-support counterfactual causality | **FAIL** | 平均 margin-drop 虽过门，但逐支持反事实答案与逐记录 two-contributor 均未达门槛。 |
| R106 permutation forward and parameter integrity | PASS | logit/trajectory permutation max delta 都为 `0.0`，参数完整。 |
| R107 result/source/cache integrity | PASS | result、source 与 cache identity 可重放一致。 |

## 4. 行为与因果量具

| 族 | 完整答案 | no-core 答案 | no-core margin drop mean / lower95 | support-0/1 反事实答案 | two-contributor point / Wilson lower |
| --- | --- | --- | --- | --- | --- |
| CPS | `12/16 = 0.75` | `2/16 = 0.125` | `6.24986 / 4.18777` | 各 `12/16 = 0.75` | `8/16 = 0.50 / 0.279996` |
| ERE | `10/16 = 0.625` | `4/16 = 0.25` | `4.10588 / 2.57923` | 各 `10/16 = 0.625` | `4/16 = 0.25 / 0.101821` |

两族每个 support 的 bootstrap mean margin-drop 都为正且 lower95 过门：CPS support-0/1 为 `10.73078/9.15024` 与 `8.81053/4.38217`，ERE support-0/1 为 `4.23656/1.04176` 与 `4.23656/1.03922`（均按 `mean/lower95`）。但这些族级均值不能替代逐记录因果性：不少 factorial cell 的某个 support drop 为零，少数较大 effect 足以抬高均值，却不能证明每条记录都真实使用两个支持对象。R105 正是为阻止这种“平均效应掩盖局部旁路”而存在。

## 5. 证据边界与路线结论

S1 说明结构性分区并非空实现：移除 core 后答案显著下降，no-core 组内输出严格不随 factorial cell 改变，置换等变也保持精确。这比 C1S 的 H0 完整 source hidden 旁路更强，证明 C1T 的隔离量具和 core-necessity 方向有效。

但固定训练配置没有把这种结构资格转化为 Overfit32 学习资格。模型既未完整拟合 32 条记录，也未在逐记录层面建立稳定的双对象因果贡献；因此不能进入 matched learned K1/K8，更不能把低 no-core accuracy 单独写成 S1 PASS。反过来，本次单 seed、固定 4,000-step Overfit32 FAIL 也不足以证明 C1T 或整个多向量 latent 架构在数学上无效；它只否决了这个冻结 identity、目标函数、优化配置与训练预算的 S1 资格。

当前没有可执行后继：C1T S2、S3、single-seed formal、C2、V2-A PASS、V2-B 和 V2-C 均未获授权。若未来继续，必须由 owner 先定义新的问题归因或 fresh successor 合同；不得从本 root 续训，也不得把同一 S1 包装成新 seed 重试。
