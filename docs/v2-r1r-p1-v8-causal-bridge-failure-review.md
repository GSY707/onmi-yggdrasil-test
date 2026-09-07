# V2-R1R P1 v8 causal-bridge 失败审阅

日期：2026-08-11  
正式状态：`FAIL_P1_V8_CAUSAL_BRIDGE`  
证据等级：sealed training-contract failure；不是架构否证，也不是 P1 通过

## 1. 核心判断

v8 没有回答“pair bridge 能否修复已经学会任务的 latent reasoner”。它实际回答的是另一个问题：一个从随机初始化开始、只见 768 条记录/族的小覆盖模型，能否同时重新学会 ordinary task、因果变化与中间状态。答案是否定的。

该结果不能升级为 causal bridge 无效。两个关键反事实都缺失：causal 臂没有共享的 broad-task competent 起点，也没有 ordinary rehearsal。尤其是 causal 臂从不读取 ordinary train，却被 B05 要求通过 ordinary retention；因此 B05 缺少训练路径前提。

## 2. 固定事实

三个正式 root 均已封存且 seal 有效：

- preflight：`PASS_P1_V8_CAUSAL_BRIDGE_PREFLIGHT`，seal `2985B2EBD2744CC9CFFA0626CE5EE67CB3333B7FA4615AA001994876DE25FE11`；
- query-cache：`PASS_P1_V8_CAUSAL_BRIDGE_QUERY_CACHE`，seal `0C3B14D3BA42A1E16BE004BF7B2180198D0823D7F6CDC4FA2236ABBFBB187591`；
- qualification：`FAIL_P1_V8_CAUSAL_BRIDGE`，seal `E425F115ABC2E4CBFAD257AA94A7319950DCE37FDA32AC9C492F5DF76E6150E4`。

三臂都完成 3,072 updates、98,304 exposures，并通过 B01、B04、B07。B02、B03、B05、B06 失败；没有 selected arm，未授权 v9 或 P2。

## 3. 为什么 v8 没有判定力

### 3.1 覆盖被重复暴露替代

v7 使用 16,384 条 ordinary train 记录和 655,360 次 exposure；v8 每臂只使用 1,536 条记录和 98,304 次 exposure。与 v7 相比，v8 的独立 ordinary 覆盖只有约 9.4%，总 exposure 只有 15%，但每条记录被重复 64 次。

结果符合典型记忆化：ordinary 臂在训练集复评为 ERE/CPS `1.0/1.0`，正式 validation 仅 `0.25/0.17578125`。因此 B05 失败不是“保留能力失败”，而是该臂从未建立可保留的 broad competence。

### 3.2 causal 臂没有 ordinary 前提

`causal_unpaired` 与 `causal_paired` 从相同随机初始化开始，但训练数据完全由 384 causal pair/族组成。它们没有看见 ordinary train，也没有继承 v7 checkpoint。用 ordinary validation 约束这两臂是合理的系统目标，却不是由当前训练路径支持的实验命题。

### 3.3 pair objective 可优化，但只在训练 pair 内成立

对 sealed checkpoint 做只读训练集复评得到：

| arm | ERE train raw / pair flip | CPS train raw / pair flip |
| --- | --- | --- |
| causal-unpaired | `0.9622 / 0.9245` | `0.5664 / 0.1458` |
| causal-paired | `0.9987 / 0.9974` | `0.7904 / 0.5807` |

pair loss 因而不是死目标：它给 CPS train pair 带来约 `+0.435` 的 pair-flip 增益。但在 128 unseen pair/族上，paired 只有 ERE/CPS `0.0859/0.0078`。优化成功、迁移失败，说明小规模 causal set 被记住，没有形成可复用的 CPS 成本比较规则。

### 3.4 temporal probe 与答案绑定仍是两个层次

三臂 temporal accuracy、zero-state drop、swapped-state drop 均通过，说明中间状态仍可承载受监督信息。与此同时答案 validation、causal flip 与 middle-state shuffle 失败。该分离再次证明：可解码状态不自动保证最终答案使用同一因果变量。

## 4. 对架构的含义

v8 没有新增“latent core 不成立”的证据。它新增了两条训练不变量：

1. 因果绑定训练必须建立在 broad-task competence 之上，不能把基础学习与修复绑定混为一个小数据 scratch 实验；
2. ordinary rehearsal 必须伴随 causal specialization，否则 retention Gate 没有训练路径支撑。

pair supervision 仍值得继续，因为它在训练内产生了大而可测的 CPS 增益。下一实验应把共同 broad checkpoint、相同混合 batch、相同 compute 固定，只改变 pair objective；若此时仍不能迁移，再转向更强的语义状态约束，而不是继续增加重复次数。

## 5. 决策

v8 保持正式 FAIL，不重跑、不改判。活动路线切换到 P1 v8R causal curriculum recovery：从 sealed v7 competent checkpoint 出发，两个 matched 微调臂同时读取完整 ordinary rehearsal 与相同 causal records，只以 pair objective 为实验变量。

v8R 使用已经揭示的 v8 optimization/audit split，因此只属于 development qualification。即使通过，也只授权用从未进入 v7/v8 model-view 的 fresh causal pairs 建立 P1 v9；不能完成 P1，更不能进入 P2。
