# V2-R1R P1 v8D：causal-decision witness 机制资格合同

日期：2026-08-11

## 1. 唯一问题

v8R 已排除 scratch、低 ordinary coverage 和单纯 readout 容量三种主要混淆。当前唯一问题是：**训练期跨反事实 final-decision witnesses，能否让同一个 shared K=8 core 在 `H_T` 中形成可迁移的 CPS 候选比较与最终归约状态，并让既有 latent-only answer head 使用该状态。**

本轮是使用已揭示 v8 split 的 development mechanism qualification，不是 fresh P1、跨 seed、K 容量或 Pareto 证据。PASS 只授权另立 fresh-seed P1 v9；FAIL 不得通过调权、换 checkpoint 或增加同类 exposure 补跑。

## 2. 模型与信息边界

部署切面不变：frozen Qwen hidden → tokenwise Boundary → shared two-layer recurrent K=8 workspace → `H_T`-only pooled linear answer head。初始化固定为 sealed v7 stripped checkpoint，不从 v8/v8R checkpoint 续训。

训练期重新实例化一个共享 `ClaimProbe`。probe 接收：

- 一个由同一 frozen Qwen 编码的自然语言 decision query；
- base 或 flip episode 的 `H_T`。

probe 不读取 source hidden、AST、teacher trace、pair role、答案 index 或 task id。query 不进入 recurrent core 或 answer head。正式答案评估前物理删除 probe，并 strict reload；删除前后答案预测必须逐项一致。

允许 simulator/label mapping 只在 training view 中生成监督 query 与真假标签；model forward 仍只有 `source_hidden`、mask、reasoning budget 与 valid-choice mask。没有 oracle span、candidate slot、三寄存器、teacher state reconstruction 或任务专属部署参数。

## 3. Decision witness

每个 witness 对同一个 query 在一个 causal pair 的 base/flip `H_T` 上给出相反 labels。这样 query 本身无法泄露 pair role，zero-state 的理论上限为 `0.5`。

### 3.1 两任务共享的 label witness

若 base 正确局部标签为 `L_b`、flip 为 `L_f`，生成：

- `Choice L_b is the correct answer.`，labels `(true,false)`；
- `Choice L_f is the correct answer.`，labels `(false,true)`。

ERE 与 CPS 都使用该 witness。

### 3.2 CPS final-reduction witness

CPS cost pair 另生成四项：

- base winner 是 unique cheapest valid option，`(true,false)`；
- flip winner 是 unique cheapest valid option，`(false,true)`；
- base winner costs less than flip winner，`(true,false)`；
- flip winner costs less than base winner，`(false,true)`。

所有 truth 必须由两条记录各自的 accepted simulator output 重放，不能只相信 causal certificate。若 winner 不是候选、任一比较对象无效、cost 不形成严格反转或 label/candidate 映射不一致，preflight 在创建 query cache 前失败。

query surface 只有 9 个 choice、5 个 candidate-optimum 和 20 个有向 candidate-comparison 模板，最多 34 个唯一文本。query cache 固定 Qwen `Qwen/Qwen3.5-2B@15852e8c...` final hidden、FP16、无截断、全量内容 hash。

## 4. 数据与调度

继续只读复用 v8R partition：每族 ordinary train 8,192 条、causal optimization 384 对、causal audit 128 对、validation 1,024 条。audit 从不参与梯度、checkpoint 或超参选择。

### 4.1 probe-only localization

先冻结 V7 Boundary/core/readout，只训练随机初始化 probe 384 updates。每 batch 含 ERE/CPS 各 8 个完整 causal pair；48 batch/epoch、8 epochs，每对恰好出现 8 次。该结果只判断原 V7 `H_T` 是否已可由 decision query 解码，不作为后续 checkpoint 选择条件。

### 4.2 joint state formation

随后从同一个 V7 model state、但保留 probe-only 后的 probe 开始 3,072 updates。每 batch32 固定：ordinary ERE/CPS 各 8 条，causal ERE/CPS 各 4 个完整 pair。ordinary 每条恰好 3 次，causal 每条恰好 32 次。

每个 causal pair 每次选择两个完整 witness query：ERE 固定两项 label witness；CPS 在六项 witness 中按 exposure index 循环选择连续两项，32 次 exposure 后每项被选择 10 或 11 次。两族每 batch 都产生相同数量的 query pairs 与 judgments。

joint 前 2,560 updates 固定 answer readout 参数，只允许 answer loss 通过固定 readout 约束 Boundary/core；最后 512 updates 才以低学习率解冻 readout。总损失固定为：

```text
L = L_answer + 0.25 * L_logit_switch + 1.0 * L_decision_witness
```

`L_logit_switch` 使用 base/flip 的 coupled difference-of-differences；`L_decision_witness` 使用 paired CE + rank。优化器 AdamW，Boundary/core/readout/probe 初始 LR 为 `1e-5/2e-5/3e-5/1e-4`，weight decay `0.01`，5% warmup、cosine 到 20%、global clip `1.0`。固定最终 update，不用 audit 选 checkpoint。

具体地，若 base/flip 正确答案为 `a/b`，则 `L_logit_switch = softplus(1 - [(z_base,a-z_base,b)+(z_flip,b-z_flip,a)])`。对同一 query 的二分类真值 logit 差 `s=z_true-z_false`，`L_decision_witness = CE + softplus(1 - direction_base*(s_base-s_flip))`。两项都不可拆成两个互不相干的单样本 margin。

## 5. Gate

固定顺序：preflight → decision-query-cache → qualification。任一阶段非 PASS 立即停止，后序 root 不得创建。

- **D01 inputs/query truth**：v7、v8、v8R roots/seals/hash 全部固定；512 对/族结构完整；34-query 上限、simulator truth、partition 与 schedule hash 全通过。
- **D02 matched compute**：probe-only `384`、joint `3,072`，所有 exposure、witness rotation、初始化、finite 与删除前后身份一致。
- **D03 decision-state transfer**：joint 后 causal audit decision-witness judgment accuracy ERE/CPS 各 `>=0.70`；相对 zero-state judgment accuracy 的 drop 各 `>=0.15`；相对 swapped-pair-state judgment accuracy 的 drop 各 `>=0.30`。同时记录更严格的 query-pair accuracy，但不把它混作本轮预注册 Gate。
- **D04 answer causal transfer**：audit ERE pair flip `>=0.70`；CPS raw `>=0.55`、pair flip `>=0.30`、prediction flip rate `>=0.40`；CPS pair 相对 v7 baseline 与 v8R replay_pair 的较高者至少 `+0.20`。
- **D05 ordinary retention**：validation ERE `>=0.95`、CPS `>=0.80`，相对 v7 baseline 两族 drop 各 `<=0.05`。
- **D06 mechanism/architecture integrity**：CPS audit judgment accuracy 相对 probe-only 至少提高 `0.15`，证明变化来自 state formation；probe 物理删除，答案 hash 不变；FP32 source-to-answer gradient 非零；部署 state dict 无 probe、task/candidate/operator 专属参数。
- **D07 stop boundary**：没有 fresh v9、K=1、baseline assessment 或 P2 root。

全部 Gate 同时通过才是 `PASS_P1_V8D_CAUSAL_DECISION_WITNESS`。若 D03 通过而 D04 失败，下一步才允许 addressable readout qualification；若 D03 失败，停止 readout 与 loss-weight 补丁，重新设计 causal state formation/Boundary 支持。

## 6. 证据边界

本轮使用已揭示的 v8 optimization/audit，不能证明 fresh generalization。它只决定机制分叉：state 是否形成、answer 是否消费。PASS 不完成 P1，不进入 P2；fresh v9 必须使用从未进入 v7/v8/v8R/v8D model-view 的新 causal seed，并重新做完整 P1 与 matched controls。
