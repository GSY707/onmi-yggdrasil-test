# V2-R1R P1 v8L：fixed-anchor causal-state ladder 机制资格合同

日期：2026-08-12

## 1. 唯一问题与停止线

v8D 已证明 CPS perturbation 进入并显著改变 `H_T`，但没有形成跨 pair 可迁移的候选比较与答案归约。v8L 只回答一个最后的机制问题：

> 当可学习 probe 被移除，且真实因果链在正确 recurrent state 上由固定语义锚点直接提供密集梯度时，同一个 shared K=8 core 能否形成可迁移的 CPS state closure？

本轮继续使用已揭示的 v8 optimization/audit split，只是 development qualification。PASS 只授权 fresh-seed P1 v9 设计；FAIL 终止当前匿名 K=8 P1 mainline，不再通过新 V8 版本调整 exposure、loss weight、readout 或辅助头。

## 2. 部署模型不变

部署切面仍为 frozen Qwen source hidden → tokenwise Boundary → shared two-layer recurrent K=8 workspace → `H_T`-only pooled 9-label answer head。初始化固定为 sealed V7 stripped checkpoint。

禁止：

- oracle source span/mask、AST、candidate slot、pair role、task id 或答案进入 model forward；
- 显式寄存器、固定 slot=entity/candidate 语义、任务专属参数或执行分支；
- query/anchor 进入 recurrent core 或 answer head；
- 可学习 ClaimProbe、teacher decoder、target adapter 或部署旁路。

训练期 teacher 只产生 `(anchor, state_index, truth_direction)`，并在 checkpoint 前完全消失。部署 state dict 必须与 V7 参数命名集合一致。

## 3. 固定语义锚点

每个 claim 文本由 frozen `Qwen/Qwen3.5-2B@15852e8c...` 编码，再经过从 V7 checkpoint 复制并永久冻结的 Boundary。每一组互斥 claim 按 `query_id` 排序，固定锚点定义为 `normalize(q_left - q_right)`；truth label 只决定 base/flip 应落在方向的哪一侧，不参与方向的朝向或几何构造。这样会消去两条自然语言模板的共同分量，避免近乎平行的正反 claim 对 core 施加相互抵消的梯度。

cache 只保存 1,448 个去重后的 512 维 semantic-contrast directions。它不拟合 center、whitening、probe 或任何 audit statistics；2,121 个 raw query 只用于计算对应的差分，随后从训练路径移除。contrast norm 必须 `>1e-4`、effective rank 必须 `>=8`，且 finite、内容 hash 全通过，否则 anchor-cache 失败。

状态评分没有参数：对指定 `H_t` 先使用部署 `final_norm`，再与 semantic-contrast anchor 做 slotwise cosine；固定温度 log-sum-exp 聚合为一个 scalar。同一 contrast 在 causal pair 的 base/flip 上真值相反，损失只要求正确方向的 score difference 超过固定 margin。由于 scorer、anchor 与聚合全部冻结，梯度只能改变 Boundary/core state。

## 4. 因果状态梯子

### 4.1 CPS

对每个 cost-mutation pair 从 simulator 与 mutation path 重放：

1. 找出包含被修改 action 的候选及 action position；
2. 在该 action 执行后的 `H_t` 生成 base/flip 累计代价 exact claims，两条 claim 的 truth direction 相反；
3. 在所有候选执行完成后的 penultimate state 生成受影响候选的最终代价 claims，以及 base winner/flip winner 的双向 cost-order claims；
4. 在 `H_T` 生成双向 unique-optimum 与局部 choice-label claims。

同一 claim 文本在 base/flip source 中完全相同。所有 labels 必须由 fresh simulator replay 得到；若 mutation 不在候选中、对应 prefix 没有 cost divergence、最终严格次序不反转、winner/label mapping 不一致或 state index 越界，preflight 失败。

### 4.2 ERE

ERE 使用同一个 scorer。在首个导致 base/flip query value 分叉的 prefix 及 `H_T`，生成 base/flip semantic-answer claims；最终再生成双向 local choice-label claims。若 pair 没有语义分叉或 labels 不相反，preflight 失败。

audit 的 anchors 可缓存和评估，但其 labels、scores、state 或结果不得进入梯度、checkpoint 选择与超参选择。

## 5. 训练路径

### 5.1 ladder bootstrap

从 V7 checkpoint 加载无 probe 模型，冻结 answer readout，只训练 Boundary/core `384` updates。batch32 包含 ERE/CPS 各 8 个完整 causal pair；每个 optimization pair恰好 8 次 exposure。每个 pair 每次按冻结轮转选择两个完整 ladder claims。

唯一损失是 fixed-anchor semantic-contrast paired rank。每对 ERE 有 3 层 contrast、每对 CPS 有 5 层 contrast；冻结轮转使 bootstrap 中 ERE 三层 exposure 为 `6/5/5`、CPS 五层为 `4/3/3/3/3`。该阶段结束后立即在 optimization/audit 上评估 anchor direction；不按 audit 选 checkpoint。

### 5.2 mixed joint closure

继续同一 deployment state `3,072` updates。每 batch32 固定 ordinary ERE/CPS 各 8 条、causal ERE/CPS 各 4 个完整 pair；ordinary 每条精确 3 次、causal 每条精确 32 次。前 2,560 updates 保持答案 pooling query 与分类头冻结，最后 512 updates 解冻这两部分；`final_norm` 全程冻结。

总损失固定为：

```text
L = L_answer + 0.25 * L_coupled_answer_switch + 1.0 * L_fixed_anchor_ladder
```

没有 auxiliary optimizer group。Boundary/core/readout 初始 LR 固定为 `1e-5/2e-5/3e-5`；AdamW、weight decay `0.01`、5% warmup、cosine 到 20%、global clip `1.0`。固定 final update。

## 6. Gate

顺序为 preflight → anchor-cache → qualification；qualification 内 bootstrap 失败即停止，不进入 joint。

- **L01 input/teacher truth**：上游 seals/hash、v8D failure、partition、mutation-path replay、ladder state index、opposite truth、lexical contrast orientation、anchor source与 cache content 全通过；audit 不进梯度，也不拟合任何 cache statistics。
- **L02 exact compute**：bootstrap `384`、joint `3,072`；exposure、rotation、初始化、finite、optimizer group 与 schedule 全等于冻结值。
- **L03 fixed-anchor state transfer**：joint 后 audit direction accuracy ERE/CPS 各 `>=0.70`；CPS `cost_trace`、`final_cost`、`cost_order`、`unique_optimum`、`choice` 五层各 `>=0.65`；CPS mean signed margin `>0`，swapped-pair direction accuracy `<=0.30`。
- **L04 answer causal transfer**：audit ERE pair flip `>=0.70`；CPS raw `>=0.55`、pair flip `>=0.30`、prediction flip `>=0.40`，且 CPS pair 相对 V7/v8R/v8D 最高参考至少 `+0.20`。
- **L05 ordinary retention**：validation ERE `>=0.95`、CPS `>=0.80`，相对 V7 drop 各 `<=0.05`。
- **L06 no-aux architecture integrity**：model/checkpoint 从未包含 ClaimProbe/anchor encoder/target adapter；部署 state dict key set 与 V7 一致。`final_norm` 始终冻结，使 fixed-anchor scorer 的坐标系在全程不可学习；最终 512 updates 只解冻 pooling query 与 answer head。FP32 source-to-answer gradient 非零；zero-update 到 joint 的 CPS anchor audit gain 至少 `0.15`；source/slot width、K/T、shared core 不变。
- **L07 stop boundary**：没有 v9、K=1、direct、text-CoT、assessment 或 P2 successor root。

全部通过才是 `PASS_P1_V8L_CAUSAL_STATE_LADDER`。PASS 不完成 P1；只允许另立 fresh causal seed 的 P1 v9。任一失败均为当前 K=8 mainline 的机制终止结论，不再建立 v8 successor。

## 7. 证据边界

fixed anchor 与 simulator teacher 是训练成本，fresh v9/P2 必须纳入 matched teacher/token/FLOPs 成本。v8L 不证明 zero-teacher learning、跨 seed、K=8 优于 K=1、latent 优于 text-CoT，也不能把开发 split 的成功写成白皮书完整成立。
