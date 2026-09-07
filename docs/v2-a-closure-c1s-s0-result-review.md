# V2-A Closure C1S S0 结果复盘

日期：2026-08-28

身份：`V2-A-CLOSURE-C1S-ADDRESSED-WORKSPACE-20260828-1`

机器终态：`PASS_V2_A_C1S_S0_QUALIFICATION`

## 1. 核心判断

C1S 已取得进入 Overfit32 的结构与测量资格。它不是对 C1/C1R dense anonymous core 的局部修补，而是把 latent state 直接切换为地址—内容配对集合、输入条件 query 和 address-routed shared transition。S0 证明这套实现满足 public-only 输入、定向写入、成对置换等变、matched K=1 参数公平、训练辅助头可剥离，并能在注册正负控上测出 functional ownership。

这不是学习结果。S0 没有读取训练数据，没有执行 optimizer step，没有写模型 checkpoint，也没有证明真实 ERE/CPS 上的地址绑定、状态闭合、答案正确率或 K=8 优于 K=1。

## 2. 唯一运行与封存

固定 root 为 `tmp/v2-a-closure-c1s-s0-preflight-20260828-1/`，sibling lease 为 `tmp/v2-a-closure-c1s-s0-preflight-20260828-1.preflight-lease.jsonl`。唯一命令只执行了一次并以 exit code 0 结束。

- source identity：`1F6FA08633ECD768A8CACCDDF0260A83D1ECB694D7AD72E0174A19917A6FA759`
- result SHA-256：`022DD05065EDE6B663E1212BA980FD2D9F2AD6A89CAEC3B3DF357FFDA0FF698D`
- evidence-seal SHA-256：`2117B1EFA80C4D55C475CC41B10238D2518AA5C34951911EACEF5CA3A5F46E3D`
- seal replay：`17/17`，无 missing、unexpected 或 mismatched
- predecessor pins：C1 formal、C1 attribution、C1R formal 的 result/seal hash 与全树 replay 在运行前后均一致
- accounting：`training_started=false`、`optimizer_steps=0`、`model_writes=0`、旧 root 写入 0

## 3. S001–S011 结果

全部 Gate 为 true。K=8 与 K=1 的 trainable parameters 均为 `34,125,197`，其中部署参数 `33,985,549`、training-only auxiliary 参数 `139,648`；slot 数只改变固定 buffer/张量形状，没有改变参数预算。

成对 reverse permutation 后 logits 与 trajectory max-abs delta 都为 `0`。one-hot source/target transition 的未选槽 delta 为 `0`，被选槽最大 delta 为 `1.705477`，inactive transition delta 为 `0`。辅助头物理剥离后没有 auxiliary state key，部署 logits delta 为 `0`。

注册量具也正确区分了三类控制：addressed positive 的 relevant mean-replace effect 为 `1.154701`、irrelevant effect 为 `0`；query swap logit L2 为 `1.414214`；duplicate-content query-swap null 为 `0`；legacy uniform-mean null 的 ownership contrast 为 `1.0`；K=1 被识别为单槽 null。

RTX 4070 Laptop GPU 上的 BF16 forward/backward 通过，所有输出与梯度均 finite，`111/111` 个有梯度张量非零，峰值显存 `320,380,928` bytes。该 backward 只用于数值连通性，不包含 optimizer step。

## 4. 证据边界与下一步

S0 只回答“架构和量具能否合法进入训练”。随机初始化的几何量只作为 diagnostic，不能作为 non-collapse 或多槽能力证据；synthetic positive/null 也不能替代真实 heldout ownership 与 causal intervention。

当前唯一授权是：另立 identity/root/lease，先冻结 ownership/state target bank 和 exposure，再实现并唯一运行 S1 Overfit32。S1 必须同时通过 answer exact、owner/query-owner、逐步 state closure、query swap、same-value/different-owner 与 recurrence/wrong-start 因果门。任一失败都原样封存停止。S2、single-seed formal、C2、V2-A PASS、V2-B 与 V2-C 当前仍为 `NOT_AUTHORIZED`。
