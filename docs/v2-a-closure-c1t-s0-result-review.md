# V2-A Closure C1T S0 结果复盘

## 1. 核心判定

C1T 的唯一 S0 preflight 与唯一 zero-training S0 均已完成并封存。正式机器终态为 `PASS_V2_A_C1T_S0_QUALIFICATION`，S001–S008 全部通过；这证明新的因果分区边界在固定 Qwen3.5-2B、真实 CUDA/BF16、完整 ERE/CPS 2×2 factorial 和持久逐卡 cache 上可按合同运行，且没有重新出现 whole-record hidden→H0 的联合上下文旁路。

本结果不是学习结果。全过程 `optimizer_steps=0`、`model_writes=0`，没有 checkpoint、参数选择或 accuracy Gate。唯一后继权限为 `C1T_S1_CONTRACT_DESIGN_ONLY`：可以设计、实现和审计 fresh S1 Overfit32 合同，但仍不授权启动 S1、S2、formal、C2 或宣称 V2-A PASS。

## 2. 封存身份

预检身份为 `V2-A-CLOSURE-C1T-CPW-S0-PREFLIGHT-20260901-1`。其 P001–P005 全过，result SHA-256 为 `BC1B18D581D36B45E144F05DEC8F143F946E687B8E50FF4D5967102D9AEF76C7`，evidence-seal SHA-256 为 `F326845A9BA4DE19444F0C5C5F6A3598C1109C5D70318E6D7BAC125B9706DDCC`，36 个封存条目全量重放一致。

正式身份为 `V2-A-CLOSURE-C1T-CPW-S0-20260901-1`。result SHA-256 为 `831BC5C2B7DE284F99CAB7E27F4A3BF3DB33FB71B4BE928B1D9A62B22A41D6A8`，evidence-seal SHA-256 为 `E4D9B3DC63D2409BD947500517B9382B42EAB8D04DB5827D0421A29A6F5FA0EE`，39 个 seal 条目无 missing、unexpected 或 mismatch。冻结 source identity 为 `709EA9B92C363BC1C90B1D336A4032BF4266D6E27D0204768C1AF83556228566`；S0 后复算未漂移。

## 3. 真实逐卡边界

source 固定为 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` 的 `Qwen3_5TextModel`，1,881,825,088 个 source 参数全部冻结。正式 S0 对 32 条记录的 192 张 object/operation/query 卡执行 192 次独立 source forward，总计 9,772 tokens，单卡 37–74 tokens；没有跨卡 batch、whole-record hidden 或旧 C1/C1S hidden 复用。source 编码用时约 120.13 秒，峰值 CUDA 分配约 3.79 GB。

持久 cache 的 `cards.pt` 为 40,289,075 bytes，SHA-256 为 `44232C7DBA87421E33DA4BD6E54F8A2792A5B720C45920FD4D573837E67A4723`；ledger SHA-256 为 `287CB720A186481DCD455C962A950370F5DAE24587065C434557B369AABFF55B`。写后 readback 重建全部 192 张卡并重算 text/token/hidden hash，`contains_targets=false`；答案、family、AST、factor、support slot、valid-choice mask 和 counterfactual index 均未进入 hidden cache。

## 4. 结构 Gate

完整任务审计覆盖 CPS/ERE 各 16 条、共 8 个 factorial group。每组四个 cell 齐全；单独翻转任一支持对象都会翻转 simulator answer，而且公开输入只改变对应对象卡。

未训练 C1T deployment graph 有 13,395,466 个参数。全部结构对照通过：

- 8 个 group 的 disable-recurrence logits 组内最大差均为 `0.0`，低于 `1e-6` Gate。这是 H0 没有读到两个 factor 联合上下文的直接结构证据。
- slot permutation 的 logit 与 trajectory 最大差均为 `0.0`。
- 单卡扰动对本卡 Boundary payload 的变化为 `0.0078125`，对其他卡为 `0.0`。
- operation→no-core logit、transition 非 target slot 写入的最大差均为 `0.0`。
- 真实 cache、完整 batch 8 的 BF16 forward/loss/backward 有限；C1T 阶段峰值 CUDA 分配约 265 MB。

随机初始化 loss 仅作有限性诊断：raw A–I answer CE `2.22744`，no-core confusion `0.04844`，support hinge `0.50063`，total `2.50198`。这些数值不构成任务能力或架构收益证据。

## 5. 证据边界与下一步

S0 关闭的是旧 C1S 之后最关键的测量前提：H0 不再能从同一份完整 source hidden 直接保留答案，任务本身也注册了真实的二因素 answer dependence。因此下一次 Overfit32 若 no-core 仍高或只有一个 contributor，就更可能是 C1T 的训练/objective/readout 重新形成捷径，而不是原任务低阶或 whole-record Boundary 污染。

S0 没有证明 recurrence 已学会、两个对象都成为功能必要条件、K=8 优于 K=1、heldout 泛化成立或 V2 架构有效。下一动作只能先冻结 S1 合同：固定 32 条训练记录、完整 factorial-group batch、真实 sealed card cache、raw A–I CE、no-core group invariance、同地址单因素 counterfactual margin、每族 Gate、单用 preflight/root 和严格 fail-stop。合同与 launch-readiness 审计完成后，仍需用户另行授权才可消费 S1。
