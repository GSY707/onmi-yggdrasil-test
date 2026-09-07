# V2-A Closure C1T S0：真实逐卡边界资格化

## 1. 目标与证据边界

C1T 的目标不是继续修补旧 SRW，而是直接切换到因果分区工作区：对象、操作与查询分别成为独立公开卡片，Qwen 只负责逐卡编码，后续状态交互只能经过公开地址和 target-only transition。S0 只回答一件事：这条新边界能否在真实固定 Qwen hidden、真实 CUDA/BF16 和完整 2×2 factorial 任务上按合同运行，并且没有重新引入 whole-record 上下文旁路。

S0 是 zero-training 资格化。允许一次 BF16 forward/backward 检查梯度有限，但禁止 `optimizer.step()`、模型 checkpoint、参数选择、阈值调整和任何学习结论。S0 PASS 只授权编写并审计 S1 Overfit32 合同；S1 训练仍需用户另行明确授权。

## 2. 固定身份与单用顺序

预检身份为 `V2-A-CLOSURE-C1T-CPW-S0-PREFLIGHT-20260901-1`，固定 root 为 `tmp/v2-a-closure-c1t-cpw-s0-preflight-20260901-1`。S0 身份为 `V2-A-CLOSURE-C1T-CPW-S0-20260901-1`，固定 root 为 `artifacts/v2-a/closure-c1t-cpw-s0-20260901-1`。两者各有同级 single-use lease；root 或 lease 任一已存在都必须拒绝启动。

执行严格分成两个互不自动串联的命令。预检只处理 CPS g00 与 ERE g00 共 8 条记录、48 张卡；它 PASS 后仅授权一次 S0。S0 重新加载固定 source，独立编码完整 32 条记录、192 张卡，并将 cache、ledger、任务 bank、测量结果、source snapshot 与 seal 写入正式 root。任何阶段异常都必须在已领取的 root 内写入 CRASH result 和 evidence seal，且禁止重试。

## 3. 固定 source 与逐卡约束

source 固定为本机离线 `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`，只加载 `Qwen3_5TextModel` 语言模型子树，不加载视觉塔与 LM head。运行时固定 Python 3.11.9、PyTorch 2.13.0+cu130、CUDA 13.0、Transformers 5.13.1、FP16 source hidden width 2048；模型与 tokenizer 的五个本地文件按 `contract.py` 的字节数和 SHA-256 逐一复核，禁止网络回退。

编码器每次 forward 只能接收一张卡的一个字符串。对象卡、操作卡、查询卡不可组成跨卡 batch，也不可先编码 whole record 再切片。每次调用生成独立 token IDs、mask 与 final hidden；相同卡文本跨记录必须得到相同 token SHA 与 hidden SHA。cache 不保存答案、family、AST、factor、support slot、valid-choice mask 或任何训练目标。

## 4. 预检 Gate

预检在领取 lease 前先检查固定 paths 未消费、source closure 完整、本地资产 SHA、唯一 RTX 4070 CUDA device、compute capability 至少 8.0、BF16 可用且空闲显存不少于 5.5 GB。领取后固定执行：8 条任务的 simulator/factorial audit；48 次真实独立 card forward；cache ledger/hash/determinism audit；C1T source-only forward；BF16 loss/backward；gradient finite；optimizer/model write accounting 为零；最后重放 source hash 并封存 evidence seal。

只有所有检查 PASS，result 才能写 `authorizes=C1T_S0_ZERO_TRAINING_ONLY`。FAIL 或 CRASH 均为 `authorizes=nothing`，S0 不得启动。

## 5. S0 Gate

S0 固定八门，按顺序 fail-stop：

1. S001：source 资产、软件、GPU/BF16 与预检 seal/授权完全一致。
2. S002：32 条任务、8 个完整 factorial group、simulator replay、单因子单卡变化和 answer flip 全通过。
3. S003：192 次真实独立编码、192 张卡、192 条 ledger；无 whole-record reuse；同文本同 token/hidden；持久化后 readback 审计一致。
4. S004：模型 forward 只含注册的 hidden/mask/address/query 字段，答案和 ledger target 均停留在 loss/evaluator 外部。
5. S005：slot permutation 等变、单卡扰动不影响其他 Boundary payload、operation 不进入 no-core、transition 只写注册 target slot。
6. S006：在每个 2×2 group 内，disable-recurrence logits 的最大差不超过 `1e-6`；这证明 H0 没有重新读取两个 factor 的 whole-record 联合上下文，不要求随机未训练模型答对。
7. S007：真实 cache、完整 batch 8、CUDA BF16 下 forward/loss/backward 和全部梯度有限。
8. S008：optimizer steps 与 model writes 均为零，source hash 未漂移，结果/缓存/ledger/source snapshot 全部进入完整 evidence seal。

S0 不以随机初始化 accuracy、margin 或 loss 数值作为科学 PASS 条件；这些只作为有限性和未来复现实验的诊断值。

## 6. 停止与授权

任一 Gate FAIL 时，后续 Gate 标为 NOT_RUN，结果原样封存并停止。不得临场修改显存下限、card token 上限、dtype、source revision、模型 seed 或实现；不得换 GPU、清洗结果后重跑、复用旧 C1/C1S hidden cache、将 S0 PASS 写成学习或架构收益。

S0 PASS 的唯一授权是 `C1T_S1_CONTRACT_DESIGN_ONLY`：可以设计、实现和审计新的 S1 Overfit32 合同，但不能消费 S1 preflight/root，不能进入 S2/formal/C2，也不能宣称 V2-A PASS。
