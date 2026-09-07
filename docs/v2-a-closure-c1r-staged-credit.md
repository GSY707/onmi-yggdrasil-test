# V2-A Closure C1R：分阶段 repair contract

本文件定义唯一身份 `V2-A-CLOSURE-C1R-STAGED-CREDIT-20260827-1`。它是对已消费 C1 证据的独立、可证伪 repair 设计，不是旧 `closure_c1` 的续跑，也不把任一局部指标提升解释成 V2-A 通过。

## 1. 问题边界与设计原则

输入只接受以下已封存事实：`artifacts/v2-a/closure-c1-failure-attribution-20260826-1/result.json` 及其 A003/A005/A006 诊断、`docs/v2-a-closure-c1-failure-attribution.md`、以及旧 C1 formal root 的只读 provenance。C1 的失败是整体 poststop Gate 失败：answer validation/OOD 和 hidden-intervention 等架构门均未通过；因此低 loss、局部 trace 改善、或方向冲突消失，都不能单独取得 C1 资格。

修复合同明确同时处理两类已归因问题：旧随机 chunk 的 exposure hygiene，以及 answer/trace gradient conflict 的因果隔离。两者不能在同一个不可分解的 joint update 中解释；因此合同把它们排成严格的两阶段链：Stage A 先建立 fresh answer-only 因果隔离，Stage B 只在 Stage A 的全部前置门通过后，冻结 Stage A 的 deployed graph，再单独训练新 trace probe。这里的“最小”指阶段内的可证伪变量和固定边界，不是声称整个 staged repair 只改变一个变量。

## 2. 唯一正式运行边界

正式输出根固定为：

* `artifacts/v2-a/closure-c1r-staged-credit-20260827-1`
* sibling lease：`artifacts/v2-a/closure-c1r-staged-credit-20260827-1.preflight-lease.jsonl`
* preflight 根：`tmp/v2-a-closure-c1r-staged-credit-preflight-20260827-1`
* preflight lease：`tmp/v2-a-closure-c1r-staged-credit-preflight-20260827-1.preflight-lease.jsonl`

先完成 sealed non-formal preflight，再以 `claim_single_use` 一次性占用 formal root。根或 sibling lease 任一已存在即原样拒绝；不得 retry、调参、换 seed、换 checkpoint、覆盖 root 或创建 successor formal。正式运行身份、source hash、输入 formal result/seal/checkpoint hash、模型配置和 seed 均写入 manifest，并由新 `closure_c1r.artifacts` 命名空间封存。preflight/formal 同时记录旧 C1 formal、cache 与 attribution 三个 root 的完整文件 metadata digest，结束时必须逐项不变；内容真实性仍由固定 result/seal hash 与 seal replay 证明。

### Stage A：answer-only causal isolation

Stage A 使用 fresh shared initialization，但保持 C1 的 model seed、order seed、model/config 和数据顺序合同不变；唯一 objective 是 answer-only，trace probe 不存在或不接收梯度。固定 endpoint 是 exactly 6144 optimizer updates，不能提前停、追加训练或选择最好 checkpoint。

Stage A optimizer 固定为 boundary/core/head learning rates `1e-4/2e-4/3e-4`、weight decay `0.01`、gradient clip `1.0`、warmup `256`；probe seed 与 overfit seed 另行固定并记录在 manifest，不得由运行时随机生成。

Stage A 与 Stage B 每步都必须验证 loss 与 clipped gradient norm 为 finite，并在预注册 history 点及最终点扫描对应 trainable parameter state；任何 NaN/Inf 作为 sealed crash 原样停止，不得继续 optimizer 或以 checkpoint 替换。

Stage A 完成后必须逐项通过 G004、G005、G006、G008、G009、G010，且 root、运行退出、结果和 seal 完整。任一项 FAIL 或缺失，Stage B 不得启动，整个 C1R 只产生诊断/失败证据，不获得 C1 资格。这样可以回答“仅 answer graph 是否足以恢复 poststop 行为”，而不会把 probe 修复冒充因果答案修复。

Stage A 是因果隔离阶段，不是最终资格阶段：它只允许我们判断 answer graph 在没有 trace 梯度干扰时是否恢复所需行为；即使 Stage A 通过，也不能跳过 Stage B。

### Stage B：deployed graph freeze + fresh probe（条件资格链）

Stage B 是条件资格链：只有 Stage A 已经通过全部前置 Gate，才有资格检验 trace probe 是否能在不改 deployed answer graph 的前提下恢复 trace readout。其 deployed graph 是 Stage A endpoint 的 exact freeze：参数、路由、输出头、normalization、数据顺序与答案评估路径均不可写。只新建并训练 trace probe；不得回写 backbone、answer head 或部署图。trajectory cache 必须写入并核对其来源 Stage A deployment-state SHA-256，不能只依赖同一进程中的对象关系。每个 epoch 对 ERE/train 与 CPS/train 的每条记录使用完整 target，按宽度 65、overlap 0 切成确定性的非重叠 blocks。一个 record 的所有 blocks 必须先做 valid-target-token mean 聚合，再执行一次 optimizer step；blocks 不是额外样本，不能改变记录级 batch 或 step 预算。

Stage B probe optimizer 固定为 trace learning rate `3e-4`、weight decay `0.01`、gradient clip `1.0`、warmup `256`；其 probe initialization 使用预注册 probe seed。

Stage B primary 启动前必须通过 frozen-probe positive control：从 train 中按冻结哈希规则抽取 ERE/CPS 各 16 条，共 32 条；在 Stage A exact-frozen trajectory 上训练 fresh probe，固定 batch 8（4+4）、100 epochs、400 updates、65-token complete blocks、warmup 100，并满足旧 Overfit32 的 all-token accuracy、content-token accuracy 与 owner-NLL-margin 三项门槛。它只证明 probe 路径和优化器在冻结轨迹上可工作，不取得架构资格；FAIL 必须在 primary Stage B 前停止。

固定 Stage B 合同为 batch 32（ERE 16 + CPS 16）、6 epochs、每 epoch 256 updates、共 1536 updates；每个 target token exact exposure 6，zero exposure 0。coverage report 必须同时证明 block 边界、完整 target、family balance、record 每 epoch 恰好一次、schedule hash 和上述 exposure。65-token complete-block schedule 是 coverage/hygiene 约束，用来消除旧随机 64-chunk 的盲区；它本身不是 capability gain、不是 C1 通过条件之外的替代物，也不能掩盖 Stage A poststop Gate 失败。

## 3. 梯度冲突与 Gate 处理

A005 显示 answer/trace 梯度存在方向性冲突，不能通过事后降低 trace 权重来“修复”；那会同时改优化目标和暴露量。Stage A 让 answer-only 因果问题单独可证伪。只有 Stage A 通过全部前置 Gate 后，Stage B 才能在被冻结的 deployed answer graph 上检验“新 probe 是否能恢复 trace 读出”。若 Stage A 通过而 Stage B 失败，结论是 probe/trace 路径仍失败；不得回到 joint training 继续混改。

整体 poststop Gate 是资格门，不是可选报告项。任何 validation/OOD、hidden intervention、route/decoder 或其他 required architecture Gate FAIL，都必须停止并保留失败证据。Stage B 的 coverage PASS 只能说明暴露合同成立，不能覆盖 answer graph 的 Gate FAIL。

## 4. 四种 repair 选择的比较

| 方案 | 改变的变量 | 归因性 | 用途 | C1 资格 |
|---|---|---|---|---|
| 仅修 exposure | 只改 chunk/schedule；answer/trace 仍混合 | 低：无法拆出梯度冲突与 graph 因果 | diagnosis/hygiene，验证 coverage 是否补齐 | 不直接取得 |
| fresh answer-only 因果隔离 | fresh shared-init + answer-only 6144 | 高：单独检验 answer graph | Stage A diagnosis，也是 C1R 必经前置 | 仅自身全 G004/5/6/8/9/10 后可进入 Stage B；不单独宣称 C1R |
| answer-only 建行为后冻结 deployed graph，再训 trace probe | 先完成 Stage A，再只训 fresh probe | 高：graph 与 probe 因果分层 | 唯一推荐的 C1R formal staged-credit 路径 | 仅 Stage A 全前置 Gate、Stage B 全 required poststop Gate 与 coverage/identity/seal 均 PASS；最多授权两个 fresh C1R seed |
| matched joint-control | 与 repair 匹配的 answer+trace 联训对照 | 可比较但同时改变多个变量 | diagnosis/负对照，解释 joint objective 的代价 | 永不单独取得本 C1R 资格 |

“fresh answer-only”与“freeze 后 fresh probe”是同一正式 C1R 的两个有序阶段，不是两个可任选的正式 arm。exposure-only 与 matched joint-control 可以作为预研诊断，但不能共享 formal root、不能越过 Stage A 门、不能授权 C2/V2-A pass。

## 5. 资源与可复现性边界

并行化只用于不改变合同的确定性计算：16 个 CPU 物理核可用于 tokenization、完整-target block 索引、hash/seal replay、评估批次准备和独立 family 数据管线；必须固定 worker 数、seed、顺序和归并顺序，避免并发改变 schedule。唯一 RTX 4070 才可用于模型训练/forward-backward；preflight 与 formal 都必须在领取 single-use root 前验证并记录实际 CUDA device name、device count、compute capability、PyTorch/CUDA 版本，设备名必须包含 `RTX 4070`。preflight 还必须以真实 64×65 packed probe 形状完成 zero-optimizer-step forward/backward，记录 valid tokens、耗时和 peak CUDA memory；OOM、非 BF16 或参数状态变化均 FAIL。核显、Windows 显示适配器或其他非-CUDA 设备不得冒充训练 GPU。若没有可验证的 RTX 4070 CUDA 设备，必须在 root 外拒绝，而不是消费身份或静默退回 CPU/核显。

预计工作量只用于容量规划，不改变 stop gate：Stage A 与旧 6144-update 量级相同（约 28.8 分钟的历史参考，实际以本机为准）；Stage B 是 1536 optimizer steps。ERE/CPS train targets 共约 1,251,500 target tokens、24,080 个 width-65 blocks；六个完整周期约 7,509,000 token exposures。由于按 record 聚合，Stage B 仍是 1536 steps，而不是 24,080 个 block steps。

## 6. 通过与授权

正式 PASS 必须同时具备：新 schema/identity、single-use lease、source identity、输入 provenance、Stage A 全部前置 Gate、Stage B exact-freeze 证明、完整 coverage report（exposure=6、zero=0）、coverage plan 与实际训练 schedule 的逐行及 SHA-256 双一致、所有 required poststop Gate、结果退出码和 evidence seal。PASS 只授权最多两个 fresh C1R seeds 的后续重复；永不授权 C2、V2-A pass、V2-B 或 V2-C。任何失败、缺证据、root/lease 碰撞或后置结果越界，均是本身份终止，不得以替代 checkpoint、调参或新 root 规避。
