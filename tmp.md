你现在执行 Project-Yggdrasil V2-A1.6「最小核心：可寻址连续状态递归」实验。

这是一次重新实现，不是在 A1.5 上继续加消融。严格遵守以下规格，不要自行改变架构，不要加入额外模块。

开始前完整阅读仓库根目录 AGENT.md。保留全部旧代码和 artifacts，不覆盖 A1.5 结果。

==================================================
一、本轮唯一研究问题
==================================================

只验证：

连续 latent state 能否通过一个共享、可寻址的 relation transition，逐步执行 COPY/SWAP，并在文本 hidden 接口下保持相同能力。

本轮禁止研究：

- 匿名 learned slots；
- K sweep；
- text-CoT 成本比较；
- audit decoder；
- Boundary-MoE；
- FFN-MoE；
- Attention Pump；
- 动态停止；
- 主动专家调用；
- 多模态；
- RL；
- verifier reward；
- teacher trace；
- 多 seed。

只跑一个 seed：20260715。

本轮分两阶段：

C0：结构化输入的 relation-addressed latent core。
C1：冻结 Qwen hidden 到同一个 C0 core 的边界接口。

C0 未通过时，禁止启动 C1。

==================================================
二、文件与目录
==================================================

新建以下文件，不修改旧 A1.5 主实现：

src/yggdrasil_v2/reasoning_medium/a1_6_data.py
src/yggdrasil_v2/reasoning_medium/a1_6_core.py
src/yggdrasil_v2/reasoning_medium/a1_6_train.py
src/yggdrasil_v2/reasoning_medium/a1_6_qwen.py
src/yggdrasil_v2/reasoning_medium/a1_6_qwen_train.py

experiments/v2_a1_6_core.py

tests/test_v2_a1_6_data.py
tests/test_v2_a1_6_core.py
tests/test_v2_a1_6_qwen.py

docs/v2-a1.6-core.md
tmp/V2-A1.6 result.md

所有新 artifact 写入：

artifacts/v2-a/a1_6/

不得删除或覆盖：

artifacts/v2-a/a1_5/
tmp/V2-A1.5 result.md
docs/v2-a1.5-latent-foundation.md

==================================================
三、A1.6 数据合同
==================================================

新 schema：

yggdrasil.v2-a1.6.relation-state-machine.v1

仍使用：

- registers：amber、cobalt、jade；
- values：A–J；
- operations：COPY、SWAP；
- 每条样本的三个初始值互不相同；
- source != target。

数据 split：

- train：8192，长度 1–4；
- validation：512，长度 2–4；
- test：512，长度 2–4；
- length_heldout：512，长度 5–6；
- relation_heldout：512，长度 2–4；
- causal_core：512，长度 2–6。

关键修正：

1. train 不再要求每一步都影响最终答案。
2. validation/test/length/relation 也不强制 all-step-necessary。
3. 只有 causal_core 要求每一步替换为 skip 后都会改变 queried final answer。
4. 不允许强制每条样本只有一个 COPY。
5. 不允许固定 COPY 的位置。
6. train 必须同时包含四种 bigram：

   COPY→COPY
   COPY→SWAP
   SWAP→COPY
   SWAP→SWAP

7. 对长度允许的位置，COPY/SWAP 的边际比例都应在 40%–60%。
8. COPY 必须出现在第 1、2、3、4 个位置，不能只出现在第一步。
9. source/target register 必须近似均衡。
10. 所有 split fingerprint 无重叠。

relation-heldout 定义：

- train/validation/test/length 中，COPY amber -> jade 完全禁止；
- relation_heldout 中，每条样本至少包含一次 COPY amber -> jade；
- amber 作为 COPY source 必须在 train 中通过 amber->cobalt 出现；
- jade 作为 COPY target 必须在 train 中通过 cobalt->jade 出现；
- amber/jade 这对寄存器必须在 train 的 SWAP 中出现；
- relation split 的长度、COPY 数量、COPY 位置和答案分布应尽量与 ordinary test 匹配。

每条 operation 需要保存：

- family；
- source；
- target；
- operation char span；
- family char span；
- source char span；
- target char span；
- state after this operation。

问题文本中还要保存：

- 三个初始 value 的独立 char spans；
- query register char span。

新增 audit-data 子命令。它必须输出：

artifacts/v2-a/a1_6/data-audit.json

audit 至少包含：

- 每个 split 的长度分布；
- family 总分布；
- 每个绝对位置的 family 分布；
- 四种 family bigram 分布；
- COPY 数量分布；
- COPY 位置分布；
- source/target 分布；
- relation holdout 检查；
- fingerprint overlap；
- causal_core necessary rate。

数据 Gate：

- train 中四种 bigram 都必须出现，且每种至少占全部 bigram 的 10%；
- train 中 position 1–4 都出现 COPY；
- 每个有效位置 COPY 比例必须在 0.35–0.65；
- relation holdout pair 在 train 中出现次数必须为 0；
- relation split 中该 pair 覆盖率必须为 1.0；
- causal_core necessary rate 必须为 1.0；
- cross-split overlap 必须为 0。

任一数据 Gate 失败时，停止，不得训练。

==================================================
四、C0 核心架构
==================================================

禁止复用 A1.5 的 SharedStructuredTransition。

实现新类：

RelationAddressedCore
RelationAddressedTransition

固定参数：

D_latent = 256
register slots = 3
attention heads 不需要用于全局 self-attention
FFN width = 512
shared transition across steps = true
T = program_length

不设置：

- control slot；
- query slot内容向量；
- mean pooling；
- 独立 answer head；
- source mean；
- absolute step embedding；
- sigmoid residual gate；
- whole-program encoder。

初始 latent state 只能由以下内容产生：

slot[amber] = value_embedding(start_amber) + register_key(amber)
slot[cobalt] = value_embedding(start_cobalt) + register_key(cobalt)
slot[jade] = value_embedding(start_jade) + register_key(jade)

初始 state 不得接收任何 operation 信息。

每个 operation 必须拆成三个独立角色：

- family latent；
- source-role latent；
- target-role latent。

禁止把 family/source/target 相加成一个 token后作为单 key cross-attention。

每一步 relation transition 严格按以下逻辑实现：

1. source-role latent 生成 source pointer logits：

   source_logits = source_query @ register_keys.T / sqrt(D)

2. target-role latent 生成 target pointer logits：

   target_logits = target_query @ register_keys.T / sqrt(D)

3. softmax 得到：

   source_weights [B, 3]
   target_weights [B, 3]

4. 从 slots 读取：

   source_read = sum(source_weights[i] * slots[i])
   target_read = sum(target_weights[i] * slots[i])

5. 将以下内容输入共享 operator MLP：

   source_read
   target_read
   family_latent

6. operator MLP 输出：

   proposed_source [B, D]
   proposed_target [B, D]
   source_write_gate [B, 1]
   target_write_gate [B, 1]

7. 使用 soft pointer 把 proposed state 写回 slots：

   next_slot[i] =
       slot[i]
       + source_weights[i] * source_write_gate * (proposed_source - slot[i])
       + target_weights[i] * target_write_gate * (proposed_target - slot[i])

source 和 target 在数据中保证不同。

除 source/target slots 外，其余 slot 应自然保持不变。

结构化 C0 中：

- source-role 和 target-role 来自 register embedding；
- family latent 来自 family embedding；
- pointer logits 使用监督 CE；
- COPY/SWAP 的实际状态变换由 operator MLP 学习；
- 不允许在代码中直接写死 COPY 或 SWAP 的赋值结果。

每一步输出全部三个 slot 的 state logits：

state_logits_t = shared_state_head(next_slots)

最终答案必须这样得到：

query_weights = one_hot(query_register)
queried_state = sum(query_weights[i] * final_slots[i])
answer_logits = shared_state_head(queried_state)

注意：

- answer 使用与逐 slot state decoding 相同的 shared_state_head；
- 不得创建独立 answer_head；
- 不得从 query/control latent 直接分类；
- 不得对 slots 做 mean pooling。

训练 loss：

state_loss = 所有有效步骤、三个 registers 的 CE
source_pointer_loss = 每步 source pointer CE
target_pointer_loss = 每步 target pointer CE
answer_loss = queried final state CE

总损失固定为：

loss =
    1.0 * state_loss
    + 0.25 * source_pointer_loss
    + 0.25 * target_pointer_loss
    + 0.5 * answer_loss

不使用 RL、cosine teacher loss 或其他辅助 loss。

==================================================
五、必须实现的完整性测试
==================================================

训练前必须通过以下单元测试：

1. 相同 start state/query、不同 operations：

   initial_slots 必须逐元素完全相同。

2. 修改未来第 k+1 步 operation：

   第 k 步 state 必须完全不变。

3. 修改第 k 步 operation：

   第 k 步及以后 state 允许变化；
   第 k-1 步及以前必须完全不变。

4. T=0：

   decoded state 必须等于 start state。

5. query 改变：

   state trajectory 不得变化；
   只有最终被读取的 register 改变。

6. answer logits 必须由 shared_state_head 产生。

7. 模型中不得存在名为 answer_head 的参数或模块。

8. transition 实例只能有一个，所有步骤共享。

9. pointer weights 每步和为 1。

10. operation family/source/target 必须作为三个独立输入进入 transition。

11. 不得存在读取完整 source text/hidden 的接口。

这些测试未通过时不得开始训练。

==================================================
六、C0 训练顺序与命令
==================================================

CLI 必须实现以下子命令和参数，使下面命令可以原样运行。

首先：

```powershell
$env:PYTHONPATH='src'
```

1. 运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_v2_a1_6_data.py `
  tests/test_v2_a1_6_core.py `
  tests/test_v2_a1_6_qwen.py -q
```

2. 生成数据：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py prepare-data `
  --output-dir artifacts/v2-a/a1_6/data `
  --seed 20260715 `
  --train-size 8192 `
  --validation-size 512 `
  --test-size 512 `
  --length-size 512 `
  --relation-size 512 `
  --causal-size 512
```

3. 审计数据：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py audit-data `
  --data-dir artifacts/v2-a/a1_6/data `
  --output artifacts/v2-a/a1_6/data-audit.json
```

必须程序化检查 Gate，不允许只在文档里人工声称通过。

4. 32-example overfit：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c0-train `
  --data-dir artifacts/v2-a/a1_6/data `
  --output-dir artifacts/v2-a/a1_6/c0-overfit32 `
  --device cuda `
  --seed 20260715 `
  --train-limit 32 `
  --validation-limit 32 `
  --steps 5000 `
  --batch-size 32 `
  --learning-rate 3e-4 `
  --validation-interval 100 `
  --early-stop-exact 0.999
```

overfit Gate：

- answer accuracy = 1.0；
- state token accuracy = 1.0；
- state full exact = 1.0；
- source pointer accuracy = 1.0；
- target pointer accuracy = 1.0。

连续两次 validation 都满足后才能提前停止。

如果 5000 steps 后仍未达到，停止整个 A1.6，不启动正式 C0。

5. 正式 C0：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c0-train `
  --data-dir artifacts/v2-a/a1_6/data `
  --output-dir artifacts/v2-a/a1_6/c0-formal `
  --device cuda `
  --seed 20260715 `
  --steps 6000 `
  --batch-size 128 `
  --learning-rate 3e-4 `
  --validation-interval 100 `
  --early-stop-patience 10
```

每次 validation 记录：

- answer accuracy；
- state token accuracy；
- state full exact；
- source pointer accuracy；
- target pointer accuracy；
- gradient norm；
- operator write gates；
- pointer entropy；
- train/validation loss。

保存：

latest.pt
best.pt
history.json
progress.json
results.json

best checkpoint 的选择分数：

score =
    state_full_exact
    + 0.25 * source_pointer_accuracy
    + 0.25 * target_pointer_accuracy

最终评测前必须重新加载 best.pt。

6. 五 split 正式评测：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c0-evaluate `
  --data-dir artifacts/v2-a/a1_6/data `
  --checkpoint artifacts/v2-a/a1_6/c0-formal/best.pt `
  --output artifacts/v2-a/a1_6/c0-formal/eval-all.json `
  --device cuda `
  --batch-size 128
```

必须评估：

train
validation
test
length_heldout
relation_heldout
causal_core

C0 Gate：

- ordinary test state full exact >= 0.995；
- length state full exact >= 0.95；
- relation-heldout state full exact >= 0.95；
- source pointer accuracy >= 0.995；
- target pointer accuracy >= 0.995；
- final answer accuracy不得明显高于对应 state full exact；
- 不允许出现 final=1.0、state full很低的脱钩。

C0 任一 Gate 失败时，停止，不启动 C1。

==================================================
七、C0 因果评测
==================================================

实现子命令：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c0-intervene `
  --data-dir artifacts/v2-a/a1_6/data `
  --split causal_core `
  --checkpoint artifacts/v2-a/a1_6/c0-formal/best.pt `
  --output artifacts/v2-a/a1_6/c0-formal/interventions.json `
  --device cuda
```

必须实现：

1. prefix fidelity：

   对每个 k，从 0 到 program_length：

   - 只执行前 k 步；
   - 模型 state 与符号 prefix-k oracle 比较；
   - 不得把 prefix-k 输出与原程序 final answer 比较。

2. task truncation：

   - 在 causal_core 上只执行前 k 步；
   - 再与完整程序 final answer 比较；
   - 用于证明完整 T 才能完成任务。

3. operation replacement：

   - 把第 k 步替换成另一个合法 COPY/SWAP；
   - 同步重算符号 oracle；
   - 比较修改后完整 state trajectory 和 final answer；
   - 不能只报告 changed prediction rate。

4. operation deletion：

   - 通过 operation_mask=false 跳过该步；
   - 不引入未训练的 NOOP family embedding；
   - 同步重算符号 oracle。

5. operation order shuffle：

   - 使用打乱后的程序重算完整 oracle；
   - 模型必须正确执行打乱后的程序。

因果 Gate：

- 所有 k 的 prefix state full exact >= 0.99；
- replacement counterfactual state full exact >= 0.95；
- deletion counterfactual state full exact >= 0.95；
- shuffled-program state full exact >= 0.95；
- causal_core 中 T=0 对完整 final answer 的准确率应接近 0；
- 不得使用 same-answer batch shuffle。

==================================================
八、C1：冻结 Qwen hidden 边界
==================================================

只有 C0 全部 Gate 通过才允许实现和运行 C1。

使用：

model_id = Qwen/Qwen3.5-2B
revision = 15852e8c16360a2fea060d615a32b45270f8a8fc

Qwen 完全冻结。

C1 不创建新 recurrent core。它必须加载：

artifacts/v2-a/a1_6/c0-formal/best.pt

并复用其中：

- register keys；
- RelationAddressedTransition；
- operator MLP；
- shared state head。

C1 只新增文本边界 adapter。

缓存必须保存独立 masks：

- start_value_masks [B, 3, L]；
- query_register_mask [B, L]；
- operation_family_masks [B, T, L]；
- operation_source_masks [B, T, L]；
- operation_target_masks [B, T, L]；
- operation_mask [B, T]。

禁止保存或使用“初始 slots cross-attend 整个 source”的路径。

C1 初始化：

- 每个 start value 只池化自己的 start_value_mask；
- 得到三个连续 value latents；
- 加上 C0 register keys；
- query hidden 只能生成三个 register 的 soft query pointer；
- query hidden 不得作为自由内容向量写入 slots。

每一步 operation：

- family latent 只来自 family mask；
- source-role latent 只来自 source mask；
- target-role latent 只来自 target mask；
- 第 t 步只能把第 t 条 operation 的三个 role latents交给 core；
- 不允许把完整 question hidden、其他 operation hidden 或 source mean 提供给 transition。

边界训练分两阶段。

阶段 C1-A：

- 冻结整个 C0 core；
- 只训练 Qwen hidden projector 和 role adapters；
- 对齐 C0 的结构化 latent/role表示；
- 使用 start value CE、query pointer CE、family CE、source pointer CE、target pointer CE；
- 可增加 normalized MSE 对齐 C0 structured embeddings；
- 不使用 teacher trace。

阶段 C1-B：

- C1-A 通过后，允许联合训练；
- boundary learning rate = 3e-4；
- core learning rate = 3e-5；
- 最多 1000 steps；
- 继续保留所有 role/pointer监督。

C1 完整性测试：

1. 相同 start/query、不同未来 operations，initial slots 必须相同。
2. 修改第 k+1 步文本，不得改变第 k 步 state。
3. query 文本只能改变 query pointer，不能改变 state trajectory。
4. 删除全部 operation spans 后，T=0 state 必须等于 start state。
5. 模型中不得存在 full-source cross-attention。
6. 模型中不得存在 source mean。
7. 答案仍由 shared state head读取 queried final slot。
8. operation intervention 时不得保留包含原 operation 的第二条输入通道。

==================================================
九、C1 命令
==================================================

1. 缓存：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c1-cache `
  --data-dir artifacts/v2-a/a1_6/data `
  --output-dir artifacts/v2-a/a1_6/c1-cache `
  --model-id Qwen/Qwen3.5-2B `
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc `
  --device cuda `
  --max-length 512 `
  --max-train-examples 4096 `
  --shard-size 64
```

发现任意 span mask 为空或 token length 超限时直接报错，禁止静默修补或截断。

2. C1-A 边界训练：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c1-train `
  --data-dir artifacts/v2-a/a1_6/data `
  --cache-dir artifacts/v2-a/a1_6/c1-cache `
  --core-checkpoint artifacts/v2-a/a1_6/c0-formal/best.pt `
  --output-dir artifacts/v2-a/a1_6/c1-boundary `
  --device cuda `
  --seed 20260715 `
  --phase boundary `
  --steps 3000 `
  --batch-size 64 `
  --boundary-learning-rate 3e-4 `
  --validation-interval 100 `
  --early-stop-patience 10
```

C1-A Gate：

- start value accuracy >= 0.995；
- query pointer accuracy >= 0.995；
- operation family accuracy >= 0.995；
- source pointer accuracy >= 0.995；
- target pointer accuracy >= 0.995；
- ordinary validation state full exact >= 0.98。

若失败，停止，不进入 joint。

3. C1-B 短程联合：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c1-train `
  --data-dir artifacts/v2-a/a1_6/data `
  --cache-dir artifacts/v2-a/a1_6/c1-cache `
  --core-checkpoint artifacts/v2-a/a1_6/c0-formal/best.pt `
  --boundary-checkpoint artifacts/v2-a/a1_6/c1-boundary/best.pt `
  --output-dir artifacts/v2-a/a1_6/c1-joint `
  --device cuda `
  --seed 20260715 `
  --phase joint `
  --steps 1000 `
  --batch-size 64 `
  --boundary-learning-rate 3e-4 `
  --core-learning-rate 3e-5 `
  --validation-interval 100 `
  --early-stop-patience 8
```

4. C1 正式评测：

```powershell
.\.venv\Scripts\python.exe experiments/v2_a1_6_core.py c1-evaluate `
  --data-dir artifacts/v2-a/a1_6/data `
  --cache-dir artifacts/v2-a/a1_6/c1-cache `
  --checkpoint artifacts/v2-a/a1_6/c1-joint/best.pt `
  --output artifacts/v2-a/a1_6/c1-joint/eval-all.json `
  --device cuda `
  --batch-size 64
```

C1 Gate：

- ordinary test state full exact >= 0.98；
- length state full exact >= 0.90；
- relation-heldout state full exact >= 0.90；
- pointer accuracy >= 0.98；
- T=0 不得直接得到完整程序 final answer；
- T=1 不得在多步 causal_core 上直接达到完整答案；
- counterfactual replacement/deletion state full exact >= 0.90。

C1 未通过时停在 A1.6，不启动任何后续架构。

==================================================
十、明确禁止的错误
==================================================

不得：

- 再次让 initial latent 读取全部 operations；
- 使用整个问题的 source cross-attention；
- 使用 query/control slot直接预测答案；
- 使用独立 answer head；
- 使用 mean pooling 读取答案；
- 把 family/source/target 相加成单一 operation token；
- 使用单 key operation cross-attention冒充关系绑定；
- 强制每条训练样本只有一个 COPY；
- 把 all-step-necessary 强加给整个训练集；
- 使用未训练的 NOOP embedding做主要干预；
- 用 changed prediction rate替代 counterfactual accuracy；
- 使用 same-answer batch shuffle；
- 跑 0.8B text baseline 与 2B latent 比较；
- 跑 text-CoT baseline；
- 扫 K；
- 跑多 seed；
- 启动 A3/A4/V2-B；
- 因一次失败删除旧代码或 artifacts；
- 把普通 IID 拟合写成架构成立。

==================================================
十一、停止规则
==================================================

严格按以下顺序停止：

1. 数据 audit 失败：停止。
2. 单元测试失败：停止。
3. C0 overfit32 失败：停止。
4. C0 ordinary/length/relation 任一 Gate 失败：停止。
5. C0 counterfactual Gate 失败：停止。
6. C1-A boundary Gate 失败：停止。
7. C1 final Gate 失败：停止在 A1.6。

停止后只报告失败层级和证据，不自行设计 A1.7，不继续试超参数网格。

==================================================
十二、最终报告格式
==================================================

最终写入：

tmp/V2-A1.6 result.md

必须按以下结构：

1. 最终 Gate 判定；
2. 数据 audit；
3. C0 overfit；
4. C0 ordinary/length/relation；
5. C0 prefix/counterfactual；
6. C1 boundary；
7. C1 final；
8. 完整性检查；
9. 已完成；
10. 未完成；
11. 当前能够声称什么；
12. 当前不能声称什么；
13. 所有可复现命令；
14. artifact 路径；
15. 若失败，明确失败发生在哪一个停止点。

报告必须优先使用：

- state full exact；
- pointer accuracy；
- counterfactual state exact；
- prefix fidelity。

final answer accuracy 只能作为派生指标，不能覆盖 state failure。

如果长时间训练预计无法在当前回合完成，必须至少完成：

- 数据和 audit；
- 所有单元测试；
- C0 overfit32；
- 给出剩余正式训练的原样命令。

不得用未收敛 smoke 替代正式结果。