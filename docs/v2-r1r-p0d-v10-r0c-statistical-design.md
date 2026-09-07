# V2-R1R P0-D v10 R0C-statistical 冻结设计

日期：2026-08-02
阶段目标：资格化 G07/G08 所需的 source-only 解析原语与真实统计 learner
证据等级：有限 measurement-component qualification；不是 production data、模型、训练或架构证据

## 1. 核心判断

R0C 不再接受“manifest 声明使用了某算法”作为证据。它只接受一组可以由手算 fixture 独立核对的实际计算：source-only parser、带 Laplace 平滑的条件多数、word/character multinomial Naive Bayes、确定性 grouped five-fold、split-local cross-validation 和 train-fit-heldout。

本阶段只验证这些测量部件是否按书面算法工作。它不读取 v9 qualification records，不生成 ERE/CPS production data，不评估真实数据是否存在捷径，也不运行 Qwen、Boundary、core、cache、GPU 或训练。R0A/R0B 的 accepted runtime 与 formal artifact 保持只读；R0C 通过也不能自动授权 R0D。

## 2. 直接切换与版本

- contract version：`r1r-p0d-v10-r0c-statistical`
- formal root：`artifacts/v2-r1r/p0d-v10-r0c-statistical-20260802-1/`
- PASS status：`PASS_R0C_STATISTICAL`
- public audit：`audit_statistical_bundle(path)`
- CLI：`audit-statistical`、`preflight-statistical`、`seal-statistical`

v10 新增 `src/yggdrasil_v2/r1_revalidation/learner/`，不得 import `tests/`、v9 `audit/` 或 accepted simulator。v9 `audit/` 与 `common/` 必须与 v9 formal source snapshot 的十个 Python 文件逐字节一致，v7/v9 两棵 prior artifact 必须按各自 evidence seal 全树复算；二者同时纳入 v10 frozen guard。v9 active CLI alias 和 `tests/v2_r1r_v9/` 测试源在 v10 直接切换时删除，其历史真相由 v9 source snapshot、文档和 sealed artifact 保留。不得建立兼容 wrapper。

## 3. 冻结资格输入

R0C control bundle 精确包含：

```text
qualification-bundle/
  manifest.json
  qualification-inputs.json
  expected-results.json
  input-seal.json
  source_snapshot/
```

`qualification-inputs.json` 只保存算法输入，不保存运行时推导结果；`expected-results.json` 只保存主设计层冻结的 golden outputs。两者分别封印，运行时不得用计算结果回写 expected。所有 JSON schema 都是 exact schema：缺字段、额外字段、重复 id、未知 label、非法 choice mask 或非有限数值均失败。

统一 label 顺序固定为 `L0 < L1 < L2`。所有 tie 都按这个局部顺序打破；预测只能在 `valid_choice_mask=true` 的 label 中选择。row id 与 group id 必须是非空唯一字符串。

资格 profile 至少覆盖：

1. ERE 可见 initial/rule/event literal 与 CPS candidate/cost/action-definition parser；
2. question-only、surface counts、choice mapping 与 mask fallback；
3. categorical seen/unseen、平票和 masked prediction；
4. lowercase Unicode word tokenizer 与 whitespace-normalized character 3–5 gram；
5. word NB、character NB 的 class/feature counts、vocabulary、exact rational score 与 prediction；
6. grouped five-fold 的 group integrity、fold map 与 balance summary；
7. split-local CV 的逐 row prediction、confusion matrix、fold-local vocabulary hash 和 fit fingerprint；
8. train-fit-heldout 的逐 row prediction、confusion、train-only vocabulary 与 fit fingerprint；
9. OOV-only、empty vocabulary、single-class fit、all-false mask、row/group overlap、duplicate id 和 insufficient-group 负控。

## 4. Source-only parser 合同

parser API 只接收 `source`、`family`、`heuristic` 与 `valid_choice_mask`；不存在 AST、answer、role、template id、certificate、split 或 hidden metadata 参数。

资格 grammar 只用于验证 parser primitive，不冒充 production renderer：

- 家族头：ERE 精确一个 `World:`，CPS 精确一个 `Planning:`；
- 共同字段：`Question:` 与 `Choices: L0=value | L1=value | L2=value`；
- ERE 可见行：`Initial:`、`Rule <name>:`、`Event <name>:`；
- CPS 可见行：`Action <name>:` 与 `Candidate L*:`，candidate 显式列出 action 序列和 raw cost。

ERE parser 必须实现 `initial_value`、`first_rule_literal`、`last_rule_literal`、`first_event_literal`、`last_event_literal`、`last_mention`。CPS parser 必须实现 `first_candidate`、`last_candidate`、`shortest_candidate`、`longest_candidate`、`lowest_raw_cost`、`highest_raw_cost`、`first_definition`。任何未声明 source line、重复 action name 或跨家族行直接失败；尤其不得静默接受 `Answer:`。如果 heuristic 原始结果不在 valid mask，返回最低序的 active label；all-false mask 直接失败。

`heuristic_prediction` 的公共签名精确为 `source, family, heuristic, valid_choice_mask`，不提供 answer、label order、certificate、split 或 hidden metadata 注入口。

source 先统一 CRLF/LF、删除行尾空白、折叠行内连续空白；该 normalization 同时用于 parser、analyzer 和 fit fingerprint。parser 不得读取 fixture expected values。

## 5. 统计 learner 合同

### 5.1 条件多数

每个 category × label 保存 raw count。预测分数为 `count + 1`；未知 category 的所有分数均为 `1`。只在 active label 内取最大值，平票按 label 顺序。禁止把 eval label 加入 count。

### 5.2 Multinomial Naive Bayes

只使用 Python 标准库，`alpha=1` 同时用于 class prior 与 feature likelihood：

```text
P(c) = (N_c + 1) / (N + C)
P(f|c) = (count(c,f) + 1) / (feature_total(c) + |V|)
```

word tokenizer 为 lowercase 后的 `(?u)\b\w+\b`；character analyzer 对 lowercase、连续空白折叠后的全文取连续 `3–5` gram。vocabulary 只由 fit rows 建立，按 Unicode code point 排序；OOV feature 丢弃。预测使用 `fractions.Fraction` 形成 exact score，不允许用浮点近似改变 tie。重复 token 必须按 multiplicity 计数，不能退化为 Bernoulli NB。

model state 必须保存 class counts、feature totals、feature-count digest、vocabulary、vocabulary SHA-256 与规范化 fit-row fingerprint。任何空 vocabulary 或单类 fit 直接失败。

### 5.3 Grouped five-fold

fold builder 先聚合 group label counts，再按 `SHA256("r1r-v10-fold|" + group_id)` 排序。每个 group 依次放入使下列三元组字典序最小的 fold：

1. 所有 label 在 folds 间的最大 count deviation；
2. fold row total 的最大 deviation；
3. fold index。

同一 group 永不拆分。少于五个 group、duplicate row id、未知 label、空 fold，或任一 fit side 变成单类时失败。

### 5.4 CV 与 train-fit-heldout

split-local CV 每个 fold 都重新创建 learner、vocabulary 与 fit fingerprint，只使用另外四 fold。输出保存 group-to-fold map、逐 row prediction、confusion matrix、每 fold vocabulary hash 和 fit fingerprint。

train-fit-heldout 只 fit 一次 train；heldout-only token 必须保持 OOV。train 与 heldout 的 row id 或 group id 有任一交集即失败。禁止复用 split-local prediction、vocabulary、model object 或 confusion counts。

confusion matrix 固定为 truth row × predicted column 的 `L0/L1/L2` 3×3 整数矩阵。

## 6. Gate、raw metrics 与独立不变量

G07 只资格化 source-only parser 与条件多数：bundle/schema/profile、question view、surface counts、heuristic prediction、精确四参数 API 与未知行拒绝、categorical counts、seen/unseen/tie/mask prediction及 parser negative controls 必须全部通过。

G08 只资格化 analyzer、NB、fold、CV 与 heldout：word tokens、character grams、vocabulary order/hash、class/feature counts、exact score/prediction、OOV/mask tie、fold map/balance、CV prediction/confusion/fold-local state、heldout prediction/confusion/train-only state 与全部负控必须全部通过。

raw metric registry 精确包含 34 项：G07 13 项、G08 21 项。它是 Gate conjunction 的唯一机器映射。每个 metric 必须至少被一个已声明且实际命中的非 holdout adversary kill；coverage 只统计声明映射，不能用同一输入异常顺带造成的级联失败补足其他 metric。schema、parser、categorical、NB、fold/CV、heldout/isolation 六个 transform family 各保留一个不参与 metric mapping 的 holdout。

输入只读与 novel grouped-pair 原子性/row-order invariance 是 runner 的独立不变量，不伪装成可由同一 expected file 自证的 raw metric。metric kill coverage 不能替代主设计层 registry 外公共函数探针。

为阻止“修改 qualification inputs、按同一实现重算 golden、重新 seal”自证，audit 另对标准化后的输入与 expected profile 检查固定语义摘要：inputs `D3D168DC1C96B34A74E90B1DF12F54B1260B1327724D9484E514762CE8598615`，expected `2F4B36C9D78BEAC5D2832C66E3DC723B0AE5116CB8691886CE31F248C78779F9`。该摘要位于两个 Gate 之外，但属于 overall PASS 必要条件；允许的 key/row order、whitespace 与 relocation 变换不会改变它。

## 7. 正向不变换与反例

正向 metamorphic 至少覆盖 JSON key order、各 section row order、fit row order、可折叠 whitespace/line ending 和 bundle relocation。变换前后 canonical report 必须逐字节一致，audit 输入保持只读。

负向 adversary 精确为 39 个：33 个 registry-mapped adversary 加六个 family holdout。它们覆盖额外 schema、seal、profile id/count、parser source/expected、hidden field、categorical fit/eval、token/gram expected、NB fit/expected state、fold group/map、CV prediction/confusion/fold vocabulary、heldout overlap/prediction/confusion/vocabulary，以及每个错误控制的期望错误码。

formal 前主设计层还必须构造未登记公共函数探针，至少包括 parser 精确签名与未知 `Answer:` 行拒绝、duplicate-token multinomial-vs-Bernoulli、Unicode/OOV mask tie、character whitespace normalization、row-order/group integrity、heldout-only signal leakage 和 train/heldout group overlap。任何已知书面违约被错误接受，R0C 主审失败，即使 machine formal 为 PASS。

## 8. Formal conjunction 与 artifact

formal root 精确包含：

```text
attempt.json
qualification-bundle/
statistical-report.json
adversary-ledger.json
coverage-ledger.json
metamorphic-ledger.json
assessment.json
run-metadata.json
evidence-seal.json
```

只有以下全部成立才输出 `PASS_R0C_STATISTICAL`：

1. v7 R0A 与 v9 R0B accepted evidence seal hash 精确、seal 全树复算通过，当前 `common/` 与 `audit/` 等于 v9 source snapshot；
2. frozen input guard 全部通过；
3. positive G07/G08 全真且连续两次 report bytes 相同；
4. missing/malformed bundle fail-closed；
5. 39/39 adversary 通过，34/34 声明 raw metric kill coverage，六个 family holdout 均通过；
6. positive metamorphic 全通过，输入只读；
7. formal root 顶层 entry 精确；
8. 输入与 expected 固定语义摘要通过；
9. evidence seal 可独立复算；
10. `authorization_created=false`。

formal command 只能调用一次。root 在求值前以 `exist_ok=false` 占用；任何失败保留 partial root、停止并主审，不修复、不覆盖、不换 suffix 重跑。通过后同样停止，执行层不得创建 R0D authorization。

## 9. 证据边界

R0C 通过只说明当前有限 numeric/source grammar 下的测量算法实现正确，并且已知泄漏与边界错误能被资格测试发现。它不说明 production renderer 可被完整解析，不说明 ERE/CPS 数据无捷径，不说明 generator、模型或 latent 架构可训练。只有 R0D 才能组合 R0A–R0C；只有后续 R3 可能关闭 P0-D。
