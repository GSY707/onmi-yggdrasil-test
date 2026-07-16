# Directory Reference

本仓库是 Project-Yggdrasil V2 的独立研究工作区。当前主线已经从旧 Stage A—AV-J-C 直接切换到 V2-A，再在 V2-A 通过后进入 V2-B。关联项目 `C:\skzy\QuickFileTransport\世界树计划` 的设计哲学是上位概念真源，本仓库不修改其源码。

## 当前真源

| 路径 | 地位与用途 |
| --- | --- |
| `docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md` | 当前唯一目标架构规范；定义连续 latent recurrence、Boundary-MoE、FFN-MoE、审计和边界。 |
| `docs/Project-Yggdrasil V2 从架构验证到商用路线图.md` | 当前唯一高层路线；按 Gate 推进 R0、R1/R2/R3、架构完整版和商用路线。 |
| `docs/next-stage-test-plan.md` | 当前最近执行真源；记录 V2-A/A1.5–A1.11 结果、A1.12 Reasoner 内部最小拆分方向、V2-B 顺序、证据口径、Gate、成本和旧路线收口。 |
| `docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md` | V2 决策形成记录；只保存推导和审阅依据，不与白皮书并行定义规范。 |
| `docs/moe-model-assembly-comparative-review-2026-07-14.md` | V2 的 Boundary-MoE/FFN-MoE 与公开模型路线的中文对照；区分已被其他模型验证的局部思想、完整架构未验证边界和 V2-B 未启动状态。 |
| `docs/DIRECTORY_REFERENCE.md` | 本索引；新代码、测试、文档和归档必须同步这里。 |
| `README.md` | 面向仓库使用者的当前状态说明；展示 A1.9 正证据、A1.10 联合失败、A1.11 一级归因和严格的未完成边界。 |
| `docs/v2-a-reasoning-medium-experiment.md` | V2-A 当前实现合同、Qwen3.5 基座、数据 schema、A0/A1/A2 结果、Gate 判定和失败边界。 |
| `docs/v2-a1.5-latent-foundation.md` | A1.5 独立 schema、P0 结构化正控制、P1 Qwen hidden 接口、P2 learned-slot formal、因果干预和停止门禁。 |
| `docs/v2-a1.9-qwen-boundary.md` | A1.9 冻结 Qwen hidden → 冻结 A1.8 structured core 的预注册合同、三 run formal/causal 结果、成本、捷径干预和证据边界。 |
| `docs/v2-a1.10-anonymous-workspace.md` | A1.10 full-token frozen-Qwen cache、匿名 K-slot workspace、通用 recurrent reasoner、方法修正、三 run formal 失败、成本和证据边界。 |
| `docs/v2-a1.11-fault-localization.md` | A1.11 Boundary × Reasoner 2×2 合同、方法纠正、严格 overfit/formal 结果、一级故障归因和 A1.12 边界。 |
| `docs/v2-a1.6-core.md` | A1.6 relation-addressed continuous state core 的数据合同、结构完整性、C0 停止点和证据边界。 |
| `docs/v2-a1.7-core.md` | A1.6 closure 归因、A1.7 受控 relation 数据、三 seed formal/causal、2×2 消融、8/12/16 步压力与成功概率真源。 |
| `docs/v2-a1.8-long-horizon.md` | A1.8 T1–16 均衡随机深度合同、三组独立 data/model seed、T20/T24 Gate、T32 诊断、稳定性、成本和归因真源。 |

## 研究辅助入口

| 路径 | 用途与边界 |
| --- | --- |
| `docs/research-agent-literature-commands.md` | 面向 Google Gemini Deep Research 的三天研究冲刺命令；覆盖 A1.5 当前故障、V2-A3/A4、V2-B、I/O、工作树/记忆、专家晋升、完整架构和商用安全，并组织增量、反证与综合。研究报告只作为决策输入，不自动修改架构真源或 Gate。 |

## 当前实现状态

V2-A0 数据/基座链路、Qwen3.5-2B/0.8B text-CoT probe 和当前结构的 V2-A1 mechanism smoke 已实现；旧 A2 的 K/T、prompt、transition、读出、full-data、teacher/masked state supervision、token-wise source adapter、step-level verifier RL、latent-attention probe 与 0.8B/2B 对照仍未达到强 text-CoT 基线。source-layer bank、query init、token mixer、latent-attention 等失败入口已从当前代码删除。新增的 A1.5 独立 schema 与 P0/P1/P2 代码已完成：P0 32-example overfit final/state full exact `1.0/1.0`，4096-example best ordinary test `1.0/1.0`，composition-heldout `0.2734/0`，length-heldout final/state full exact `1.0/0.2266`；真实 Qwen3.5-2B FP16 hidden cache 已分片生成，P1 4096-cache formal ordinary validation final/state `1.0/1.0`，小 probe 仅作为 underfit 诊断；P2 K=8 formal ordinary test `1.0/1.0`、composition `0.2773/0`、length `1.0/0.6484`，same-answer composition shuffle 失败；A1.5 0.8B no-cap zero-shot 128 条 test/composition/length formal 分别为 `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`，2-shot test/composition/length final/state 也分别为 `0.1328/0.0156`、`0.1953/0`、`0.0938/0`（格式解析率高但 state 失败）。A1.5 未通过，A3/A4/V2-B 均未启动。A1.5 真源见 `docs/v2-a1.5-latent-foundation.md` 与 `tmp/V2-A1.5 result.md`；旧 A2 结果仍只作历史 probe，不得与 A1.5 混写。

A1.6 已作为失败证据保留：data audit 与 overfit32 通过，正式 C0 ordinary/length trajectory full 为 `1.0/1.0`，relation-heldout 为 `0.765625`，按 Gate 停止。只读 checkpoint 诊断显示 oracle-reset one-step 与 predicted hard re-embed diagnostic 在 test/length/relation/causal 均为 `1.0`，失败定位为 continuous latent closure；旧 final-vs-trajectory Gate 无效，旧 relation split 也混入多变量。因果干预与 C1 未启动。

A1.7 是当前结构化 core 证据：独立受控数据合同把唯一 holdout 冻结为 `COPY amber→jade`。三个初始化 seed 的 5–6 步 formal C0 与 causal intervention 均通过；目标 seed 的 test/length/relation trajectory full 为 `1.0/0.996094/0.998047`。但 2×2 消融表明短程通过不能只归因于地址/内容分离或 closure；closure 的明确作用是减缓长程漂移。三个目标 checkpoint 的 8/12/16 步严格压力 Gate 为 `0/3`，T16 supported 为 `0.894531/0.941406/0.894531`。当前对 V2-A 形成 matched text-CoT Pareto 的工程判断为 `40%–55%`、中心约 `48%`。该结论只覆盖三寄存器结构化合成 core；Qwen/C1、匿名 workspace、完整 V2-A、A3/A4/V2-B 均未启动。真源见 `docs/v2-a1.7-core.md`、`tmp/V2-A1.7 result.md` 与 `artifacts/v2-a/a1_7/core-assessment-summary.json`。

A1.8 是当前最新 core 证据：保持 A1.7 架构和 closure 不变，把训练切换为 T1–16 batch 内均衡随机深度。三组独立 data/model seed 的 short、T8/12/16、OOD T20/T24、relation 和 causal Gate 全部通过；T16 supported/relation 最低为 `0.996094/1.0`，T24 为 `1.0/0.996094`，诊断性 T32 为 `0.988281/1.0`。跨 run 及相对 A1.7 fingerprints overlap 为 `0`。证据否定 T16 必然内在发散并支持 horizon mismatch，但没有 compute-matched 地拆分长度覆盖和 transition exposure；仍不覆盖 Qwen hidden、匿名 workspace 或 matched text-CoT Pareto。A1.8 完成时的工程概率为 V2-A Pareto `45%–60%`、中心约 `53%`。真源见 `docs/v2-a1.8-long-horizon.md`、`tmp/V2-A1.8 result.md` 与 `artifacts/v2-a/a1_8/assessment-summary.json`。

A1.9 是当前最新 boundary 证据：冻结 Qwen3.5-2B 与三组已经通过的 A1.8 core，只训练 value、family 和共享 register adapter。三个独立 data/core/adapter seed 的 cache audit、formal 与 hidden causal Gate 全部通过；T16 supported/relation 最低 `1.0/0.996094`，T24 为 `1.0/1.0`，诊断性 T32 为 `0.996094/1.0`，mapping 最低 `1.0`。query swap 和 same-answer/different-trajectory 跟随新 oracle 为 `1.0`，no-hidden 与 independent role shuffle trajectory 为 `0`；core hash 未改变、跨 run fingerprint overlap 为 `0`。证据只覆盖 oracle-role-segmented Qwen hidden，不覆盖 learned full-text role extraction、匿名 workspace 或 matched text-CoT Pareto。当前 V2-A Pareto 工程判断为 `48%–63%`、中心约 `55%`。真源见 `docs/v2-a1.9-qwen-boundary.md`、`tmp/V2-A1.9 result.md` 与 `artifacts/v2-a/a1_9/assessment-summary.json`。

A1.10 是联合目标配置的失败证据：在同一轮删除 oracle span mask 和显式三寄存器 scaffold，使用 frozen Qwen3.5-2B 的完整 last-hidden + attention mask、匿名 `K=8` learned slots，以及两层参数共享的通用 recurrent Transformer。修正版 overfit32 通过，但三个独立 data/model seed 正式 Gate 为 `0/3`，hidden interventions 按停止规则未运行。该阶段自身不能单独归因，后续 A1.11 已补齐正交诊断。真源见 `docs/v2-a1.10-anonymous-workspace.md`、`tmp/V2-A1.10 result.md` 与 `artifacts/v2-a/a1_10/assessment-summary.json`。

A1.11 是当前最新故障定位证据。Boundary 臂在无 oracle span、保留 frozen A1.8 core 的修正合同下，overfit32 trajectory/answer/mapping 全为 `1.0`，但 source/target pointer 最低均为 `0.9642857143`，严格 Gate failed；只读 hard re-embedding 全部恢复 `1.0`，所以存在连续 latent → frozen address geometry 缺陷，但 task-level formal 未运行。Reasoner 臂直接使用 exact symbolic typed roles，不加载 Qwen/cache/adapter/core；overfit32 全通过，三组 formal 稳定 `0/3`，全部主要 trajectory 为 `0–0.003906`，训练集均衡诊断 trajectory 也只有 `0–0.003906`。这足以否定纯组合故障并把主要独立失败源定位到 anonymous binding／generic transition／当前 readout objective 这一整臂，但尚未拆开三者。当前 V2-A matched Pareto 工程判断为 `22%–35%`、中心约 `28%`。真源见 `docs/v2-a1.11-fault-localization.md`、`tmp/V2-A1.11 result.md` 与 `artifacts/v2-a/a1_11/assessment-summary.json`。

## 当前 V2-A 代码与测试

| 路径 | 用途 | 当前证据地位 |
| --- | --- | --- |
| `pyproject.toml` | 当前 V2-A Python 依赖、包布局和 pytest 配置；Torch CUDA 运行时由本机环境选择。 | 当前工程合同 |
| `uv.lock` | 当前 Python 依赖解析锁文件；不等同于 GPU 运行时证据。 | 环境复现辅助 |
| `experiments/v2_a_reasoning_medium.py` | 统一 CLI：数据生成、direct/answer-only/text-CoT baseline、latent training/resume。 | 当前 V2-A 运行入口 |
| `src/yggdrasil_v2/reasoning_medium/data.py` | `symbolic-state-machine.v4` 数据生成器、heldout split、依赖和泄漏 manifest。 | A0 数据合同 |
| `src/yggdrasil_v2/reasoning_medium/prompts.py` | prompt contract v3、三路/encoder prompt、可选 demonstrations、答案解析和语义终止。 | A0/A1 计量组件 |
| `src/yggdrasil_v2/reasoning_medium/model.py` | 冻结 Qwen 文本塔提取、latent queries、copied/MLP transitions、token-wise source adapter、source 初始化/逐步 reread、mean/flatten readout 和 no-bypass 检查。 | A1/A2 mechanism 与诊断实现 |
| `src/yggdrasil_v2/reasoning_medium/train.py` | hidden-cache v5（last hidden + 有效程序步骤 mask）、中间 state/query-state labels、可选 teacher/state 训练辅助、step-level self-critical verifier RL、fit/评估、干预、checkpoint/resume 和结果 JSON。 | A1/A2 probe 实现；verifier-RL 当前只形成失败 probe，不是 Gate 证据 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_data.py` | A1.5 v1 必要步骤数据、长度/组合 split、operation spans/masks、反事实检查和弱基线。 | A1.5 contract/mechanism |
| `src/yggdrasil_v2/reasoning_medium/a1_5_model.py` | P0 结构化显式 register-slot recurrent core 与 shared self/cross-attention + FFN transition。 | A1.5 P0 surrogate positive control |
| `src/yggdrasil_v2/reasoning_medium/a1_5_train.py` | P0 overfit/full training、逐步 state CE、checkpoint、弱基线和 no-op/delete/shuffle/truncate/T0/T1/disable-delta 干预。 | A1.5 P0 probe；composition 因果未通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_data.py` | A1.6 relation-state-machine 数据、字符 span、causal_core 必要性和 audit Gate。 | A1.6 数据合同；audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_core.py` | 独立 RelationAddressedCore/Transition；三路 role pointer、共享 operator 和 shared state head。 | A1.6 C0 实现；formal relation Gate 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_train.py` | C0 固定 loss、checkpoint、评测和 prefix/counterfactual 代码。 | 已实现；formal relation Gate 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_qwen.py` | 冻结 Qwen hidden span cache 与复用 C0 core 的边界 adapter。 | 已实现；C1 未启动 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_qwen_train.py` | C1 boundary/joint 训练入口和缓存 batch 处理。 | 已实现；C1 未启动 |
| `src/yggdrasil_v2/reasoning_medium/a1_6_closure_diagnostic.py` | A1.6 只读 checkpoint 的 free/oracle-reset/hard-reembed/first-error 诊断。 | A1.6 closure 失败归因证据；hard re-embed 仅诊断 |
| `experiments/v2_a1_6_core.py` | A1.6 prepare/audit、C0 train/evaluate/intervene、C1 cache/train/evaluate CLI。 | A1.6 历史入口；停止在 formal relation Gate |
| `src/yggdrasil_v2/reasoning_medium/a1_7_data.py` | A1.7 12-combination universe、唯一 relation holdout、分层随机 sequence 与 causal necessary audit。 | A1.7 当前数据合同；audit Gate 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_core.py` | 默认 content-only slots、address-only register keys、唯一 shared transition、last-active shared state readout，以及显式 address-mixed 消融开关。 | A1.7 目标 core 与受控结构消融 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_train.py` | 可配置 stop-gradient prototype closure、trajectory/final/query metrics、identity/non-collapse、训练与 intervention。 | A1.7 formal、causal 与 closure 真消融实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_stress.py` | 与正式数据零 fingerprint 重叠的 8/12/16 步 supported/relation 压力生成、审计与逐长度 Gate。 | A1.7 长程漂移评测实现；目标 0/3 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_7_assessment.py` | 多 seed、2×2 消融和压力结果的机器可读聚合与判断。 | A1.7 核心证据汇总实现 |
| `experiments/v2_a1_7_core.py` | A1.6 diagnostic、A1.7 prepare/audit/train/evaluate/intervene/stress/assessment 统一 CLI。 | 当前 A1.7 运行入口；不含 C1 |
| `src/yggdrasil_v2/reasoning_medium/a1_8_data.py` | T1–16 training、short、T8/12/16、T20/24/32、relation、causal 数据生成，外部 fingerprint 排除和十二项 audit。 | A1.8 独立数据合同；三 run audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_8_train.py` | batch 内长度均衡训练、最差长度 checkpoint scoring、formal Gate、逐步稳定性、因果与 CUDA 成本 benchmark。 | A1.8 正式训练/评测实现；不更改 A1.7 core |
| `src/yggdrasil_v2/reasoning_medium/a1_8_assessment.py` | 三 run、cross-run overlap、A1.7 对照、成本和总 Gate 的机器可读聚合。 | A1.8 总判定实现 |
| `experiments/v2_a1_8_core.py` | A1.8 prepare/audit/train/evaluate/intervene/benchmark/summarize CLI。 | 当前结构化 core 最新入口；不含 Qwen boundary |
| `src/yggdrasil_v2/reasoning_medium/a1_9_cache.py` | 固定 revision Qwen3.5-2B 的 oracle role span pooled hidden 分片 cache 与严格 cache audit。 | A1.9 真实 Qwen boundary 输入合同；三 run audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_model.py` | 冻结 A1.8 core 与 value/family/shared-register 三类窄 adapter；无 full-source、答案头或 query-state 写入。 | A1.9 adapter-only 结构合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_train.py` | T1–16 均衡 boundary 训练、最差长度 checkpoint、formal Gate、core hash 与 cached CUDA 成本。 | A1.9 三 run formal 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_interventions.py` | prefix、operation replacement/deletion/shuffle、query swap、same-answer、no-hidden 与 role shuffle。 | A1.9 hidden causal Gate 三 run 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_9_assessment.py` | cache/formal/causal/core hash/cross-run/cost 的机器可读总判定与证据边界。 | A1.9 总 Gate `3/3` passed |
| `experiments/v2_a1_9_boundary.py` | A1.9 cache/audit/train/evaluate/intervene/benchmark/assess 统一 CLI。 | 当前 frozen-Qwen boundary 入口；不含 learned full-text reader |
| `src/yggdrasil_v2/reasoning_medium/a1_10_cache.py` | 固定 revision Qwen3.5-2B 的完整 last-hidden + attention mask 分片 cache；不保存 spans、operation mask、role tensor 或 input IDs。 | A1.10 full-text 输入合同；三 run cache audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_10_model.py` | 匿名 `K=8` learned workspace、两层共享通用 recurrent Transformer、workspace-only state/answer heads。 | A1.10 联合架构实现；无显式寄存器 scaffold |
| `src/yggdrasil_v2/reasoning_medium/a1_10_train.py` | 全局 16-step T1–16 训练、formal Gate、训练子集诊断和 cached CUDA 成本；短程序 final state label 在剩余步骤重复。 | A1.10 三 run formal `0/3`；联合合同失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_10_interventions.py` | prefix、替换/删除/打乱、query swap、same-answer、no-source/source shuffle、disable recurrence 与 slot permutation。 | 已实现；因 ordinary formal 失败而未执行正式干预 |
| `src/yggdrasil_v2/reasoning_medium/a1_10_assessment.py` | overfit、cache/formal/干预先后、cross-run、成本和证据边界的机器可读总判定。 | A1.10 总 Gate `0/3` failed |
| `experiments/v2_a1_10_anonymous_workspace.py` | A1.10 full-token cache/audit/train/evaluate/intervene/benchmark/assess 统一 CLI。 | 当前 A1.10 历史入口；不得绕过 formal 停止规则 |
| `src/yggdrasil_v2/reasoning_medium/a1_11_models.py` | learned full-text typed reader → frozen core，以及 exact-symbolic typed roles → anonymous reasoner 两个正交模型。 | A1.11 模型合同；不保留旧 continuous-adapter Reasoner 入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_11_train.py` | 两臂 overfit/formal、strict Gate、hard re-embedding 只读诊断、checkpoint 与数据/核心完整性。 | Boundary strict overfit failed；Reasoner formal `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_11_interventions.py` | 两臂 ordinary-formal 后 counterfactual、disable recurrence 与 slot permutation。 | 已实现；本轮无 run 通过 ordinary formal，故未执行 |
| `src/yggdrasil_v2/reasoning_medium/a1_11_assessment.py` | A1.9/A1.10 参考格、两臂 overfit/formal、停止顺序与 2×2 一级归因。 | A1.11 机器总判定；纯组合解释 rejected |
| `experiments/v2_a1_11_localization.py` | Boundary/Reasoner train/evaluate、Gate 后 intervention 和 assess 统一 CLI。 | 当前 A1.11 历史入口；下一阶段应新建 A1.12 合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p1.py` | Qwen3.5-2B FP16 hidden cache、operation span token mask、P1 hidden-to-latent interface。 | A1.5 P1 mechanism/surrogate probe |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p1_train.py` | P1 hidden cache training、start/query/operation warm-up、state/final CE、best reload 和多 split 诊断。 | A1.5 P1 surrogate formal；ordinary validation 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p2.py` | K=8 learned multi-slot workspace、共享 transition、无答案旁路和 permutation probe。 | A1.5 P2 formal core；composition 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_p2_train.py` | P2 frozen-hidden training、state/final loss、same-answer shuffle、operation counterfactual 和轨迹干预。 | A1.5 P2 formal training/evidence |
| `src/yggdrasil_v2/reasoning_medium/baseline.py` | 旧 V2-A0 文本基线与 Qwen3.5 文本塔加载。 | A0 smoke/probe 实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_5_text_baseline.py` | A1.5 matched visible text-CoT baseline、逐步 state parser、Qwen3.5-0.8B/2B 选择、无人工总输出 token cap 的语义终止生成。 | A1.5 smoke/probe；需与 latent 使用同样本数比较 |
| `tests/test_v2_a_data.py` | split 可复现、heldout/length、解析器和泄漏断言。 | 单元验证 |
| `tests/test_v2_a_model.py` | latent shape、trajectory、干预和 no-bypass 断言。 | 单元验证 |
| `tests/test_v2_a_train.py` | step-level verifier policy-gradient 的 mask、reward 统计和反传断言。 | 单元验证 |
| `tests/test_v2_a1_5.py` | A1.5 split/necessity/span contract、P0 shared transition、variable T 和 no-gate 断言。 | A1.5 单元验证 |
| `tests/test_v2_a1_5_p1_p2.py` | P1 span/warm-up shape、P2 learned-slot/permutation smoke 断言。 | A1.5 单元验证 |
| `tests/test_v2_a1_6_data.py` | A1.6 split、span、causal necessary 和 audit contract。 | A1.6 单元验证 |
| `tests/test_v2_a1_6_core.py` | operation-free initialization、prefix boundary、query isolation、shared transition/head 和 pointer contract。 | A1.6 单元验证 |
| `tests/test_v2_a1_6_qwen.py` | C1 core reuse、无 full-source path 和空 span 失败检查。 | A1.6 单元验证 |
| `tests/test_v2_a1_7_data.py` | 受控 11+1 relation support、位置分层、overlap 和 causal necessary contract。 | A1.7 数据完整性测试 |
| `tests/test_v2_a1_7_core.py` | 默认地址/内容分离、显式 address-mixed 消融、operation-free initial state、shared transition、T=0、query isolation、last-active identity、closure stop-gradient 与零权重真消融。 | A1.7 core 完整性测试 |
| `tests/test_v2_a1_7_stress.py` | 长度/位置平衡、唯一 holdout、base fingerprint 零重叠和 stress audit Gate。 | A1.7 长程压力合同测试 |
| `tests/test_v2_a1_8_data.py` | T1–32 split、relation holdout、外部数据排除和跨数据 seed fingerprint 零重叠。 | A1.8 数据合同测试 |
| `tests/test_v2_a1_8_train.py` | 16 长度 batch 均衡采样和 T32 连续扰动稳定性诊断。 | A1.8 训练/诊断合同测试 |
| `tests/test_v2_a1_9_boundary.py` | oracle role cache 审计、共享 register adapter、冻结 core、无 full-source path 和 mapping/trajectory 反传合同。 | A1.9 边界完整性单元测试 |
| `tests/test_v2_a1_10_anonymous_workspace.py` | full-token cache 禁止项、匿名 slot 对称性、通用共享 recurrence、无 scaffold/bypass、统一 recurrent budget 和干预合同。 | A1.10 架构与实验合同单元测试 |
| `tests/test_v2_a1_11_localization.py` | Boundary 无 span/frozen core、Reasoner exact-symbolic/no-Qwen/no-core、固定 recurrent budget 与 2×2 分类。 | A1.11 正交隔离合同单元测试 |
| `tests/conftest.py` | 为当前 CPU torchvision wheel 预声明缺失的 NMS operator，保证 Transformers 测试收集可重复；不改变模型运行语义。 | 测试环境隔离 |

本地 `artifacts/v2-a/` 被 `.gitignore` 忽略；阶段结果路径、配置和证据等级必须以对应阶段文档与本目录索引为准，不把未索引的本地文件当作 repo truth。

## 公开路线对照入口

| 主题 | 路径 | 用途 |
| --- | --- | --- |
| MoE/模型组装对照 | `docs/moe-model-assembly-comparative-review-2026-07-14.md` | 对照 VLMo、Uni-MoE、MoME、Uni-Med、DeepSeek-VL2、MoE-LLaVA、DeepSeekMoE/DeepSeek-V3、Qwen3、SMoES；记录成果、差异和 V2-B 未完成项。 |

## A1.6 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_6/data/` | seed `20260715` 的六 split relation-state-machine 数据。 | 数据合同证据 |
| `artifacts/v2-a/a1_6/data-audit.json` | 长度/family/bigram/COPY 位置、relation holdout、causal necessary 和 fingerprint overlap 审计。 | audit Gate 通过 |
| `artifacts/v2-a/a1_6/c0-overfit32/` | C0 5000-step overfit checkpoint、history、progress 和 results；fit Gate 通过，validation 仅作 few-shot diagnostic。 | C0 overfit sanity evidence |
| `artifacts/v2-a/a1_6/c0-formal/` | fresh formal C0 checkpoint、history、progress、results 和六 split `eval-all.json`；relation-heldout Gate 失败。 | C0 formal evidence；停止在正式 Gate |
| `artifacts/v2-a/a1_6/diagnostics/closure-diagnostic.json` | free recurrence、oracle-reset one-step、predicted hard re-embed 与首次错误 step。 | A1.6 只读失败归因；不改变旧 checkpoint |

## A1.7 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_7/data/` | seed `20260715`、11 个训练组合 + 唯一 `COPY amber→jade` holdout 的六 split 数据。 | A1.7 数据合同证据 |
| `artifacts/v2-a/a1_7/data-audit.json` | joint/position/sequence/marginal/overlap/causal necessary 审计。 | 十项 data Gate 全部通过 |
| `artifacts/v2-a/a1_7/c0-overfit32/` | 32-example fit-only checkpoint/results；step 200 五项指标全 `1.0`。 | A1.7 overfit sanity evidence |
| `artifacts/v2-a/a1_7/c0-formal/` | seed `20260715` 的 fresh formal checkpoint、history、results、`eval-all.json`、`interventions.json`。 | A1.7 目标 seed formal + causal Gate 通过 |
| `artifacts/v2-a/a1_7/multiseed/seed-{20260716,20260717}/` | 两个附加初始化 seed 的独立 checkpoint、formal 评测与因果干预。 | 与目标 seed 合计 3/3 短程 formal + causal 通过 |
| `artifacts/v2-a/a1_7/ablations/` | content-only/address-mixed × closure on/off 的另外三种受控训练、formal 和 intervention 结果。 | 四组合均过短程 Gate；短程必要性归因不成立 |
| `artifacts/v2-a/a1_7/stress-data/` | seed `20260718`、supported/relation 各 768 条、长度 8/12/16 的零重叠压力数据。 | A1.7 长程评测合同 |
| `artifacts/v2-a/a1_7/stress-data-audit.json` | 长度/位置/holdout/support/fingerprint 审计。 | stress data Gate 全部通过 |
| `artifacts/v2-a/a1_7/stress/` | 三个目标 seed 与三种消融 checkpoint 的逐长度压力结果。 | 目标严格 Gate 0/3；closure 显著缓解 T16 漂移 |
| `artifacts/v2-a/a1_7/core-assessment-summary.json` | formal、causal、2×2 消融、stress 的机器可读聚合和边界判断。 | A1.7 当前核心评估总表 |

## A1.8 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_8/overfit32/` | 每个 T1–16 两条记录的 length-balanced overfit checkpoint/history/results。 | step 500 停止，16 长度 trajectory/pointers 全 `1.0` |
| `artifacts/v2-a/a1_8/runs/run-{1,2,3}/data/` | 三个独立 data seed，各 22,784 条 T1–32 数据与 manifest。 | A1.8 formal 数据；彼此及 A1.7 overlap `0` |
| `artifacts/v2-a/a1_8/runs/run-{1,2,3}/data-audit.json` | 长度、support、relation 位置、diversity、causal necessity、内部/外部 overlap audit。 | 三组全部通过 |
| `artifacts/v2-a/a1_8/runs/run-{1,2,3}/formal/` | fresh random-depth checkpoints、history、results、formal eval/stability 和 interventions。 | 三组 formal + causal 全部通过 |
| `artifacts/v2-a/a1_8/cost-benchmark.json` | RTX 4070 Laptop GPU 上的训练/推理延迟、peak memory、参数、KV 与 FLOPs proxy。 | A1.8 structured core 成本证据；非 text-CoT Pareto |
| `artifacts/v2-a/a1_8/assessment-summary.json` | 三 run metrics、cross-run overlap、A1.7 delta、成本与总 Gate。 | A1.8 机器可读最终真源；passed |

## A1.9 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_9/overfit32/` | 真实 Qwen oracle-role cache/audit 与 T1–16 length-balanced fit-only checkpoint/results。 | mapping、pointer、trajectory、answer 全 `1.0`；overfit Gate 通过 |
| `artifacts/v2-a/a1_9/runs/run-{1,2,3}/cache/` | 三组独立 data seed，各 `22,784` 条 role-pooled Qwen hidden 分片与 manifest。 | A1.9 formal 输入；三组 fingerprint overlap `0` |
| `artifacts/v2-a/a1_9/runs/run-{1,2,3}/cache-audit.json` | fingerprint、shape、finite、operation count、split overlap、full-source 禁止项审计。 | 三组 cache audit 全部通过 |
| `artifacts/v2-a/a1_9/runs/run-{1,2,3}/formal/` | adapter-only checkpoints、history、results、formal eval 与 formal 后 hidden interventions。 | 三组 formal + hidden causal 全部通过 |
| `artifacts/v2-a/a1_9/cost-benchmark.json` | RTX 4070 Laptop GPU 上的 cache manifest 编码成本、cached train/inference latency、memory、参数和 KV 口径。 | A1.9 成本证据；非在线端到端、非 text-CoT Pareto |
| `artifacts/v2-a/a1_9/assessment-summary.json` | overfit、三 run、cross-run overlap、成本、证据边界和总 Gate。 | A1.9 机器可读最终真源；passed |
| `tmp/V2-A1.9 result.md` | A1.9 执行结果、异常、成本、完成/未完成项和下一阶段建议。 | 人类可读结果交接 |

## A1.10 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_10/overfit32/` | 64 条 T1–16 full-token cache、audit、fresh checkpoint、history 和逐长度评测。 | 修正 recurrent budget 后 trajectory/final/answer 全 `1.0`；fit-only sanity 通过 |
| `artifacts/v2-a/a1_10/runs/run-{1,2,3}/cache/` | 三组独立 data seed，各 `22,784` 条完整 Qwen last-hidden 分片与 attention mask。 | A1.10 formal 输入；合计 `9,046,290` source tokens、`38.36` GB，fingerprint overlap `0` |
| `artifacts/v2-a/a1_10/runs/run-{1,2,3}/cache-audit.json` | fingerprint、shape、finite、attention mask、split overlap 和 oracle/scaffold 禁止项审计。 | 三组 cache audit 全部通过 |
| `artifacts/v2-a/a1_10/runs/run-{1,2,3}/formal/` | fresh anonymous-reasoner checkpoints、history、formal eval 与训练子集 fit 诊断。 | 三组 formal `0/3`；trajectory 近零，未进入正式干预 |
| `artifacts/v2-a/a1_10/cost-benchmark.json` | RTX 4070 Laptop GPU 上的 cache manifest 编码成本、cached reasoner train/inference latency、memory 和参数口径。 | A1.10 成本证据；排除在线 Qwen 编码，非 text-CoT Pareto |
| `artifacts/v2-a/a1_10/assessment-summary.json` | overfit、三 run、formal-before-intervention、cross-run、成本、证据边界和总 Gate。 | A1.10 机器可读最终真源；failed，hidden interventions 未执行 |
| `tmp/V2-A1.10 result.md` | A1.10 方法修正、正式结果、成本、完成/未完成项、概率更新和 A1.11 建议。 | 人类可读结果交接 |

## A1.11 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_11/overfit32/boundary/` | learned full-text reader 的 fresh checkpoint/history/results，以及只读 hard-reembed diagnostic。 | strict pointer Gate failed；诊断全通过，formal stopped |
| `artifacts/v2-a/a1_11/overfit32/reasoner/` | exact-symbolic input、fresh anonymous reasoner checkpoint/history/results。 | T1–16 trajectory/final/answer 全 `1.0`；fit-only passed |
| `artifacts/v2-a/a1_11/runs/run-{1,2,3}/reasoner/formal/` | 三组独立 data/reasoner seed 的 checkpoint、history、训练子集诊断和六 split formal eval。 | formal `0/3`；ordinary 后干预未运行 |
| `artifacts/v2-a/a1_11/assessment-summary.json` | A1.9/A1.10 参考格、两臂 Gate、执行顺序、证据边界和一级归因。 | 当前 A1.11 机器真源；combination-only rejected，architecture not validated |
| `tmp/V2-A1.11 result.md` | 方法纠正、两臂结果、可能机制、完成/未完成项和 A1.12 建议。 | 人类可读结果交接 |

## A1.5 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_5/data-large/manifest.json` | A1.5 v1 4096/256 split、必要性与泄漏 manifest。 | 当前数据合同证据 |
| `artifacts/v2-a/a1_5/p0-overfit-final/` | 32-example P0 overfit、checkpoint、history 和 results。 | surrogate positive-control |
| `artifacts/v2-a/a1_5/p0-final/` | 4096-example P0 best checkpoint、best-eval、训练曲线和干预结果。 | surrogate probe；A1.5 未通过 |
| `artifacts/v2-a/a1_5/p1-hidden-cache-final/` | Qwen3.5-2B FP16 分片 hidden 与 operation span mask。 | mechanism smoke |
| `artifacts/v2-a/a1_5/p1-final-small/` | P1 small real-hidden underfit 训练与 warm-up/state 诊断。 | surrogate probe；训练预算不足 |
| `artifacts/v2-a/a1_5/p1-hidden-cache-large/` | Qwen3.5-2B FP16 4096/256 全 split hidden cache、显式 span mask 和版本 manifest。 | mechanism-smoke；无静默截断 |
| `artifacts/v2-a/a1_5/p1-final-large/` | P1 4096-cache warm-up/joint formal checkpoint、best reload 和 eval-all。 | surrogate-probe；ordinary validation 通过 |
| `artifacts/v2-a/a1_5/p2-final-large/` | P2 K=8 formal checkpoint、训练历史、结果和 test/composition/length 干预。 | surrogate-probe；composition 失败 |
| `artifacts/v2-a/a1_5/text-test-0-08b-one.json` | A1.5 schema 上 Qwen3.5-0.8B zero-shot、无人工总输出 cap 单样本 text-CoT smoke。 | smoke；final/state `0/0` |
| `artifacts/v2-a/a1_5/text-test-0-08b-formal.json` | Qwen3.5-0.8B matched test zero-shot 128 条；无 token cap，118 semantic terminal、10 safety timeout，parse/final/state `0.1797/0.0625/0.0234`。 | probe；需结合 termination_reason 解读 |
| `artifacts/v2-a/a1_5/text-composition-0-08b-formal.json` | Qwen3.5-0.8B matched composition-heldout zero-shot 128 条；无 token cap，parse/final/state `0.4063/0.3828/0.3750`。 | probe；final 低于 prefix-2 |
| `artifacts/v2-a/a1_5/text-length-0-08b-formal.json` | Qwen3.5-0.8B matched length-heldout zero-shot 128 条；无 token cap，parse/final/state `0.0625/0/0`。 | probe；长度外推失败 |
| `artifacts/v2-a/a1_5/text-test-2-08b-formal.json` | Qwen3.5-0.8B matched test 2-shot 128 条；无 token cap，parse/final/state `0.9766/0.1328/0.0156`，3 条 timeout。 | probe；格式改善但 state 失败 |
| `artifacts/v2-a/a1_5/text-composition-2-08b-formal.json` | Qwen3.5-0.8B matched composition 2-shot 128 条；无 token cap，parse/final/state `0.9219/0.1953/0`，7 条 timeout。 | probe；composition state 失败 |
| `artifacts/v2-a/a1_5/text-length-2-08b-formal.json` | Qwen3.5-0.8B matched length 2-shot 128 条；无 token cap，parse/final/state `0.7656/0.0938/0`，29 条 timeout。 | probe；长度 state 失败 |
| `artifacts/v2-a/a1_5/text-{composition,length}-0-08b-formal.json.partial.json` | no-cap 批次的逐步进度快照；最终 JSON 已完成时仅作终止/进度追溯，不作为质量结果。 | audit trace |
| `artifacts/v2-a/a1_5/p2/smoke.json` | K=8 learned-slot synthetic smoke。 | mechanism smoke |

## 保留的用户资料

| 路径 | 用途 |
| --- | --- |
| `AGENT.md` | 项目协作提示和实验习惯。 |
| `note.txt` | 用户笔记；按要求保留，不删除。 |
| `下一件事.txt` | 用户待办笔记；保留。 |
| `HF token.txt` | 用户本地文件；保留，不在文档中展开其内容。 |
| `.gitignore` | 本地环境、缓存和未来实验产物的忽略规则。 |

## 旧路线归档

`archive/legacy-proxy-route-2026-07-11/` 是一次完整的非删除归档，保留切换前的原始相对目录：

| 子目录或文件 | 内容 | 当前地位 |
| --- | --- | --- |
| `docs/` | 旧阶段报告、V1 白皮书、旧路线和旧命令清单 | 历史思想与代理证据 |
| `experiments/` | Stage A—AV-J-C 及相关代理脚本 | 不再运行 |
| `src/` | 旧代理公共实现 | 不再作为 V2 基座 |
| `tests/` | 绑定旧契约的测试 | 不再作为当前验收 |
| `artifacts/` | 旧 smoke/probe/formal 输出 | 只用于追溯和失败分析 |
| `thinking/` | 旧路线思考稿 | 历史材料 |
| `pyproject.toml` | 旧实验依赖和 pytest 配置 | 随旧代码归档 |
| `README-legacy.md` | 归档前 README | 历史入口 |

归档中的报告可能仍引用原始根目录路径；这些引用用于还原当时的实验语境，不构成当前可运行入口。后续 V2 实现必须新建代码和测试，不在旧代理代码上增加兼容 wrapper。

## 维护规则

新增 V2 文件应先归入明确的 V2-A/V2-B 语义包，再更新本索引。实验结果只能在 evidence level、配置、seed、heldout、消融、成本和失败边界齐全后进入当前证据；旧 Stage 结果不得静默升级为 V2 架构证据。任何再次废弃的当前文件都移动到带日期的 `archive/`，不保留并列旧入口。
