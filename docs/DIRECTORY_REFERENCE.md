# Directory Reference

本仓库是 Project-Yggdrasil V2 的独立研究工作区。当前主线已经从旧 Stage A—AV-J-C 直接切换为 V2-A 推理介质、V2-B 静态多模态模型核、V2-C 多层多线程树图全双工智能体；严格按 A→B→C Gate 推进。关联项目 `C:\skzy\QuickFileTransport\世界树计划` 的设计哲学是上位概念真源，本仓库不修改其源码。

## 当前真源

| 路径 | 地位与用途 |
| --- | --- |
| `docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md` | 当前唯一目标架构规范；定义连续语义 recurrence、离散地址/控制 + 连续 payload 的合规混合 core、静态 Boundary-MoE/FFN-MoE，以及 V2-C 系统级全双工 Boundary-MoE 与人类目标能力边界。 |
| `docs/Project-Yggdrasil V2 从架构验证到商用路线图.md` | 当前唯一高层路线；按 Gate 推进 R0/R1、R2/R3 静态模型核、V2-C 架构完整版和商用路线；旧 A1-A4 后续路线已被直接替换。 |
| `docs/next-stage-test-plan.md` | 当前最近执行真源；记录 V2-A/A1.5–A1.21P 的故障定位、V2-R1R 跨任务重验证、V2-B 静态模型核边界、V2-C 停止边界、证据口径、Gate 和成本。 |
| `docs/v2-c-hierarchical-full-duplex-agent-experiment.md` | V2-C 统一系统实验合同；定义多层、多线程、责任树 + 受控图边、全双工消息、主动 Boundary、必要工具、人类目标连续性、Flat/同步基线、C0-C5 Gate 和停止规则；当前设计完成但未实施。 |
| `docs/v2-a-route-reassessment-2026-07-17.md` | A1.18B 后路线重审；定义合规混合 core，以及 A1.19H generalized hybrid core、A1.20B/A1.20C full-text boundary、A1.21P Pareto、A1.22A audit 的顺序与停机门；2026-08-01 已同步 B→C 高层路线覆盖决定。 |
| `docs/v2-a1.20b-full-text-boundary.md` | A1.20B 无 oracle full-text Boundary 合同、启动前吞吐优化、run-1 eligibility 失败、联合 mapping、oracle 替换、梯度信用审计与停线结论。 |
| `docs/v2-a1.20c-boundary-repair.md` | A1.20C 分层 compiler × execution-credit 合同、token anchor、ST hard-forward、overfit32 失败、真实梯度冲突与停线结论。 |
| `docs/v2-a1.20d-full-text-mechanism-repair.md` | A1.20C 后的 post-stop 双向 section 推断修复；记录 N5 截断根因、三遍 decode、完整 split/hidden causal/routing schedule 诊断与非 formal 边界。 |
| `docs/v2-a1.21p-pareto.md` | K=1 容量负基线、在线 canonical/routing Pareto smoke、路径稳定性、六项正式缺口和 A1.22A 停机判定。 |
| `docs/v2-r1-revalidation-task-design.md` | A1.21P 后的 R1 唯一实现级冻结设计；定义 ERE/CPS、统一 view、R1R-Latent v1、公平路径与 Gate；第 16 节覆盖 generator v1，冻结 P0-D v2 的去捷径、split、claim、tokenizer 与审计合同。 |
| `docs/v2-r1r-p0-result.md` | V2-R1R generator v1 的 smoke/formal 机器结果、数据规模与失败重试证据；旧 10/10 已被主设计层否决，只是 rejected diagnostic。 |
| `docs/v2-r1r-p0-main-review.md` | P0-D v1 主设计层独立复核；记录 ERE/CPS surface 满分捷径、claim/split/token/certificate 缺口及 v2 修订理由；当前 P0-D 路线判定真源之一，P0-M 禁止。 |
| `docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md` | V2 决策形成记录；只保存推导和审阅依据，不与白皮书并行定义规范。 |
| `docs/moe-model-assembly-comparative-review-2026-07-14.md` | V2 的 Boundary-MoE/FFN-MoE 与公开模型路线的中文对照；区分已被其他模型验证的局部思想、完整架构未验证边界和 V2-B 未启动状态。 |
| `docs/DIRECTORY_REFERENCE.md` | 本索引；新代码、测试、文档和归档必须同步这里。 |
| `README.md` | 面向仓库使用者的当前状态说明；展示 A1.19H/A1.20D 正证据、A1.21P 停止结论、V2-R1R P0-D v1 否决与 v2 重做边界、V2-C 仅设计状态。 |
| `docs/v2-a-reasoning-medium-experiment.md` | V2-A 当前实现合同、Qwen3.5 基座、数据 schema、A0/A1/A2 结果、Gate 判定和失败边界。 |
| `docs/v2-a1.5-latent-foundation.md` | A1.5 独立 schema、P0 结构化正控制、P1 Qwen hidden 接口、P2 learned-slot formal、因果干预和停止门禁。 |
| `docs/v2-a1.9-qwen-boundary.md` | A1.9 冻结 Qwen hidden → 冻结 A1.8 structured core 的预注册合同、三 run formal/causal 结果、成本、捷径干预和证据边界。 |
| `docs/v2-a1.10-anonymous-workspace.md` | A1.10 full-token frozen-Qwen cache、匿名 K-slot workspace、通用 recurrent reasoner、方法修正、三 run formal 失败、成本和证据边界。 |
| `docs/v2-a1.11-fault-localization.md` | A1.11 Boundary × Reasoner 2×2 合同、方法纠正、严格 overfit/formal 结果、一级故障归因和 A1.12 边界。 |
| `docs/v2-a1.12-reasoner-root-cause.md` | A1.12 binding × cursor 正交拆分、三臂 formal 失败和排除结论。 |
| `docs/v2-a1.13-transition-closure-root-cause.md` | A1.13 transition identity × latent closure 正交矩阵、三 seed formal/causal 结果和 state/answer 分离判定。 |
| `docs/v2-a1.13f-fixed-budget-method-audit.md` | A1.13F 固定 4000-step 审计；排除 early-stop 作为单因素臂不稳定的主要解释。 |
| `docs/v2-a1.15-closed-coupled-core.md` | A1.15 query-coupled readout 合同、overfit 正控制和 fresh-seed formal `0/3` 结果。 |
| `docs/v2-a1.16-redundant-answer-loss.md` | A1.16 删除重复 answer CE 的唯一变量实验；fresh-seed formal `0/3`。 |
| `docs/v2-a1.17-paired-objective-initialization-audit.md` | A1.17 同 seed/同共享初始化的 objective × initialization 配对审计合同与最终归因。 |
| `docs/v2-a1.18-training-scaffold.md` | A1.18 FINAL-SAUX `2/3`、A1.18B TSAUX paired/fresh 六 seed formal/causal、部署剥离、机制结论和 state-target 来源边界。 |
| `docs/v2-a1.6-core.md` | A1.6 relation-addressed continuous state core 的数据合同、结构完整性、C0 停止点和证据边界。 |
| `docs/v2-a1.7-core.md` | A1.6 closure 归因、A1.7 受控 relation 数据、三 seed formal/causal、2×2 消融、8/12/16 步压力与成功概率真源。 |
| `docs/v2-a1.8-long-horizon.md` | A1.8 T1–16 均衡随机深度合同、三组独立 data/model seed、T20/T24 Gate、T32 诊断、稳定性、成本和归因真源。 |

## 研究辅助入口

| 路径 | 用途与边界 |
| --- | --- |
| `docs/research-agent-literature-commands.md` | 面向 Google Gemini Deep Research 的历史研究冲刺命令；其中旧 A1-A4 仅保留为专题分组，当前正式路线已将 I/O、工作树/记忆、专家晋升和整机评测聚合为 V2-C。研究报告只作为决策输入，不自动修改架构真源或 Gate。 |
| `docs/thinking-draft-synthesis-2026-08-01.md` | 对 `thinking/` 五篇近期草稿的主题重建、逐命题判断和 V2-C 聚合映射，并把项目初期 V1 白皮书单列为历史参照；明确排除 `thinking/research/`，不修改原稿，也不自动升级为实验通过证据。 |

## 当前实现状态

V2-A0 数据/基座链路、Qwen3.5-2B/0.8B text-CoT probe 和当前结构的 V2-A1 mechanism smoke 已实现；旧 A2 的 K/T、prompt、transition、读出、full-data、teacher/masked state supervision、token-wise source adapter、step-level verifier RL、latent-attention probe 与 0.8B/2B 对照仍未达到强 text-CoT 基线。source-layer bank、query init、token mixer、latent-attention 等失败入口已从当前代码删除。新增的 A1.5 独立 schema 与 P0/P1/P2 代码已完成：P0 32-example overfit final/state full exact `1.0/1.0`，4096-example best ordinary test `1.0/1.0`，composition-heldout `0.2734/0`，length-heldout final/state full exact `1.0/0.2266`；真实 Qwen3.5-2B FP16 hidden cache 已分片生成，P1 4096-cache formal ordinary validation final/state `1.0/1.0`，小 probe 仅作为 underfit 诊断；P2 K=8 formal ordinary test `1.0/1.0`、composition `0.2773/0`、length `1.0/0.6484`，same-answer composition shuffle 失败；A1.5 0.8B no-cap zero-shot 128 条 test/composition/length formal 分别为 `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`，2-shot test/composition/length final/state 也分别为 `0.1328/0.0156`、`0.1953/0`、`0.0938/0`（格式解析率高但 state 失败）。A1.5 未通过，V2-A3/V2-A4/V2-B 均未启动。A1.5 真源见 `docs/v2-a1.5-latent-foundation.md` 与 `tmp/V2-A1.5 result.md`；旧 A2 结果仍只作历史 probe，不得与 A1.5 混写。

A1.6 已作为失败证据保留：data audit 与 overfit32 通过，正式 C0 ordinary/length trajectory full 为 `1.0/1.0`，relation-heldout 为 `0.765625`，按 Gate 停止。只读 checkpoint 诊断显示 oracle-reset one-step 与 predicted hard re-embed diagnostic 在 test/length/relation/causal 均为 `1.0`，失败定位为 continuous latent closure；旧 final-vs-trajectory Gate 无效，旧 relation split 也混入多变量。因果干预与 C1 未启动。

A1.7 是当前结构化 core 证据：独立受控数据合同把唯一 holdout 冻结为 `COPY amber→jade`。三个初始化 seed 的 5–6 步 formal C0 与 causal intervention 均通过；目标 seed 的 test/length/relation trajectory full 为 `1.0/0.996094/0.998047`。但 2×2 消融表明短程通过不能只归因于地址/内容分离或 closure；closure 的明确作用是减缓长程漂移。三个目标 checkpoint 的 8/12/16 步严格压力 Gate 为 `0/3`，T16 supported 为 `0.894531/0.941406/0.894531`。当前对 V2-A 形成 matched text-CoT Pareto 的工程判断为 `40%–55%`、中心约 `48%`。该结论只覆盖三寄存器结构化合成 core；Qwen/C1、匿名 workspace、完整 V2-A、V2-A3/V2-A4/V2-B 均未启动。真源见 `docs/v2-a1.7-core.md`、`tmp/V2-A1.7 result.md` 与 `artifacts/v2-a/a1_7/core-assessment-summary.json`。

A1.8 是当前最新 core 证据：保持 A1.7 架构和 closure 不变，把训练切换为 T1–16 batch 内均衡随机深度。三组独立 data/model seed 的 short、T8/12/16、OOD T20/T24、relation 和 causal Gate 全部通过；T16 supported/relation 最低为 `0.996094/1.0`，T24 为 `1.0/0.996094`，诊断性 T32 为 `0.988281/1.0`。跨 run 及相对 A1.7 fingerprints overlap 为 `0`。证据否定 T16 必然内在发散并支持 horizon mismatch，但没有 compute-matched 地拆分长度覆盖和 transition exposure；仍不覆盖 Qwen hidden、匿名 workspace 或 matched text-CoT Pareto。A1.8 完成时的工程概率为 V2-A Pareto `45%–60%`、中心约 `53%`。真源见 `docs/v2-a1.8-long-horizon.md`、`tmp/V2-A1.8 result.md` 与 `artifacts/v2-a/a1_8/assessment-summary.json`。

A1.9 是当前最新 boundary 证据：冻结 Qwen3.5-2B 与三组已经通过的 A1.8 core，只训练 value、family 和共享 register adapter。三个独立 data/core/adapter seed 的 cache audit、formal 与 hidden causal Gate 全部通过；T16 supported/relation 最低 `1.0/0.996094`，T24 为 `1.0/1.0`，诊断性 T32 为 `0.996094/1.0`，mapping 最低 `1.0`。query swap 和 same-answer/different-trajectory 跟随新 oracle 为 `1.0`，no-hidden 与 independent role shuffle trajectory 为 `0`；core hash 未改变、跨 run fingerprint overlap 为 `0`。证据只覆盖 oracle-role-segmented Qwen hidden，不覆盖 learned full-text role extraction、匿名 workspace 或 matched text-CoT Pareto。当前 V2-A Pareto 工程判断为 `48%–63%`、中心约 `55%`。真源见 `docs/v2-a1.9-qwen-boundary.md`、`tmp/V2-A1.9 result.md` 与 `artifacts/v2-a/a1_9/assessment-summary.json`。

A1.10 是联合目标配置的失败证据：在同一轮删除 oracle span mask 和显式三寄存器 scaffold，使用 frozen Qwen3.5-2B 的完整 last-hidden + attention mask、匿名 `K=8` learned slots，以及两层参数共享的通用 recurrent Transformer。修正版 overfit32 通过，但三个独立 data/model seed 正式 Gate 为 `0/3`，hidden interventions 按停止规则未运行。该阶段自身不能单独归因，后续 A1.11 已补齐正交诊断。真源见 `docs/v2-a1.10-anonymous-workspace.md`、`tmp/V2-A1.10 result.md` 与 `artifacts/v2-a/a1_10/assessment-summary.json`。

A1.11 是一级故障定位证据。Boundary 臂在无 oracle span、保留 frozen A1.8 core 的修正合同下，overfit32 trajectory/answer/mapping 全为 `1.0`，但 source/target pointer 最低均为 `0.9642857143`，严格 Gate failed；只读 hard re-embedding 全部恢复 `1.0`，所以存在连续 latent → frozen address geometry 缺陷，但 task-level formal 未运行。Reasoner 臂直接使用 exact symbolic typed roles，不加载 Qwen/cache/adapter/core；overfit32 全通过，三组 formal 稳定 `0/3`，全部主要 trajectory 为 `0–0.003906`，训练集均衡诊断 trajectory 也只有 `0–0.003906`。这足以否定纯组合故障并把主要独立失败源定位到 anonymous binding／generic transition／当前 readout objective 这一整臂，但尚未拆开三者。当前 V2-A matched Pareto 工程判断为 `22%–35%`、中心约 `28%`。真源见 `docs/v2-a1.11-fault-localization.md`、`tmp/V2-A1.11 result.md` 与 `artifacts/v2-a/a1_11/assessment-summary.json`。

A1.12–A1.17 是当前最新二级根因证据。A1.12 BIND/CURSOR/BOTH 均为 formal `0/3`；A1.13 GENERIC-CLOSURE state `0/3`、STRUCTURED-CE `1/3`、STRUCTURED-CLOSURE state `3/3`/full `1/3`；A1.13F fixed 4000-step 排除 early-stop 主因。A1.15/A1.16 的 query-coupled + CE/noCE fresh seed 均 state/full `0/3`。A1.17 在 A1.13 三个成功 model/data seed 上验证全部共享初始 tensor 位相等后重跑，两条 coupled 臂仍均 state/full `0/3`。机器分类 `independent_answer_auxiliary_gradient_required`：当前 state 学习依赖独立 pooled-answer objective 的全局辅助梯度，但该旁路不能形成可靠因果答案。根因在 exact-symbolic 三寄存器合同内已定位，架构未通过；下一阶段只允许 training-only QAUX/SAUX 目标重设。真源见 `docs/v2-a1.12-reasoner-root-cause.md` 至 `docs/v2-a1.17-paired-objective-initialization-audit.md`、`tmp/V2-A1.12-A1.17 root-cause result.md` 与各阶段 assessment。

A1.18/A1.18B 是当前最新训练机制证据。QAUX/FINAL-SAUX overfit32 通过；FINAL-SAUX paired formal/causal 为 `2/3`，证明 final-only 全局完整状态梯度方向正确但 seed 不稳定。TSAUX 用一组跨步共享训练头在每个递归步从 global workspace mean 预测完整 state；三个 paired seed 与三个 fresh model seed 的 formal/causal 均为 `3/3`。六个通过部署模型的 causal trajectory/answer 都为 `1.0`，全部反事实 Gate 通过，disable-recurrence 与 wrong-start trajectory 都为 `0`；formal 前辅助参数已物理删除。机器分类 `per_step_global_state_credit_assignment_confirmed`、`mechanism_solved=true`。结论只覆盖 exact-symbolic 三寄存器 core；逐步 oracle state target 的开放任务来源仍未解决。真源见 `docs/v2-a1.18-training-scaffold.md`、`tmp/V2-A1.18 result.md` 与 `artifacts/v2-a/a1_18b/assessment-summary.json`。

2026-07-17 白皮书与路线重审已接受 `S_t=(A_t,H_t)` 合规混合 core：离散/prototype-anchored sidecar 只承载身份、地址、类型和控制，连续 `H_t` 承载语义推理。A1.19H-H1/H2 已完成 formal/causal `3/3`；A1.20D 又以三遍双向 section decode 形成 full-text mechanism 强诊断。A1.21P 因任务同构、baseline 不公平和 formal 证据缺失而停止，`a121p_passed=false`、`a122a_authorized=false`。2026-08-01 已由主设计层冻结 V2-R1R：A1.20D 只作正控制，以 ERE/CPS 检验同一 token-wise Boundary 与 shared recurrent core；训练使用可剥离的通用 claim verifier，不再使用任务专用 compiler 或显式寄存器。P0-D v1 package、正式数据与旧机器审计已完成，但旧 10/10 因 ERE/CPS 满分捷径、错误 claim、split 污染和审计缺口被主设计层否决；第 16 节 v2 修订已冻结。尚无有效 P0-D、Qwen hidden cache、Boundary/core、P0-M 或训练 artifact。V2-C 统一系统实验合同已完成并收口旧 A1-A4 远期路线，但必须继续等待 V2-A/V2-B。

## V2-R1R P0-D 代码、测试与产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `src/yggdrasil_v2/r1_revalidation/` | 独立 schema/view、nonce symbols、typed simulator、ERE/CPS generator、renderer 和 audit；当前内容是待 v2 修订的 generator v1，不含 model/cache/train。 | rejected P0-D v1 实现，不能训练 |
| `experiments/v2_r1_revalidation.py` | v1 `generate-p0` / `audit-p0` CLI；正式输出默认拒绝覆盖。 | 待按第 16 节升级 v2 |
| `tests/test_v2_r1r_data.py` | v1 simulator、causal pair、fingerprint、模板隔离、model view 和 token limit 测试。 | 旧测试通过但覆盖不足 |
| `tests/test_v2_r1r_audit.py` | v1 overlap、unigram leakage 和 forbidden-field 审计测试。 | 旧测试通过但漏检关系型捷径 |
| `docs/v2-r1r-p0-result.md` | v1 smoke/formal 机器结果、Gate 与停止判定。 | rejected diagnostic |
| `docs/v2-r1r-p0-main-review.md` | v1 主设计层复核和 v2 修订依据。 | 当前复核结论 |
| `artifacts/v2-r1r/p0-v1/` | v1 formal ERE/CPS JSONL、manifest、audit、heuristics 和 assessment。 | 旧机器 10/10；主设计层 rejected |

## V2-R1R P0-D 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-r1r/p0-v1/data/ere/` | v1 ERE train、validation、OOD 与 causal pairs JSONL。 | rejected diagnostic；含 seed-literal 满分捷径 |
| `artifacts/v2-r1r/p0-v1/data/cps/` | v1 CPS train、validation、OOD 与 causal pairs JSONL。 | rejected diagnostic；含 candidate 位置满分捷径 |
| `artifacts/v2-r1r/p0-v1/manifest.json` | v1 generator/version/seed、旧设计 hash、环境、命令、时间和文件 SHA-256。 | v1 provenance 记录 |
| `artifacts/v2-r1r/p0-v1/audit.json` | 旧十项 P0-D 机器审计明细。 | 自判通过但规格覆盖不足 |
| `artifacts/v2-r1r/p0-v1/heuristics.json` | 旧 unigram、claim 数量和标签/选择平衡统计。 | 漏检结构化捷径与 claim 真值 |
| `artifacts/v2-r1r/p0-v1/p0-assessment.json` | 旧 `passed` conjunction。 | 文件值 `true`，路线判定 rejected |
| `artifacts/v2-r1r/p0-smoke-v1/` | 每族 64 train、32 validation/OOD 的 smoke 数据与审计。 | smoke，不是 formal |
| `artifacts/v2-r1r/p0-v1-failed-*/` | 三次被保留的 formal 失败重试及其修复前证据。 | 诊断归档，不是最终结果 |

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
| `src/yggdrasil_v2/reasoning_medium/a1_12_models.py` | exact-symbolic A1.11 Reasoner 的 entity binding 与 aligned cursor 两个正交开关。 | A1.12 诊断模型；三臂均不足 |
| `src/yggdrasil_v2/reasoning_medium/a1_12_train.py` | A1.12 overfit/formal、统一 Gate、checkpoint 与评估。 | A1.12 三臂 formal `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_12_interventions.py` | ordinary formal 后的结构反事实、recurrence 与 start-state 干预。 | 因 formal 全失败未执行 |
| `src/yggdrasil_v2/reasoning_medium/a1_12_assessment.py` | binding × cursor 三臂机器汇总与严格分类。 | `binding_and_cursor_insufficient` |
| `experiments/v2_a1_12_root_cause.py` | A1.12 train/evaluate/intervene/assess CLI。 | A1.12 可复现实验入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_13_models.py` | relation-addressed transition、soft prototype closure、query-coupled answer 与训练期 QAUX/SAUX/TSAUX 受控头。 | A1.13–A1.18B 诊断核心；不是完整 V2-A |
| `src/yggdrasil_v2/reasoning_medium/a1_13_train.py` | A1.13–A1.17 训练、formal Gate、可选固定预算与 answer-loss 唯一变量。 | transition/closure/objective 归因实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_13_interventions.py` | relation/OOD/causal ordinary pass 后的反事实与 recurrence 干预。 | A1.13 joint run-2 causal 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_13_assessment.py` | transition × closure 2×2 聚合、state/full 分离和自适应分类。 | A1.13 joint state `3/3`、full `1/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_13f_assessment.py` | 固定预算三臂审计与 early-stop 判定。 | early-stop primary=false |
| `src/yggdrasil_v2/reasoning_medium/a1_15_assessment.py` | query-coupled 三 seed formal/causal 聚合。 | A1.15 state/full `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_16_assessment.py` | no-answer-CE 三 seed formal/causal 与触发条件聚合。 | A1.16 state/full `0/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_17_assessment.py` | 配对 seed、共享初始化逐 tensor 等同性、两种 coupled objective 与 A1.13 reference 的机器归因。 | A1.17 最终根因判定实现 |
| `experiments/v2_a1_13_transition_closure.py` | A1.13 train/evaluate/intervene/assess 与 A1.13F audit CLI。 | transition × closure 主入口 |
| `experiments/v2_a1_15_closed_coupled_core.py` | A1.15 train/evaluate/intervene/assess CLI。 | query-coupled + answer CE 入口 |
| `experiments/v2_a1_16_redundant_answer_loss.py` | A1.16 train/evaluate/intervene/assess CLI。 | query-coupled + no answer CE 入口 |
| `experiments/v2_a1_17_paired_objective_initialization.py` | 聚合 A1.17 paired reference、coupled-CE 与 coupled-noCE。 | A1.17 assessment 入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_18_train.py` | QAUX/FINAL-SAUX/TSAUX 训练、state-only checkpoint、固定 formal budget、辅助剥离部署与 formal loader。 | A1.18/A1.18B 长期训练合同实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_18_interventions.py` | 只接受辅助剥离部署模型的 structural/query/same-answer/recurrence/start-state 因果干预。 | TSAUX paired/fresh causal 均 `3/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_18b_assessment.py` | FINAL-SAUX、TSAUX paired/fresh、seed/budget/target/deployment integrity 与机制分类。 | `per_step_global_state_credit_assignment_confirmed` |
| `experiments/v2_a1_18_training_scaffold.py` | QAUX/FINAL-SAUX/TSAUX train/evaluate/intervene CLI。 | A1.18/A1.18B 统一执行入口 |
| `experiments/v2_a1_18b_trajectory_state_scaffold.py` | A1.18B 过拟合、FINAL-SAUX、TSAUX paired/fresh 的机器总判定入口。 | 当前机制 assessment 入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_data.py` | opaque-handle 编码、随机语义到物理 slot 映射、批量索引/GPU 搬运与缓存期地址完整性校验。 | A1.19H 共享数据边界；热路径不再执行 CUDA 标量校验 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_model.py` | equality-only opaque address sidecar、exchangeable continuous entity payload、共享 transition/state head 与 query-coupled answer。 | A1.19H generalized hybrid core 实现 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_train.py` | H1 fixed-cardinality 训练、辅助剥离部署、formal 与状态评测。 | H1 formal/causal `3/3` |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_h2_data.py` | N2/N3/N4 训练、N5 heldout、relation heldout、独立数据 seed 与 overlap audit。 | H2 variable-cardinality 数据合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_h2_train.py` | H2 固定预算训练、一次性 CPU 编码/GPU tensor cache、预生成采样、无逐步 CUDA 同步的 loss、cached formal 与吞吐等价基准。 | run-3 fresh 重跑入口；60-step 参数级等价基准为 `10.74×` |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_{interventions,assessment}.py` | H1 structural/address/recurrence 干预与三 seed 汇总。 | H1 机器 Gate 已通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_19h_h2_{interventions,assessment}.py` | H2 N5/address/recurrence 干预与 H1+H2 总判定。 | H2 formal/causal `3/3`；A1.19H 已关闭通过 |
| `experiments/v2_a1_19h_hybrid_core.py` | H1/H2 prepare、audit、train、benchmark、evaluate、intervene、assess 统一 CLI。 | A1.19H 可复现历史入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_cache.py` | H2 数据上的完整 Qwen last-hidden 分片 cache、strict audit 与 mmap dataset；禁止保存 span/role/entity mask/operation mask/input IDs。 | A1.20B full-text 输入合同 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_model.py` | 共享 entity/operation typed queries，自主预测 entity/operation presence、value、family、source/target/query pointer，并把连续 value payload 送入冻结 A1.19H core。 | A1.20B Boundary 实现；无 oracle mask 输入 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_train.py` | Boundary factorized mapping/state loss、冻结 core hash、N/T 均衡采样、overfit/formal evaluator、同步 mmap training 与吞吐对照。 | run-1 fixed-5000 已完成；formal eligibility 失败 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_diagnostics.py` | 选择性 oracle 替换、逐样本联合 mapping exact 与 state CE→离散控制 logits 梯度信用审计。 | A1.20B 失败归因；oracle 只作诊断 |
| `src/yggdrasil_v2/reasoning_medium/a1_20b_assessment.py` | overfit/cache/training/oracle/gradient evidence 的机器总判定与停线列表。 | `full_text_entity_binding_and_program_extraction_failure` |
| `experiments/v2_a1_20b_full_text_boundary.py` | cache/audit/train/evaluate、diagnose、gradient-audit、assess-failure CLI。 | A1.20B 可复现失败入口；不得继续 run-2/run-3 |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_supervision.py` | 从 tokenizer offset 生成 entity name/value、operation family/source/target、query 的训练专用 token anchor target，并审计 target 不进入 forward。 | A1.20C compiler supervision；overfit/run-1 audit 通过 |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_model.py` | flat/hierarchical/factorized compiler、shared entity/operation Viterbi、三遍双向 section decode 与 frozen-core bridge。 | A1.20C 历史失败实现 + A1.20D post-stop 机制修复 |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_train.py` | anchor + mapping + state loss、分离 reader/schedule seed、优化 scope、formal eligibility、hard-forward equivalence、core hash 与 checkpoint。 | A1.20D 路径稳定性可复现训练基础；当前非 fresh formal |
| `src/yggdrasil_v2/reasoning_medium/a1_20c_diagnostics.py` | 对实际 checkpoint 计算 state-vs-local 梯度夹角、失败 cell、tail loss 与 section 诊断。 | A1.20C 失败归因和 A1.20D 根因搜索辅助 |
| `experiments/v2_a1_20c_boundary_repair.py` | prepare/audit、训练、schedule seed、优化 scope 与诊断 CLI。 | A1.20C/A1.20D 可复现入口 |
| `src/yggdrasil_v2/reasoning_medium/a1_21p_k1.py` | 单 slot anonymous recurrent reasoner 的训练、formal evaluator 与完整性报告。 | A1.21P 容量负基线；validation trajectory `0.106934` |
| `src/yggdrasil_v2/reasoning_medium/a1_21p_pareto.py` | routing surface data、section/capacity 诊断、跨域 probe、official-chat 在线 direct/text-CoT/hybrid preflight 与正式 assessment。 | A1.20D/A1.21P 机器评估入口；`a121p_passed=false` |
| `experiments/v2_a1_21p_pareto.py` | prepare/probe/K1/preflight/assess 统一 CLI。 | 当前 A1.21P 可复现入口；A1.22A 未授权 |
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
| `tests/test_v2_a1_12_root_cause.py` | binding/cursor 唯一变量、forward shape、训练合同和分类。 | A1.12 完整性单元测试 |
| `tests/test_v2_a1_13_transition_closure.py` | transition/closure/coupled/noCE 合同、loss 差分、配对初始化逐 tensor 等同和 A1.13F–A1.17 分类。 | A1.13–A1.17 完整性单元测试 |
| `tests/test_v2_a1_18_training_scaffold.py` | QAUX 初始化等同、SAUX/TSAUX shape/gradient、query-coupled answer、训练期限定、物理部署剥离和 A1.18B 分类。 | A1.18/A1.18B 完整性单元测试 |
| `tests/test_v2_a1_19h_hybrid_core.py` | opaque-handle/slot 等变、辅助剥离、H2 N5 合同、缓存编码等价、冻结 loss 与缓存期严格地址校验。 | A1.19H 完整性与训练优化回归测试 |
| `tests/test_v2_a1_20b_full_text_boundary.py` | continuous-payload core 等价、Boundary 无 oracle mask/role 输入、冻结 core 梯度隔离、N/T sampler、oracle 诊断选择与 hard-control state-credit 断裂。 | A1.20B 架构/训练/归因合同测试 |
| `tests/test_v2_a1_20c_boundary_repair.py` | token anchor、flat/hierarchical/factorized no-oracle forward、shared Viterbi、三遍 section decode、ST/state-credit 与训练 schedule 合同。 | A1.20C/A1.20D 回归测试通过 |
| `tests/test_v2_a1_21p_pareto.py` | trace parser、official prompt、routing 语义保持、Pareto dominance、K1 容量/梯度和 formal assessment 停机合同。 | A1.21P 8 项回归测试通过 |
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

## A1.12–A1.17 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_12/overfit32/{bind,cursor,both}/` | 三条唯一变量臂的 length-balanced fit-only checkpoint/results。 | 三臂 overfit 均通过 |
| `artifacts/v2-a/a1_12/runs/run-{1,2,3}/{bind,cursor,both}/` | 九个 fresh formal checkpoint、训练与六 split 评估。 | formal 全 `0/3`；无干预 |
| `artifacts/v2-a/a1_12/assessment-summary.json` | binding × cursor 聚合。 | `binding_and_cursor_insufficient` |
| `artifacts/v2-a/a1_13/overfit32/` | GENERIC-CLOSURE、STRUCTURED-CE、STRUCTURED-CLOSURE fit-only 正控制。 | 三臂均通过 |
| `artifacts/v2-a/a1_13/runs/run-{1,2,3}/` | transition × closure formal 与 Gate 后干预。 | 联合臂 state `3/3`、full `1/3`；run-2 causal 通过 |
| `artifacts/v2-a/a1_13/assessment-summary.json` | state/full 分离的三臂与 A1.12 reference 聚合。 | overall `seed_unstable_inconclusive`；联合 state 稳定 |
| `artifacts/v2-a/a1_13f/runs/run-{1,2,3}/` | GENERIC-CE、GENERIC-CLOSURE、STRUCTURED-CE 的固定 4000-step 审计；复用原本已满预算的配对 run。 | state `0/3`、`0/3`、`1/3` |
| `artifacts/v2-a/a1_13f/assessment-summary.json` | 固定预算与 ordinary formal 的方法学判定。 | `fixed_budget_seed_instability_persists`；early-stop primary=false |
| `artifacts/v2-a/a1_15/` | query-coupled + answer CE overfit、三 fresh formal 与 assessment。 | overfit passed；state/full `0/3` |
| `artifacts/v2-a/a1_16/` | query-coupled + no answer CE overfit、三 fresh formal 与 assessment。 | overfit passed；state/full `0/3` |
| `artifacts/v2-a/a1_17/runs/run-{1,2,3}/{coupled_ce,coupled_noce}/` | A1.13 成功 seed 的两种 paired objective；共享初始化位相等。 | 两臂 state/full 均 `0/3` |
| `artifacts/v2-a/a1_17/assessment-summary.json` | reference、paired seed、共享初始化 tensor identity 与两臂正式结果。 | `independent_answer_auxiliary_gradient_required`；root localized，architecture not validated |
| `tmp/V2-A1.12-A1.17 root-cause result.md` | 故障排除链、最终判断、证据边界、完成/未完成项与 A1.18 建议。 | 当前人类可读根因交接 |

## A1.18/A1.18B 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_18/overfit/{qaux,saux}/` | QAUX 与 FINAL-SAUX length-balanced overfit32、训练checkpoint和辅助剥离部署artifact。 | 两臂严格 overfit Gate 通过 |
| `artifacts/v2-a/a1_18/runs/run-{1,2,3}/saux/` | FINAL-SAUX paired fixed-4000 training、部署formal与通过run干预。 | formal/causal `2/3`；final-only state credit seed 不稳定 |
| `artifacts/v2-a/a1_18b/overfit/tsaux/` | 在 FINAL-SAUX 失败seed上的 TSAUX overfit32 正控制。 | step 4000 严格通过 |
| `artifacts/v2-a/a1_18b/runs/run-{1,2,3}/tsaux/` | TSAUX paired fixed-4000 training、辅助剥离formal与causal。 | formal/causal `3/3` |
| `artifacts/v2-a/a1_18b/fresh/run-{1,2,3}/tsaux/` | model seed `20261821/22/23` 的 TSAUX fresh fixed-4000 training、部署formal与causal。 | formal/causal `3/3` |
| `artifacts/v2-a/a1_18b/assessment-summary.json` | overfit、FINAL-SAUX、TSAUX paired/fresh、完整性、成本和边界的机器总判定。 | `mechanism_solved=true`；当前机制机器真源 |
| `tmp/V2-A1.18 result.md` | A1.18/A1.18B 结果、根因、完成/未完成边界，以及已被 2026-07-17 路线重审后移的原 A1.19 建议。 | 当前人类可读机制交接 |

## A1.19H 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_19h/h1/` | H1 overfit32、三组 fixed-cardinality formal、structural/address/recurrence 干预与 assessment。 | formal/causal `3/3`；`opaque_handle_hybrid_core_confirmed` |
| `artifacts/v2-a/a1_19h/h2/overfit32/` | N2/N3/N4、T1–16 cell-balanced fit-only checkpoint/results。 | 严格 overfit Gate 通过 |
| `artifacts/v2-a/a1_19h/h2/runs/run-{1,2,3}/data/` | 三组独立 data seed 的 N2–N5、relation/OOD/causal 数据。 | 三组 data audit 通过；跨 run fingerprint 排除生效 |
| `artifacts/v2-a/a1_19h/h2/runs/run-{1,2}/formal/` | 两组 fixed-4000 variable-cardinality checkpoint、部署模型和 formal eval。 | N2–N5 与 relation Gate 全通过；causal intervention 全通过 |
| `artifacts/v2-a/a1_19h/h2/throughput-optimization.json` | 旧逐步 CUDA 编码/同步与新 GPU-cache 管线的同初始化同 batch 序列 60-step 对照。 | `10.744×`；final loss delta `0`；parameter max delta `0` |
| `artifacts/v2-a/a1_19h/h2/runs/run-1/formal/formal-eval-optimized-regression.json` | 优化后 forward/cached evaluator 对旧 run-1 checkpoint 的全 split 回归。 | 原/新 gates 与 splits 逐字段完全相同 |
| `artifacts/v2-a/a1_19h/h2/runs/run-3/formal/` | model/data/mapping seed `20261963/20261953/20261973` 的 fresh optimized fixed-4000 formal。 | formal/causal 通过；训练 `155.52s`、`25.72 step/s` |
| `artifacts/v2-a/a1_19h/assessment-summary.json` | H1、H2 overfit、三数据 audit、三 run formal/causal 与完整性聚合。 | `generalized_hybrid_core_confirmed`；A1.19H 机器真源 |
| `docs/v2-a1.19h-hybrid-core.md` | 架构合同、方法修正、H1/H2 结果、吞吐等价与证据边界。 | A1.19H 人类可读正式记录 |
| `tmp/V2-A1.19H result.md` | 完成/未完成项、异常修正与 A1.20B 交接。 | 当前阶段结果交接 |

## A1.20B 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_20b/overfit32/` | 32 条 N2–N4/T1–16 full-token cache、audit、fresh checkpoint 与逐 cell 结果。 | mapping/trajectory/final/answer 全 `1.0`；fit-only sanity 通过 |
| `artifacts/v2-a/a1_20b/qwen-cache-batch-benchmark.json` | batch 8/12/16 的真实 full-text Qwen 编码 wall time、real tokens/s 与 peak allocation。 | 选择 batch 12，`2602.43 real tokens/s` |
| `artifacts/v2-a/a1_20b/runs/run-1/train-cache/` | train 16,384 + validation 2,048 条 FP16 Qwen last-hidden mmap shards；共 `5,594,417` source tokens。 | full-text Boundary run-1 输入；无 oracle 字段 |
| `artifacts/v2-a/a1_20b/runs/run-1/train-cache-audit.json` | schema/hash/source/shape/finite/fingerprint/overlap/no-oracle/no-truncation Gate。 | 全部通过 |
| `artifacts/v2-a/a1_20b/training-pipeline-benchmark.json` | 同 schedule/initialization 的同步 mmap 与后台 pinned prefetch 20-step 对照。 | prefetch 仅 `0.939×` 且非参数级等价；正式保留同步路径 |
| `artifacts/v2-a/a1_20b/runs/run-1/formal/results.json` | reader seed `20262011` 的 fixed-5000 Boundary training 与 2048 条 heldout validation。 | trajectory/final/answer `0.297363/0.409668/0.608887`；formal eligibility 失败 |
| `artifacts/v2-a/a1_20b/runs/run-1/formal/failure-diagnostic.json` | 15 个选择性 oracle 条件、联合 Boundary/program/address exact 与 N/T 分组。 | full oracle core `1.0`；value/source/target binding 为 primary failure |
| `artifacts/v2-a/a1_20b/runs/run-1/formal/gradient-credit-audit.json` | state CE-only 与 full factorized loss 对各 Boundary logits 的梯度范数。 | state CE 到六类离散控制 logits 全为 `0`；hard-control credit 断裂确认 |
| `artifacts/v2-a/a1_20b/assessment-summary.json` | overfit、cache、run-1、oracle、gradient、未运行阶段与证据边界的机器聚合。 | `a120b_passed=false`；A1.21P/A1.22A 不允许 |
| `docs/v2-a1.20b-full-text-boundary.md` | 正式合同、优化、结果、根因与路线判断。 | A1.20B 人类可读正式记录 |
| `tmp/V2-A1.20B result.md` | 完成/未完成项与停线交接。 | 当前阶段结果交接 |

## A1.20C 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_20c/overfit32/supervision/` | overfit32 train/validation token anchor target 与 manifest。 | 训练专用；不进入 forward |
| `artifacts/v2-a/a1_20c/overfit32/supervision-audit.json` | token offset、shape/range、source 对齐与 no-forward-input 审计。 | 全部 Gate 通过 |
| `artifacts/v2-a/a1_20c/runs/run-1/supervision/` | 16,384 train + 2,048 validation 的 compiler anchor target。 | 已准备；正式 matrix 因 target overfit 失败未使用 |
| `artifacts/v2-a/a1_20c/runs/run-1/supervision-audit.json` | run-1 compiler supervision 审计。 | 全部 Gate 通过 |
| `artifacts/v2-a/a1_20c/smoke/hierarchical__straight_through/` | 2-step CUDA smoke、checkpoint 与 hard-forward equivalence。 | state/answer logit delta `0/0` |
| `artifacts/v2-a/a1_20c/overfit32/hierarchical__straight_through/results.json` | fresh fixed-5000 目标臂 overfit32 结果。 | trajectory/final/state-token/answer `0.875/0.90625/0.964474/1.0`；strict Gate 失败 |
| `artifacts/v2-a/a1_20c/overfit32/hierarchical__straight_through/failure-diagnostic.json` | 四个失败 N4 cell、state-vs-local 梯度夹角、late regression 与停止列表。 | 机器分类 `anchor_localization_solved_but_execution_objectives_conflict` |
| `docs/v2-a1.20c-boundary-repair.md` | 2×2 合同、实现、overfit、梯度冲突、Gate 与下一修复边界。 | A1.20C 人类可读正式记录 |
| `tmp/V2-A1.20C result.md` | 已完成/未完成项与停线交接。 | 当前阶段结果交接 |

## A1.20D/A1.21P 运行产物

| 路径 | 内容 | 当前地位 |
| --- | --- | --- |
| `artifacts/v2-a/a1_21p/routing-{inrange,n5}-entity-capacity-threshold.json` | N4 伪第五实体与 N5 真第五实体在粗/refined section boundary 下的 pair-gain 扫描。 | A1.20D 根因证据 |
| `artifacts/v2-a/a1_21p/joint-shared-family-canonical-formal-bidirectional-probe.json` | 同一 checkpoint 的 canonical 八 split 完整矩阵。 | 全部诊断 Gate 通过；源 artifact `formal_gate=false` |
| `artifacts/v2-a/a1_21p/joint-shared-family-canonical-hidden-causal-audit.json` | normal、zero hidden、batch roll、token reverse 与 core integrity。 | hidden causal Gate 通过 |
| `artifacts/v2-a/a1_21p/joint-shared-family-routing-ood-{bidirectional-probe,seed2,seed3}.json` | 三个 continuation schedule 的 routing OOD/N5 probes。 | 路径稳定性 `3/3`；非 fresh model/data seed |
| `artifacts/v2-a/a1_21p/k1/full-probe/results.json` | K=1 2048 validation 容量基线。 | trajectory/final/answer `0.106934/0.333008/0.550293` |
| `artifacts/v2-a/a1_21p/p{0-current,1-routing}-family-chat-smoke5.json` | official chat template 下 direct/text-CoT/hybrid 在线五条 Pareto smoke。 | candidate 正证据；非正式样本规模 |
| `artifacts/v2-a/a1_21p/routing-data-v2-ood-small/manifest.json` | routing 数据来源、split 和语义变换声明。 | `surface_only_transform=true`；不得作为新代数 |
| `artifacts/v2-a/a1_21p/assessment-summary.json` | 机制门、正式门、来源 SHA-256、完成/缺失项与停机判定。 | `a121p_passed=false`；`a122a_authorized=false` |
| `docs/v2-a1.20d-full-text-mechanism-repair.md` | 三遍双向 section 修复、结果、架构解释和证据边界。 | A1.20D 人类可读记录 |
| `docs/v2-a1.21p-pareto.md` | K1、在线 smoke、六项正式缺口和下一轮合同。 | A1.21P 人类可读正式记录 |
| `tmp/V2-A1.20D-A1.21P result.md` | 完成/未完成项与 A1.22A 停机交接。 | 当前阶段结果交接 |

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
| `thinking/`（不含 `thinking/research/`） | 被 `.gitignore` 排除的用户原始思考稿；五篇近期草稿只读并保留原文，`Project-Yggdrasil 未来多模态潜空间智能体架构.md` 作为项目初期 V1 历史参照；正式整理见 `docs/thinking-draft-synthesis-2026-08-01.md`。 |
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

新增 V2 文件应先归入明确的 V2-A/V2-B/V2-C 语义包，再更新本索引。实验结果只能在 evidence level、配置、seed、heldout、消融、成本和失败边界齐全后进入当前证据；旧 Stage 结果不得静默升级为 V2 架构证据。V2-C C0-C4 只提供系统组件归因，不能单独冒充 C5 `integrated-system`。任何再次废弃的当前文件都移动到带日期的 `archive/`，不保留并列旧入口。
