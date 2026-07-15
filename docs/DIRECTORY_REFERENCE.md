# Directory Reference

本仓库是 Project-Yggdrasil V2 的独立研究工作区。当前主线已经从旧 Stage A—AV-J-C 直接切换到 V2-A，再在 V2-A 通过后进入 V2-B。关联项目 `C:\skzy\QuickFileTransport\世界树计划` 的设计哲学是上位概念真源，本仓库不修改其源码。

## 当前真源

| 路径 | 地位与用途 |
| --- | --- |
| `docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md` | 当前唯一目标架构规范；定义连续 latent recurrence、Boundary-MoE、FFN-MoE、审计和边界。 |
| `docs/Project-Yggdrasil V2 从架构验证到商用路线图.md` | 当前唯一高层路线；按 Gate 推进 R0、R1/R2/R3、架构完整版和商用路线。 |
| `docs/next-stage-test-plan.md` | 当前最近执行真源；冻结 V2-A/V2-A1.5/V2-B 的实验顺序、证据口径、Gate、消融、成本指标和旧路线收口。 |
| `docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md` | V2 决策形成记录；只保存推导和审阅依据，不与白皮书并行定义规范。 |
| `docs/moe-model-assembly-comparative-review-2026-07-14.md` | V2 的 Boundary-MoE/FFN-MoE 与公开模型路线的中文对照；区分已被其他模型验证的局部思想、完整架构未验证边界和 V2-B 未启动状态。 |
| `docs/DIRECTORY_REFERENCE.md` | 本索引；新代码、测试、文档和归档必须同步这里。 |
| `README.md` | 面向仓库使用者的当前状态说明；只展示 V2-A 当前入口和未通过 Gate 的边界。 |
| `docs/v2-a-reasoning-medium-experiment.md` | V2-A 当前实现合同、Qwen3.5 基座、数据 schema、A0/A1/A2 结果、Gate 判定和失败边界。 |
| `docs/v2-a1.5-latent-foundation.md` | A1.5 独立 schema、P0 结构化正控制、P1 Qwen hidden 接口、P2 learned-slot formal、因果干预和停止门禁。 |

## 研究辅助入口

| 路径 | 用途与边界 |
| --- | --- |
| `docs/research-agent-literature-commands.md` | 面向 Google Gemini Deep Research 的三天研究冲刺命令；覆盖 A1.5 当前故障、V2-A3/A4、V2-B、I/O、工作树/记忆、专家晋升、完整架构和商用安全，并组织增量、反证与综合。研究报告只作为决策输入，不自动修改架构真源或 Gate。 |

## 当前实现状态

V2-A0 数据/基座链路、Qwen3.5-2B/0.8B text-CoT probe 和当前结构的 V2-A1 mechanism smoke 已实现；旧 A2 的 K/T、prompt、transition、读出、full-data、teacher/masked state supervision、token-wise source adapter、step-level verifier RL、latent-attention probe 与 0.8B/2B 对照仍未达到强 text-CoT 基线。source-layer bank、query init、token mixer、latent-attention 等失败入口已从当前代码删除。新增的 A1.5 独立 schema 与 P0/P1/P2 代码已完成：P0 32-example overfit final/state full exact `1.0/1.0`，4096-example best ordinary test `1.0/1.0`，composition-heldout `0.2734/0`，length-heldout final/state full exact `1.0/0.2266`；真实 Qwen3.5-2B FP16 hidden cache 已分片生成，P1 4096-cache formal ordinary validation final/state `1.0/1.0`，小 probe 仅作为 underfit 诊断；P2 K=8 formal ordinary test `1.0/1.0`、composition `0.2773/0`、length `1.0/0.6484`，same-answer composition shuffle 失败；A1.5 0.8B no-cap zero-shot 128 条 test/composition/length formal 分别为 `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`，2-shot test/composition/length final/state 也分别为 `0.1328/0.0156`、`0.1953/0`、`0.0938/0`（格式解析率高但 state 失败）。A1.5 未通过，A3/A4/V2-B 均未启动。A1.5 真源见 `docs/v2-a1.5-latent-foundation.md` 与 `tmp/V2-A1.5 result.md`；旧 A2 结果仍只作历史 probe，不得与 A1.5 混写。

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
| `tests/conftest.py` | 为当前 CPU torchvision wheel 预声明缺失的 NMS operator，保证 Transformers 测试收集可重复；不改变模型运行语义。 | 测试环境隔离 |

本地 `artifacts/v2-a/` 被 `.gitignore` 忽略；阶段结果路径、配置和证据等级必须以 `docs/v2-a-reasoning-medium-experiment.md` 为索引，不把未索引的本地文件当作 repo truth。

## 公开路线对照入口

| 主题 | 路径 | 用途 |
| --- | --- | --- |
| MoE/模型组装对照 | `docs/moe-model-assembly-comparative-review-2026-07-14.md` | 对照 VLMo、Uni-MoE、MoME、Uni-Med、DeepSeek-VL2、MoE-LLaVA、DeepSeekMoE/DeepSeek-V3、Qwen3、SMoES；记录成果、差异和 V2-B 未完成项。 |

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
