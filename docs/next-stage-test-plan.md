# 下一阶段测试任务规划

## 口径

本规划把“我们需要更多挑战，这个架构也是”理解为：下一阶段不只是提高数据规模或多跑 seed，而是要更直接地挑战架构假设本身。测试目标应从“toy 链路能跑通”升级为“单个专家或旧 baseline 不能直接解决，必须依赖统一 latent bus、主动读取、对象/计数 slot、输出专家和长程记忆共同工作”。

这里仍然区分两件事：

1. 架构闭环验证：证明某条结构在可控任务中有因果作用。
2. 任务质量证明：证明真实世界任务上达到可用质量。

下一阶段优先做前者，但任务必须比现有 Stage AM/AJ 更难；不能把低熵饱和任务继续拉长训练后当成新证据。

## 当前基线

已经比较强的正信号：

- Stage AM：结构化 evidence 上，cell/count/pair 统一 latent bus + count 输入专家 + decoded position compare 后，四任务 all-task answer 达到 98.24%。
- Stage AN：同一统一 bus 接 answer-token writer 后，3 seed 平均 answer-token sequence exact 达到 98.11%，证明统一 bus 不只支撑分类头。
- Stage AQ：加入 relation delta 与 deterministic truth-table 过程状态，并把 truth-table state 强注入 answer writer 后，3 seed relation sequence exact 达到 99.74%，no-evidence relation truth-table 为 50.52%。
- Stage AK/AL：relation-only 的 object/pair latent bus、query retrieval、compare 和 yes/no answer writer 已接近 99%-100%。
- Stage AJ/AP：Transformer patch 输入到 latent 再完整重绘的链路能跑；copy-only 辅助监督正式单 seed 达到 100% scene exact；edit/generation 正式单 seed 达到 99.02%/100% scene exact。
- Stage E/F/I/J/K：多模态 shared latent、工具历史、memory、UI/DOM 和 audit loop 在合成环境中可以闭合。

仍然没有证明的部分：

- Stage AM 还只是结构化合成 evidence，不证明真实视觉计数或 detector/segmentation 到 count slots。
- Stage AM 已由 Stage AN 补上 answer-token writer 门禁，但仍是结构化 evidence，不证明真实语言生成能力。
- Stage AM relation 只有 92.97%，Stage AN 为 92.45%；Stage AQ 已拉到 99.74%，但依赖显式 deterministic truth-table 结构偏置。
- Stage AP 已补完 Stage AJ edit/generation 正式单 seed 长训；仍未完成 edit/generation 的多 seed 统计和更高熵图像任务。
- 当前没有真实文件、真实 UI 页面、真实 OCR/layout、长程探索成本、局部记忆污染和工具失败恢复的综合挑战。

## 总原则

每个新测试都必须同时有这些门禁：

1. 主任务结果：answer exact、scene exact、foreground/object exact 或 episode success，不能只看 cosine、MSE 或 loss。
2. 因果消融：no-evidence、shuffled-evidence、no-source、no-image、no-tool-history 等必须显著下降。
3. 对照基线：direct structured baseline、oracle/strong expert baseline、旧架构 baseline 至少保留一个。
4. 稳定性：除快速 smoke 外，正式结论默认至少 3 seeds；长训任务可先 1 seed 方向验证，但必须标成 probe。
5. 证据落盘：每轮必须写 JSON、样例 PNG/HTML/PDF 或 trace，并同步实验报告和 `docs/DIRECTORY_REFERENCE.md`。
6. 失败也要可用：失败实验必须给出下一步决策，不允许只留下“可能调参可修”的模糊结论。

## 优先级矩阵

| 优先级 | 测试任务 | 挑战的架构假设 | 最小可执行形态 | 通过门槛 | 失败后处理 |
| --- | --- | --- | --- | --- | --- |
| P0 | 实验基础设施硬化 | 更长训练必须可恢复，否则无法挑战更难任务 | 给 Stage AJ/后续新脚本补 checkpoint、resume、定期样例、统一 JSON schema 和超时保护 | 中断后 resume 的最终指标与不中断 run 误差可接受；JSON 中记录 config、seed、best checkpoint、时间、设备 | 没有恢复能力前，不跑 30 分钟以上正式长训 |
| P1 | Stage AN：Stage AM 统一 bus 接 answer-token latent writer | 统一 bus 是否能支撑文本 token latent 输出，而不是只支撑分类头 | 复用 color/shape/count/relation 四任务，把 answer writer 从分类头升级为 Stage AC 风格 answer-token latent writer | 3 seeds 平均 answer token exact >= 95%，relation >= 90%，no-evidence 下降到接近随机或明显低于 full | 如果分类头高、token writer 低，优先修输出 token 分离/contrastive，不回退到分类头 |
| P1 | Stage AO：像素到 cell/count slots | count 输入专家是否能从真实像素或更高熵合成图像产生一等 count slots | 用合成多对象图片替代结构化 evidence，训练 detector/segmentation/counting expert 产出 cell/count/pair slots，再接 Stage AM bus | object table 或 mask IoU >= 95%，count answer >= 90%，all-task answer >= 85%，no-image/no-evidence 明显下降 | 如果 detector/count slots 不稳，先单独修专家，不把失败归因到 reasoner |
| P1 | Stage AP：Stage AJ supervised edit/generation 正式长训 | 图像输出专家能否从 latent 完整重绘并执行编辑，而不是只 copy | 在 checkpoint/resume 后补 d_model 192、4096/512/512、edit/generate 正式单 seed，再扩到 3 seeds | copy 门禁保持 100%；edit scene exact >= 95%；no-source <= 10%；foreground/object 指标优先于全图 MSE | 若 edit 低但 copy 高，改可修改对象表；不继续拉旧无监督 `memory_tree_edit` |
| P1 | Stage AQ：relation delta/truth-table slots | relation 是否需要显式过程状态，而不是 raw pair compare | 在 Stage AM 上加入 delta slots、truth-table/process supervision，测试 relation 从 92.97% 拉回 98%+ | 3 seeds relation answer >= 98%，no-evidence relation 不能同步升高，process probe 可读 | 若只靠 compare loss 提升，判为指标修补，不作为架构进步 |
| P2 | Stage AR：跨专家不可单解任务 | 单个专家能 100% 时，不能证明 MoE/latent 组合价值 | 设计 image + text rule + telemetry + memory 四路任务，任意缺一路都无法答对 | full >= 90%，每个 no-modality ablation 至少下降 30 个百分点；direct all-input baseline 记录成本 | 如果 direct baseline 同等且更便宜，保留负结果，不强行宣称 latent 优势 |
| P2 | Stage AS：真实文件证据审计 | 架构是否能处理真实文件边界和引用，而不是 synthetic DSL | 小型 HTML/PDF/CSV/截图组合，模型必须调用受控工具抽取证据并输出带引用 audit report | 引用命中率、结论准确率、工具步骤合法率均 >= 85%；no-tool-history 明显下降 | 若 OCR/layout 是瓶颈，接预训练 OCR/layout expert，不让 tiny latent 承担全部感知 |
| P2 | Stage AT：长程局部记忆和探索成本 | memory tree/work tree 是否真的降低长任务成本 | 局部地图、遮挡、错误探测、不可重置探测成本、多目标任务，要求写 memory 并复用 | 同等预算下 memory/active-read 明显优于 no-memory；错误记忆注入后可检测或修正 | 如果只是短 horizon 100%，继续加路径长度、干扰和错误观察，不算通过 |
| P3 | Stage AU：预训练专家 + hard negative grounding | 真实专家接入后，latent bus 是否承载视觉/文档 grounding | DocVQA/OCR/layout/CLIP 或小型 VLM expert，加入同 prompt 风格但图像事实不同的 hard negatives | full 明显高于 text-only/CLIP-only；no-image 和 shuffled-image 必须掉分 | 如果 text-only 接近 full，任务不是 grounding 测试，重做数据集 |

## P0 落地状态

2026-07-04 已先在 Stage AJ 图像 IO 长训脚本落地 P0 基础设施：磁盘级 `latest.pt`/`best.pt`、`--resume`、训练中样例、`schema_version=2` 的结果 JSON、CPU/GPU 设备选择、时间预算跳过保护和 `--stop-after-steps` 恢复链故障注入。验证记录见 `docs/omni-transformer-stage-aj-transformer-image-io-fidelity-experiment.md` 的 P0 小节。

后续 Stage AN/AO/AP/AQ 不应重新发明一套恢复格式；如果需要长训，应复用这套字段口径：latest 用于继续训练，best 只用于最终评估，结果 JSON 必须记录 checkpoint 目录、resume 状态、设备、seed、best checkpoint、训练步数和是否因时间/测试注入停止。

## P1 已完成状态

2026-07-04 已完成 Stage AN、Stage AP 与 Stage AQ：

- Stage AN：`docs/omni-transformer-stage-an-answer-token-latent-writer-prep.md`，正式结果 `artifacts/omni_transformer_stage_an_answer_token_writer/formal_results.json`，answer-token sequence exact 98.11%，relation sequence exact 92.45%。
- Stage AP：`docs/omni-transformer-stage-ap-formal-edit-generation-experiment.md`，正式结果 `artifacts/omni_transformer_stage_ap_formal_edit_generate/formal_edit_generate_result.json`，edit scene exact 99.02%，edit no-source 4.10%，text generation scene exact 100%。
- Stage AQ：`docs/omni-transformer-stage-aq-relation-process-supervision-experiment.md`，正式通过结果 `artifacts/omni_transformer_stage_aq_relation_process_supervision/formal_results_v3.json`，relation sequence exact 99.74%，relation truth-table 100%，no-evidence relation truth-table 50.52%。

Stage AQ 的关键经验是：只加 learned truth-table head 不够；deterministic truth-table 可读但弱注入仍不够；必须把 truth-table state 作为强过程状态接入 answer writer。

## 推荐执行顺序

1. P0 已先在 Stage AJ 落地；后续长训脚本要复用同一恢复与结果 schema。
2. Stage AN 和 Stage AQ 已完成；继续维护它们作为 Stage AM 代码路径的统一 bus / token writer / relation process 基线。
3. Stage AP 已完成正式单 seed；图像输出路线下一步应转向多 seed 或更高熵图像任务。
4. 然后做 Stage AO。它会把 Stage AM/AQ 从结构化 evidence 推到像素 expert，这才是真正挑战 cell/count/pair slots 的下一步。
5. P2/P3 作为第二轮扩展，不要和 P1 混成一个超大实验。跨专家、真实文件、长程记忆、预训练专家各自都有不同失败点，必须分开归因。

## 旧路线收口规则

这些内容保留为历史证据或负对照，但不应再作为下一阶段主线：

- Stage AI 低熵单物体图像生成/编辑已经饱和，不再继续加 seed 或拉长训练。
- 无辅助监督的 `memory_tree_copy`/`memory_tree_edit` 已经证明对象保真不足，只作为负对照；下一步不继续拉长旧配置。
- 纯 32-slot resampler 或 Attention Pump 作为唯一信息通道的路线已经多次丢信息，不再作为默认架构候选。
- naive MoE staged 训练已出现灾难性遗忘；没有 replay、蒸馏、冻结或正则化时不再跑同类 staged 实验。
- 只看全图 pixel MSE、latent cosine 或 loss 的测试不再作为通过证据。
- 废旧测试如果继续锁旧契约，应删除或改成当前契约；不要为了兼容旧设计保留过渡断言。

## 担忧和不确定点

1. P1 任务已经会把本机 4070 Laptop GPU 推到较长训练，checkpoint/resume 是硬前置。
2. Stage AO 的 detector/segmentation/counting expert 如果从零训练，可能先卡在感知专家本身；需要把“专家质量不足”和“latent bus 不行”分开报告。
3. Stage AN 可能暴露 answer-token latent collapse。即使 cosine 很高，也必须以 token exact 和 no-evidence gap 判断。
4. Stage AS/P3 可能需要下载真实数据或预训练模型；涉及网络、缓存、HF token 和磁盘占用时要单独记录。
5. 本规划没有承诺真实产品能力，只规划下一批能挑战架构假设的本地实验任务。
