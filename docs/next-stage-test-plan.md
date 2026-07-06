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
| P2 | Stage AV：70M 从零集成 Micro-Omni | 不接预训练专家时，单一更大模型是否能同时承载视觉、工具和 memory 能力 | 70M 级 Transformer，从零训练，统一接 image/text/tool/memory/state，覆盖 AR/AS/AT 子任务 | overall >= 85%，每任务 >= 80%，对应 no-image/no-tool/no-memory 消融明显下降 | 若视觉子任务随机，先分阶段预训练视觉 reader，不继续只靠扩大参数量 |
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

## P2 已启动状态

2026-07-04 已完成 Stage AR 视觉中心跨专家不可单解任务：

- Stage AR：`docs/omni-transformer-stage-ar-cross-expert-visual-experiment.md`，正式结果 `artifacts/omni_transformer_stage_ar_cross_expert_visual/formal_results.json`。
- 3 seed 平均 latent full answer accuracy 为 100.00%，no-image / no-text-rule / no-telemetry / no-memory 消融分别降到 23.96% / 25.07% / 32.10% / 30.21%。
- direct all-input baseline 为 51.04%，没有同等解决。

Stage AR 的结论只覆盖受控合成视觉 + 受控 DSL text-rule + 离散 telemetry/memory 的四路因果依赖；不证明真实 VLM grounding、自然语言指令跟随或长程 agent 状态。agent 状态路线保留给 Stage AT 或 AR 后续加强版。

2026-07-04 已完成 Stage AS 真实文件证据审计：

- Stage AS：`docs/omni-transformer-stage-as-file-audit-experiment.md`，正式结果 `artifacts/omni_transformer_stage_as_file_audit/formal_results.json`。
- 3 seed 平均 conclusion accuracy、citation accuracy、report exact、tool step legality 均为 100.00%。
- no-tool-history conclusion accuracy 为 25.00%，相对 full 下降 75.00 个百分点。

Stage AS 的结论覆盖真实落盘 HTML/CSV/PDF/截图文件和受控工具读取链路；不证明复杂 PDF layout、OCR、开放网页审计或自由规划型 tool-use agent。

2026-07-04 已完成 Stage AT 长程局部记忆和探索成本任务：

- Stage AT：`docs/omni-transformer-stage-at-memory-exploration-experiment.md`，正式结果 `artifacts/omni_transformer_stage_at_memory_exploration/formal_results.json`。
- 3 seed 平均 episode success 为 100.00%，强 no-memory oracle episode success 为 0.00%，success gap 为 100.00 个百分点。
- full mean scans 为 4.00，no-memory oracle mean scans 为 10.00；full mean cost 为 38.00，no-memory oracle mean cost 为 47.96。
- corrupt-memory success 为 99.93%，corrupt-memory correction rate 为 99.80%。

Stage AT 的结论覆盖受控局部地图、遮挡 zone、主动 scan/write memory、多目标复用和错误 memory 修正；不证明真实导航、真实视觉 SLAM、开放式 memory tree 自动扩展或自由规划型长期 agent。

2026-07-04 已启动 Stage AV 70M 从零集成 Micro-Omni，当前是中间负结果，不进入正式 3 seed：

- Stage AV：`docs/omni-transformer-stage-av-from-scratch-micro-omni-experiment.md`，probe 结果 `artifacts/omni_transformer_stage_av_from_scratch_micro_omni/probe_70m_residual_results.json`。
- 70M 档配置为 `d_model=768`、`layers=10`、`heads=12`、`latent_tokens=8`，参数量 70,994,707；本机 RTX 4070 Laptop 8GB 上 probe 峰值 CUDA allocated 约 1,810.62 MB。
- probe overall answer accuracy 为 64.32%，memory exploration 为 100.00%，file-audit tools 为 69.53%，cross-expert visual 只有 23.44%。
- `no_tool_history` 从 64.32% 降到 32.68%，`no_memory` 降到 31.77%；但 `no_image` 基本不降，视觉 grounding 未闭合。

Stage AV 说明“本机训练 70M 级从零集成模型”可行，但“单阶段多任务自然学会视觉+工具+memory”未证明。下一步不应直接扩大参数或跑 3 seed，应先分阶段预训练视觉 zone/color reader，再接统一 answer head。

2026-07-04 已切换 Stage AV-B 分阶段潜空间训练方案，并完成 smoke：

- Stage AV-B：`docs/omni-transformer-stage-avb-staged-latent-training.md`，脚本 `experiments/omni_transformer_stage_avb_staged_latent_training.py`。
- 训练顺序改为 `text_base -> image_translate -> tool_memory_translate -> latent_reason -> joint`。
- 70M batch 256 smoke 参数量为 71,025,455，峰值 CUDA allocated 约 3,094.87 MB，说明本机 8GB 显存仍有余量。
- 长训建议由用户本地启动，优先用 batch 256；若 `nvidia-smi -l 2` 显示长期低于 60W，再提到 384/512。

Stage AV-B 目前只有 smoke，没有能力结论。它替代 AV v0 继续推进，AV v0 作为“单阶段端到端不自然闭合视觉 grounding”的负结果保留。

用户本地已完成 Stage AV-B long probe：

- 结果：`artifacts/omni_transformer_stage_avb_staged_latent_training/long_result.json`。
- overall answer accuracy 为 74.61%，未达到 85% 门槛。
- `file_audit_tools` 为 95.01%，`memory_exploration` 为 100.00%，但 `cross_expert_visual` 只有 28.95%。
- `no-tool-history` 降到 14.16%，说明工具历史有因果作用；但 `no-image` 为 73.93%、`shuffled-image` 为 74.61%，几乎不降，说明视觉 grounding 未进入最终答案。
- unique train tokens 约 516k，processed tokens 约 87.1M，主要是反复重采样小数据集。

判断：Stage AV-B long probe 不通过。下一步不继续对小数据加步数，应接 Stage AV-C sharded dataset pipeline，并处理 image translator 遗忘。

2026-07-04 已补 Stage AV-C 大规模合成数据流水线 smoke：

- Stage AV-C：`docs/omni-transformer-stage-avc-dataset-pipeline.md`，脚本 `experiments/omni_transformer_stage_avc_dataset_pipeline.py`。
- 当前样本长度为 63 tokens；1B token 约需 15,873,015 examples。
- smoke 数据集 `artifacts/omni_transformer_stage_avc_dataset_pipeline/smoke_dataset/manifest.json` 包含 1,408 examples、88,704 tokens，验证了 train/val/test/heldout sharded `.pt` 输出。
- 结论：AV-B 不能继续用几千条小数据长训做结论；下一步应让训练脚本接 `--dataset-manifest`，先跑 10M token 级别，再考虑 100M/1B。

Stage AV-C 目前只完成数据生成流水线，没有训练结论。

2026-07-05 已补 Stage AV-D 1B-first 数据集设计：

- Stage AV-D：`docs/omni-transformer-stage-avd-1b-first-dataset-design.md`。
- 新口径是先设计 1B token 母分布，再定向裁剪 10M token probe；10M 不是随机缩小版，而是保留关键结构、降低文本/schema/horizon 等部分熵。
- 已根据用户纠正把“模态翻译”改成“外部表征 ↔ latent 双向互译”，并在 1B/10M 配比里加入 `bidirectional_translation` 数据块。
- AV-C 当前生成器只保留为 shard/manifest 输出链路 smoke，不作为正式数据分布。
- 下一步应新增 AV-D generator，支持 `--scale 10m|100m|1b`，但先只生成 10M，并让 AV-B 从 `--dataset-manifest` 读取。

2026-07-05 已新增 Stage AV-E 双向互译 smoke/probe：

- Stage AV-E：`docs/omni-transformer-stage-ave-bidirectional-latent-external.md`，脚本 `experiments/omni_transformer_stage_ave_bidirectional_latent_external.py`。
- 链路为 `source external -> source latent -> target external` 与 `target external -> target latent -> source external`，同时训练 source/target reconstruction、双向 translation、latent alignment 和 answer。
- 小 probe 中 source-to-target token accuracy 为 28.40%，target-to-source token accuracy 为 28.24%，answer 仍接近随机，sequence exact 为 0。
- 结论：AV-E 方向更贴近架构理念，但当前只是早期负/诊断结果；不能用 latent cosine 代替外部互译 exact。

2026-07-05 已准备 Stage AV-F 10M 级双向互译数据集和 AV-E manifest 训练入口：

- Stage AV-F：`docs/omni-transformer-stage-avf-10m-bidirectional-dataset.md`，脚本 `experiments/omni_transformer_stage_avf_10m_bidirectional_dataset.py`。
- 10M manifest：`artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m/manifest.json`。
- 数据规模为 184,000 examples、10,120,000 pair tokens、train unique pair tokens 8,800,000。
- AV-E 训练脚本已支持 `--dataset-manifest`、`--gpu-resident-data`、checkpoint/resume 和 batched eval。
- 70M AV-E capacity smoke：75,970,770 参数；batch 256 峰值 CUDA allocated 约 4,753.32 MB，batch 512 峰值约 8,923.00 MB。
- 结论：10M 级训练入口已准备好；下一步应由用户启动长训，能力判断必须看互译 sequence exact 和 source/target 双向指标，不只看 answer accuracy。

2026-07-05 已补 Stage AV-G 严格分阶段 10M 训练入口：

- Stage AV-G：`docs/omni-transformer-stage-avg-strict-staged-10m-training.md`，脚本 `experiments/omni_transformer_stage_avg_strict_staged_10m_training.py`。
- AV-G 不触碰正在跑的 AV-E joint baseline；输出和 checkpoint 独立在 `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/`。
- 默认 schedule 为 `text_latent_base 400 -> external_codec 1200 -> bidirectional_translate 2200 -> latent_reason 1400 -> joint_debug 800`。
- 已做 CPU smoke 和 CPU resume smoke，验证 stage 切换、manifest IO、checkpoint/resume 和结果 JSON。
- 没有在本轮跑 70M CUDA smoke，避免抢正在运行的 AV-E 10M 训练 GPU。
- 后续严格架构路线应以 AV-G 为准；AV-E 当前长训保留为 joint baseline/负对照。

2026-07-05 用户本地完成 AV-E joint baseline 和 AV-G strict staged 两批 10M 长训：

- AV-E joint：`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_result.json`。
- AV-G strict：`artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_result.json`。
- 两批都是 75,970,770 参数、AV-F 10M manifest、batch 256、6000 steps、processed pair tokens 84,480,000。
- AV-E joint 最终 answer test accuracy 为 97.25%，但四个互译 sequence exact 全为 0；判为 answer shortcut，不通过。
- AV-G strict 最终 answer test accuracy 为 77.85%，四个互译 sequence exact 仍全为 0；严格 schedule 没有修复 external codec。
- 两批 token accuracy 均平台在约 72%-74%；按位置诊断显示 task/tool/memory 等低熵位置接近 100%，但 image zone color 和 variable text positions 只有约 30%-34%。
- 结论：下一步不应继续扩大同一 AV-E/AV-G 配置，而应新增 decoder/codec 修复实验。当前 `decode(latent)` 的 mean-pool latent + position query 不足以绑定每个可变字段。

2026-07-05 已改走 Stage AV-H 文本锚定视觉编辑数据集和训练入口：

- Stage AV-H：`docs/omni-transformer-stage-avh-text-anchored-visual-edit.md`。
- 数据集脚本：`experiments/omni_transformer_stage_avh_text_anchored_visual_edit_dataset.py`。
- 训练脚本：`experiments/omni_transformer_stage_avh_text_anchored_visual_edit_training.py`。
- 10M manifest：`artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/dataset_10m/manifest.json`。
- 数据规模：21,900 examples、10,030,200 tokens、train unique tokens 8,244,000。
- 任务链路为 source text + source image + edit instruction -> target text -> target object record -> target image，避免 AV-F 无文本锚点高熵互译。
- 68M capacity smoke：68,416,419 参数，batch 128，peak CUDA allocated 6,537.45 MB。
- 2026-07-05 已修 AV-H 训练性能：移除每步 GPU->CPU 同步点、按 stage 裁剪输出头、缓存 record 渲染常量，并新增 `--micro-batch-size` 与 `--optimizer` 开关。
- 68M perf-fix smoke：effective batch 128、micro batch 64、Adafactor，10 step peak CUDA allocated 4,893.83 MB，训练窗口 1.52 step/s。
- 已做 dataset smoke、training CPU smoke、10M loader smoke、CUDA capacity smoke 和 perf-fix smoke；下一步可用 micro batch 64 启动 AV-H 10M 长训。

用户本地已完成 AV-H 10M 长训：

- 结果：`artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_result.json`。
- 成本：68,416,419 参数、6,000 steps、processed tokens 351,744,000、elapsed 3,502.67 sec、peak CUDA allocated 4,894.82 MB。
- 原始 result JSON 旧口径 `joint_debug` 指标显示 test target record exact 95.77%、heldout 92.69%，但该模式输入包含 `target_record`，有 teacher-forcing / copy 泄漏，不能算端到端通过。
- 已补诊断：`artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_stage_mode_diagnostic.json`。
- 非泄漏关键指标：`joint_no_target_record` target record exact test 19.38%、heldout 1.46%；`image_ground` source record exact test 1.69%、heldout 0.00%；target text exact 仍接近 0。
- 样例 contact sheet 显示预测图像有颜色/位置弱信号，但明显模糊，不能只看 MSE。
- 结论：AV-H 10M 不通过。它提供了 teacher-forced record/image output 上限信号，但没有证明文本/图像到 target record、source image grounding 和端到端图像编辑链路。
- 训练脚本已修正后续评估口径：兼容主指标映射到 `joint_no_target_record`，并保留 `joint_teacher_record` 作为 teacher-forced 上限。

2026-07-05 已新增 Stage AV-I 整图 latent token 容量诊断：

- Stage AV-I：`docs/omni-transformer-stage-avi-whole-image-latent-capacity.md`。
- 脚本：`experiments/omni_transformer_stage_avi_whole_image_latent_capacity.py`。
- 设计：整图 encoder -> latent token bank -> 整图 decoder，不切 patch；黑色背景转透明 alpha，RGB loss 只算对象像素，alpha loss 监督全图透明度；ordered prefix 按短到长 curriculum 解锁，greedy prefix 选择最小透明整图重建 loss 的 token，不再使用前景交集。
- CUDA smoke：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/transparent_curriculum_cuda_smoke_v2_result.json`。
- 12/24/32 的 3000-step 全 train probe：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_result.json`，汇总 `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_summary.json`。
- 结果：12 token ordered final 的 test transparent loss 0.002505、object RGB MSE 0.002271、alpha IoU 0.9972；24/32 没有优于 12，后续 token 在 2 token 后边际收益很小。
- 已修正脚本：`d_model` 不再必须同时表示 encoder hidden width 和 latent token width；新参数 `latent_dim`、`encoder_width`、`decoder_width` 可单独控制，避免 token 数被单 token 宽度掩盖。
- 2 维 token 极限 probe：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result.json`，汇总 `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_summary.json`。
- 2 维结果：1/2/4/8/16/32 token 的 test alpha IoU 从 0.6213 升到 0.8166，说明从 1 开始的 token 容量曲线有效；但即使 32 token、64 个 latent scalars，也远不如 `latent_dim=192, latent_tokens=12`，且样图有明显 ghost/object 混叠。
- 已加入 compact prefix guidance：对未达到最终 token 数的 prefix 加 `0.5 * occupied_alpha_area + 0.5 * (1 - target_alpha_coverage)`，引导前序 token 少占面积但尽量覆盖目标区域。
- compact + 7000-step probe：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result.json`，汇总 `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json`。
- compact 结果：32 token 最好，test transparent loss 0.044031、object RGB MSE 0.029897、alpha IoU 0.9091；64 token alpha IoU 0.9074，接近但未超过 32。相比无 compact 3000-step，32 token alpha IoU 从 0.8166 升到 0.9091。
- 已加入 precision compact：coverage 奖励从线性 `x` 改成 `x^n`，`n` 默认从 1 线性增到 4，用来减少半透明/低置信覆盖的奖励。
- precision compact probe：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json`，汇总 `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json`。
- precision compact 结果：16 token 最好，test transparent loss 0.042794、object RGB MSE 0.030811、alpha IoU 0.9117；4/8/16 token 优于线性 compact，但 32/64 token 比线性 compact 变差。
- 已修正 precision schedule：`n` 在每个 active-token curriculum 段内都重新从 0.5 到 5，使用两端快、中间慢的非线性曲线。
- segment precision compact probe：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json`，汇总 `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json`。
- segment precision compact 结果：32 token 综合最好，test transparent loss 0.037481、object RGB MSE 0.024237、alpha IoU 0.9127；64 token alpha IoU 最高，test 0.9221、heldout 0.9103，但 object RGB MSE 和综合 loss 比 32 差。
- 2026-07-06 已把 AV-I 脚本切到 `schema_version=5`：compact prefix guidance 改为最大化前景正确像素，惩罚背景误涂、前景颜色错误和漏前景；相邻 prefix 新增 residual routing loss，要求后续 prefix 保留前序正确像素、限制正确区域外 delta、优先修正前序错误像素。
- correct-pixel residual CPU smoke：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/correct_pixel_residual_cpu_smoke_result.json`，验证新 loss、history、soft/hard correct pixel 指标、residual 指标和样例输出链路可跑。
- correct-pixel residual 正式 probe：`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result.json`，汇总 `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_summary.json`。
- correct-pixel residual 结果：本轮不通过。16 token 是本轮最佳，test loss 0.057707、object RGB MSE 0.030764、soft/hard correct pixel 0.4757/0.4139；64 token 只有 alpha IoU 最高 0.8665，但 test loss 0.081500、object RGB MSE 0.050253、soft/hard correct pixel 0.3534/0.2455，wrong color 0.4885。与上一轮 segment precision compact 相比，16/32/64 的公共指标全部退化。
- 结论：当前 AV-H 渲染图对宽 token 低熵，但对 2 维 token 仍有容量压力。分段 precision compact 仍是当前最好基线；第一版 correct-pixel residual 权重组合失败，下一步应修 loss/schedule，先把 correct/wrong pixel 保留为指标，再弱化或分阶段启用 wrong-color / residual delta，并补 scene/record inverse 指标和更高熵图像；只有 token 增加收益趋平且仍不能复原时，再进入参数量 sweep。

2026-07-06 已把 AV 线主目标重新收口为“先打通潜空间思考”：

- 路线文档：`docs/omni-latent-reasoning-route-and-goals.md`。
- 当前目标不是继续追 AV-H/AV-I 图像编辑或整图重建，而是先证明 `external <-> latent workspace <-> latent reasoning <-> external` 的闭环。
- 图像由我们自己构造，teacher/parser/renderer/trace 都可作为训练期监督；但推理期不能输入 teacher trace，也不能把 target record 作为泄漏输入。
- 下一阶段建议走 Stage AV-J：先用 `scene_record + text + operation + teacher trace` 训练 latent workspace、active read/process state 和 target record/answer，不把像素图作为第一主任务。
- AV-J 通过门槛必须同时看 source/target record exact、answer exact、trace/read accuracy、teacher-free eval 和 no-source/no-operation/no-trace 消融；answer 单独高不算通过。
- 图像外设应在 AV-J 后接入：先验证 `record -> image` 与 `image -> latent -> record` 的双向翻译，再进入图像编辑/生成。

2026-07-06 已准备 Stage AV-J 潜空间状态推理核心训练入口，停在大型训练前：

- Stage AV-J：`docs/omni-transformer-stage-avj-latent-reasoning-core.md`。
- 脚本：`experiments/omni_transformer_stage_avj_latent_reasoning_core.py`。
- 任务从单点问答升级为 record 级状态变换：`conditional_recolor`、`same_row_move`、`count_delete_or_add`。
- 模型输入 `source scene record + text operation`，输出 `target scene record + answer tokens`，并监督 `read_a/read_b/edit_slot/condition/edit_action/edit field` teacher trace；推理期不输入 teacher trace。
- 已完成 CPU smoke、CPU resume smoke、CUDA capacity smoke 和单元测试 `tests/test_stage_avj_latent_reasoning_core.py`。
- CUDA smoke 使用 `d_model=192/layers=3/heads=4/batch=128`，参数量 4,863,087，只证明 AMP、GPU resident data、checkpoint 和 eval variants 可跑，不提供能力结论。
- 70M 档已修正为 `d_model=480/layers=8/heads=8`，参数量 71,694,351；batch256 2-step capacity 通过，peak CUDA allocated 约 4,629.29 MB。
- `d_model=768/layers=10/heads=12` 在 AV-J 当前结构下约 225.8M，不是 70M；batch512 20-step capacity 在本轮 180 秒工具窗口内未完成，未产出 JSON。
- 文档中已给出修正后的 70M 训练命令和 70M batch256 20-step capacity 命令；大型训练由用户启动。

用户本地已完成 AV-J 70M 长训：

- 结果：`artifacts/omni_transformer_stage_avj_latent_reasoning_core/train_70m_result.json`，汇总 `artifacts/omni_transformer_stage_avj_latent_reasoning_core/train_70m_summary.json`。
- 成本：71,694,351 参数、17,000 steps、elapsed 2,982.64 sec、peak CUDA allocated 5,007.75 MB。
- test full source record exact 99.78%，target record exact 67.77%，answer sequence exact 98.78%。
- heldout full source record exact 99.95%，target record exact 67.19%，answer sequence exact 98.97%。
- test read_a/read_b/edit_slot 为 91.46% / 72.22% / 71.19%，condition/edit_action 为 98.51% / 99.66%。
- test no_source target record exact 8.62%，no_operation target record exact 0.00%，no_process target record exact 11.74%；但 no_source/no_process answer sequence exact 仍有 86.13% / 80.32%。
- 分任务 target record exact：conditional_recolor 约 91%，same_row_move 约 62%，count_delete_or_add 约 50%。
- 结论：AV-J 70M 不通过。source codec、operation 条件和 answer token 已学会，但 target record 状态改写未闭合，answer head 有明显模板/动作捷径。下一步应做 AV-J-B：弱化/后移 answer loss，增加 per-slot edit delta / copy-vs-update gate，强化 read_b/edit_slot hard negative，并把 count add/delete 拆成显式 count、delete slot、insert cell 过程态。

2026-07-06 已准备 Stage AV-J-B 长链 trace / verifier 训练入口，停在大型训练前：

- Stage AV-J-B：`docs/omni-transformer-stage-avjb-trace-verifier.md`。
- 脚本：`experiments/omni_transformer_stage_avjb_trace_verifier.py`。
- AV-J-B 复用 AV-J 的 record/text 数据分布，但加入 same-row candidate mask、count value、copy-vs-update gate、字段级 target accuracy、record verifier，并把 answer loss 后移且默认降到 0.05。
- 新 schedule 为 `codec -> trace_sft -> target_verifier -> joint`，目标是用 oracle solver 生成的结构化 trace 引导模型形成自己的 latent chain，而不是继续背 answer 模板。
- 已完成 `tests/test_stage_avjb_trace_verifier.py`、CPU smoke、CPU resume smoke、CUDA smoke 和 70M batch256 2-step capacity。
- 70M 档配置仍为 `d_model=480/layers=8/heads=8`，参数量 72,161,403；batch256 2-step capacity peak CUDA allocated 约 4,628.25 MB。
- 文档中已给出 AV-J-B 70M 长训命令；大型训练由用户启动。

## 推荐执行顺序

1. P0 已先在 Stage AJ 落地；后续长训脚本要复用同一恢复与结果 schema。
2. Stage AN 和 Stage AQ 已完成；继续维护它们作为 Stage AM 代码路径的统一 bus / token writer / relation process 基线。
3. Stage AP 已完成正式单 seed；图像输出路线下一步应转向多 seed 或更高熵图像任务。
4. Stage AO 已完成后，Stage AR 已先验证视觉中心四路专家因果依赖，Stage AS 已验证小型真实文件边界和引用审计链路，Stage AT 已验证受控长程局部记忆的探索成本收益。
5. Stage AV 已证明 70M 本机从零训练可承受，但单阶段集成未通过；Stage AV-B 到 AV-G 证明“分阶段”和“互译”方向必要，但 external codec/translation exact 没闭合。
6. AV-H/AV-I 保留为图像外设与 latent token 容量诊断，不再作为当前主线。AV-J 70M 已完成但未通过；AV-J-B 已准备好长训入口，下一步由用户启动 AV-J-B 70M，重点看 target record exact、candidate/count/copy gate 和 answer shortcut 是否改善。
7. P2 的 AR/AS/AT/AV 不要混成一个超大实验。跨专家、真实文件、长程记忆、从零集成、数据规模、预训练专家各自都有不同失败点，必须分开归因。

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
