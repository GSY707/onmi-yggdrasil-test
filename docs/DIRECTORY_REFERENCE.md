# Directory Reference

本仓库是 `Project-Yggdrasil 未来多模态潜空间智能体架构` 的独立可行性测试工作区。关联项目 `C:\skzy\QuickFileTransport\世界树计划` 只作为概念真源读取，本仓库不修改其源码。

"docs/Project-Yggdrasil 未来多模态潜空间智能体架构.md" 是本项目需要测试的架构的白皮书。
"note.txt" 是用户的笔记，可以参考其中的内容，但不要删。

## 顶层摘要

| 路径 | 用途 |
| --- | --- |
| `README.md` | 项目简短介绍、近期图像生成/编辑成果、Stage E+F 多模态融合成功结果、目标路线、初步证明和最小验证入口 |
| `docs/experiment-general-lessons.md` | 当前实验按任务线沉淀出的失败、修复和通用规则，覆盖统一 latent bus、专家分工、输出头、图像保真、长训恢复和旧路线收口 |
| `docs/omni-latent-reasoning-route-and-goals.md` | 2026-07-06 AV 线正确路线回收：先打通 `external <-> latent workspace <-> latent reasoning <-> external`，把自构造图像和 teacher 作为外设监督源，建议下一步 Stage AV-J |
| `docs/from-scratch-training-vram-quantization.md` | 从 0 训练完整架构验证的显存、量化、省显存边界和最低价整机购卡建议 |
| `pyproject.toml` | Python 测试配置、pytest 可选依赖、DocVQA 数据读取可选依赖与预训练/LLaVA 实验依赖 |
| `.gitignore` | 忽略本地虚拟环境、pytest 缓存、大模型 checkpoint 和本地 HF token 文件 |
| `experiments/text_to_latent_thought.py` | 纯文本思考训练到特殊 latent token 内部思考的本地 GPU 验证脚本 |
| `experiments/text_to_latent_sweep.py` | 多 seed、多 move-count 批量验证入口与聚合统计 |
| `experiments/multimodal_latent_pipeline.py` | 更接近多模态架构的连续潜变量管线实验：纯文本训练、潜变量翻译、潜变量/文本并行思考、潜变量思考到文本输出 |
| `experiments/multimodal_latent_comparison.py` | 连续潜变量管线的 6 组对照/消融实验：direct、visible CoT、scratch、no-codec、bad-codec、wide direct |
| `experiments/heterogeneous_latent_input.py` | 异构输入实验：文本动作 + 非文本地形 tensor，通过 latent codec 外设推理终点 |
| `experiments/heterogeneous_baseline_comparison.py` | 异构输入同等信息 baseline 对比：直接规则、learned transition、增强 raw direct、长 move 泛化与成本统计 |
| `experiments/visual_multimodal_stage_ab.py` | 真实像素输入 Stage A/B 实验：可控视觉地图、未见风格泛化、视觉 parser/CNN/latent/direct baseline |
| `experiments/visual_multimodal_stage_c.py` | 真实像素输入 Stage C 实验：每张地图随机符号语义绑定，允许主动探测恢复临时 legend |
| `experiments/visual_multimodal_stage_d.py` | 真实像素输入 Stage D 实验：局部可见多帧观察、探测成本、短期 legend 记忆复用 |
| `experiments/multimodal_fusion_latent_flow.py` | Stage E+F 多模态融合与潜空间数据流动实验：图像、文本、遥测输入进入 shared latent bus，含 baseline、ablation 与 slot probe |
| `experiments/omni_transformer_stage_h.py` | Stage H Tiny Omni Transformer 实验：统一 token stream、H1 direct、H2 latent bottleneck、文本 DSL 输出与 latent probe |
| `experiments/omni_transformer_stage_h3_h4.py` | Stage H3/H4 实验：latent token 容量曲线、counterfactual query、未见视觉风格和 held-out 组合测试 |
| `experiments/omni_transformer_stage_h5.py` | Stage H5 实验：视觉增强泛化与可组合规则 held-out 组合泛化 |
| `experiments/omni_transformer_stage_h6.py` | Stage H6 实验：多输出格式，覆盖 FULL、ACTION_ONLY、READ_VISUAL、READ_GOAL、READ_TELEMETRY |
| `experiments/omni_transformer_stage_i_agent.py` | Stage I Tiny Omni Tool-Using Agent 实验：多步工具调用、工具历史、记忆写入、latent bottleneck agent rollout |
| `experiments/omni_transformer_stage_jk_audit_ui.py` | Stage J/K Tiny Omni Agent 实验：多模态证据审计与 UI+DOM 操作，覆盖截图/图像、文本规则/目标、结构化证据/DOM、memory、工具历史与 latent bottleneck rollout |
| `experiments/omni_transformer_stage_l_docvqa.py` | Stage L DocVQA 真实文档问答边界实验：真实文档图像、问题、OCR 行候选进入 direct/latent bottleneck Transformer，评估答案证据行选择 |
| `experiments/omni_transformer_stage_m_moe_multimodal_llm.py` | Stage M Tiny MoE 多模态 LLM 实验：Vision/Text/Spatial/Counting/Chart 功能专家、router、attention pump、latent thought 与文本输出专家 |
| `experiments/omni_transformer_stage_n_strong_experts.py` | Stage N 强专家 MoE 与 baseline 成本对比实验：oracle symbolic experts、strong direct/latent decoder、raw classifier、参数量和预测成本 |
| `experiments/omni_transformer_stage_o_pretrained_experts.py` | Stage O 真实预训练专家接入实验：frozen CLIP image/text experts、linear classifier、direct decoder、MoE latent decoder、专家编码成本 |
| `experiments/omni_transformer_stage_p_llava_moe.py` | Stage P LLaVA-Instruct-150K MoE/latent 实验：官方 LLaVA JSON、COCO 图像缓存、CLIP token experts、Attention Pump latent、字符输出头与候选答案 scorer |
| `experiments/omni_transformer_stage_q_tiny_moe_vlm_from_scratch.py` | Stage Q 从零 Tiny MoE-VLM 实验：COCO 像素、字符级 prompt/answer encoder、tiny CNN vision expert、MoE/Attention Pump latent、候选答案 scorer 与 GPU 常驻数据优化 |
| `experiments/omni_transformer_stage_r_curriculum_vlm.py` | Stage R 低熵强监督 curriculum VLM 实验：从单颜色到多物体 caption 的合成视觉问答阶梯，测 direct 与 MoE latent 能力上限 |
| `experiments/omni_transformer_stage_s_scorer_reconstruction.py` | Stage S scorer 与信息还原诊断实验：answer cross-attention scorer、no-pump 专家 token 直连、各层 reconstruction probe |
| `experiments/omni_transformer_stage_t_preserve_latent.py` | Stage T 保信息 latent 压缩器实验：raw expert token、route-weighted token、summary token、32-slot resampler 对照与 reconstruction probe |
| `experiments/omni_transformer_stage_u_object_slots.py` | Stage U 扩展视觉输入、对象/空间/计数专家与输出 token sweep 实验：64 patch tokens、object slots、spatial/counting experts、no-patch/no-object 消融 |
| `experiments/omni_transformer_stage_v_expert_alignment.py` | Stage V 专家语义空间对齐与信息丢失诊断实验：共享 semantic decoder 监督、单 expert 信息还原、跨 expert transfer 语言一致性测试 |
| `experiments/omni_transformer_stage_w_alignment_mechanisms.py` | Stage W 四种专家对齐机制实验：共享 semantic decoder、slot targets、contrastive alignment、训练期 common semantic bus，推理期直出 latent |
| `experiments/omni_transformer_stage_x_decomposed_diagnostics.py` | Stage X 分解诊断实验：冻结 expert tokens 后分别测试信息存在、对象化、跨 expert 互读、参数/Transformer 读头和最终 scorer 瓶颈 |
| `experiments/omni_transformer_stage_y_parallel_input_experts.py` | Stage Y 并行直读输入专家架构修正实验：对比旧串联、并行直读图像专家、合并视觉输入专家，并把 latent 推理从输入专家中拆出 |
| `experiments/omni_transformer_stage_z_supervised_teacher_diagnostics.py` | Stage Z 显式监督与训练期教师实验：object/spatial/count 专家监督、teacher distillation、信息丢失和潜变量互读诊断 |
| `experiments/omni_transformer_stage_aa_token_alignment_diagnostics.py` | Stage AA token 级对齐实验：teacher/cross-expert same-position contrastive、信息丢失和潜变量互读诊断 |
| `experiments/omni_transformer_stage_ab_text_moe_alignment.py` | Stage AB 文本 latent 对齐、latent-to-answer 输出专家和 MoE 推理实验：对齐 prompt/answer latent、候选答案 latent scorer 与 routed reasoner |
| `experiments/omni_transformer_stage_ac_latent_reasoning.py` | Stage AC/AD/AE/AF/AG/AH Q/A 潜空间、外部信息互译与答案 token 潜空间推理实验：分阶段训练 text codec、evidence codec、latent reasoner，并支持 reasoner-only readout/trace、active-read agent、MoE staged、teacher-forced multi-step query trace、query alignment/process supervision 消融与 no/shuffled evidence 门禁 |
| `experiments/omni_transformer_stage_ai_image_generation_editing.py` | Stage AI 图像生成与源图编辑输出专家实验：prompt direct、latent output、latent image edit、no-source/source-no-edit 消融与 36 小时时间上限 |
| `experiments/omni_transformer_stage_aj_transformer_image_io_fidelity.py` | Stage AJ Transformer 图像输入到 latent 再完整重绘保真实验：patch-token image encoder、fixed latent baseline、记忆树式逐层残差 latent、copy/edit/generation 对象属性/mask 辅助监督、Transformer patch decoder、no-source 消融、批量 GPU template parser、前景/背景分解指标、P0 checkpoint/resume 长训恢复 |
| `experiments/omni_transformer_stage_ak_unified_latent_bus.py` | Stage AK/AL/AM/AN/AQ 统一潜空间硬化实验：pair/object、cell、count latent bus，text query retrieval、decoded position compare、分类 answer writer、answer-token writer、relation delta/truth-table 过程监督、no-evidence 门禁与 AMP/GPU 常驻训练 |
| `experiments/omni_transformer_stage_ao_pixel_to_slots.py` | Stage AO 像素到 cell/count/pair slots 训练前准备实验：合成像素图、pixel cell expert、soft count slots、复用统一 bus/answer-token/relation process、zero-image 门禁与 sweep 聚合 |
| `experiments/omni_transformer_stage_ar_cross_expert_visual.py` | Stage AR 视觉中心跨专家不可单解任务：image、text rule、telemetry、memory 四路输入共同决定答案，含 no-modality 消融、direct baseline、样例 PNG 与 sweep 聚合 |
| `experiments/omni_transformer_stage_as_file_audit.py` | Stage AS 真实文件证据审计实验：生成 HTML/CSV/PDF/截图文件，通过受控工具读取证据，输出 audit conclusion、citation pattern、tool legality 与 no-tool-history 消融 |
| `experiments/omni_transformer_stage_at_memory_exploration.py` | Stage AT 长程局部记忆和探索成本实验：局部遮挡地图、主动 scan/write memory、多目标复用、强 no-memory oracle 对照和错误 memory 修正 |
| `experiments/omni_transformer_stage_av_from_scratch_micro_omni.py` | Stage AV 70M 从零集成 Micro-Omni 实验：单一 Transformer 接 image/text/tool/memory/state，覆盖 AR/AS/AT 子任务、容量 smoke、probe 和消融 |
| `experiments/omni_transformer_stage_avb_staged_latent_training.py` | Stage AV-B 分阶段潜空间训练实验：text base、image translator、tool/memory translator、latent reasoner、joint tuning，GPU 常驻数据、AMP、checkpoint/resume 和 batch smoke |
| `experiments/omni_transformer_stage_avc_dataset_pipeline.py` | Stage AV-C 大规模合成数据流水线：生成 train/val/test/heldout sharded `.pt` 数据集、manifest、token 规模估算和任务分布统计 |
| `experiments/omni_transformer_stage_ave_bidirectional_latent_external.py` | Stage AV-E 潜变量与外部表征双向互译实验：source/target external、latent edit、source->target、target->source、reconstruction、answer 和 heldout probe |
| `experiments/omni_transformer_stage_avf_10m_bidirectional_dataset.py` | Stage AV-F 10M 级双向互译数据集生成器：按 AV-D 1B-first 设计定向裁剪 source/target external pairs，输出 AV-E 可读 manifest 和 tensor shards |
| `experiments/omni_transformer_stage_avg_strict_staged_10m_training.py` | Stage AV-G 严格分阶段 10M 训练入口：text latent base、external codec、bidirectional translation、latent reason 和 short joint debug，输出独立 AV-G checkpoint |
| `experiments/omni_transformer_stage_avh_text_anchored_visual_edit_dataset.py` | Stage AV-H 文本锚定视觉编辑数据集生成器：source text/source image/edit instruction 到 target text/target record/target image，训练时由 record 渲染图像 |
| `experiments/omni_transformer_stage_avh_text_anchored_visual_edit_training.py` | Stage AV-H 10M 训练入口：text latent、image grounding、edit reason、image output、joint debug 五阶段训练，含 68M 配置、AMP、GPU 常驻数据、micro-batch 累积、Adafactor/AdamW 开关和 checkpoint/resume |
| `experiments/omni_transformer_stage_avi_whole_image_latent_capacity.py` | Stage AV-I 整图 latent token 容量诊断：不切 patch，整图 encoder 到 latent token bank，透明背景 RGBA 重建，ordered prefix 按短到长 curriculum 解锁，支持拆分 `latent_dim`、`encoder_width`、`decoder_width`，并用 schema v5 correct-pixel compact guidance 与 residual routing 约束前序正确像素、错误像素惩罚和后续 token 残差分工 |
| `src/latent_space_agent_feasibility/` | 纯 Python 原型，实现 LOD、注意力泵、主动采样、记忆树节点/关联边、KV 分支剪枝、离线对齐与渐进式生成计划 |
| `tests/` | 可运行验证用例，覆盖白皮书核心命题的接口和不变量 |
| `tests/test_stage_aj_checkpointing.py` | Stage AJ P0 checkpoint/resume 辅助逻辑测试，覆盖 runtime 控制参数兼容和显式 CPU 设备选择 |
| `artifacts/text_to_latent_thought/results.json` | 训练实验结果；运行脚本后生成，不保存模型 checkpoint |
| `artifacts/text_to_latent_thought/sweep_results.json` | move=6/8/10、3 seeds 的聚合结果 |
| `artifacts/text_to_latent_thought/sweep_runs/` | sweep 的每次 run 明细 JSON |
| `artifacts/multimodal_latent_pipeline/result.json` | 连续潜变量管线单次正式结果 |
| `artifacts/multimodal_latent_pipeline/sweep_results.json` | 连续潜变量管线 move=8/10/12、3 seeds 的聚合结果 |
| `artifacts/multimodal_latent_pipeline/sweep_runs/` | 连续潜变量管线 sweep 的每次 run 明细 JSON |
| `artifacts/multimodal_latent_comparison/sweep_results.json` | 连续潜变量管线 6 组 baseline/消融对照聚合结果 |
| `artifacts/multimodal_latent_comparison/sweep_runs/` | baseline/消融对照的每次 run 明细 JSON |
| `artifacts/heterogeneous_latent_input/sweep_results.json` | 异构输入实验 move=8/12/16、3 seeds 的聚合结果 |
| `artifacts/heterogeneous_latent_input/sweep_runs/` | 异构输入实验的每次 run 明细 JSON |
| `artifacts/heterogeneous_baseline_comparison/sweep_results.json` | 异构输入同等信息 baseline 对比的聚合结果 |
| `artifacts/heterogeneous_baseline_comparison/sweep_runs/` | 异构输入同等信息 baseline 对比的每次 run 明细 JSON |
| `artifacts/visual_multimodal_stage_ab/sweep_results.json` | 真实像素输入 Stage A/B 多 seed 聚合结果 |
| `artifacts/visual_multimodal_stage_ab/sweep_runs/` | 真实像素输入 Stage A/B 每次 run 明细 JSON |
| `artifacts/visual_multimodal_stage_ab/sweep_runs/samples/` | Stage A/B 渲染样例 PNG，验证输入确实是像素图像 |
| `artifacts/visual_multimodal_stage_c/sweep_results.json` | 真实像素输入 Stage C 主动探测多 seed 聚合结果 |
| `artifacts/visual_multimodal_stage_c/sweep_runs/` | 真实像素输入 Stage C 每次 run 明细 JSON |
| `artifacts/visual_multimodal_stage_c/sweep_runs/samples/` | Stage C 随机符号地图样例 PNG |
| `artifacts/visual_multimodal_stage_d/sweep_results.json` | 真实像素输入 Stage D 局部可见/legend 记忆多 seed 聚合结果 |
| `artifacts/visual_multimodal_stage_d/sweep_runs/` | 真实像素输入 Stage D 每次 run 明细 JSON |
| `artifacts/visual_multimodal_stage_d/sweep_runs/samples/` | Stage D 隐藏全图调试图与局部观察帧条带 PNG |
| `artifacts/multimodal_fusion_latent_flow/sweep_results.json` | Stage E+F 多模态融合与潜空间数据流动多 seed 聚合结果 |
| `artifacts/multimodal_fusion_latent_flow/sweep_runs/` | Stage E+F 每次 run 明细 JSON |
| `artifacts/multimodal_fusion_latent_flow/sweep_runs/*/samples/` | Stage E+F 设备面板图像样例 PNG |
| `artifacts/omni_transformer_stage_h/sweep_results.json` | Stage H Tiny Omni Transformer H1/H2 多 seed 聚合结果 |
| `artifacts/omni_transformer_stage_h/sweep_runs/` | Stage H H1/H2 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_h/sweep_runs/*/samples/` | Stage H 设备面板图像样例 PNG |
| `artifacts/omni_transformer_stage_h3_h4/sweep_results.json` | Stage H3/H4 latent 容量曲线与 OOD 边界多 seed 聚合结果 |
| `artifacts/omni_transformer_stage_h3_h4/sweep_runs/` | Stage H3/H4 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_h3_h4/sweep_runs/*/samples/` | Stage H4 训练风格与 OOD 风格设备面板样例 PNG |
| `artifacts/omni_transformer_stage_h5/sweep_results.json` | Stage H5 视觉增强与可组合规则泛化多 seed 聚合结果 |
| `artifacts/omni_transformer_stage_h5/sweep_runs/` | Stage H5 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_h5/sweep_runs/*/samples/` | Stage H5 增强风格与 held-out 风格设备面板样例 PNG |
| `artifacts/omni_transformer_stage_h6/sweep_results.json` | Stage H6 多输出格式多 seed 聚合结果 |
| `artifacts/omni_transformer_stage_h6/sweep_runs/` | Stage H6 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_i_agent/sweep_results.json` | Stage I tool-using agent 多 seed 聚合结果 |
| `artifacts/omni_transformer_stage_i_agent/sweep_runs/` | Stage I 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_i_agent/sweep_runs/*/samples/` | Stage I agent 面板图像样例 PNG |
| `artifacts/omni_transformer_stage_jk_audit_ui/sweep_results.json` | Stage J/K 多模态证据审计与 UI+DOM agent 多 seed 聚合结果 |
| `artifacts/omni_transformer_stage_jk_audit_ui/sweep_runs/` | Stage J/K 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_jk_audit_ui/sweep_runs/*/samples/` | Stage J/K 证据审计图像与 UI 截图样例 PNG |
| `artifacts/omni_transformer_stage_l_docvqa/sweep_results.json` | Stage L DocVQA 真实文档问答多 seed 聚合结果 |
| `artifacts/omni_transformer_stage_l_docvqa/sweep_runs/` | Stage L 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_l_docvqa/sweep_runs/*/samples/` | Stage L 真实 DocVQA 缩略图样例与候选 OCR 行 JSON |
| `artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_results.json` | Stage M Tiny MoE 多模态 LLM 多 seed 聚合结果，含 teacher-forced、greedy、消融与 latent probe |
| `artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_runs/` | Stage M 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_runs/*/samples/` | Stage M 合成多模态图像网格与样本 prompt/answer/route JSON |
| `artifacts/omni_transformer_stage_n_strong_experts/sweep_results.json` | Stage N 强专家 MoE 与 baseline 成本对比多 seed 聚合结果，含准确率、训练秒数、预测 ms/example、参数量和消融 |
| `artifacts/omni_transformer_stage_n_strong_experts/sweep_runs/` | Stage N 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_n_strong_experts/sweep_runs/*/samples/` | Stage N 合成多模态图像网格与样本 prompt/answer JSON |
| `artifacts/omni_transformer_stage_o_pretrained_experts/sweep_results.json` | Stage O 真实预训练 CLIP 专家多 seed 聚合结果，含 zero-shot、classifier、direct/latent decoder、专家编码成本、参数量和消融 |
| `artifacts/omni_transformer_stage_o_pretrained_experts/sweep_runs/` | Stage O 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_o_pretrained_experts/sweep_runs/*/samples/` | Stage O 合成多模态图像网格与样本 prompt/answer JSON |
| `artifacts/omni_transformer_stage_p_llava_moe/data/` | Stage P LLaVA JSON、COCO 图像和 CLIP token feature 本地缓存；用于避免重复下载/编码，不保存训练 checkpoint |
| `artifacts/omni_transformer_stage_p_llava_moe/sweep_results.json` | Stage P LLaVA-Instruct-150K MoE/latent 多 seed 聚合结果，含 CLIP zero-shot、字符输出头、候选 scorer、消融、参数量和预测成本 |
| `artifacts/omni_transformer_stage_p_llava_moe/sweep_runs/` | Stage P 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_p_llava_moe/sweep_runs/*/samples/` | Stage P COCO 图像样例网格与 LLaVA prompt/answer 样本 JSON |
| `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/data/` | Stage Q LLaVA JSON、COCO 图像和从零实验 pixel tensor 本地缓存；用于避免重复下载/像素预处理，不保存训练 checkpoint |
| `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_results.json` | Stage Q 从零 Tiny MoE-VLM 多 seed 聚合结果，含 text-only/direct/mean-pool/MoE latent、消融、参数量、训练秒数和预测成本 |
| `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_runs/` | Stage Q 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_runs/*/samples/` | Stage Q COCO 图像样例网格与 LLaVA prompt/answer 样本 JSON |
| `artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_results.json` | Stage R 低熵强监督 curriculum VLM 多 seed 聚合结果，含每个 level 的 random/direct/MoE/no-image、visual gap 和能力上限 |
| `artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_runs/` | Stage R 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_runs/*/*/samples/` | Stage R 各难度 level 的合成视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_results.json` | Stage S scorer 与信息还原诊断多 seed 聚合结果，含 cross+pump、cross+no-pump、no-image 和 reconstruction probe |
| `artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_runs/` | Stage S 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_runs/*/*/samples/` | Stage S 各难度 level 的合成视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_t_preserve_latent/sweep_results.json` | Stage T 保信息 latent 压缩器多 seed 聚合结果，含 wide residual、32-slot resampler、no-image 和 reconstruction probe |
| `artifacts/omni_transformer_stage_t_preserve_latent/sweep_runs/` | Stage T 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_t_preserve_latent/sweep_runs/*/*/samples/` | Stage T 各难度 level 的合成视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_u_object_slots/sweep_results.json` | Stage U 扩展视觉输入、对象专家与输出 token sweep 多 seed 聚合结果，含 patch-wide、object experts、no-image、no-patch、no-object 消融、参数量和预测成本 |
| `artifacts/omni_transformer_stage_u_object_slots/sweep_runs/` | Stage U 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_u_object_slots/sweep_runs/*/*/samples/` | Stage U 多物体合成视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_v_expert_alignment/sweep_results.json` | Stage V 专家语义空间对齐与信息丢失诊断多 seed 聚合结果，含 ranking-only/supervised、信息还原率、信息丢失率、cross-expert transfer 和 language gap |
| `artifacts/omni_transformer_stage_v_expert_alignment/sweep_runs/` | Stage V 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_v_expert_alignment/sweep_runs/*/samples/` | Stage V 多物体语义监督样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_w_alignment_mechanisms/sweep_results.json` | Stage W 四种专家对齐机制多 seed 聚合结果，含 ranking-only、shared decoder、slot targets、contrastive、训练期 common bus、消融和跨 expert 诊断 |
| `artifacts/omni_transformer_stage_w_alignment_mechanisms/sweep_runs/` | Stage W 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_w_alignment_mechanisms/sweep_runs/*/samples/` | Stage W 多物体语义监督样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_x_decomposed_diagnostics/sweep_results.json` | Stage X 分解诊断多 seed 聚合结果，含 semantic transfer、answer reconstruction、object table、frozen scorer 与读头参数对比 |
| `artifacts/omni_transformer_stage_x_decomposed_diagnostics/sweep_runs/` | Stage X 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_x_decomposed_diagnostics/sweep_runs/*/samples/` | Stage X 多物体语义监督样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_y_parallel_input_experts/sweep_results.json` | Stage Y 并行直读输入专家多 seed 聚合结果，含 serial/parallel/merged 对照、消融、参数量、预测成本和 answer reconstruction |
| `artifacts/omni_transformer_stage_y_parallel_input_experts/sweep_runs/` | Stage Y 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_y_parallel_input_experts/sweep_runs/*/samples/` | Stage Y 多物体视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/sweep_results.json` | Stage Z 显式监督与训练期教师多 seed 聚合结果，含专家消融、训练头指标、信息丢失、semantic/count transfer 和成本 |
| `artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/sweep_runs/` | Stage Z 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/sweep_runs/*/samples/` | Stage Z 多物体视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_aa_token_alignment_diagnostics/sweep_results.json` | Stage AA token 级对齐多 seed 聚合结果，含 token 检索、信息丢失、semantic/count transfer、消融和成本 |
| `artifacts/omni_transformer_stage_aa_token_alignment_diagnostics/sweep_runs/` | Stage AA 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_aa_token_alignment_diagnostics/sweep_runs/*/samples/` | Stage AA 多物体视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_ab_text_moe_alignment/sweep_results.json` | Stage AB 文本 latent 对齐与 MoE 推理多 seed 聚合结果，含 latent-to-answer、prompt-answer retrieval、MoE gate、消融和成本 |
| `artifacts/omni_transformer_stage_ab_text_moe_alignment/sweep_runs/` | Stage AB 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ab_text_moe_alignment/sweep_runs/*/samples/` | Stage AB 多物体视觉问答样例 PNG 与 samples.json |
| `artifacts/omni_transformer_stage_ac_latent_reasoning/sweep_results.json` | Stage AC Q/A 潜空间、外部信息互译与答案 token 潜空间推理单 seed 聚合结果，含 text/evidence codec 保真、latent reasoner、no-evidence 与 shuffled-evidence 消融 |
| `artifacts/omni_transformer_stage_ac_latent_reasoning/sweep_runs/` | Stage AC 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ad_reasoner_trace/color_readout_reader_results.json` | Stage AD color-only reasoner-only readout/trace 聚合结果，含冻结 codec、reader 还原、no/shuffled evidence 消融 |
| `artifacts/omni_transformer_stage_ad_reasoner_trace/color_readout_reader_runs/` | Stage AD color-only 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ad_reasoner_trace/all_readout_reader_results.json` | Stage AD 四任务 reasoner-only readout/trace 聚合结果，含按任务族 answer exact、trace 与 reader 指标 |
| `artifacts/omni_transformer_stage_ad_reasoner_trace/all_readout_reader_runs/` | Stage AD 四任务每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ae_active_read/cell_lookup_results.json` | Stage AE cell lookup 主动读取聚合结果，含 cell query、reader 和 no/shuffled evidence 消融 |
| `artifacts/omni_transformer_stage_ae_active_read/cell_lookup_runs/` | Stage AE cell lookup 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ae_active_read/count_only_results.json` | Stage AE count-only 主动读取聚合结果，含 count-pair query、count trace 与证据依赖消融 |
| `artifacts/omni_transformer_stage_ae_active_read/count_only_runs/` | Stage AE count-only 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ae_active_read/relation_only_results.json` | Stage AE relation-only 主动读取聚合结果，含 left/right object query、relation-op query 与 pair reader 指标 |
| `artifacts/omni_transformer_stage_ae_active_read/relation_only_runs/` | Stage AE relation-only 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ae_active_read/all_active_results.json` | Stage AE 四任务 active-read agent 聚合结果，含 query policy、reader、no/shuffled evidence 和按任务族指标 |
| `artifacts/omni_transformer_stage_ae_active_read/all_active_runs/` | Stage AE 四任务 active-read 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ae_active_read/all_active_trace2_results.json` | Stage AE 高 trace 权重诊断聚合结果，用于验证单纯加大 trace loss 不能修复 query policy |
| `artifacts/omni_transformer_stage_ae_active_read/all_active_trace2_runs/` | Stage AE 高 trace 权重诊断每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_af_moe_staged/all_staged_results.json` | Stage AF 四任务 MoE staged 聚合结果，含 gate 塌缩、遗忘和按任务族准确率 |
| `artifacts/omni_transformer_stage_af_moe_staged/all_staged_runs/` | Stage AF 四任务 MoE staged 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_af_moe_staged/all_mixed_results.json` | Stage AF 四任务 MoE mixed 聚合结果，含 supervised gate、reader 和 no/shuffled evidence 消融 |
| `artifacts/omni_transformer_stage_af_moe_staged/all_mixed_runs/` | Stage AF 四任务 MoE mixed 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_af_moe_staged/all_staged_replay4_results.json` | Stage AF 四任务 MoE staged+replay 聚合结果，用于验证 replay 对遗忘的缓解效果 |
| `artifacts/omni_transformer_stage_af_moe_staged/all_staged_replay4_runs/` | Stage AF 四任务 MoE staged+replay 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_af_moe_staged/relation_only_moe_results.json` | Stage AF relation-only MoE 聚合结果，定位 relation query policy 仍未闭合 |
| `artifacts/omni_transformer_stage_af_moe_staged/relation_only_moe_runs/` | Stage AF relation-only MoE 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ag_teacher_forced_trace/smoke_results.json` | Stage AG trace_multistep smoke 聚合结果，用于验证 teacher-forced query trace 分支可执行 |
| `artifacts/omni_transformer_stage_ag_teacher_forced_trace/smoke_runs/` | Stage AG smoke 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ag_teacher_forced_trace/relation_only_results.json` | Stage AG relation-only teacher-forced multi-step query trace 聚合结果，显示 relation compare/answer 写入仍未闭合 |
| `artifacts/omni_transformer_stage_ag_teacher_forced_trace/relation_only_runs/` | Stage AG relation-only 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ag_teacher_forced_trace/all_trace_results.json` | Stage AG 四任务 teacher-forced multi-step query trace 聚合结果，含 full/no/shuffled 与 teacher-forced query 上限 |
| `artifacts/omni_transformer_stage_ag_teacher_forced_trace/all_trace_runs/` | Stage AG 四任务每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/smoke_results.json` | Stage AH query/process 新指标 smoke 聚合结果 |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/smoke_runs/` | Stage AH smoke 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_query_alignment_results.json` | Stage AH relation-only CLIP-style query alignment 聚合结果，显示 query 小幅改善但 reader 被破坏 |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_query_alignment_runs/` | Stage AH query alignment 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_process_results.json` | Stage AH relation-only process supervision 聚合结果，显示 teacher-forced compare 上限改善 |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_process_runs/` | Stage AH process supervision 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_combined_results.json` | Stage AH relation-only query alignment + process supervision 聚合结果，显示组合未叠加收益 |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_combined_runs/` | Stage AH combined 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_query_alignment_detached_results.json` | Stage AH detached-key query alignment 聚合结果，显示固定 evidence key 可保住 reader 但 query 改善不足 |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/relation_query_alignment_detached_runs/` | Stage AH detached-key query alignment 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/all_process_results.json` | Stage AH 四任务 process-only 聚合结果，显示 process supervision 对 full 和 teacher-forced 上限的影响 |
| `artifacts/omni_transformer_stage_ah_query_process_ablation/all_process_runs/` | Stage AH 四任务 process-only 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ak_unified_latent_bus/smoke_results.json` | Stage AK 统一 latent bus smoke 聚合结果，用于验证 object bus、query retrieval、compare 和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_ak_unified_latent_bus/smoke_runs/` | Stage AK smoke 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_ak_unified_latent_bus/relation_results.json` | Stage AK relation-only 统一 latent bus 聚合结果，显示 pair readout、query retrieval 和 compare 同时闭合 |
| `artifacts/omni_transformer_stage_ak_unified_latent_bus/relation_runs/` | Stage AK relation-only 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_al_unified_bus_answer/smoke_results.json` | Stage AL 统一 latent bus 接答案输出头 smoke 聚合结果，用于验证 answer writer 指标和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_al_unified_bus_answer/smoke_runs/` | Stage AL smoke 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_al_unified_bus_answer/relation_answer_results.json` | Stage AL relation-only 统一 latent bus + answer writer 聚合结果，显示 model-selected answer 闭合 |
| `artifacts/omni_transformer_stage_al_unified_bus_answer/relation_answer_runs/` | Stage AL relation-only answer writer 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_am_hardest_unified_bus/count_expert_only_results.json` | Stage AM count-only 诊断聚合结果，显示一等 count 输入专家能把 count table、selected count 和 count answer 全部闭合 |
| `artifacts/omni_transformer_stage_am_hardest_unified_bus/count_expert_only_runs/` | Stage AM count-only 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_am_hardest_unified_bus/position_compare_all_task_results.json` | Stage AM 历史最高难度四任务聚合结果，显示 cell/count/pair 统一 bus + decoded position compare 后 all-task answer 闭合 |
| `artifacts/omni_transformer_stage_am_hardest_unified_bus/position_compare_all_task_runs/` | Stage AM 最终 all-task 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_am_hardest_unified_bus/*_smoke_results.json` | Stage AM 各结构改动 smoke 结果，用于验证 count slot、count 输入专家和 position compare 的 JSON 输出链路 |
| `artifacts/omni_transformer_stage_am_hardest_unified_bus/*_all_task_results.json` | Stage AM 失败/对照 all-task 聚合结果，用于比较 pair-head count、softmax count、additive count、compare 权重和 position compare |
| `artifacts/omni_transformer_stage_an_answer_token_writer/smoke_results.json` | Stage AN answer-token writer smoke 聚合结果，用于验证 token writer 训练、sequence exact 指标和聚合 JSON 输出链路 |
| `artifacts/omni_transformer_stage_an_answer_token_writer/smoke_runs/` | Stage AN smoke 每次 run 明细 JSON |
| `artifacts/omni_transformer_stage_an_answer_token_writer/formal_results.json` | Stage AN 正式 3 seed 训练聚合结果；answer-token sequence exact 98.11%，relation sequence exact 92.45%，no-evidence sequence exact 27.93% |
| `artifacts/omni_transformer_stage_an_answer_token_writer/formal_runs/` | Stage AN 正式训练每次 seed 明细 JSON，包含训练轨迹、full/no-evidence 指标和耗时 |
| `artifacts/omni_transformer_stage_aq_relation_process_supervision/formal_results.json` | Stage AQ v1 正式 3 seed 失败结果；learned truth-table 未稳，relation sequence exact 89.32% |
| `artifacts/omni_transformer_stage_aq_relation_process_supervision/formal_results_v2.json` | Stage AQ v2 正式 3 seed 中间结果；deterministic truth-table 已可读但弱注入不足，relation sequence exact 96.61% |
| `artifacts/omni_transformer_stage_aq_relation_process_supervision/formal_results_v3.json` | Stage AQ v3 正式 3 seed 通过结果；truth-table 强注入后 relation sequence exact 99.74%，truth-table 100%，no-evidence truth-table 50.52% |
| `artifacts/omni_transformer_stage_aq_relation_process_supervision/formal_runs_v3/` | Stage AQ v3 每个 seed 明细 JSON，包含 delta、truth-table、answer-token 与 no-evidence 指标 |
| `artifacts/omni_transformer_stage_aq_relation_process_supervision/smoke_results*.json` | Stage AQ 各版本 smoke 结果，用于验证过程监督字段、truth-table 计算和聚合 JSON 输出链路 |
| `artifacts/omni_transformer_stage_ao_pixel_to_slots/smoke_results.json` | Stage AO 像素到 slots 2-step smoke 聚合结果，用于验证训练循环、zero-image 消融和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_ao_pixel_to_slots/smoke_runs/` | Stage AO smoke 每次 seed 明细 JSON |
| `artifacts/omni_transformer_stage_ao_pixel_to_slots/probe_results.json` | Stage AO 80-step 单 seed 短 probe 聚合结果；颜色/形状与 cell slots 有学习信号，但 count table 仍弱 |
| `artifacts/omni_transformer_stage_ao_pixel_to_slots/probe_runs/` | Stage AO 短 probe 每次 seed 明细 JSON |
| `artifacts/omni_transformer_stage_ao_pixel_to_slots/formal_results.json` | Stage AO 正式 3 seed 聚合结果；cell/color/shape、learned count table、count answer 和全任务 answer 闭合，pixel soft count 诊断 81.45% 未过 95% |
| `artifacts/omni_transformer_stage_ao_pixel_to_slots/formal_runs/` | Stage AO 正式训练每个 seed 明细 JSON，包含 full/no-image 指标、训练轨迹和耗时 |
| `artifacts/omni_transformer_stage_ar_cross_expert_visual/smoke_results.json` | Stage AR 视觉中心跨专家任务 smoke 聚合结果，用于验证脚本、样例 PNG、消融评估和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_ar_cross_expert_visual/probe_results_v2.json` | Stage AR 修正 text-rule DSL parser 后的单 seed probe 结果；latent full 100%，四路缺模态消融均显著下降 |
| `artifacts/omni_transformer_stage_ar_cross_expert_visual/formal_results.json` | Stage AR 正式 3 seed 聚合结果；latent full 100%，no-image/no-text-rule/no-telemetry/no-memory 分别降到 23.96%/25.07%/32.10%/30.21%，direct baseline 51.04% |
| `artifacts/omni_transformer_stage_ar_cross_expert_visual/formal_runs/` | Stage AR 正式训练每个 seed 明细 JSON、训练轨迹、消融指标和样例 PNG/JSON |
| `artifacts/omni_transformer_stage_as_file_audit/smoke_results.json` | Stage AS 真实文件证据审计 smoke 聚合结果，用于验证文件生成、工具读取、训练循环和 JSON/PNG 输出链路 |
| `artifacts/omni_transformer_stage_as_file_audit/probe_results.json` | Stage AS 单 seed probe 聚合结果；conclusion/citation/report/tool legality 均 100%，no-tool-history 回到 25% |
| `artifacts/omni_transformer_stage_as_file_audit/formal_results.json` | Stage AS 正式 3 seed 聚合结果；conclusion/citation/report/tool legality 均 100%，no-tool-history conclusion 25%，gap 75pp |
| `artifacts/omni_transformer_stage_as_file_audit/formal_runs/` | Stage AS 正式训练每个 seed 明细 JSON、训练轨迹、样例 HTML/CSV/PDF/截图文件和 sample audit JSON |
| `artifacts/omni_transformer_stage_at_memory_exploration/smoke_results.json` | Stage AT 长程局部记忆 smoke 聚合结果，用于验证训练循环、rollout、样例地图 PNG 和 trace 输出链路 |
| `artifacts/omni_transformer_stage_at_memory_exploration/probe_v8_results.json` | Stage AT 首轮主动建图策略的单 seed probe 结果；full success 100%，no-memory oracle success 0%，scan 4 vs 10 |
| `artifacts/omni_transformer_stage_at_memory_exploration/formal_results.json` | Stage AT 正式 3 seed 聚合结果；episode success 100%，no-memory oracle 0%，scan reduction 6.00，cost reduction 9.96，corrupt-memory correction 99.80% |
| `artifacts/omni_transformer_stage_at_memory_exploration/formal_runs/` | Stage AT 正式训练每个 seed 明细 JSON、训练轨迹、rollout 指标、样例局部地图 PNG 和 memory/no-memory/corrupt trace |
| `artifacts/omni_transformer_stage_av_from_scratch_micro_omni/smoke_results.json` | Stage AV 小模型 smoke 聚合结果，用于验证脚本、训练循环、样例 PNG 和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_av_from_scratch_micro_omni/larger_smoke_results.json` | Stage AV 70M 容量 smoke 结果；参数量 70,994,707，RTX 4070 Laptop 8GB 上峰值 CUDA allocated 约 1,375 MB |
| `artifacts/omni_transformer_stage_av_from_scratch_micro_omni/probe_70m_residual_results.json` | Stage AV 70M residual evidence readout probe 结果；overall 64.32%，memory 100%，file-tool 69.53%，visual 23.44%，未过正式门槛 |
| `artifacts/omni_transformer_stage_av_from_scratch_micro_omni/probe_70m_residual_runs/` | Stage AV 70M probe 每 seed 明细 JSON、训练轨迹、样例 PNG 和 samples.json |
| `artifacts/omni_transformer_stage_avb_staged_latent_training/smoke_result.json` | Stage AV-B 小模型 smoke 结果，用于验证分阶段训练、checkpoint、样例和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_avb_staged_latent_training/large_smoke_result.json` | Stage AV-B 70M smoke 结果；参数量 71,025,455，峰值 CUDA allocated 约 1,505 MB |
| `artifacts/omni_transformer_stage_avb_staged_latent_training/large_batch128_smoke_result.json` | Stage AV-B 70M batch 128 smoke 结果；峰值 CUDA allocated 约 2,184 MB |
| `artifacts/omni_transformer_stage_avb_staged_latent_training/large_batch256_smoke_result.json` | Stage AV-B 70M batch 256 smoke 结果；峰值 CUDA allocated 约 3,095 MB，用于长训 batch 起点 |
| `artifacts/omni_transformer_stage_avb_staged_latent_training/long_result.json` | Stage AV-B 用户本地 long probe 结果；overall 74.61%，tool 95.01%，memory 100%，visual 28.95%，no-image/shuffled-image 不降，未过门槛 |
| `artifacts/omni_transformer_stage_avb_staged_latent_training/*_checkpoints/` | Stage AV-B smoke checkpoint 目录，验证 `latest.pt` 写入与 `--resume` 路径 |
| `artifacts/omni_transformer_stage_avc_dataset_pipeline/smoke_dataset/manifest.json` | Stage AV-C sharded dataset smoke manifest；1,408 examples、88,704 tokens、train/val/test/heldout split 和任务分布 |
| `artifacts/omni_transformer_stage_avc_dataset_pipeline/smoke_dataset/` | Stage AV-C 小规模 sharded 数据集 smoke，包含 `.pt` shard、tokens/answers/tasks tensor 和 heldout split |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/smoke_results.json` | Stage AV-E 双向互译 smoke 聚合结果，用于验证 source/target external、latent edit、heldout、PNG 样例和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/probe_results.json` | Stage AV-E 小 probe 聚合结果；source-to-target token accuracy 28.40%，target-to-source 28.24%，answer 接近随机，未过门槛 |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/probe_runs/` | Stage AV-E probe 每 seed 明细 JSON、训练轨迹、source/target PNG 和 samples.json |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/manifest_smoke_gpu_result.json` | Stage AV-E 读取 AV-F smoke manifest 的 CUDA smoke 结果，验证 manifest 训练入口、checkpoint 和 batched eval |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/dataset10m_loader_smoke_gpu_result.json` | Stage AV-E 读取 AV-F 10M manifest 的 CUDA loader smoke 结果，验证完整 train 分片可读 |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/dataset10m_70m_capacity_smoke_result.json` | Stage AV-E 70M batch 256 capacity smoke；75,970,770 参数，峰值 CUDA allocated 约 4,753.32 MB |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/dataset10m_70m_batch512_capacity_smoke_result.json` | Stage AV-E 70M batch 512 capacity smoke；75,970,770 参数，峰值 CUDA allocated 约 8,923.00 MB |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_result.json` | Stage AV-E 10M joint baseline 长训结果；answer test 97.25%，但 source/target recon 与双向 translation sequence exact 全为 0，判为 answer shortcut 负结果 |
| `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_checkpoints/` | Stage AV-E 10M joint baseline `latest.pt`/`best.pt` checkpoint 目录 |
| `artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/smoke_dataset/manifest.json` | Stage AV-F 双向互译数据集 smoke manifest；896 examples、49,280 pair tokens，用于验证 AV-E manifest 训练入口 |
| `artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m/manifest.json` | Stage AV-F 10M 级双向互译数据集 manifest；184,000 examples、10,120,000 pair tokens、train unique pair tokens 8,800,000 |
| `artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m/` | Stage AV-F 10M 级 sharded `.pt` 数据集，包含 source_tokens、target_tokens、answers、ops 和 target_zones |
| `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/strict_cpu_smoke_result.json` | Stage AV-G 严格分阶段 CPU smoke，验证 5 个 stage 的 loss、manifest IO、checkpoint 和 samples 输出，不占用正在跑的 CUDA 长训 |
| `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/strict_cpu_resume_smoke_result.json` | Stage AV-G CPU resume smoke，验证 `latest.pt` 恢复后继续 strict staged schedule |
| `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/strict_cpu_smoke_checkpoints/` | Stage AV-G CPU smoke checkpoint 目录，验证 `latest.pt` 恢复链；未来 70M 长训应另用 `train_10m_70m_strict_checkpoints` |
| `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_result.json` | Stage AV-G 10M strict staged 长训结果；answer test 77.85%，但 source/target recon 与双向 translation sequence exact 全为 0，说明 strict schedule 未修复 external codec |
| `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_checkpoints/` | Stage AV-G 10M strict staged `latest.pt`/`best.pt` checkpoint 目录 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/smoke_dataset/manifest.json` | Stage AV-H 文本锚定视觉编辑 smoke dataset manifest；896 examples、410,368 tokens，用于验证 dataset schema、shards 和样例渲染 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/dataset_10m/manifest.json` | Stage AV-H 10M 文本锚定视觉编辑数据集 manifest；21,900 examples、10,030,200 tokens、train unique tokens 8,244,000 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/dataset_10m/samples/` | Stage AV-H 10M 数据集 source/target PNG 和样例 JSON，用于人工检查 source record、edit instruction 与 target record/image 是否一致 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/smoke_result.json` | Stage AV-H 小模型 CPU training smoke，验证五阶段 loss、manifest IO、checkpoint 和 samples 输出链路 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/dataset10m_loader_smoke_result.json` | Stage AV-H 读取完整 10M manifest 的 loader smoke，验证 train/val/test/heldout 分片可读 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/dataset10m_68m_capacity_smoke_result.json` | Stage AV-H 68M batch 96 CUDA capacity smoke；68,416,419 参数，峰值 CUDA allocated 约 5,004.43 MB |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/dataset10m_68m_batch128_capacity_smoke_result.json` | Stage AV-H 68M batch 128 CUDA capacity smoke；68,416,419 参数，峰值 CUDA allocated 约 6,537.45 MB，作为建议长训起点 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/perf_fix_micro_cpu_smoke_result.json` | Stage AV-H micro-batch 性能修复后 CPU smoke，验证梯度累积、stage 输出头裁剪和 JSON 输出链路 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/dataset10m_68m_batch128_micro64_perf_fix_smoke_result.json` | Stage AV-H 性能修复后推荐 CUDA smoke；effective batch 128、micro batch 64、Adafactor，峰值 CUDA allocated 约 4,893.83 MB，训练窗口 1.52 step/s |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/dataset10m_68m_batch128_micro96_perf_fix_smoke_result.json` | Stage AV-H micro batch 96 对照 smoke；峰值 CUDA allocated 约 6,696.90 MB，训练窗口慢于 micro 64，不作为默认长训配置 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_result.json` | Stage AV-H 10M 68M 长训结果；旧 joint_debug 口径 target record exact 高但含 target_record teacher-forcing 泄漏，不能作为端到端通过证据 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_stage_mode_diagnostic.json` | Stage AV-H 10M 长训后补非泄漏诊断；text_latent、image_ground、edit_reason、image_output、joint_teacher_record 和 joint_no_target_record 分模式评估 |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/nonleaky_eval_cpu_smoke_result.json` | Stage AV-H 训练脚本修正非泄漏多模式评估后的 CPU smoke，用于验证主指标不再默认使用 target_record teacher-forcing |
| `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/samples/contact_sheet.png` | Stage AV-H 10M 长训样例 source/target/pred contact sheet；显示预测图有弱颜色/位置但明显模糊，MSE 不能单独作为图像闭合证据 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/smoke_result.json` | Stage AV-I 整图 latent 容量 smoke；CPU 2-step 验证 ordered/greedy prefix、整图 decoder 和样例输出链路 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/transparent_curriculum_cuda_smoke_v2_result.json` | Stage AV-I 透明背景 curriculum CUDA smoke；验证 RGBA、AMP、greedy eval 和输出目录不覆盖旧探针 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_100step_v2_result.json` | Stage AV-I 1/2/4/8/12/16/24 token 100-step CUDA probe；暴露短训会过度涂前景，不能判断 12 token 是否足够 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_12_24_1000step_result.json` | Stage AV-I 12 vs 24 token 1000-step 旧 CUDA probe；保留为黑背景/前景指标历史结果，不作为当前透明背景判断 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_result.json` | Stage AV-I 透明背景 12/24/32 token 3000-step 全 train probe；12 token test transparent loss 0.002505、alpha IoU 0.9972，24/32 未优于 12 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_summary.json` | Stage AV-I 透明背景 3000-step probe 汇总；抽取 final metrics、prefix curve、curriculum 和样例 contact sheet 路径 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/latent_dim2_cpu_smoke_result.json` | Stage AV-I `latent_dim=2` CPU smoke；验证 encoder/latent/decoder 宽度拆分和 1/2 token 输出链路 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/compact_prefix_cpu_smoke_v2_result.json` | Stage AV-I compact prefix CPU smoke；验证 active=1 时 compact area/coverage loss 已进入训练 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/precision_compact_cpu_smoke_result.json` | Stage AV-I precision compact CPU smoke；验证 coverage 奖励从 `x` 到动态 `x^n`，n 从 1 增至 4 且 precision coverage 指标进入 history |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/segment_precision_compact_cpu_smoke_result.json` | Stage AV-I segment precision compact CPU smoke；验证每个 active-token 段内 n 都从 0.5 重置到 5，且两端快、中间慢 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/correct_pixel_residual_cpu_smoke_result.json` | Stage AV-I schema v5 correct-pixel residual CPU smoke；验证前景正确像素 compact、wrong background/wrong color/missed foreground 指标、相邻 prefix residual routing loss 和样例输出链路 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/correct_pixel_residual_cpu_smoke_result_latent_2_samples/` | Stage AV-I correct-pixel residual CPU smoke 的 2-token 样例目录，含 target、ordered/greedy prefix PNG 和 `samples.json` |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result.json` | Stage AV-I schema v5 correct-pixel residual 2 维 token 正式 probe；本轮失败，16 token 本轮最佳，64 token 仍只在 alpha IoU 上最高但综合重建和 correct pixel 退化 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_summary.json` | Stage AV-I correct-pixel residual 正式 probe 汇总；记录 soft/hard correct pixel、wrong background、wrong color、missed foreground、与 segment precision compact 的公共指标对比和失败判断 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result.json` | Stage AV-I `latent_dim=2`、1/2/4/8/16/32/64 token 3000-step 全 train 极限 probe；32 token 最好但仍明显 ghost/object 混叠，64 token 退化需等步长复查 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_summary.json` | Stage AV-I 2 维 token 极限 probe 汇总；记录 latent scalars、final metrics、prefix curve、curriculum 和样例 contact sheet 路径 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result.json` | Stage AV-I 2 维 token compact + 7000-step 全 train probe；32 token 最好，test loss 0.044031、alpha IoU 0.9091，64 token 接近但未超过 32 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json` | Stage AV-I 2 维 token compact + 7000-step 汇总；记录 area/coverage 指标、final metrics、prefix curve 和样例 contact sheet 路径 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json` | Stage AV-I 2 维 token precision compact + 7000-step 全 train probe；n 从 1 增到 4，16 token 最好，test loss 0.042794、alpha IoU 0.9117，但 32/64 比线性 compact 变差 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json` | Stage AV-I 2 维 token precision compact 汇总；记录 precision coverage、final metrics、prefix curve、history 中动态 n 和样例 contact sheet 路径 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json` | Stage AV-I 2 维 token segment precision compact + 7000-step 全 train probe；每段 n=0.5->5、两端快中间慢，32 token 综合最好，test loss 0.037481、alpha IoU 0.9127 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json` | Stage AV-I 2 维 token segment precision compact 汇总；记录分段 n history、final metrics、prefix curve、precision coverage 和样例 contact sheet 路径 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/latent_12_samples/ordered_contact_sheet.png` | Stage AV-I 12-token ordered prefix 样例图，展示 prefix 1/2/4/8/12 的整图重建演化 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/latent_24_samples/ordered_contact_sheet.png` | Stage AV-I 24-token ordered prefix 样例图，展示 prefix 1/2/4/8/16/24 的整图重建演化 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_result_latent_12_samples/ordered_contact_sheet_checker.png` | Stage AV-I 12-token 透明背景 ordered prefix 棋盘底样例；显示 1-2 token 已重建大部分结构 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_result_latent_24_samples/ordered_contact_sheet_checker.png` | Stage AV-I 24-token 透明背景 ordered prefix 棋盘底样例；用于对照 12 token 和后续 token 边际收益 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_result_latent_32_samples/ordered_contact_sheet_checker.png` | Stage AV-I 32-token 透明背景 ordered prefix 棋盘底样例；用于观察更长 token bank 未带来明显改进 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result_latent_32_samples/ordered_contact_sheet_checker.png` | Stage AV-I 2 维 32-token ordered prefix 棋盘底样例；当前极窄 latent 最好档，仍有 ghost/object 混叠 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result_latent_64_samples/ordered_contact_sheet_checker.png` | Stage AV-I 2 维 64-token ordered prefix 棋盘底样例；显示长 token bank 在固定 3000 step curriculum 下退化 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_32_samples/ordered_contact_sheet_checker.png` | Stage AV-I compact 2 维 32-token ordered prefix 棋盘底样例；当前 compact 最好档，ghost/object 混叠明显低于旧 3000-step |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples/ordered_contact_sheet_checker.png` | Stage AV-I compact 2 维 64-token ordered prefix 棋盘底样例；显示 64 token 接近但未超过 32，仍有颜色/位置错配 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_16_samples/ordered_contact_sheet_checker.png` | Stage AV-I precision compact 2 维 16-token ordered prefix 棋盘底样例；当前 precision compact 最好档，中等 token 精确性优于线性 compact |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples/ordered_contact_sheet_checker.png` | Stage AV-I precision compact 2 维 64-token ordered prefix 棋盘底样例；显示 n=1->4 对长 token bank 未带来收益，仍有颜色/位置错配 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_32_samples/ordered_contact_sheet_checker.png` | Stage AV-I segment precision compact 2 维 32-token ordered prefix 棋盘底样例；当前综合重建 loss 最好档 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples/ordered_contact_sheet_checker.png` | Stage AV-I segment precision compact 2 维 64-token ordered prefix 棋盘底样例；alpha IoU 最高但颜色/对象误差高于 32 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result_latent_16_samples/ordered_contact_sheet_checker.png` | Stage AV-I correct-pixel residual 2 维 16-token ordered prefix 棋盘底样例；本轮综合指标最佳但低于上一轮 segment precision compact |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result_latent_32_samples/ordered_contact_sheet_checker.png` | Stage AV-I correct-pixel residual 2 维 32-token ordered prefix 棋盘底样例；显示对象收缩、缺失和错色，未超过 16 |
| `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result_latent_64_samples/ordered_contact_sheet_checker.png` | Stage AV-I correct-pixel residual 2 维 64-token ordered prefix 棋盘底样例；显示后续 token 压背景/修 mask 但颜色和对象错误更重 |
| `artifacts/omni_transformer_stage_ai_image_generation_editing/result.json` | Stage AI 单 seed GPU 图像生成/编辑早期结果，保留为正式 sweep 前的参考 |
| `artifacts/omni_transformer_stage_ai_image_generation_editing/smoke_result.json` | Stage AI 小规模 smoke 结果，用于快速验证脚本、GPU、JSON 和 PNG 输出链路 |
| `artifacts/omni_transformer_stage_ai_image_generation_editing/samples/result/` | Stage AI 正式单 seed 的源图、目标图、prompt direct、latent output、latent image edit 和 no-source PNG 样例 |
| `artifacts/omni_transformer_stage_ai_image_generation_editing/samples/smoke_result/` | Stage AI smoke 的独立 PNG 样例，避免覆盖正式结果样例 |
| `artifacts/omni_transformer_stage_ai_image_generation_editing/sweep_results.json` | Stage AI 36 小时内正式 sweep 聚合结果；3 seeds 主任务均 100%，用于证明该任务已饱和 |
| `artifacts/omni_transformer_stage_ai_image_generation_editing/sweep_runs/` | Stage AI 多 seed sweep 每次 run 明细 JSON 与对应样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/result.json` | Stage AJ 中等单 seed Transformer 图像 IO 保真结果，含 copy/edit、foreground/background MSE、nearest scene 和消融 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/smoke_result.json` | Stage AJ 小规模 smoke 结果，用于快速验证脚本、GPU、JSON 和 PNG 输出链路 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/memory_tree_smoke_result.json` | Stage AJ 记忆树式逐层残差 latent smoke 结果，验证 memory_tree_copy/edit 训练、JSON 和 PNG 输出链路 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/supervised_copy_smoke_result.json` | Stage AJ copy-only 辅助监督 smoke 结果，验证 memory_tree_supervised_copy、对象属性/mask 辅助 loss、JSON 和 PNG 输出链路 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/supervised_copy_probe_result.json` | Stage AJ copy-only 辅助监督中等 probe 结果；memory_tree_supervised_copy 达到 scene exact 100%、foreground MSE 0.005730、aux scene/mask IoU 100% |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/formal_supervised_copy_result.json` | Stage AJ copy-only 辅助监督正式单 seed 结果；memory_tree_supervised_copy 达到 scene exact 100%、foreground MSE 0.000794、aux scene/mask IoU 100% |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/supervised_edit_generate_smoke_result.json` | Stage AJ 监督 edit/generation smoke 结果，验证 memory_tree_supervised_edit、text_supervised_generate、no-source 消融和样例输出链路 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/supervised_edit_generate_probe_result.json` | Stage AJ 监督 edit/generation 中等 probe 结果；edit scene exact 99.22%、edit no-source 5.08%、text generation 100% |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/result/` | Stage AJ 中等单 seed 的 background/source/target、transformer copy/edit/no-source PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/smoke_result/` | Stage AJ smoke 的独立 PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/memory_tree_smoke_result/` | Stage AJ memory-tree smoke 的独立 PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/supervised_copy_smoke_result/` | Stage AJ copy-only 辅助监督 smoke 的独立 PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/supervised_copy_probe_result/` | Stage AJ copy-only 辅助监督 probe 的 source/target/copy PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/formal_supervised_copy_result/` | Stage AJ copy-only 辅助监督正式单 seed 的 source/target/copy PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/supervised_edit_generate_smoke_result/` | Stage AJ 监督 edit/generation smoke 的 source/target/edit/generate PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/supervised_edit_generate_probe_result/` | Stage AJ 监督 edit/generation probe 的随机背景 edit、no-source 和规范背景 generation PNG 样例 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/formal_memory_tree_result.json` | Stage AJ 单 seed memory-tree 正式长训结果；edit scene exact 11.91%、no-source 3.71%、copy 0.59%，说明编辑信号增强但完整重绘保真失败 |
| `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/samples/formal_memory_tree_result/` | Stage AJ memory-tree 正式长训 PNG 样例，含 background/source/target、copy/edit/no-source 输出 |
| `artifacts/omni_transformer_stage_ap_formal_edit_generate/formal_edit_generate_result.json` | Stage AP 正式单 seed edit/generation 长训结果；edit scene exact 99.02%、edit no-source 4.10%、text generation 100% |
| `artifacts/omni_transformer_stage_ap_formal_edit_generate/checkpoints/` | Stage AP 正式长训 latest/best checkpoint、step 级样例 PNG/JSON 和恢复续跑状态 |
| `artifacts/omni_transformer_stage_ap_formal_edit_generate/samples/formal_edit_generate_result/` | Stage AP 正式结果样例 PNG，含 source、target、edit、edit_no_source 和 generate 输出 |
| `artifacts/p0_stage_aj_checkpoint_resume_smoke/partial_result.json` | Stage AJ P0 interrupted smoke 结果，使用 `--stop-after-steps 2` 验证 step 2 停止和 checkpoint 落盘 |
| `artifacts/p0_stage_aj_checkpoint_resume_smoke/resumed_result.json` | Stage AJ P0 resume smoke 结果，使用 `--resume` 从 step 2 续到 step 4 |
| `artifacts/p0_stage_aj_checkpoint_resume_smoke/uninterrupted_result.json` | Stage AJ P0 uninterrupted 对照结果，用于比较 resume 与不中断训练的关键指标差异 |
| `artifacts/p0_stage_aj_checkpoint_resume_smoke/resume_checkpoints/` | Stage AJ P0 resume smoke 的 latest/best checkpoint 和 step 级样例 PNG/JSON |
| `docs/feasibility-report.md` | 中文可行性结论、关联项目概念映射、截至 Stage AM 的全部实验结论和未证明边界 |
| `docs/next-stage-test-plan.md` | 下一阶段测试任务规划，定义 P0-P3 挑战矩阵、Stage AN/AO/AP/AQ/AR/AS/AT/AV 等任务、通过门槛、旧路线收口规则、完成状态和担忧 |
| `docs/experiment-general-lessons.md` | 从当前全部实验中按任务线抽取的失败、修复和跨任务通用规则，只保留后续所有任务都应遵守的原则和担忧 |
| `docs/from-scratch-training-vram-quantization.md` | 从 0 训练完整架构验证的显存档位、量化收益、无效量化、购卡优先级和正式验证下限 |
| `docs/text-to-latent-thought-experiment.md` | 纯文本思考训练迁移到特殊 latent token 内部思考的本机 GPU 实验报告 |
| `docs/heterogeneous-latent-input-experiment.md` | 文本动作 + 非文本地形 tensor 的异构输入潜变量实验报告，说明可行性证据与非优劣对照边界 |
| `docs/heterogeneous-baseline-comparison-experiment.md` | 异构输入同等信息 baseline 对比报告，说明结构化 baseline 同准确率且成本更低 |
| `docs/visual-multimodal-stage-ab-experiment.md` | 真实像素输入 Stage A/B 实验报告，说明同风格可行和未见风格泛化不足 |
| `docs/visual-multimodal-stage-c-experiment.md` | 真实像素输入 Stage C 主动探测实验报告，说明随机符号绑定下探测恢复语义 |
| `docs/visual-multimodal-stage-d-experiment.md` | 真实像素输入 Stage D 局部可见与短期记忆实验报告，说明 legend 记忆降低探测成本 |
| `docs/multimodal-fusion-latent-flow-experiment.md` | Stage E+F 多模态融合与潜空间数据流动实验报告，说明 shared latent bus、ablation 与 slot probe 结论 |
| `docs/omni-transformer-stage-h-experiment.md` | Stage H Tiny Omni Transformer H1/H2 实验报告，说明 direct 与 latent bottleneck 结论 |
| `docs/omni-transformer-stage-h3-h4-experiment.md` | Stage H3/H4 实验报告，说明 latent 容量断点、counterfactual query 与 OOD 边界 |
| `docs/omni-transformer-stage-h5-h6-experiment.md` | Stage H5/H6 实验报告，说明视觉/组合泛化与多输出格式结论 |
| `docs/omni-transformer-stage-i-agent-experiment.md` | Stage I Tiny Omni Tool-Using Agent 实验报告，说明 agent loop、工具历史、记忆和 latent bottleneck 结论 |
| `docs/omni-transformer-stage-jk-audit-ui-experiment.md` | Stage J/K 多模态证据审计与 UI+DOM agent 实验报告，说明 evidence audit、DOM 操作、消融与未证明边界 |
| `docs/omni-transformer-stage-l-docvqa-experiment.md` | Stage L DocVQA 真实文档问答实验报告，说明真实数据负结果和预训练专家需求 |
| `docs/omni-transformer-stage-m-moe-multimodal-llm-experiment.md` | Stage M Tiny MoE 多模态 LLM 实验报告，说明多专家聚合、latent 输出、消融和专家分工边界 |
| `docs/omni-transformer-stage-n-strong-experts-comparison.md` | Stage N 强专家 MoE 与 baseline 成本对比报告，说明强专家收益、规则 baseline 成本和 MoE 性价比边界 |
| `docs/omni-transformer-stage-o-pretrained-experts-experiment.md` | Stage O 真实预训练 CLIP 专家接入报告，说明真实专家信号、任务不匹配边界和专家编码成本 |
| `docs/omni-transformer-stage-p-llava-moe-experiment.md` | Stage P LLaVA-Instruct-150K MoE/latent 报告，说明真实指令数据链路、latent scorer 正信号、direct/CLIP baseline 和从零长文本生成边界 |
| `docs/omni-transformer-stage-q-tiny-moe-vlm-from-scratch-experiment.md` | Stage Q 从零 Tiny MoE-VLM 报告，说明 CPU 瓶颈修正、真实 LLaVA/COCO 负结果和预训练专家必要性 |
| `docs/omni-transformer-stage-r-curriculum-vlm-experiment.md` | Stage R 低熵强监督 curriculum VLM 报告，说明 from-scratch direct 可学范围、MoE latent 当前上限和 counting 短板 |
| `docs/omni-transformer-stage-s-scorer-reconstruction-experiment.md` | Stage S scorer 与信息还原诊断报告，说明 Attention Pump/latent 信息丢失证据和 no-pump 恢复效果 |
| `docs/omni-transformer-stage-t-preserve-latent-experiment.md` | Stage T 保信息 latent 压缩器报告，说明 raw/weighted token 保真、summary token、32-slot 反例和 token 成本口径 |
| `docs/omni-transformer-stage-u-object-slots-experiment.md` | Stage U 扩展视觉输入、对象专家与输出 token sweep 报告，说明 higher-entropy 多物体任务、专家消融负结果和下一步强专家方向 |
| `docs/omni-transformer-stage-v-expert-alignment-experiment.md` | Stage V 专家语义空间对齐与信息丢失诊断报告，说明带监督训练收益、专家语言不通证据、信息丢失率和 common semantic bus 下一步 |
| `docs/omni-transformer-stage-w-alignment-mechanisms-experiment.md` | Stage W 四种专家对齐机制报告，说明 shared decoder 正信号、slot/contrastive/bus 边界和训练期 common bus 口径 |
| `docs/omni-transformer-stage-x-decomposed-diagnostics-experiment.md` | Stage X 分解诊断报告，说明 Stage T/W 指标差异、objectization 缺失、Transformer 读头边界和 scorer 瓶颈 |
| `docs/omni-transformer-stage-y-parallel-input-experts-experiment.md` | Stage Y 并行直读输入专家架构修正报告，说明串联输入专家错误、parallel direct 正信号和功能专家仍未分工 |
| `docs/omni-transformer-stage-z-supervised-teacher-diagnostics-experiment.md` | Stage Z 显式监督、训练期教师和潜变量互读诊断报告，说明功能专家开始承载任务、teacher 强但互读未解决 |
| `docs/omni-transformer-stage-aa-token-alignment-diagnostics-experiment.md` | Stage AA token 级对齐报告，说明 token 对齐降低信息丢失、部分改善互读但最终 scorer 未充分使用专家 |
| `docs/omni-transformer-stage-ab-text-moe-alignment-experiment.md` | Stage AB 文本 latent 对齐、latent-to-answer 专家和 MoE 推理报告，说明文本对齐正信号、输出专家弱信号和朴素 MoE 负结果 |
| `docs/omni-transformer-stage-ac-latent-reasoning-experiment.md` | Stage AC Q/A 潜空间、外部信息互译与答案 token 潜空间推理报告，说明前两步高保真成功但普通 latent reasoner 未闭合 |
| `docs/omni-transformer-stage-ad-latent-reasoner-readout-experiment.md` | Stage AD 只调潜变量推理专家 readout/trace 报告，说明 reasoner-only 正信号、count/relation 边界和 selector 对齐担忧 |
| `docs/omni-transformer-stage-ae-active-read-agent-experiment.md` | Stage AE 潜变量推理专家主动读取 agent 报告，说明 active read 正信号、证据依赖增强和 query policy 失败点 |
| `docs/omni-transformer-stage-af-moe-staged-reasoner-experiment.md` | Stage AF MoE 推理专家与分任务阶段训练报告，说明 gate 可训、staged 遗忘、replay 缓解和 relation query 仍失败 |
| `docs/omni-transformer-stage-ag-teacher-forced-query-trace-experiment.md` | Stage AG teacher-forced multi-step query trace 报告，说明正确读取后的上限、自由 query 瓶颈和 relation compare 负结果 |
| `docs/omni-transformer-stage-ah-query-process-ablation-experiment.md` | Stage AH query alignment 与 process supervision 逐项消融报告，说明各自修复的问题与副作用 |
| `docs/omni-transformer-stage-ai-image-generation-editing-experiment.md` | Stage AI 图像生成与源图编辑输出专家报告，说明正式 sweep 饱和、消融有效和真实图像外推边界 |
| `docs/omni-transformer-stage-aj-transformer-image-io-fidelity-experiment.md` | Stage AJ Transformer 图像输入到 latent 再完整重绘保真报告，说明 fixed baseline 负结果、memory-tree 正式长训信号、copy 门禁失败、copy/edit/generation 辅助监督正信号、P0 checkpoint/resume 恢复验证和下一步对象保真门禁 |
| `docs/omni-transformer-stage-ap-formal-edit-generation-experiment.md` | Stage AP 正式 edit/generation 长训报告，说明 d_model 192 正式单 seed 下 edit/generation 闭合、no-source 消融和边界 |
| `docs/omni-transformer-stage-ak-unified-latent-bus-experiment.md` | Stage AK 统一潜空间硬化报告，说明统一 object/pair bus 如何闭合 query retrieval 与 relation compare |
| `docs/omni-transformer-stage-al-unified-bus-answer-experiment.md` | Stage AL 统一潜空间接答案输出头报告，说明硬 latent bus 到 yes/no answer writer 的最小链路闭合 |
| `docs/omni-transformer-stage-am-hardest-unified-bus-experiment.md` | Stage AM 历史最高难度统一 bus 报告，说明 cell/count slots 一等化、count 输入专家和 decoded position compare 如何闭合四任务 |
| `docs/omni-transformer-stage-an-answer-token-latent-writer-prep.md` | Stage AN 统一 bus 接 answer-token latent writer 的训练前准备、正式命令、正式 3 seed 结果、门禁判断和担忧 |
| `docs/omni-transformer-stage-ao-pixel-to-slots-prep.md` | Stage AO 像素到 cell/count/pair slots 的训练前准备、smoke/probe、正式 3 seed 结果、pixel count 诊断未过门槛和担忧 |
| `docs/omni-transformer-stage-aq-relation-process-supervision-experiment.md` | Stage AQ relation delta/truth-table 过程监督报告，记录 v1/v2 失败与 v3 通过结果 |
| `docs/omni-transformer-stage-ar-cross-expert-visual-experiment.md` | Stage AR 视觉中心跨专家不可单解任务报告，记录四路输入因果依赖、direct baseline、probe 修正、正式 3 seed 结果和边界 |
| `docs/omni-transformer-stage-as-file-audit-experiment.md` | Stage AS 真实文件证据审计报告，记录 HTML/CSV/PDF/截图受控工具读取、引用准确率、no-tool-history 消融、正式 3 seed 结果和边界 |
| `docs/omni-transformer-stage-at-memory-exploration-experiment.md` | Stage AT 长程局部记忆和探索成本报告，记录主动建图、强 no-memory oracle、错误 memory 修正、正式 3 seed 结果和边界 |
| `docs/omni-transformer-stage-av-from-scratch-micro-omni-experiment.md` | Stage AV 70M 从零集成 Micro-Omni 报告，记录本机容量、probe 正负信号、视觉 grounding 未闭合和下一步分阶段训练建议 |
| `docs/omni-transformer-stage-avb-staged-latent-training.md` | Stage AV-B 分阶段潜空间训练报告，记录新训练顺序、GPU 优化、smoke、batch 256 长训建议和功耗监控口径 |
| `docs/omni-transformer-stage-avc-dataset-pipeline.md` | Stage AV-C 大规模合成数据流水线报告，记录 sharded dataset、1B token 规模估算、smoke 和下一步接训练脚本 |
| `docs/omni-transformer-stage-avd-1b-first-dataset-design.md` | Stage AV-D 1B-first 数据集设计，定义 1B 母分布、10M 定向降熵裁剪、任务熵、hard negative、split 和训练阶段映射 |
| `docs/omni-transformer-stage-ave-bidirectional-latent-external.md` | Stage AV-E 双向互译实验报告，记录 source external ↔ latent ↔ target external、probe 结果、latent cosine 不可靠和下一步数据设计要求 |
| `docs/omni-transformer-stage-avf-10m-bidirectional-dataset.md` | Stage AV-F 10M 级双向互译数据集报告，记录 manifest schema、已生成数据规模、CUDA smoke、70M capacity smoke 和建议长训命令 |
| `docs/omni-transformer-stage-avg-strict-staged-10m-training.md` | Stage AV-G 严格分阶段 10M 训练报告，记录 stage schedule、独立输出路径、CPU smoke/resume 和严格架构长训命令 |
| `docs/omni-transformer-stage-avh-text-anchored-visual-edit.md` | Stage AV-H 文本锚定视觉编辑数据集与训练报告，记录与 AI/AJ/AP 的区别、10M manifest、五阶段训练、训练性能修复、68M 长训负结果、非泄漏诊断和担忧 |
| `docs/omni-transformer-stage-avi-whole-image-latent-capacity.md` | Stage AV-I 整图 latent token 容量诊断报告，记录透明背景 RGBA、短到长 curriculum、宽 token 12/24/32 probe、2 维 token 极限 probe、compact / precision / segment precision compact probe、schema v5 correct-pixel residual smoke 与正式负结果、低熵判断和下一步 loss/schedule 修正 |
| `docs/archive/README-2026-07-04-legacy-command-list.md` | 2026-07-04 重写 README 前的旧版命令清单和逐阶段实验链接归档，仅作记录性查阅 |
| `docs/DIRECTORY_REFERENCE.md` | 当前文件，保持测试工作区结构可检索 |

## 目录树

```text
.
|-- README.md
|-- .gitignore
|-- pyproject.toml
|-- experiments/
|   |-- heterogeneous_baseline_comparison.py
|   |-- heterogeneous_latent_input.py
|   |-- multimodal_latent_comparison.py
|   |-- multimodal_fusion_latent_flow.py
|   |-- multimodal_latent_pipeline.py
|   |-- omni_transformer_stage_h.py
|   |-- omni_transformer_stage_h3_h4.py
|   |-- omni_transformer_stage_h5.py
|   |-- omni_transformer_stage_h6.py
|   |-- omni_transformer_stage_i_agent.py
|   |-- omni_transformer_stage_jk_audit_ui.py
|   |-- omni_transformer_stage_l_docvqa.py
|   |-- omni_transformer_stage_m_moe_multimodal_llm.py
|   |-- omni_transformer_stage_n_strong_experts.py
|   |-- omni_transformer_stage_o_pretrained_experts.py
|   |-- omni_transformer_stage_p_llava_moe.py
|   |-- omni_transformer_stage_q_tiny_moe_vlm_from_scratch.py
|   |-- omni_transformer_stage_r_curriculum_vlm.py
|   |-- omni_transformer_stage_s_scorer_reconstruction.py
|   |-- omni_transformer_stage_t_preserve_latent.py
|   |-- omni_transformer_stage_u_object_slots.py
|   |-- omni_transformer_stage_v_expert_alignment.py
|   |-- omni_transformer_stage_w_alignment_mechanisms.py
|   |-- omni_transformer_stage_x_decomposed_diagnostics.py
|   |-- omni_transformer_stage_y_parallel_input_experts.py
|   |-- omni_transformer_stage_z_supervised_teacher_diagnostics.py
|   |-- omni_transformer_stage_aa_token_alignment_diagnostics.py
|   |-- omni_transformer_stage_ab_text_moe_alignment.py
|   |-- omni_transformer_stage_ac_latent_reasoning.py
|   |-- omni_transformer_stage_ai_image_generation_editing.py
|   |-- omni_transformer_stage_aj_transformer_image_io_fidelity.py
|   |-- omni_transformer_stage_ak_unified_latent_bus.py
|   |-- omni_transformer_stage_ao_pixel_to_slots.py
|   |-- omni_transformer_stage_ar_cross_expert_visual.py
|   |-- omni_transformer_stage_as_file_audit.py
|   |-- omni_transformer_stage_at_memory_exploration.py
|   |-- omni_transformer_stage_av_from_scratch_micro_omni.py
|   |-- omni_transformer_stage_avb_staged_latent_training.py
|   |-- omni_transformer_stage_avc_dataset_pipeline.py
|   |-- omni_transformer_stage_ave_bidirectional_latent_external.py
|   |-- omni_transformer_stage_avf_10m_bidirectional_dataset.py
|   |-- omni_transformer_stage_avg_strict_staged_10m_training.py
|   |-- omni_transformer_stage_avh_text_anchored_visual_edit_dataset.py
|   |-- omni_transformer_stage_avh_text_anchored_visual_edit_training.py
|   |-- omni_transformer_stage_avi_whole_image_latent_capacity.py
|   |-- text_to_latent_sweep.py
|   |-- text_to_latent_thought.py
|   |-- visual_multimodal_stage_ab.py
|   |-- visual_multimodal_stage_c.py
|   `-- visual_multimodal_stage_d.py
|-- docs/
|   |-- archive/
|   |   `-- README-2026-07-04-legacy-command-list.md
|   |-- DIRECTORY_REFERENCE.md
|   |-- experiment-general-lessons.md
|   |-- feasibility-report.md
|   |-- next-stage-test-plan.md
|   |-- from-scratch-training-vram-quantization.md
|   |-- heterogeneous-baseline-comparison-experiment.md
|   |-- heterogeneous-latent-input-experiment.md
|   |-- multimodal-latent-comparison-experiment.md
|   |-- multimodal-latent-pipeline-experiment.md
|   |-- multimodal-fusion-latent-flow-experiment.md
|   |-- omni-transformer-stage-h-experiment.md
|   |-- omni-transformer-stage-h3-h4-experiment.md
|   |-- omni-transformer-stage-h5-h6-experiment.md
|   |-- omni-transformer-stage-i-agent-experiment.md
|   |-- omni-transformer-stage-jk-audit-ui-experiment.md
|   |-- omni-transformer-stage-l-docvqa-experiment.md
|   |-- omni-transformer-stage-m-moe-multimodal-llm-experiment.md
|   |-- omni-transformer-stage-n-strong-experts-comparison.md
|   |-- omni-transformer-stage-o-pretrained-experts-experiment.md
|   |-- omni-transformer-stage-p-llava-moe-experiment.md
|   |-- omni-transformer-stage-q-tiny-moe-vlm-from-scratch-experiment.md
|   |-- omni-transformer-stage-r-curriculum-vlm-experiment.md
|   |-- omni-transformer-stage-s-scorer-reconstruction-experiment.md
|   |-- omni-transformer-stage-t-preserve-latent-experiment.md
|   |-- omni-transformer-stage-u-object-slots-experiment.md
|   |-- omni-transformer-stage-v-expert-alignment-experiment.md
|   |-- omni-transformer-stage-w-alignment-mechanisms-experiment.md
|   |-- omni-transformer-stage-x-decomposed-diagnostics-experiment.md
|   |-- omni-transformer-stage-y-parallel-input-experts-experiment.md
|   |-- omni-transformer-stage-z-supervised-teacher-diagnostics-experiment.md
|   |-- omni-transformer-stage-aa-token-alignment-diagnostics-experiment.md
|   |-- omni-transformer-stage-ab-text-moe-alignment-experiment.md
|   |-- omni-transformer-stage-ac-latent-reasoning-experiment.md
|   |-- omni-transformer-stage-ad-latent-reasoner-readout-experiment.md
|   |-- omni-transformer-stage-ae-active-read-agent-experiment.md
|   |-- omni-transformer-stage-af-moe-staged-reasoner-experiment.md
|   |-- omni-transformer-stage-ag-teacher-forced-query-trace-experiment.md
|   |-- omni-transformer-stage-ah-query-process-ablation-experiment.md
|   |-- omni-transformer-stage-ai-image-generation-editing-experiment.md
|   |-- omni-transformer-stage-aj-transformer-image-io-fidelity-experiment.md
|   |-- omni-transformer-stage-ap-formal-edit-generation-experiment.md
|   |-- omni-transformer-stage-ak-unified-latent-bus-experiment.md
|   |-- omni-transformer-stage-al-unified-bus-answer-experiment.md
|   |-- omni-transformer-stage-am-hardest-unified-bus-experiment.md
|   |-- omni-transformer-stage-an-answer-token-latent-writer-prep.md
|   |-- omni-transformer-stage-ao-pixel-to-slots-prep.md
|   |-- omni-transformer-stage-aq-relation-process-supervision-experiment.md
|   |-- omni-transformer-stage-ar-cross-expert-visual-experiment.md
|   |-- omni-transformer-stage-as-file-audit-experiment.md
|   |-- omni-transformer-stage-at-memory-exploration-experiment.md
|   |-- omni-transformer-stage-av-from-scratch-micro-omni-experiment.md
|   |-- omni-transformer-stage-avb-staged-latent-training.md
|   |-- omni-transformer-stage-avc-dataset-pipeline.md
|   |-- omni-transformer-stage-avd-1b-first-dataset-design.md
|   |-- omni-transformer-stage-ave-bidirectional-latent-external.md
|   |-- omni-transformer-stage-avf-10m-bidirectional-dataset.md
|   |-- omni-transformer-stage-avg-strict-staged-10m-training.md
|   |-- omni-transformer-stage-avh-text-anchored-visual-edit.md
|   |-- omni-transformer-stage-avi-whole-image-latent-capacity.md
|   |-- text-to-latent-thought-experiment.md
|   |-- visual-multimodal-stage-ab-experiment.md
|   |-- visual-multimodal-stage-c-experiment.md
|   `-- visual-multimodal-stage-d-experiment.md
|-- src/
|   `-- latent_space_agent_feasibility/
|       |-- __init__.py
|       `-- core.py
`-- tests/
    |-- test_architecture_feasibility.py
    `-- test_stage_aj_checkpointing.py
```

## 快速查找

| 想找什么 | 看哪里 |
| --- | --- |
| 项目简短介绍、近期图像生成/编辑成果、Stage E+F 多模态融合成功结果、目标路线和初步证明 | `README.md` |
| 多模态潜空间架构是否可行 | `docs/feasibility-report.md` |
| 下一阶段测试任务规划、挑战矩阵、通过门槛和旧路线收口规则 | `docs/next-stage-test-plan.md` |
| 当前实验中按任务线沉淀的失败、修复和通用经验 | `docs/experiment-general-lessons.md` |
| 从 0 训练完整架构验证至少需要多少显存、量化能省多少、最低价整机买什么显卡 | `docs/from-scratch-training-vram-quantization.md` |
| 纯文本训练迁移到特殊 latent token 内部思考是否可行 | `experiments/text_to_latent_thought.py` 与 `artifacts/text_to_latent_thought/results.json` |
| 多 seed / 更长 move 的稳定性 | `experiments/text_to_latent_sweep.py` 与 `artifacts/text_to_latent_thought/sweep_results.json` |
| 更接近多模态架构的连续潜变量管线 | `experiments/multimodal_latent_pipeline.py` 与 `artifacts/multimodal_latent_pipeline/` |
| 连续潜变量管线实验结论 | `docs/multimodal-latent-pipeline-experiment.md` |
| 连续潜变量管线与 6 组 baseline 对比 | `experiments/multimodal_latent_comparison.py` 与 `artifacts/multimodal_latent_comparison/` |
| 6 组 baseline 对照结论 | `docs/multimodal-latent-comparison-experiment.md` |
| 异构输入 latent 外设实验 | `experiments/heterogeneous_latent_input.py` 与 `artifacts/heterogeneous_latent_input/` |
| 异构输入 latent 外设可行性与对照边界 | `docs/heterogeneous-latent-input-experiment.md` |
| 异构输入同等信息 baseline 对比 | `experiments/heterogeneous_baseline_comparison.py` 与 `artifacts/heterogeneous_baseline_comparison/` |
| 异构输入 baseline 对比结论 | `docs/heterogeneous-baseline-comparison-experiment.md` |
| 真实像素输入 Stage A/B 实验 | `experiments/visual_multimodal_stage_ab.py` 与 `artifacts/visual_multimodal_stage_ab/` |
| 真实像素输入 Stage A/B 结论 | `docs/visual-multimodal-stage-ab-experiment.md` |
| 真实像素输入 Stage C 主动探测实验 | `experiments/visual_multimodal_stage_c.py` 与 `artifacts/visual_multimodal_stage_c/` |
| 真实像素输入 Stage C 主动探测结论 | `docs/visual-multimodal-stage-c-experiment.md` |
| 真实像素输入 Stage D 局部可见/记忆实验 | `experiments/visual_multimodal_stage_d.py` 与 `artifacts/visual_multimodal_stage_d/` |
| 真实像素输入 Stage D 局部可见/记忆结论 | `docs/visual-multimodal-stage-d-experiment.md` |
| Stage E+F 多模态融合与潜空间数据流动实验 | `experiments/multimodal_fusion_latent_flow.py` 与 `artifacts/multimodal_fusion_latent_flow/` |
| Stage E+F 多模态融合与潜空间数据流动结论 | `docs/multimodal-fusion-latent-flow-experiment.md` |
| Stage H Tiny Omni Transformer H1/H2 实验 | `experiments/omni_transformer_stage_h.py` 与 `artifacts/omni_transformer_stage_h/` |
| Stage H Tiny Omni Transformer H1/H2 结论 | `docs/omni-transformer-stage-h-experiment.md` |
| Stage H3/H4 latent 容量与 OOD 边界实验 | `experiments/omni_transformer_stage_h3_h4.py` 与 `artifacts/omni_transformer_stage_h3_h4/` |
| Stage H3/H4 latent 容量与 OOD 边界结论 | `docs/omni-transformer-stage-h3-h4-experiment.md` |
| Stage H5 视觉增强与可组合规则泛化实验 | `experiments/omni_transformer_stage_h5.py` 与 `artifacts/omni_transformer_stage_h5/` |
| Stage H6 多输出格式实验 | `experiments/omni_transformer_stage_h6.py` 与 `artifacts/omni_transformer_stage_h6/` |
| Stage H5/H6 视觉/组合泛化与多输出格式结论 | `docs/omni-transformer-stage-h5-h6-experiment.md` |
| Stage I Tiny Omni Tool-Using Agent 实验 | `experiments/omni_transformer_stage_i_agent.py` 与 `artifacts/omni_transformer_stage_i_agent/` |
| Stage I Tiny Omni Tool-Using Agent 结论 | `docs/omni-transformer-stage-i-agent-experiment.md` |
| Stage J/K 多模态证据审计与 UI+DOM agent 实验 | `experiments/omni_transformer_stage_jk_audit_ui.py` 与 `artifacts/omni_transformer_stage_jk_audit_ui/` |
| Stage J/K 多模态证据审计与 UI+DOM agent 结论 | `docs/omni-transformer-stage-jk-audit-ui-experiment.md` |
| Stage L DocVQA 真实文档问答实验 | `experiments/omni_transformer_stage_l_docvqa.py` 与 `artifacts/omni_transformer_stage_l_docvqa/` |
| Stage L DocVQA 真实文档问答边界结论 | `docs/omni-transformer-stage-l-docvqa-experiment.md` |
| Stage M Tiny MoE 多模态 LLM 实验 | `experiments/omni_transformer_stage_m_moe_multimodal_llm.py` 与 `artifacts/omni_transformer_stage_m_moe_multimodal_llm/` |
| Stage M 多专家聚合与 latent 输出结论 | `docs/omni-transformer-stage-m-moe-multimodal-llm-experiment.md` |
| Stage N 强专家 MoE 与 baseline 成本对比实验 | `experiments/omni_transformer_stage_n_strong_experts.py` 与 `artifacts/omni_transformer_stage_n_strong_experts/` |
| Stage N 强专家收益和成本结论 | `docs/omni-transformer-stage-n-strong-experts-comparison.md` |
| Stage O 真实预训练 CLIP 专家接入实验 | `experiments/omni_transformer_stage_o_pretrained_experts.py` 与 `artifacts/omni_transformer_stage_o_pretrained_experts/` |
| Stage O 真实预训练专家信号和成本结论 | `docs/omni-transformer-stage-o-pretrained-experts-experiment.md` |
| Stage P LLaVA-Instruct-150K MoE/latent 实验 | `experiments/omni_transformer_stage_p_llava_moe.py` 与 `artifacts/omni_transformer_stage_p_llava_moe/` |
| Stage P 真实 LLaVA 指令数据链路、latent scorer 与生成边界 | `docs/omni-transformer-stage-p-llava-moe-experiment.md` |
| Stage Q 从零 Tiny MoE-VLM 实验 | `experiments/omni_transformer_stage_q_tiny_moe_vlm_from_scratch.py` 与 `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/` |
| Stage Q 从零训练链路、CPU 瓶颈修正与视觉 grounding 负结果 | `docs/omni-transformer-stage-q-tiny-moe-vlm-from-scratch-experiment.md` |
| Stage R 低熵强监督 curriculum VLM 能力上限实验 | `experiments/omni_transformer_stage_r_curriculum_vlm.py` 与 `artifacts/omni_transformer_stage_r_curriculum_vlm/` |
| Stage R from-scratch direct 可学范围、MoE latent 上限与 counting 短板 | `docs/omni-transformer-stage-r-curriculum-vlm-experiment.md` |
| Stage S scorer 与信息还原诊断实验 | `experiments/omni_transformer_stage_s_scorer_reconstruction.py` 与 `artifacts/omni_transformer_stage_s_scorer_reconstruction/` |
| Stage S Attention Pump 信息丢失、no-pump 恢复与 reconstruction probe 结论 | `docs/omni-transformer-stage-s-scorer-reconstruction-experiment.md` |
| Stage T 保信息 latent 压缩器实验 | `experiments/omni_transformer_stage_t_preserve_latent.py` 与 `artifacts/omni_transformer_stage_t_preserve_latent/` |
| Stage T raw/weighted token 保真、32-slot 反例和 token 成本口径 | `docs/omni-transformer-stage-t-preserve-latent-experiment.md` |
| Stage U 扩展视觉输入、对象专家与输出 token sweep 实验 | `experiments/omni_transformer_stage_u_object_slots.py` 与 `artifacts/omni_transformer_stage_u_object_slots/` |
| Stage U higher-entropy 多物体任务、no-patch/no-object 消融和强专家边界 | `docs/omni-transformer-stage-u-object-slots-experiment.md` |
| Stage V 专家语义空间对齐与信息丢失诊断实验 | `experiments/omni_transformer_stage_v_expert_alignment.py` 与 `artifacts/omni_transformer_stage_v_expert_alignment/` |
| Stage V 带监督训练、专家语言不通、信息丢失率与 common semantic bus 下一步 | `docs/omni-transformer-stage-v-expert-alignment-experiment.md` |
| Stage W 四种专家对齐机制实验 | `experiments/omni_transformer_stage_w_alignment_mechanisms.py` 与 `artifacts/omni_transformer_stage_w_alignment_mechanisms/` |
| Stage W shared decoder、slot targets、contrastive 与训练期 common bus 结论 | `docs/omni-transformer-stage-w-alignment-mechanisms-experiment.md` |
| Stage X 信息存在、对象化、跨专家互读与最终读头分解诊断实验 | `experiments/omni_transformer_stage_x_decomposed_diagnostics.py` 与 `artifacts/omni_transformer_stage_x_decomposed_diagnostics/` |
| Stage X objectization 缺失、Transformer 读头边界和 scorer 瓶颈结论 | `docs/omni-transformer-stage-x-decomposed-diagnostics-experiment.md` |
| Stage Y 并行直读输入专家架构修正实验 | `experiments/omni_transformer_stage_y_parallel_input_experts.py` 与 `artifacts/omni_transformer_stage_y_parallel_input_experts/` |
| Stage Y 输入专家直读图像、latent 推理拆分和功能专家分工边界 | `docs/omni-transformer-stage-y-parallel-input-experts-experiment.md` |
| Stage Z 显式监督、训练期教师与互读诊断实验 | `experiments/omni_transformer_stage_z_supervised_teacher_diagnostics.py` 与 `artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/` |
| Stage Z 信息丢失、功能专家承载和潜变量互读结论 | `docs/omni-transformer-stage-z-supervised-teacher-diagnostics-experiment.md` |
| Stage AA token 级对齐与互读诊断实验 | `experiments/omni_transformer_stage_aa_token_alignment_diagnostics.py` 与 `artifacts/omni_transformer_stage_aa_token_alignment_diagnostics/` |
| Stage AA token 对齐降低信息丢失但 scorer 未充分使用专家的结论 | `docs/omni-transformer-stage-aa-token-alignment-diagnostics-experiment.md` |
| Stage AB 文本 latent 对齐、latent-to-answer 专家和 MoE 推理实验 | `experiments/omni_transformer_stage_ab_text_moe_alignment.py` 与 `artifacts/omni_transformer_stage_ab_text_moe_alignment/` |
| Stage AB 文本对齐正信号、输出专家弱信号和朴素 MoE 负结果 | `docs/omni-transformer-stage-ab-text-moe-alignment-experiment.md` |
| Stage AC Q/A 潜空间、外部信息互译与答案 token 潜空间推理实验 | `experiments/omni_transformer_stage_ac_latent_reasoning.py` 与 `artifacts/omni_transformer_stage_ac_latent_reasoning/` |
| Stage AC 前两步高保真成功、普通 latent reasoner 未闭合的结论 | `docs/omni-transformer-stage-ac-latent-reasoning-experiment.md` |
| Stage AD 只调潜变量推理专家 readout/trace 实验 | `experiments/omni_transformer_stage_ac_latent_reasoning.py` 与 `artifacts/omni_transformer_stage_ad_reasoner_trace/` |
| Stage AD reasoner-only 正信号、count/relation 边界和 selector 对齐担忧 | `docs/omni-transformer-stage-ad-latent-reasoner-readout-experiment.md` |
| Stage AE 潜变量推理专家主动读取 agent 实验 | `experiments/omni_transformer_stage_ac_latent_reasoning.py` 与 `artifacts/omni_transformer_stage_ae_active_read/` |
| Stage AE active read 正信号、证据依赖增强和 query policy 失败点 | `docs/omni-transformer-stage-ae-active-read-agent-experiment.md` |
| Stage AF MoE 推理专家与分任务阶段训练实验 | `experiments/omni_transformer_stage_ac_latent_reasoning.py` 与 `artifacts/omni_transformer_stage_af_moe_staged/` |
| Stage AF gate 可训、staged 遗忘、replay 缓解和 relation query 仍失败 | `docs/omni-transformer-stage-af-moe-staged-reasoner-experiment.md` |
| Stage AG teacher-forced multi-step query trace 实验 | `experiments/omni_transformer_stage_ac_latent_reasoning.py` 与 `artifacts/omni_transformer_stage_ag_teacher_forced_trace/` |
| Stage AG 正确读取后的上限、自由 query 瓶颈和 relation compare 负结果 | `docs/omni-transformer-stage-ag-teacher-forced-query-trace-experiment.md` |
| Stage AH query alignment 与 process supervision 逐项消融实验 | `experiments/omni_transformer_stage_ac_latent_reasoning.py` 与 `artifacts/omni_transformer_stage_ah_query_process_ablation/` |
| Stage AH 各思路分别修复了哪些问题与带来哪些副作用 | `docs/omni-transformer-stage-ah-query-process-ablation-experiment.md` |
| Stage AK/AL/AM/AN/AQ 统一潜空间硬化实验 | `experiments/omni_transformer_stage_ak_unified_latent_bus.py` 与 `artifacts/omni_transformer_stage_ak_unified_latent_bus/`、`artifacts/omni_transformer_stage_al_unified_bus_answer/`、`artifacts/omni_transformer_stage_am_hardest_unified_bus/`、`artifacts/omni_transformer_stage_an_answer_token_writer/`、`artifacts/omni_transformer_stage_aq_relation_process_supervision/` |
| Stage AK 先统一 object/pair latent bus 后 query 和 compare 同时闭合的结论 | `docs/omni-transformer-stage-ak-unified-latent-bus-experiment.md` |
| Stage AL 统一潜空间接答案输出头实验 | `experiments/omni_transformer_stage_ak_unified_latent_bus.py` 与 `artifacts/omni_transformer_stage_al_unified_bus_answer/` |
| Stage AL 硬 latent bus 到 yes/no answer writer 的最小输出链路 | `docs/omni-transformer-stage-al-unified-bus-answer-experiment.md` |
| Stage AM 历史最高难度四任务统一 bus 实验 | `experiments/omni_transformer_stage_ak_unified_latent_bus.py` 与 `artifacts/omni_transformer_stage_am_hardest_unified_bus/` |
| Stage AM cell/count slots 一等化、count 输入专家和 decoded position compare 结论 | `docs/omni-transformer-stage-am-hardest-unified-bus-experiment.md` |
| Stage AN 统一 bus 接 answer-token writer 训练前准备、正式命令和 3 seed 结果 | `docs/omni-transformer-stage-an-answer-token-latent-writer-prep.md`、`experiments/omni_transformer_stage_ak_unified_latent_bus.py` 与 `artifacts/omni_transformer_stage_an_answer_token_writer/` |
| Stage AO 像素到 cell/count/pair slots 训练前准备、smoke/probe、正式结果和 pixel count 诊断 | `docs/omni-transformer-stage-ao-pixel-to-slots-prep.md`、`experiments/omni_transformer_stage_ao_pixel_to_slots.py` 与 `artifacts/omni_transformer_stage_ao_pixel_to_slots/` |
| Stage AQ relation delta/truth-table 过程监督实验、失败迭代和通过结果 | `docs/omni-transformer-stage-aq-relation-process-supervision-experiment.md`、`experiments/omni_transformer_stage_ak_unified_latent_bus.py` 与 `artifacts/omni_transformer_stage_aq_relation_process_supervision/` |
| Stage AR 视觉中心跨专家不可单解任务、四路缺模态消融和 direct baseline | `docs/omni-transformer-stage-ar-cross-expert-visual-experiment.md`、`experiments/omni_transformer_stage_ar_cross_expert_visual.py` 与 `artifacts/omni_transformer_stage_ar_cross_expert_visual/` |
| Stage AS 真实文件证据审计、引用准确率和 no-tool-history 消融 | `docs/omni-transformer-stage-as-file-audit-experiment.md`、`experiments/omni_transformer_stage_as_file_audit.py` 与 `artifacts/omni_transformer_stage_as_file_audit/` |
| Stage AT 长程局部记忆、探索成本、强 no-memory oracle 和错误 memory 修正 | `docs/omni-transformer-stage-at-memory-exploration-experiment.md`、`experiments/omni_transformer_stage_at_memory_exploration.py` 与 `artifacts/omni_transformer_stage_at_memory_exploration/` |
| Stage AV 70M 从零集成 Micro-Omni、本机容量和视觉 grounding 负结果 | `docs/omni-transformer-stage-av-from-scratch-micro-omni-experiment.md`、`experiments/omni_transformer_stage_av_from_scratch_micro_omni.py` 与 `artifacts/omni_transformer_stage_av_from_scratch_micro_omni/` |
| Stage AV-B 分阶段潜空间训练、GPU 常驻数据、checkpoint/resume 和 batch 256 长训建议 | `docs/omni-transformer-stage-avb-staged-latent-training.md`、`experiments/omni_transformer_stage_avb_staged_latent_training.py` 与 `artifacts/omni_transformer_stage_avb_staged_latent_training/` |
| Stage AV-C 大规模合成数据流水线、1B token 估算和 sharded smoke 数据集 | `docs/omni-transformer-stage-avc-dataset-pipeline.md`、`experiments/omni_transformer_stage_avc_dataset_pipeline.py` 与 `artifacts/omni_transformer_stage_avc_dataset_pipeline/` |
| Stage AV-D 1B-first 母分布、10M 定向裁剪和高熵数据设计 | `docs/omni-transformer-stage-avd-1b-first-dataset-design.md` |
| Stage AV-E 潜变量与外部表征双向互译、source->target、target->source、latent edit 和 10M joint 负结果 | `docs/omni-transformer-stage-ave-bidirectional-latent-external.md`、`experiments/omni_transformer_stage_ave_bidirectional_latent_external.py` 与 `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_result.json` |
| Stage AV-F 10M 级双向互译数据集、AV-E/AV-G 训练入口和 10M 负结果对照 | `docs/omni-transformer-stage-avf-10m-bidirectional-dataset.md`、`experiments/omni_transformer_stage_avf_10m_bidirectional_dataset.py`、`artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m/manifest.json`、`artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_result.json` 与 `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_result.json` |
| Stage AV-G 严格分阶段 10M 训练、独立 checkpoint、CPU smoke/resume 和 10M strict staged 负结果 | `docs/omni-transformer-stage-avg-strict-staged-10m-training.md`、`experiments/omni_transformer_stage_avg_strict_staged_10m_training.py` 与 `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_result.json` |
| Stage AV-H 文本锚定视觉编辑数据集、10M manifest、五阶段训练入口、训练性能修复、68M 长训负结果和非泄漏诊断 | `docs/omni-transformer-stage-avh-text-anchored-visual-edit.md`、`experiments/omni_transformer_stage_avh_text_anchored_visual_edit_dataset.py`、`experiments/omni_transformer_stage_avh_text_anchored_visual_edit_training.py`、`artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/dataset_10m/manifest.json`、`artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_result.json` 与 `artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_stage_mode_diagnostic.json` |
| Stage AV-I 整图 latent token 容量诊断、透明背景 curriculum、宽 token probe、2 维 token 极限曲线、compact/precision compact 前缀约束和 schema v5 correct-pixel residual 训练语义 | `docs/omni-transformer-stage-avi-whole-image-latent-capacity.md`、`experiments/omni_transformer_stage_avi_whole_image_latent_capacity.py`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_result.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_transparent_curriculum_12_24_32_3000step_summary.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_result.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_tokens_1_2_4_8_16_32_64_3000step_summary.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_result.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_result.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_segment_precision_compact_tokens_1_2_4_8_16_32_64_7000step_summary.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/correct_pixel_residual_cpu_smoke_result.json`、`artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_result.json` 与 `artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/probe_latent_dim2_correct_pixel_residual_tokens_1_2_4_8_16_32_64_7000step_summary.json` |
| Stage AI 图像生成与源图编辑输出专家实验 | `experiments/omni_transformer_stage_ai_image_generation_editing.py` 与 `artifacts/omni_transformer_stage_ai_image_generation_editing/` |
| Stage AI 低熵图像生成/编辑任务饱和与消融结论 | `docs/omni-transformer-stage-ai-image-generation-editing-experiment.md` |
| Stage AJ Transformer 图像 IO 保真实验 | `experiments/omni_transformer_stage_aj_transformer_image_io_fidelity.py` 与 `artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/` |
| Stage AJ fixed baseline 负结果、memory-tree 正式长训、copy 门禁失败与 copy/edit/generation 辅助监督正信号 | `docs/omni-transformer-stage-aj-transformer-image-io-fidelity-experiment.md` |
| Stage AJ P0 checkpoint/resume 长训恢复、step 样例和恢复等价 smoke | `experiments/omni_transformer_stage_aj_transformer_image_io_fidelity.py`、`tests/test_stage_aj_checkpointing.py` 与 `artifacts/p0_stage_aj_checkpoint_resume_smoke/` |
| Stage AP 正式 edit/generation 长训、no-source 消融和样例 | `docs/omni-transformer-stage-ap-formal-edit-generation-experiment.md` 与 `artifacts/omni_transformer_stage_ap_formal_edit_generate/` |
| 本轮 GPU 训练实验结论 | `docs/text-to-latent-thought-experiment.md` |
| 如何运行验证 | `README.md` |
| 旧 README 的逐阶段长命令清单 | `docs/archive/README-2026-07-04-legacy-command-list.md` |
| LOD 统一维度与正交层级编码 | `src/latent_space_agent_feasibility/core.py` 的 `mean_pool_lod`、`lod_embedding` |
| MoE 变长专家输出聚合 | `src/latent_space_agent_feasibility/core.py` 的 `CrossAttentionPump` |
| 主动采样与记忆树写入 | `src/latent_space_agent_feasibility/core.py` 的 `HierarchicalMediaIndex`、`MemoryTree` |
| 树状 KV Cache 剪枝模拟 | `src/latent_space_agent_feasibility/core.py` 的 `KVCacheTree` |
| 离线自我对齐边界 | `src/latent_space_agent_feasibility/core.py` 的 `OfflineAlignmentLab` |
| 渐进式生成降级策略 | `src/latent_space_agent_feasibility/core.py` 的 `progressive_generation_plan` |
