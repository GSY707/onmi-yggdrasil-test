# Directory Reference

本仓库是 `Project-Yggdrasil 未来多模态潜空间智能体架构` 的独立可行性测试工作区。关联项目 `C:\skzy\QuickFileTransport\世界树计划` 只作为概念真源读取，本仓库不修改其源码。

"docs/Project-Yggdrasil 未来多模态潜空间智能体架构.md" 是本项目需要测试的架构的白皮书。
"note.txt" 是用户的笔记，可以参考其中的内容，但不要删。

## 顶层摘要

| 路径 | 用途 |
| --- | --- |
| `README.md` | 运行方式、验证边界与记忆树口径 |
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
| `src/latent_space_agent_feasibility/` | 纯 Python 原型，实现 LOD、注意力泵、主动采样、记忆树节点/关联边、KV 分支剪枝、离线对齐与渐进式生成计划 |
| `tests/` | 可运行验证用例，覆盖白皮书核心命题的接口和不变量 |
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
| `docs/feasibility-report.md` | 中文可行性结论、关联项目概念映射和未证明边界 |
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
|   |-- text_to_latent_sweep.py
|   |-- text_to_latent_thought.py
|   |-- visual_multimodal_stage_ab.py
|   |-- visual_multimodal_stage_c.py
|   `-- visual_multimodal_stage_d.py
|-- docs/
|   |-- DIRECTORY_REFERENCE.md
|   |-- feasibility-report.md
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
|   |-- text-to-latent-thought-experiment.md
|   |-- visual-multimodal-stage-ab-experiment.md
|   |-- visual-multimodal-stage-c-experiment.md
|   `-- visual-multimodal-stage-d-experiment.md
|-- src/
|   `-- latent_space_agent_feasibility/
|       |-- __init__.py
|       `-- core.py
`-- tests/
    `-- test_architecture_feasibility.py
```

## 快速查找

| 想找什么 | 看哪里 |
| --- | --- |
| 多模态潜空间架构是否可行 | `docs/feasibility-report.md` |
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
| 本轮 GPU 训练实验结论 | `docs/text-to-latent-thought-experiment.md` |
| 如何运行验证 | `README.md` |
| LOD 统一维度与正交层级编码 | `src/latent_space_agent_feasibility/core.py` 的 `mean_pool_lod`、`lod_embedding` |
| MoE 变长专家输出聚合 | `src/latent_space_agent_feasibility/core.py` 的 `CrossAttentionPump` |
| 主动采样与记忆树写入 | `src/latent_space_agent_feasibility/core.py` 的 `HierarchicalMediaIndex`、`MemoryTree` |
| 树状 KV Cache 剪枝模拟 | `src/latent_space_agent_feasibility/core.py` 的 `KVCacheTree` |
| 离线自我对齐边界 | `src/latent_space_agent_feasibility/core.py` 的 `OfflineAlignmentLab` |
| 渐进式生成降级策略 | `src/latent_space_agent_feasibility/core.py` 的 `progressive_generation_plan` |
