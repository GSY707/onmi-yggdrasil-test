# 未来多模态潜空间智能体架构可行性验证报告

## 范围

本轮验证对象是：

- `C:\skzy\QuickFileTransport\世界树计划\docs\research\archive\future-planning\Project-Yggdrasil 未来多模态潜空间智能体架构.md`

关联真源参考：

- `C:\skzy\QuickFileTransport\世界树计划\docs\specs\work-tree-protocol-v0.2.md`
- `C:\skzy\QuickFileTransport\世界树计划\docs\specs\work-tree-graph-fork-parallel-protocol-v0.1.md`
- `C:\skzy\QuickFileTransport\世界树计划\docs\research\specifications\concepts\记忆树核心设计.md`
- `C:\skzy\QuickFileTransport\世界树计划\docs\architecture\runtime-principles-for-newcomers.md`
- `C:\skzy\QuickFileTransport\世界树计划\packages\python-sdk\src\yggdrasil_sdk\runtime_kernel\work_tree_graph.py`

验证方式是当前仓库里的纯 Python 原型、单元测试和本地 PyTorch toy 训练实验。它只能证明接口、形状、资源、存储不变量和合成任务训练链路可以闭合，不能证明真实模型训练效果。

## 结论

整体结论：白皮书里的工程方向有可落地的接口骨架，最适合作为 `世界树计划` 现有工作树、记忆树、Fork runtime 的下一层实验分支；但其中“潜空间自我对齐”“真实多模态推理主导权转移”“provider KV Cache 物理剪枝”还没有被本轮验证证明，需要真实模型、服务端 KV API 或训练实验。

| 命题 | 本轮结果 | 证据 |
| --- | --- | --- |
| 大脑与 I/O 外设解耦，零初始化残差接入 | 接口可行 | `ResidualLatentBridge` 证明 `gate=0` 时文本路径不被扰动，维度漂移会失败 |
| 全局锁定 `d_model`，用 LOD token 数表达细节 | 可行但有容量边界 | `mean_pool_lod` 保持特征维度不变；`lod_embedding` 在 `level < d_model` 时正交 |
| 级联主动采样与记忆树写入 | 可行，且与现有记忆树/工作树口径一致 | `HierarchicalMediaIndex` 先读宏观摘要再按指令取细节；`MemoryTree` 只写 LOD 摘要节点、URI 引用和定向关联边，拒绝原始 payload |
| 逻辑树索引驱动 KV Cache 分支剪枝 | 调度层可行，物理层未证明 | `KVCacheTree` 能释放死分支 token 并保留主干；真实 provider 是否能 drop KV segment 取决于 serving stack |
| 内部宪法驱动离线自我对齐 | 只能证明流程约束，不证明收敛 | `OfflineAlignmentLab` 强制离线产生 LoRA proposal，不修改 base model |
| 注意力泵聚合变长专家输出 | 形状与路由接口可行 | `CrossAttentionPump` 按路由权重和输入长度输出有界 token 序列 |
| 树高控制 LOD / Progressive Generation | 调度策略可行 | `progressive_generation_plan` 根据资源给出草图、草稿、完整三档输出计划 |
| 纯文本思考迁移到特殊 latent token 内部思考 | 小模型 toy task 初步可行，但收益不稳定 | 单次基线 `latent_from_text` 85.84%；9 次 sweep overall `latent_from_text` 84.22%，高于 `latent_scratch` 75.00%，但 move=6 平均低于 scratch |
| 连续潜变量管线：文本训练、潜变量翻译、并行思考、文本输出 | 更接近多模态架构，前几步可行 | 9 次 sweep overall：纯文本 final 98.16%，并行 final 94.48%，并行 step text 98.75%，并行 codec step 75.25%，latent thought + text output final 96.40% |
| 连续潜变量管线 6 组对照 | pipeline 有效，但 codec alignment 不是 final 性能来源 | 6 次 comparison：full pipeline 97.07%，visible CoT 99.58%，latent scratch good codec 93.07%，no codec alignment 100.00%，bad codec 98.52% |
| 异构输入潜变量链路：文本动作 + 非文本地形 tensor | 核心链路可行，且不再依赖 visible text CoT 迁移假象；未证明路线更优 | 9 次 sweep overall：good codec 100.00%，oracle 100.00%；text-only 3.50% 与 bad codec 3.97% 只是信息可见性检查 |
| 异构输入同等信息 baseline | 低熵结构化任务上，非 latent baseline 同准确率且成本更低 | 3 次 sweep，训练 move=8、测试 move=8/16/32：latent_good_codec、direct_rule_features、learned_effective_move、learned_transition_table 均 100%；learned_effective_move 仅 64 参数、0.09 秒训练 |
| 真实像素输入 Stage A/B | Stage A 链路可行；Stage B 风格泛化不足，未证明 latent 更优 | 3 次 sweep：Stage A 中 parser/CNN terrain/CNN latent 均 100%；Stage B final：CNN terrain 8.82%/4.62%，CNN latent 15.01%/7.49%，oracle 100% |
| 真实像素输入 Stage C 主动探测 | 随机符号语义绑定下，无探测接近猜测；完整探测可恢复 oracle | 3 次 sweep：passive no probe overall 8.45%，CNN semantic no probe 4.02%，probe 1/2 为 11.39%/19.24%，rule/CNN/latent symbol + 4 probes 均 100% |
| 真实像素输入 Stage D 局部可见与短期记忆 | 局部多帧可行；legend 记忆把探测成本从路径长度压到符号种类数 | 3 次 sweep：no probe 7.82%；每步探测 100% 但平均 18.67 次探测；legend memory budget 4 为 100%，平均 3.67 次探测 |
| Stage E+F 多模态融合与潜空间数据流动 | 图像、文本、遥测三路可进入 shared latent bus，并输出完整文本答案；潜空间流动可被 ablation 和 probe 验证 | 3 次 sweep：shared latent fusion action/text 均 100%；early concat 也 100%；单模态 26.56%/28.12%/31.25%，late fusion 38.02%；清零任一路 latent 后 action 掉到 26.56%-36.98%，slot probe 本模态 100%、交叉方向 25% |
| Stage H Tiny Omni Transformer H1/H2 | 统一 token stream 的 decoder-only Transformer 可处理图像 patch、文本 token、遥测 token；latent bottleneck 可承载多模态信息并生成文本答案 | 3 次 sweep：H1 direct answer/action/format 均 100%；H2 latent bottleneck answer/action/format 均 100%；H2 no-latent-access answer 0%、action 17.19%；H2 latent probe visual/goal/telemetry 均 100% |
| Stage H3/H4 latent 容量与 OOD 边界 | `K=0` 与 `K>=1` 有清晰容量断点；counterfactual query 可行；未见视觉风格和随机 held-out 组合暴露泛化边界 | 3 次 sweep：H3 K=0 answer 1.56%、action 17.19%，K=1/2/4/8/16 answer 均 100%；H4 counterfactual only 100%；unseen visual style answer 0.52%、visual 0.52%；held-out target triples answer/action 4.17% |
| Stage H5/H6 视觉/组合泛化与多输出格式 | 视觉增强修复增强族内 held-out style；可组合规则修复 held-out 组合；多输出格式可与 latent bottleneck 同时工作 | H5：held-out visual style answer 98.70%、visual 100%，far OOD answer 54.69%；compositional held-out triples answer/action 100%；H6：held-out triples + held-out style answer 96.61%，READ_VISUAL/GOAL/TELEMETRY 均 100%，FULL 87.16% |
| Stage I Tiny Omni Tool-Using Agent | 从单轮回答扩展到多步工具调用、工具历史、记忆写入和最终报告；latent bottleneck agent 可闭环完成 synthetic agent episode | 3 次 sweep：direct agent step/rollout 均 100%；latent bottleneck agent step/rollout 均 100%；四类 episode success 均 100%；no tool history 0%，no memory 14.52%，no latent access 0%；latent probe fault 98.34%、phase 100% |
| Stage J/K 多模态证据审计与 UI+DOM Agent | latent bottleneck agent 可以融合像素观察、文本规则/目标、结构化证据/DOM、memory 和工具历史，并闭环输出审计结论或 UI 操作 | 3 次 sweep：Stage J audit direct/latent step/rollout 均 100%；Stage K UI+DOM direct/latent step/rollout 均 100%；audit no history/latent 均 0%，no memory 15.89%，no image 66.21%，no structured 66.24%，no policy text 34.70%；UI no history/DOM/latent 均 0%，no image 52.15%，no goal text 34.08% |
| Stage L DocVQA 真实文档问答 | 真实 DocVQA 上，tiny 从零 direct 只有弱信号，latent bottleneck 接近随机；证明当前 toy 架构不能直接跨到真实文档问答 | 3 次 sweep，8 候选 OCR 行选择：random 12.50%，lexical overlap 19.21%，direct 21.48%，latent bottleneck 14.65%；no latent access 12.57%；latent probe label 15.95%、question type 61.07% |
| Stage M Tiny MoE 多模态 LLM | 按功能区分的专家、router、attention pump、latent thought 和文本输出专家可以端到端训练闭合；但专家分工还不够硬 | 3 次 sweep：direct routed experts exact 75.65%、semantic 76.11%；MoE latent bottleneck exact 79.30%、semantic 80.01%；greedy exact 同为 79.30%；router 100%；no latent access 0%，no prompt 21.16%，shuffled image 34.83%；latent probe task 100%、target 79.69%、route 99.54% |
| Stage N 强专家 MoE 与 baseline 成本对比 | 强专家能消除 Stage M 的 counting/spatial 短板，并让专家分工变硬；但低熵 synthetic 任务上规则 baseline 成本最低，MoE 不能宣称性价比更优 | 3 次 sweep：deterministic strong expert renderer 100%、0 参数、0 训练、0.016 ms/example；strong direct 100%、283,563 参数、19.43 秒训练、1.497 ms/example；strong MoE latent 100%、283,563 参数、22.41 秒训练、1.484 ms/example；raw classifier 80.92%、144,405 参数、26.93 秒训练、0.439 ms/example |
| Stage O 真实预训练 CLIP 专家 | 真实 frozen CLIP image/text experts 可以接入 latent bus，但普通 CLIP 与几何计数/空间/图表任务不匹配；未证明真实预训练专家自动优于 toy 专家 | 3 次 sweep：CLIP zero-shot 50.52%；CLIP linear classifier 68.23%，151,484,374 总参数，with-CLIP 3.719 ms/example；CLIP direct decoder 67.97%，152,359,153 总参数，with-CLIP 5.625 ms/example；CLIP MoE latent 68.36%，with-CLIP 5.580 ms/example；no latent access 0%，no image 44.40%，no text 6.12% |
| Stage P LLaVA-Instruct-150K MoE/latent | 真实 LLaVA 指令数据链路可闭合；latent scorer 明显高于随机且依赖 latent，但 direct scorer 和 CLIP zero-shot 更强；从零字符长答案生成失败 | 3 次 sweep，8 候选答案 ranking：CLIP zero-shot 86.98%；scorer direct 57.03%，scorer MoE Attention-Pump latent 42.45%，scorer text-only 49.48%；scorer MoE no latent access 13.28%，no text 21.88%，no image 42.97%；字符级 MoE latent 9.38%，接近/低于随机 12.50% |
| Stage Q 从零 Tiny MoE-VLM | 从零训练链路可闭合且 CPU 瓶颈已修正；但真实 LLaVA/COCO 上仅略高于随机，未学出视觉 grounding | 3 次 sweep，8 候选答案 ranking：scratch text-only 13.80%，direct 14.58%，mean-pool latent 14.58%，MoE Attention-Pump latent 14.58%；no latent access 8.85%，no image 15.63%，no text 10.16%；Stage P CLIP zero-shot 86.98% |
| Stage R 低熵强监督 Curriculum VLM | 从零 direct 路径能学低/中熵视觉匹配；MoE Attention-Pump latent 的稳定上限只有单颜色识别；counting 是共同短板 | 3 次 sweep：L1 color direct/MoE 均 100%；L2 object direct 100%、MoE 46.03%；L3 position direct 99.22%、MoE 16.60%；L4 count direct 19.86%、MoE 19.40%；L5 relation direct 94.14%、MoE 25.20%；L6 caption direct 79.30%、MoE 15.82% |
| Stage S Scorer 与信息还原诊断 | 只改最终 scorer 不能修复；移除 Attention Pump 后 L2/L5 大幅恢复；probe 证实信息主要在 pump/thought latent 丢失 | 3 次 sweep：cross+pump L1/L2/L5 为 16.47%/9.96%/12.96%；cross+no-pump L1/L2/L5 为 100%/94.40%/77.67%；L2 no-pump expert concat 可还原 93.16%，pump latent 仅 2.73%；L5 no-pump expert concat 83.14%，pump latent 12.37% |
| Stage T 保信息 Latent 压缩器 | 保留 raw + route-weighted expert tokens 并追加 summary tokens 后，L2/L3/L5 明显恢复；纯 32-slot resampler 仍失败 | 3 次 sweep：wide residual L1/L2/L3/L5 为 100%/98.63%/73.76%/92.06%，no-image 回到 11.59%/12.17%/12.76%/12.17%；32-slot resampler L1/L2/L3/L5 为 16.15%/13.35%/12.70%/12.04%；L4 counting 仍 19.40% |
| Stage U 扩展视觉输入、对象专家与输出 token sweep | 64 patch raw evidence tokens 有弱视觉信号；learned object/spatial/counting experts 没有稳定、因果地超过 patch-wide；输出 token 数不是主瓶颈 | 2 seeds x 3 output counts：patch-wide 16.80%/19.04%/18.85%；object expert 20.80%/15.33%/17.68%；no-patch expert 14.06%-15.72%，no-object experts 不降；随机 12.50% |

## 与世界树计划的概念映射

这里的记忆树不是 RAG，也不是文档块检索库。按 `世界树计划` 的记忆树概念，它是 LOD 节点树：父节点是高层总结，子节点是更高精度展开；节点可以带 URI / 证据引用，但不保存高熵原始多模态 payload；节点之间通过有向关联边表达证据、因果、上下文或推理关系；读取时按深度返回子树，按广度返回关联节点，并且读到节点时同时暴露子节点名和关联目标名。

现有 `WorkTreeNode.detailLevel`、父子节点 LOD、`executionSummary` / `failureSummary`、`producedEvidenceRefs` 与记忆树的父子摘要、关联边、URI 指针可以共同承载白皮书中的“低熵摘要 + URI 指针 + 按需展开”思路。不要另造第二套任务树，也不要把记忆树退化成 RAG。

`relationIds`、`pendingInformationItems`、`runtime_hints` 可以作为主动采样和注意力路由的上层信号；它们应继续是信息线索，不应被误用成硬依赖。硬依赖仍然属于 `dependsOn`。

Fork 的父上下文锚点和 child 执行焦点可以映射到“逻辑树索引 + 分支局部推演”。但 Fork 结果必须通过摘要、证据和结构化结果回父节点合并，不应把完整推演流水账灌回父上下文。

## 不成立或未证明的部分

- “潜空间接管绝对主导权”没有被证明；这需要真实训练曲线、灾难性遗忘指标和下游任务质量评估。
- “内部宪法无监督演进”没有被证明；本轮只证明离线提案流程可以防止 live 权重漂移。
- “KV Cache drop() 瞬间释放显存”没有被证明；本轮只有逻辑索引模拟，真实效果依赖模型服务端是否支持 segment 级 KV 回收。
- “视频等高熵输入不会拖垮推理”需要真实多模态 extractor、索引延迟、召回率和成本评估。
- “特殊 latent token 思考优于直接答案训练”没有被证明；9 次 sweep overall `latent_from_text` 84.22%，低于 `direct_scratch` 87.03%。本轮只能说明机制能跑通，且更长任务上文本思考预训练对 latent 平均有帮助。
- 连续潜变量管线虽然更接近目标架构，但仍是合成文本任务；它没有证明真实图像、视频或动作外设能形成同样稳定的 latent space。当前最弱项是 codec diagnostic，final-only 后约 66% 的中间 latent 仍可被 codec 直接解码。
- 6 组对照显示：good codec alignment 对最终答案不是必要条件，甚至低于 no-codec / bad-codec final baseline。当前 codec 的价值更像“可解释/可翻译约束”，不是性能收益。
- 异构输入实验已经排除了“只是在迁移 visible text CoT”的主要疑点：任务答案必须依赖非文本地形 tensor，good codec 达到 oracle 100.00%。但 text-only 和 bad codec 接近随机只说明它们看不见或无法正确读取地形，对“路线是否更好”没有直接比较意义；raw direct 失败也只是一个弱小型 baseline 的结果。该实验仍是低熵离散合成任务，不等价于真实图像、视频或传感器外设。
- 同等信息 baseline 对比显示：在低熵、可解析、规则清晰的异构输入上，latent codec 不是更优方案。`direct_rule_features`、`learned_effective_move`、`learned_transition_table` 都与 `latent_good_codec` 一样达到 100%，但前两者训练成本显著更低。这里的价值不在 latent 本身，而在把异构输入转成可稳定参与推理的中间表示。
- 真实像素 Stage A/B 显示：同训练风格下，视觉 parser、CNN-to-terrain、CNN-to-latent 都能闭合；未见风格下，训练 palette parser、CNN-to-terrain 和 CNN-to-latent 都明显掉点。CNN-to-latent 在 Stage B 高于 CNN-to-terrain，但绝对准确率仍低，不能证明 latent 已解决跨风格泛化。
- 真实像素 Stage C 验证了主动测试的必要性：当每张地图随机绑定视觉符号与地形语义时，不探测路线只能接近猜测；1/2 次探测只能部分恢复；完整探测后 rule/CNN/latent 符号路线都能达到 100%。这支持“先试验、再推理”的架构方向，但仍不证明 latent 本身优于结构化符号路线。
- 真实像素 Stage D 进一步证明短期记忆的价值：在局部可见、多帧观察下，每步重复探测可以正确但成本随路径长度增长；legend 记忆把已探测符号语义复用起来，`move=32` 从 32 次探测降到约 3.97 次探测，同时保持 100%。
- Stage E+F 证明了多模态融合和潜空间数据流动可以被本地实验诊断：shared latent fusion 能从图像、文本、遥测三路输出完整文本答案，删掉或打乱任一路 latent 都会明显掉点，slot probe 显示三个 pre-fusion slots 只携带本模态因素。但 early concat 同样达到 100%，所以这仍不是 latent 优越性证明；它只证明 latent bus 是一个可行且可诊断的数据流动方案。
- Stage H 把主体换成统一 decoder-only Transformer：H1 direct 证明普通 omni prefix-to-answer 链路可行；H2 latent bottleneck 禁止答案直接看原始输入后仍达 100%，并且断开答案到 latent 的访问会让 answer exact 变成 0。这是目前最接近最终架构目标的本地证据：特殊 latent token 可以作为内部 scratchpad 承载图像、文本、遥测融合后的信息。
- Stage H 仍是合成低熵任务：图像不是照片级自然图像，输出是固定 DSL，未验证开放式自然语言、未见风格泛化、复杂组合泛化或真实 omni 规模训练收益。
- Stage H3/H4 进一步校准了边界：当前低熵任务里 1 个 latent token 已足够，说明任务太小；counterfactual query 可以被同一 latent bottleneck 模型处理；但未见视觉风格几乎全崩，随机策略表的 held-out 组合也不能自然泛化。这些负结果说明下一步必须引入视觉增强和可组合规则，而不是只继续堆同分布训练。
- Stage H5/H6 说明这两个负点可以被定向修复：多风格增强把增强族 held-out style 从 H4 的 0.52% 提到 98.70%，可组合规则把 held-out 组合从 4.17% 提到 100%。多输出格式在最难的 `held-out triples + held-out style` 上达到 96.61%，但 FULL 格式只有 87.16%，说明复合字段自回归输出仍是弱点。远距离视觉 OOD 仍只有 54.69%，不能宣称真实视觉泛化已解决。
- Stage I 把实验推进到 agent loop：模型必须在多步中调用工具、读取工具结果、写 memory 并最终报告。latent bottleneck agent 与 direct agent 都能闭环 100%，断开工具历史或 latent 都会变成 0%。这说明当前 tiny omni Transformer 架构可以承载“观察-工具-历史-记忆-下一步动作”的基本 agent 控制闭环。
- Stage I 仍是 imitation learning：训练来自 expert trajectory teacher forcing，工具是合成 DSL，episode 很短，还没有真实工具、长程任务、错误恢复或世界树计划真实记忆树读写。
- Stage J/K 把 agent loop 推到两个更接近真实目标的受控场景：多模态证据审计和 UI+DOM 操作。两者的 latent bottleneck agent 都能 100% closed-loop 完成，且消融显示工具历史、DOM/结构化证据、图像、文本目标和 latent scratchpad 都是有效因子。
- Stage J/K 仍然不是开放环境验证：证据审计材料、截图、DOM、工具和 memory 都是 synthetic token/DSL；没有真实浏览器 DOM、CSS layout、异步事件、PDF/网页证据、长程导航、自我探索或错误恢复训练。
- Stage L 直接接入 DocVQA 真实文档图像、自然语言问题、答案和 OCR 行候选，结果是明确负向边界：direct 只比 lexical overlap baseline 高约 2.27 个百分点，latent bottleneck 低于 lexical overlap 且接近随机。这说明真实文档问答不能靠 tiny 从零模型和低分辨率缩略图硬学出来。
- Stage L 不推翻架构方向；它说明下一步必须引入预训练视觉/OCR/text/layout 专家，把专家输出送入 latent bus，而不是让 latent bottleneck 同时承担 OCR、文档语义和答案定位。
- Stage M 修正了前一版 tiny omni 实验没有体现 MoE 的问题：功能专家、router、attention pump、latent thought、文本输出专家可以在本机 GPU 上闭合，且断开 latent 后答案归零。它支持“多专家输出经注意力泵进入 latent，再由输出专家生成”的核心链路。
- Stage M 仍不能证明 MoE latent 更优；MoE latent 只比 direct routed experts 高约 3.65 个百分点，且任务仍是合成模板输出。更重要的是，`wrong route` 仍有 68.88%，单独去掉 `spatial/counting/chart` 专家掉点不大，说明专家分工还没有成为强控制面。counting 只有 47.90%，spatial 只有 63.40%，需要对象级表示、空间归纳偏置或专家级辅助监督。
- Stage N 说明换成强专家后，Stage M 的能力短板可以被修复：strong MoE latent 和 strong direct 都达到 100%，且去掉 spatial/counting/chart 专家会让对应 20% 任务族失败，专家分工比 Stage M 更硬。
- Stage N 也说明强专家路线的工程代价必须单独算：deterministic strong expert renderer 也达到 100%，且 0 参数、0 训练、预测约 0.016 ms/example；strong MoE latent 是 283,563 参数、22.41 秒训练、1.484 ms/example。低熵规则任务上，neural latent decoder 没有性价比优势。MoE 的价值应来自开放任务组合、跨专家融合、输出统一和 agent 控制，而不是可规则求解任务上的单点准确率。
- Stage O 把 oracle strong expert 换成真实 frozen CLIP image/text expert。结果是中等正信号但边界明确：CLIP zero-shot 已有 50.52%，训练小头/decoder 后约 68%，说明真实 pretrained features 可接入并可学习；但 CLIP MoE latent 低于 Stage M/N，且 chart/spatial 仍弱。这不是架构否定，而是专家能力和任务不匹配：普通图文对齐 CLIP 不是 OCR/layout、对象计数、空间关系或图表读取专家。
- Stage O 的成本也很关键：frozen CLIP 本身 151,277,313 参数，CLIP 编码约 3.68 ms/example；cached downstream 虽便宜，但真实在线成本由专家编码主导。后续比较必须持续拆分 frozen expert 参数、adapter 参数、expert encoding 成本和 downstream 成本。
- Stage P 接入真实 LLaVA-Instruct-150K JSON 与 COCO 图像，说明官方指令数据、真实图像、CLIP token experts、router、Attention Pump、latent thought 和文本候选输出可以在本机 GPU 上闭合。`scorer MoE Attention-Pump latent` 42.45% 明显高于 8 候选随机 12.50%，且 `no latent access` 掉到 13.28%，证明 latent 数据流不是空壳。
- Stage P 同时暴露两个边界：第一，direct scorer 57.03% 高于 MoE latent scorer，CLIP zero-shot 86.98% 远高于所有小训练头，所以没有证明架构更优；第二，字符级从零长答案生成头几乎失败，MoE latent 只有 9.38%，说明真实 LLaVA 长文本输出需要预训练 LM 输出专家或 LoRA，而不是本机 tiny 字符头。
- Stage P 的 `no image modality` 为 42.97%，几乎不低于 full 42.45%，说明当前候选答案评估主要被 prompt/answer 语义支撑，不是强视觉 grounding 测试。下一步应加入同 prompt 类型、同语言风格但图像事实不同的 hard negative。
- Stage Q 去掉所有 frozen pretrained experts，从 COCO 像素和字符 token 从零训练 tiny MoE-VLM。链路可以训练，优化后 3 seed 约 153 秒完成，但 Top-1 只有 14.58%，接近随机 12.50%；`no image modality` 15.63% 不降反升，说明没有学出有效视觉 grounding。它支持 Stage P 判断：真实多模态路线需要预训练视觉/文本/语言专家，from-scratch tiny 只适合作为接口压力测试。
- Stage R 把 Stage Q 的数据换成低熵强监督 synthetic curriculum 后，结论变得更精确：同一个 from-scratch tiny CNN + 字符 answer scorer 的 direct 路径可以学会颜色、颜色+形状、位置、二物体关系和三物体 caption ranking；但 MoE Attention-Pump latent 只稳定解决 L1 单颜色，L2 只有 46.03%，L3/L5/L6 接近随机附近。失败核心不是像素不可学，而是当前 latent bottleneck / Attention Pump / scorer 训练路径太弱。
- Stage R 的 L4 counting 让 direct 和 MoE 都回到随机附近，说明当前 tiny CNN token 没有对象级计数归纳偏置；这类任务需要 object slot、detector、counting expert 或显式辅助监督。
- Stage S 进一步定位了 Stage R 的信息丢失位置：只把最终 scorer 改成 answer cross-attention、但仍经过 Attention Pump 时，L1/L2/L5 分别只有 16.47%/9.96%/12.96%；去掉 Attention Pump，让 answer 直接 cross-attend weighted expert tokens 后，L1/L2/L5 提到 100%/94.40%/77.67%，且 no-image 回到随机附近。
- Stage S 的 reconstruction probe 证明上游专家 token 中确实保留了信息：L2 的 no-pump `vision`/`fusion`/`expert_concat` 可还原 96.94%/92.51%/93.16%，但 pump latent/thought latent 只有 2.73%/3.19%；L5 的 no-pump `expert_concat` 可还原 83.14%，pump latent 只有 12.37%。这直接支持“当前 Attention Pump/latent 压缩造成信息丢失”。
- Stage S 同时说明 L3 position 有训练预算因素：no-pump 360 steps 平均 26.95%，单 seed 1000 steps 可到 77.9%，但 pump 路径仍接近随机。L4 counting 依旧所有路径接近随机，仍需对象级计数专家。
- Stage T 按“宁愿多 token，也要保信息”的原则替换 latent 压缩器：`wide_residual_latent` 保留 raw expert tokens 与 route-weighted expert tokens，并追加 summary tokens，总计约 72 latent tokens。结果 L2/L3/L5 分别达到 98.63%/73.76%/92.06%，而 no-image 回到随机附近，说明保真 latent bus 能恢复 Stage S 暴露的信息丢失。
- Stage T 的反例同样重要：`slot_resampler_latent` 使用 32 个 latent slot，但不保留原 token，L2/L3/L5 仍为 13.35%/12.70%/12.04%。这说明问题不是简单的 latent token 数量，而是有没有 raw/residual 信息通道。
- Stage T 仍未解决 L4 counting，继续证明 counting 需要 object-slot/counting expert。L3 位置组合虽大幅改善但未到 90%+，下一步需要显式坐标、patch position 或 spatial expert。
- Stage U 按这个方向加了 64 个 patch raw tokens、object slot、spatial、counting experts，并 sweep 8/16/32 个 output summary tokens。结果只证明扩大 patch 视觉输入有弱信号：`patch_wide_latent` 为 16.80%/19.04%/18.85%，高于随机 12.50% 和 no-image 13.18%-14.75%，但远未解决多物体关系/计数任务。
- Stage U 没有证明 learned object/spatial/counting experts 有因果收益：`object_slot_spatial_wide_latent` 只有 output=8 达到 20.80%，output=16/32 低于 patch-wide；`no_object_experts` 基本不降，`no_patch_expert` 掉到 14.06%-15.72%。这说明视觉证据仍主要在 raw patch bus 中，当前 QueryResampler 式 object slot 不能替代对象检测/显式空间专家。
- Stage U 的 output token sweep 也没有单调收益：patch-wide 最好是 output=16，object variant 最好是 output=8。当前瓶颈不是追加 summary/output tokens 太少，而是上游对象级归纳偏置和监督不足。

## 推荐下一步

1. 先把当前原型保留为独立实验，不并入 `世界树计划` runtime 主线。
2. 下一步真实多模态 LLM 应保留 Stage M/N/O 的 MoE/attention pump 骨架，但按任务选真实专家：DocVQA 用 OCR/text/layout/document experts，UI 用 DOM/screenshot/layout experts，计数/空间用 object detector/spatial relation experts。
3. 不要继续在几何 synthetic 任务上硬压 CLIP；它不适合精确计数、left-of 判断和图表读取。
4. 下一阶段任务应刻意设计成“单个专家不能直接完成，必须跨专家融合后才能答”，否则 deterministic baseline 会继续 100%，无法证明 MoE 的组合价值。
5. LLaVA 路线下一步应把输出专家换成小型预训练 LM + LoRA，或者先用 frozen text scorer 做 hard-negative grounding，不要继续用从零字符级长答案头。
6. 从零 tiny VLM 不应继续直接放大 LLaVA；若坚持 from-scratch，应先做低熵 curriculum（caption matching、对象/属性、多选 VQA、合成到真实），再升到 LLaVA。
7. Stage T 之后，latent bus 的默认方向应切到保真 token：先保留 raw evidence tokens，再追加 route-weighted tokens 和 summary tokens；后续 token 成本由记忆树/工作树做分层选择和摘要，不在第一层压缩器里强行丢信息。
8. 不应继续使用纯 resampler/Attention Pump 作为唯一信息通道；如果要压缩，也必须有 residual passthrough 或 object/slot token 保真旁路。
9. Stage U 已经说明“只加 learned object slot/counting resampler”不够；下一步应直接切到带辅助监督的 object slot、detector/segmentation/grounding expert，或至少加入 occupancy、color、shape、cell/box、count 辅助 loss。
10. L3/L4 位置组合和 counting 下一步仍应保留 raw patch evidence tokens，同时让更强 object/spatial/counting experts 产生可验证的对象表或对象 token；不要再把纯 resampler 当作唯一对象归纳偏置。
11. DocVQA 下一步应接入预训练专家：OCR/text/layout encoder 或现有 VLM，把高分辨率文档理解交给专家，再测试 latent bus 与 agent decoder。
12. 真实工具边界可并行推进：用 Playwright/本地 HTML 页面替换合成 UI+DOM DSL，保留截图 + DOM + 工具历史 + latent bottleneck 的评估结构。
13. 证据审计下一步应接入真实文件形态：小型 HTML/PDF/CSV/截图组合，要求模型调用受控工具抽取证据并输出带引用的 audit report。
14. 如果继续走真实多模态路线，应把 Stage D 的短期 legend 扩展成局部地图记忆：加入遮挡、错误探测、不可重置探测成本、探索路径规划和多目标任务。
15. 如果要继续验证 KV 剪枝，需要选定一个可控推理服务栈，确认是否暴露 prefix/segment 级 KV 生命周期 API。
16. 若要验证潜空间训练路线，应单独建训练实验，不要把未验证训练假设混入现有 work-tree runtime。
