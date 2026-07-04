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

整体结论：白皮书里的工程方向有可落地的接口骨架，最适合作为 `世界树计划` 现有工作树、记忆树、Fork runtime 的下一层实验分支；但其中“潜空间自我对齐”“真实多模态推理主导权转移”“provider KV Cache 物理剪枝”还没有被本轮验证证明，需要真实模型、服务端 KV API 或训练实验。Stage Y 到 AM 的新结论进一步收窄了路线：输入专家应并行直读外部信息，显式监督和 token/text 对齐都有正信号；Q/A 与外部信息互译可以做到高保真，但普通 latent reasoner 不能自动完成答案 token 潜空间推理。只调 latent reasoner、加入显式 readout/trace 后有强正信号；主动读取 agent 后证据依赖性更强，但 query policy 尤其是 relation 的左右对象查询还没有闭合。MoE gate 可以学会任务族路由，但 naive 分阶段训练会灾难性遗忘，replay 只能部分缓解。Teacher-forced multi-step query trace 在四任务上打开了正确读取后的上限；逐项消融进一步说明 process supervision 能修正确读取后的 compare 上限，朴素 CLIP-style query alignment 只能小幅改善 query，且容易破坏 evidence reader。Stage AK 把统一 object/pair latent bus 前置后，relation 的 pair readout、text query retrieval 和 compare 同时达到约 99%-100%；Stage AL 再接 answer writer 后 model-selected answer 达到 99.61%；Stage AM 把任务拉回 color/shape/count/relation 四任务后，证明 cell/count slots 必须作为一等 latent bus，加入 count 输入专家和 decoded position compare 后 all-task model answer 达到 98.24%。这支持“先把潜空间练硬，再接 reasoner/输出头”的路线。Stage AI 把验证切到图像输出专家后很快饱和，只能证明低熵图像生成/编辑链路能闭合；Stage AJ 改成全 Transformer patch 输入、latent 和 patch decoder 后，无辅助监督 memory-tree 仍丢对象，但对象属性/mask 辅助监督能把合成源图完整重绘打到 100%。Stage AP 补完正式 edit/generation 长训：编辑 scene exact 99.02%，no-source 4.10%，文本生成 100%。这证明低熵合成单物体上的 latent 图像输出专家正式闭合，但仍不能外推到真实照片或复杂编辑质量。

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
| Stage V 专家语义空间对齐与信息丢失诊断 | 带监督语义 loss 提升下游和 no-patch 结果，证明监督方向有效；但跨 expert transfer 没有改善，说明专家仍未形成统一语言 | 2 seeds：ranking-only top1 19.92%、no-patch 14.06%、diagonal info 57.20%、offdiag 40.69%；supervised top1 23.57%、no-patch 21.22%、diagonal info 63.10%、offdiag 40.52%；trained shared decoder 读 patch/wide 约 92%，读 object/spatial/count 约 55%-59% |
| Stage W 四种专家对齐机制与训练期 common semantic bus | 共享 semantic decoder 是当前最有效低成本对齐；显式 common bus 可作训练期 teacher/诊断，但不应进入最终推理路径；四机制全开没有超过最小监督 baseline | 2 seeds：ranking-only top1 13.41%；shared grid decoder 22.66%、no-patch 21.88%、offdiag 39.95%、language gap 15.71%；all-four train-only bus top1 21.09%、common bus info 58.58%、`inference_uses_common_bus=false` |
| Stage X 信息存在、对象化、跨专家互读与最终读头分解诊断 | 问题是多瓶颈叠加：Transformer 能读 patch/wide 语义，但 object slots 没对象化，answer 信息仍弱，最终 scorer 也没充分利用 token | 2 seeds：shared patch/wide Transformer cell info 89.45%/86.36%，object slots 49.78%；object table set exact 0%、最高 recall 11.45%；answer reconstruction 最高 20.96%；frozen scorer raw expert concat 27.08% 高于原 full 22.53% |
| Stage Y 并行直读输入专家架构修正 | 串联输入专家是错误拓扑；并行直读图像专家明显优于旧串联，但 object/spatial/count 仍未自然分工 | 2 seeds：serial_chain top1 11.07%、parallel_direct 18.62%、merged_direct 14.32%；parallel_direct no-image 12.37%、no-patch 13.02%、no-object/spatial/count 仍 18.62%；answer reconstruction 最高 10.03% |
| Stage Z 显式监督与训练期教师诊断 | 显式监督让功能专家开始承载任务信息；teacher 提高 top-1，但 student latent 只部分继承 teacher 语义，互读仍未解决 | 2 seeds：baseline_parallel top1 20.31%、supervised_no_teacher 22.66%、supervised_teacher 26.30%；supervised_no_teacher 去掉 object/spatial/count 后降到 12.24%；teacher bus cell info 98.61%、scene exact 80.73%；semantic offdiag 最高 37.42% |
| Stage AA token 级对齐与互读诊断 | token 对齐显著降低 object/spatial/count 信息丢失，并部分改善互读；但最终 scorer 没充分使用对齐后的功能专家，top-1 未超过简单显式监督 | 2 seeds：supervised_no_teacher top1 24.61%、token_aligned_teacher 22.79%；token_aligned semantic occupied color/shape 65.70%/48.13%，count positive diag 77.69%；no-object/spatial/count 仍 21.09%；teacher retrieval top1 14.28% |
| Stage AB 文本 latent 对齐、latent-to-answer 专家与 MoE 推理 | prompt/answer 文本 latent 对齐和 latent-to-answer runtime scorer 有正信号；朴素 MoE reasoner 没成功，router 近似均匀且没有形成专家分工 | 2 seeds：text_latent_aligned top1 23.83%，高于 token_aligned_teacher 21.09%；latent-answer candidate top1 15.36%，略高于随机 12.5%；prompt/latent-answer retrieval 只有 3%-4%；MoE top1 20.57%，gate entropy 1.37 接近 log(4)=1.386 |
| Stage AC Q/A 潜空间、外部信息互译与答案 token 潜空间推理 | 前两步高保真成立，但第三步普通 latent reasoner 失败；这证明“互译能力”不是“潜空间推理能力”的充分条件 | 1 seed：Q/A question/answer exact 均 100%；evidence occupancy/color/shape 为 100%，count table exact 91.80%；latent reasoner full answer 40.04%，no-evidence 27.34%，shuffled-evidence 26.56%；color-only 诊断中前两步 100%，reasoner 仍约 37.5% |
| Stage AD 只调潜变量推理专家 readout/trace | 只调整 latent reasoner 后出现强正信号：显式 evidence readout 和 trace 监督能把答案 token latent 推理从弱证据使用提升到可用范围；但 selector、count、relation 未完全闭合 | 1 seed：四任务 full answer 74.41%，no/shuffled 均 28.71%；color-only full 91.80%，no 29.30%，shuffled 27.34%；reasoner reader occupancy/color/shape 为 100%，count table 75.20%；relation answer 57.81%，trace relation 51.56% |
| Stage AE 潜变量推理专家主动读取 agent | 主动读取能启动，且比 readout 更依赖证据；但四任务准确率低于 Stage AD，失败点进一步定位为 query policy，尤其是 relation 的左右对象 query | 1 seed：cell lookup full 90.23%，no 23.44%，shuffled 20.70%；count-only full 83.98%，no 12.11%，shuffled 8.59%；四任务 full 69.73%，no 2.93%，shuffled 29.30%；relation-only full 54.69%，no 61.33%，left/right pair query 34.38%/28.91%，但 pair reader row/col 100%/99.87% |
| Stage AF MoE 推理专家与分任务阶段训练 | MoE gate 能被监督到 100%，但没有自动修复 query policy；naive staged 严重遗忘，staged+replay 有缓解但低于 mixed/AE/AD | 1 seed：staged full 13.09%，gate 25%，前三任务 0%；mixed MoE full 69.34%，gate 100%；staged+replay full 64.45%，gate 100%；relation-only MoE full 53.12%，no-evidence 61.33%，left/right pair query 35.94%/30.08% |
| Stage AG Teacher-forced multi-step query trace | 暂停 MoE 是正确方向；teacher-forced query 在四任务上显著打开上限，但自由 query 仍弱，relation-only 仍未被正确 query 解开 | 1 seed：四任务 full 58.79%，no 16.80%，shuffled 21.29%，teacher-forced queries 83.59%；relation-only full 59.77%，no 60.55%，teacher-forced queries 59.77%；pair reader row/col 约 96%-97% |
| Stage AH Query alignment 与 process supervision 消融 | Process supervision 修了正确读取后的 compare 上限；朴素 CLIP-style query alignment 只小幅修 query，且会破坏 reader；detached-key 能保住 reader 但 query 改善不足 | 1 seed：relation query-only left/right 46.48%/47.27% 但 pair reader row/col 38.72%/34.74%；process-only teacher-forced 66.80%，row/col forced 100%；四任务 process-only full 62.70%，teacher-forced 84.18% |
| Stage AK 统一潜空间硬化 | 先定义并训练统一 object/pair latent bus 后，relation 的对象检索和位置比较同时闭合；这支持“潜空间未统一”是 Stage AE-AH 的主因之一 | 1 seed：pair occupancy exact 99.22%，pair row/col 100%/100%，left/right retrieval 100%/100%，teacher/model compare 99.61%；no-evidence compare 60.55% |
| Stage AL 统一潜空间接答案输出头 | 硬化后的统一 object/pair latent bus 可以直接支撑答案输出；最小链路 `bus -> retrieval -> compare -> answer` 闭合 | 1 seed：pair row/col 100%/100%，left/right retrieval 100%/100%，teacher/model compare 99.61%，teacher/model answer 99.61%，no-evidence answer 60.35% |
| Stage AM 历史最高难度统一 bus | 四任务混训时，cell/count 必须是一等 slot；count 不能靠普通 attention 自然学出，需要 count 输入专家或等价计数归纳偏置；relation compare 应读 decoded position state | 1 seed all-task：model answer 98.24%，color/shape/count answer 100%，relation answer 92.97%，count table exact 100%，no-evidence answer 27.54%；count-only：count table、selected count、answer 均 100% |
| Stage AI 图像生成与源图编辑输出专家 | 低熵合成图像生成/编辑链路可闭合，但正式 sweep 已饱和，不能继续区分 direct、latent output 或编辑架构强弱 | 3 seeds：prompt_direct、latent_output、latent_image_edit 的 nearest/head scene exact 均 100%；no-source nearest scene exact 3.42%，source-no-edit 0%；平均总耗时 250.092 秒，未触发 36 小时时间上限 |
| Stage AJ/AP Transformer 图像 IO 保真 | 全 Transformer patch 输入到 latent 再 patch decoder 完整重绘的链路可跑通；无辅助监督 memory-tree 有源图依赖但 copy 失败；对象属性/mask 辅助监督在正式规模闭合 copy、edit 和 generation | fixed baseline：transformer_edit scene exact 3.12%、transformer_copy 0%；无辅助 memory-tree：edit 11.91%、copy 0.59%；supervised copy formal：scene exact 100%、foreground MSE 0.000794；Stage AP formal：edit 99.02%、edit no-source 4.10%、source-no-edit 0%、text generate 100% |

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
- Stage V 直接测试“潜变量空间是否炸了”：对每个 expert token 训练/评估 16 cell occupancy/color/shape 还原，并做 source expert probe -> target expert transfer。ranking-only 的 diagonal cell info 为 57.20%，offdiag 只有 40.69%，language gap 16.50%，说明 expert 表示确实没有自然统一。
- Stage V 加共享 semantic decoder 监督后，下游 top1 从 19.92% 提到 23.57%，`no_patch` 从 14.06% 提到 21.22%，diagonal cell info 从 57.20% 提到 63.10%。这证明带监督训练能减少部分信息丢失，并让非 patch 路径承载更多视觉证据。
- Stage V 同时显示当前监督还没解决“语言不通”：offdiag transfer 40.69% -> 40.52%，没有改善；trained shared decoder 能读 patch/wide 到约 92%，但 object/spatial/count 只有约 55%-59%。所以问题不是单纯输出 scorer 弱，而是 object/spatial/count experts 没有形成可靠对象语义空间。
- Stage W 测试了四种更强对齐机制：共享 semantic grid decoder、object slot targets、cross-expert contrastive、训练期 common semantic bus。所有 mode 的诊断都保持 `inference_uses_common_bus=false`，最终 scorer 仍直接读 latent，不把显式 bus 当运行时中间层。
- Stage W 的最强主结果来自最小机制：`shared_grid_decoder` 把 Full Top-1 从 13.41% 提到 22.66%，`no_patch` 从 11.85% 提到 21.88%，且 language gap 从 19.32% 降到 15.71%。这支持“带监督公共读法”作为专家对齐 baseline。
- Stage W 没有证明四机制叠加更优：`all_four_train_only_bus` 的 common bus 自身 cell info 为 58.58%，但最终 Top-1 为 21.09%，低于 `shared_grid_decoder`；offdiag cell info 也只有 34.98%。因此显式 bus 适合早期 teacher/诊断/蒸馏，不适合作为最终架构的推理依赖。
- Stage X 把 Stage T/W 的差异拆开：Stage T 的 answer-class reconstruction 证明答案信息可保留；Stage W/X 的 semantic transfer 更严，要求完整场景语义和跨 expert 互读。两者不是同一指标，不能直接拿百分比比较。
- Stage X 显示 empty-cell baseline 约 59.1%，所以 55%-60% 的 cell info 基本不说明读懂场景。shared + Transformer 能把 patch/wide 分别读到 89.45%/86.36%，occupied color 接近 99%，说明 patch/wide 中确实有真实语义；但 object slots 只有 49.78%，scene exact 为 0，说明对象专家没有成型。
- Stage X 也显示参数/Transformer 不是唯一答案：MLP large 没有稳定优于 MLP small；Transformer 能读 patch/wide，但不能修复 object slots 或 patch/wide -> object transfer。object table set exact 全部 0%，最高 set recall 只有 11.45%。
- Stage X 进一步证明最终读头也弱：冻结 token 后单独训练 candidate scorer，shared 模型的 raw expert concat 能到 27.08%，高于原 full 22.53%。因此后续要同时修 objectization、alignment 和 readout，而不是只加一个 loss。
- Stage Y 修正了一个架构错误：object/spatial/count 等输入专家不应该串在 patch tokens 后面逐级变换，而应该并行直读外部图像；latent reasoner 和 answer scorer 再读这些输入专家输出。`parallel_direct` 从旧串联的 11.07% 提到 18.62%，且 no-image/no-patch 降到接近随机，说明它确实开始使用图像证据。
- Stage Y 也暴露了新的负结果：`parallel_direct` 去掉 object/spatial/count 后不掉点，answer reconstruction 最高只有 10.03%。拓扑问题修了，但功能专家仍不会自然分工，答案信息也没有充分压进 latent。
- Stage Z 证明显式监督是必要的：`supervised_no_teacher` 去掉 object/spatial/count 后从 22.66% 掉到 12.24%，说明这些专家开始真正承载任务信息。训练期 teacher 自身很强，cell info 98.61%、scene exact 80.73%，并把 top-1 推到 26.30%。
- Stage Z 的边界是 teacher 不等于最终解决方案：teacher 主要让 patch/teacher 路径更强，student latent 没完整继承语义；semantic offdiag 最高只有 37.42%，count positive offdiag 还下降，说明显式监督会让专家更专门化，但不会自动生成统一 latent 语言。
- Stage AA 把 scene/cell teacher 改成同位置 token 对齐后，object/spatial/count 的可读语义明显增强：semantic occupied color/shape 到 65.70%/48.13%，count positive diag 到 77.69%，teacher/cross-expert same-position retrieval 显著高于随机。
- Stage AA 的新发现是瓶颈从“信息不存在”转向“信息没被用上”：`token_aligned_teacher` top-1 只有 22.79%，低于 `supervised_no_teacher` 24.61%；去掉 object/spatial/count 仍有 21.09%。最终 scorer/route 没把对齐后的功能专家作为主证据。
- Stage AB 进一步测试输出端：对齐 prompt/answer 文本 latent 并加入 latent-to-answer runtime scorer 后，`text_latent_aligned` 达到 23.83%，高于同轮 `token_aligned_teacher` 21.09%，说明输出端对齐是正向尝试。
- Stage AB 同时给出两个负结果：latent-answer candidate top-1 只有 15.36%，只是略高于 12.5% 随机；retrieval 只有 3%-4% 且 cosine 很高，存在 collapse 风险。朴素 4-expert MoE reasoner top-1 只有 20.57%，gate entropy 1.37 接近均匀分配，说明“换成 MoE + balance loss”没有形成有效路由。
- Stage AC 按“问题/答案潜空间 -> 外部信息潜空间 -> 潜空间推理出答案 token -> 文本输出”的新路线重建实验。结果显示第一步和第二步可以同时高保真：问题与答案 token latent 都 100% 还原，外部事实表 latent 也 100% 还原 occupancy/color/shape。
- Stage AC 的关键负结果是第三步：即使前两步成功，普通 Transformer latent reasoner 在四任务上只有 40.04% answer word exact，no-evidence 为 27.34%，shuffled-evidence 为 26.56%。这说明 reasoner 有弱证据使用，但没有形成可靠潜空间推理。
- Stage AC 的 color-only 诊断更硬：只做 row/column -> color lookup 时，Q/A 与 evidence codec 都达到 100%，把 reasoner 从 1500 步拉到 5000 步仍约 37%。所以这不是简单训练步数问题。
- Stage AC 再次证明 cosine 不够：full answer latent cosine 为 94.96%，但答案 exact 只有 40.04%。后续不能把 latent cosine 或 MSE 当作潜空间推理成功证据。
- Stage AD 只调整 latent reasoner：文本 codec 和 evidence codec 在 reasoner 训练阶段冻结，新增 reasoner 内部的 cell/count readout、trace 监督和 reader 监督。四任务 answer word exact 从 40.04% 提到 74.41%，no/shuffled 均为 28.71%，说明收益来自证据使用。
- Stage AD 的 color-only 诊断把旧的约 39.45% 提到 91.80%，且 reasoner reader 能从冻结 evidence latent 100% 还原 occupancy/color/shape/count table。这支持新的判断：第三步不是只靠更多训练步数，而是需要能读潜空间证据并把结果写回 answer-token latent 的专门结构。
- Stage AD 仍未证明完整潜空间推理。target cell trace 只有 62.11%，relation answer 57.81%，trace relation 51.56%；当前 readout 更像可训练的潜空间值读取器，还不是严格的“先选对象/格子 -> 执行比较/计数 -> 输出答案 token latent”的完整链。
- Stage AE 把第三步改成主动读取 agent：reasoner 先发 query，queryable reader 从冻结 evidence latent 返回 observation，再写 answer-token latent。Cell lookup 达到 90.23%，count-only 达到 83.98%，说明 active read 能启动。
- Stage AE 四任务 full 为 69.73%，低于 Stage AD 的 74.41%，但 no-evidence 只有 2.93%，比 Stage AD 的 28.71% 更能说明答案依赖外部读取。它牺牲了部分准确率，换来了更接近最终 agent runtime 的证据依赖结构。
- Stage AE 的核心失败点是 query policy：relation-only 中 pair reader 已能把对象 row/col 读到 100%/99.87%，但 left/right pair query 只有 34.38%/28.91%，且 no-evidence 61.33% 高于 full 54.69%。所以 relation 不是 reader 没信息，而是 reasoner 没学会从问题生成正确对象查询。
- Stage AE 的高 trace 权重诊断也失败：`reasoner_trace_weight=2.0` 的四任务 full 为 69.34%，没有超过默认 69.73%。后续需要 teacher-forced query、分步 imitation、query contrastive/retrieval，而不是只把 trace loss 乘大。
- Stage AF 给 active reasoner 加 MoE answer writer 和 relation state writer。Mixed 训练下 MoE gate 能到 100%，但四任务 full 只有 69.34%，没有超过 Stage AE 的 69.73%，说明“专家路由正确”不等于“query policy 正确”。
- Stage AF 的 naive staged 训练是明确负结果：四任务 full 只有 13.09%，gate accuracy 25%，前三个任务 answer exact 为 0%。这是灾难性遗忘，不是专家分化成功。
- Stage AF 的 staged+replay 能把 full 拉回 64.45%，gate 100%，说明 replay 能缓解遗忘；但仍低于 mixed MoE，也低于 Stage AE/AD。
- Stage AF relation-only MoE 仍失败：full 53.12%，no-evidence 61.33%，left/right pair query 35.94%/30.08%。这再次说明 relation 的核心不是缺 MoE 专家，而是缺分步 query policy 和流程监督。
- Stage AG 暂停 MoE，改为 `trace_multistep` 和 `--query-teacher-forcing train`。四任务 teacher-forced queries 从 full 58.79% 提到 83.59%，说明正确读取 observation 后，answer-token latent 写入路径有明显上限。
- Stage AG 的自由查询仍然弱：四任务 left/right pair query 只有 35.94%/39.84%，relation trace 52.34%。所以 query policy 仍需要 contrastive/retrieval 或 scheduled sampling，不能只靠 CE。
- Stage AG 的新负发现是 relation-only teacher forcing 没有打开上限：full 59.77%，teacher-forced queries 59.77%，no-evidence 60.55%。pair reader row/col 已约 97%，说明 relation compare 和 answer latent 写入需要更硬的 row/col/delta/truth-table 中间监督。
- Stage AH 逐项验证后，process supervision 是当前最干净的正信号：relation-only teacher-forced 从 59.77% 提到 66.80%，四任务 full 从 58.79% 到 62.70%，teacher-forced 从 83.59% 到 84.18%，且 no-evidence 仍约 16.60%。
- Stage AH 的朴素 CLIP-style query alignment 有副作用：left/right query 从约 35%/45% 提到 46.48%/47.27%，但 pair reader row/col 掉到 38.72%/34.74%，说明 query-object 对齐会破坏 evidence token 可读性。
- Stage AH detached-key query alignment 保住 reader row/col 到 98.40%/96.15%，但 query 只有 39.06%/41.80%。所以“冻结 evidence tower，只训 query tower”方向更安全，但当前训练信号仍不够强。
- Stage AH combined 没有叠加收益：teacher-forced 只有 59.77%，说明 query alignment、reader loss、process loss 的梯度目标互相牵制。下一步应改成分阶段预训练，而不是同时加权。
- Stage AK 按“先把潜空间练硬”的思路，先训练统一 pair/object latent bus，不接最终 answer decoder。结果 pair slot row/col、left/right query retrieval、relation op 和 model selected compare 全部达到约 99%-100%。
- Stage AK 的 no-evidence 诊断也符合预期：left/right retrieval 仍为 100%，因为问题文本本身包含对象 color-shape；但 row/col 回到约 25% 随机附近，compare 只有 60.55%。这说明 full compare 的 99.61% 来自 evidence latent 中的位置事实。
- Stage AK 说明 Stage AE-AH 的主要失败不必先归因到架构错误；更可能是潜空间一开始没有统一，后期补 loss 把多个局部 latent 空间硬接在一起。
- Stage AL 在同一个统一 bus 上接最小 answer writer，model-selected answer 达到 99.61%，与 compare 99.61% 对齐。no-evidence answer 只有 60.35%，说明输出头确实依赖 evidence slot 中的位置事实。
- Stage AL 的结论是阶段性的：它不是完整 answer-token decoder，也不是四任务 reasoner；但已经证明硬潜空间可以支撑 `query -> observe/select -> compare -> answer` 的最小输出链。
- Stage AM 把任务拉回 `color_at_cell`、`shape_at_cell`、`count_color_shape`、`relation_yes_no` 四任务。只把 count head 接在 pair slots 上时，count answer 只有 42.19%；独立 count slots 但仍用 softmax attention 时，count answer 约 54%-56%；additive/sigmoid count 聚合也失败。这说明 count slot 需要一等 count 输入专家或等价计数归纳偏置。
- Stage AM 的 count 输入专家闭合了 count-only：count table exact、selected count value 和 count answer 都达到 100%，no-evidence count answer 只有 9.57%。这验证了“cell/count slots 没做硬会挂”这个担忧。
- Stage AM 最终 all-task 在单 seed 上达到 model answer 98.24%，color/shape/count answer 均 100%，relation answer 92.97%，count table exact 100%，no-evidence answer 27.54%。relation 一开始只有约 80%，把 compare context 从 raw pair slot 改为 decoded row/col position state 后升到 92.97%。
- Stage AM 的边界也很明确：当前 count 输入专家利用的是结构化 evidence 中的 one-hot color/shape，不证明真实图像计数；真实任务需要 detector/segmentation/counting expert 先产生同等质量的 count slots。
- Stage AI 证明了低熵图像输出链路能闭合，但正式 sweep 已经饱和：prompt direct、latent output、latent image edit 都达到 100% scene exact。因此它不能再证明 latent 输出专家更优，也不能外推到真实图像生成能力。
- Stage AI 的 no-source/source-no-edit 消融仍有效：no-source 只有 3.42%，source-no-edit 为 0%，说明源图编辑不是简单猜测或复制。但任务仍只是 64x64 单物体合成图，不含自然图像、多对象、遮挡、局部 mask、风格迁移或扩散采样。
- Stage AJ 更贴近“整图输入 -> latent -> 完整重绘”，fixed baseline 结果是负的：Transformer patch decoder 可以学到背景纹理，foreground/object 保真失败；`transformer_edit` scene exact 只有 3.12%，`transformer_copy` 为 0%。
- Stage AJ 暴露了一个指标风险：全图 pixel MSE 会被背景主导。`transformer_edit` 的 background MSE 只有 0.002343，但 foreground MSE 是 0.182054，说明只看全图误差会掩盖对象语义失败。
- Stage AJ 的 `memory_tree_copy/memory_tree_edit` 正式单 seed 已完成：edit scene exact 从 fixed baseline 的 3.12% 提升到 11.91%，no-source 只有 3.71%，说明源图依赖增强；但 copy scene exact 只有 0.59%，foreground MSE 0.222561，完整重绘保真仍未闭合。
- Stage AJ 的 memory-tree 正信号主要来自位置编辑：`position_edit` scene exact 为 28.02%，`color_edit` 只有 1.95%，`shape_edit` 只有 3.98%。视觉抽检也显示生成的是模糊物体或多重影子，而不是稳定对象。
- Stage AJ 的 copy-only 辅助监督是强正信号：`memory_tree_supervised_copy` 在 4096/512/512、1500 step 正式单 seed 上达到 test scene exact 100%、foreground MSE 0.000794、aux scene 100%、aux mask IoU 100%。这说明 copy 失败点主要是 latent 没有对象表约束，而不是 Transformer patch decoder 无法重绘对象。
- Stage AJ 的监督 edit/generation probe 先打开能力边界：`memory_tree_supervised_edit` 在 2048/256/256、600 step 上达到 scene exact 99.22%，而 no-source 只有 5.08%；`text_supervised_generate` 在规范背景上达到 100% scene exact。
- Stage AP 已补完正式 edit/generation 长训：d_model 192、train/val/test 为 4096/512/512、edit/generate 各 1500 step、总耗时 3039.263 秒。正式 test 中 `memory_tree_supervised_edit` scene exact 99.02%，no-source 4.10%，source-no-edit 0%；`text_supervised_generate` scene exact 100%。这说明源图 latent 对保留未修改属性和背景有因果作用，也说明规范背景文本生成链路正式闭合。
- Stage AJ 训练瓶颈曾被评估路径放大：旧版 nearest-template 指标逐样本构造 120 个候选图，并在一个 batch 内重复解析两遍，导致 CPU/Python 调度拖住 GPU。现已改为批量 GPU template parser、训练期 `train_eval_size` 子集和 `--torch-num-threads 1`。
- Stage AI/AJ 都不能外推到人类审美质量、真实照片保真度或复杂编辑一致性。后续必须加入 foreground/object/scene exact、样例 PNG、局部保持、完整复制和编辑一致性门禁。

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
11. Stage V 之后，所有新增 expert 都应先通过信息还原和 cross-expert transfer 诊断，再看最终任务准确率；否则容易把 raw patch/wide latent 的收益误判成 expert 收益。
12. Stage W 之后，common semantic bus 只能作为初期训练 teacher、诊断器或蒸馏目标；最终 scorer/输出专家仍应直读 latent。contrastive alignment 与 slot-level targets 需要继续做，但要避免把显式公共语义表变成运行时主路径。
13. Stage Y 之后，输入专家拓扑应固定为“外部信息直读 -> latent reasoner -> latent 输出/answer scorer”，不要再回到 Stage U 那种串联专家链路。
14. Stage Z/AA 之后，object/spatial/count 专家应保留显式监督和 token-level 对齐，但蒸馏目标要从全局 scene/cell bus 继续下沉到 object/cell-level，例如 DETR-like set prediction、Hungarian matching、objectness、cell/box、color、shape 和同对象 token contrastive。
15. Stage AA/AB 之后，下一步重点应改最终 aggregator/scorer：让答案候选显式 cross-attend 到 object/spatial/count 的 aligned cell/object tokens，或者加入 expert usage supervision；继续只加全局 align loss 的边际收益已经很低。
16. Stage AB 之后，latent-to-answer 专家应从轻量候选 scorer 升级为更强的 contrastive/autoregressive 输出专家；同时要控制 latent/answer cosine collapse，不能只看 cosine 高。
17. Stage AB 之后，MoE router 需要任务族、证据类型或专家使用监督；不要再只把普通 reasoner 替换成 MoE 并加 balance loss。
18. Stage AC/AD/AE 之后，第三步必须单独设计训练目标：target cell token、selected object token、count accumulator、relation pair token 等中间 latent 操作应被显式监督。
19. Stage AC/AD/AE 之后，evidence latent 不能只要求“decoder 能读出事实”；它还必须对 reasoner 可操作，例如固定 cell/object/count-pair token 坐标、可检索对象表或可微 lookup 结构。Stage AD/AE 已证明 reasoner 内部 readout/queryable reader 是正向路径，但 selector 和 query policy 还没严格对齐。
20. Stage AC/AD/AE 之后，answer-token latent 需要离散分离或 token-level contrastive 约束；只用 MSE/cosine 靠近答案 latent 会产生高 cosine、低 exact 的假成功。
21. Stage AM 之后，不应继续优先做 MoE；先把统一 latent bus 接回 Stage AC 的 answer-token latent writer，验证分类答案头能否升级到文本 answer token latent。
22. Stage AM 已验证 cell/count slots 必须一等化；下一步应把当前结构化 count 输入专家替换成可训练的 detector/segmentation/counting expert，再测真实或更高熵视觉任务。
23. Stage AM 之后，relation 还没回到 Stage AL 的 99.61%；下一步应加入 relation truth-table/delta slots 或更硬的 process supervision，而不是只加 compare loss 权重。
24. Stage AF 之后，若继续 staged 训练，必须有 replay buffer、蒸馏、正则化或冻结策略；不要再做单向无 replay 的阶段训练。
25. Stage AC/AD/AE/AF/AG 下一轮应加入 direct structured baseline，确认任务本身和训练预算不是瓶颈。
26. DocVQA 下一步应接入预训练专家：OCR/text/layout encoder 或现有 VLM，把高分辨率文档理解交给专家，再测试 latent bus 与 agent decoder。
27. 真实工具边界可并行推进：用 Playwright/本地 HTML 页面替换合成 UI+DOM DSL，保留截图 + DOM + 工具历史 + latent bottleneck 的评估结构。
28. 证据审计下一步应接入真实文件形态：小型 HTML/PDF/CSV/截图组合，要求模型调用受控工具抽取证据并输出带引用的 audit report。
29. 如果继续走真实多模态路线，应把 Stage D 的短期 legend 扩展成局部地图记忆：加入遮挡、错误探测、不可重置探测成本、探索路径规划和多目标任务。
30. 如果要继续验证 KV 剪枝，需要选定一个可控推理服务栈，确认是否暴露 prefix/segment 级 KV 生命周期 API。
31. 若要验证潜空间训练路线，应单独建训练实验，不要把未验证训练假设混入现有 work-tree runtime。
32. Stage AI 之后，不要继续在低熵单物体模板上加 seed 或拉长训练；它已经饱和。图像生成/编辑验证应直接切到 Stage AJ 这类“整图输入、prompt 不可见背景、latent 完整重绘”的保真任务。
33. Stage AJ/AP 之后，图像输出路线可以继续用 Transformer patch decoder；`memory_tree_supervised_copy`、`memory_tree_supervised_edit` 和 `text_supervised_generate` 都已有正式单 seed 正结果。
34. Stage AJ 之后，copy 重绘仍是编辑前置门禁：源图完整重绘必须先过 foreground scene exact，再测试 edit prompt 修改能力；不能只看背景 MSE 或全图 pixel MSE。
35. 下一步应把对象属性/mask 读头扩展为可修改对象表，并提高任务熵；不要直接把旧无辅助 `memory_tree_edit` 继续拉长训练。
36. 可以小规模测试 patch size 4 是否改善对象边界，但必须和当前 supervised copy 对照，同时记录显存、吞吐和 foreground scene exact；不要把更高 token 成本误判为架构进步。
37. 更长图像训练前应加 checkpoint、中间 PNG 样例和断点恢复；当前 Stage AI/AJ 脚本只适合快速 GPU 验证和 sweep 聚合。
