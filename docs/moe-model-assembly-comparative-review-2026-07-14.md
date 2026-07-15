# Project-Yggdrasil 的 MoE 与模型组装思路：公开路线对照

日期：2026-07-14  
检索范围：截至 2026-07-14 可公开检索的论文、技术报告和官方模型说明

## 结论

Project-Yggdrasil 当前 V2 的 MoE 设计不是一个尚未有人尝试的孤立点子，但也没有证据表明已有公开模型完整采用了本项目的组合架构。

已经被其他模型分别验证的部分包括：

1. 在 Transformer/LLM 内部使用稀疏 FFN-MoE，以较低 active compute 换取更大的总参数容量。
2. 为不同模态、视觉子任务或任务族保留不同专家，再通过 connector、投影层或共享模块汇合。
3. 同时在视觉/输入侧和语言/核心侧做两层专家，并用渐进式训练、任务条件或模态信号引导路由。
4. 先让专家形成稳定的专长，再谈路由、负载和通信优化。

尚未被公开结果完整证明的部分是 V2 的整体闭环：

> 异构边界专家 → 统一宽度的连续 latent workspace → 内部 FFN-MoE → 显式 READ/REASON/AUDIT/EMIT/STOP 控制 → 可因果审计的 latent readout。

因此，本项目的主要差异不在“发明了 MoE”，而在于试图把 MoE 放到“异构外设组装 + 连续潜变量推理核心 + 显式控制面”这一更大的系统边界内。这个系统级组合仍属于待验证架构假设。

## 1. 本项目到底提出了什么

当前 V2 白皮书把专家体系拆成两层：

- `Boundary-MoE`：输入/输出边界专家可以拥有不同 tokenizer、内部宽度、token 数、空间结构和候选表，只在进入 latent workspace 前投影到统一 `D_latent`；第一版按显式 modality/type 路由。
- `FFN-MoE`：latent Transformer 内部的每个 latent token 在 residual stream 宽度不变的前提下选择 FFN expert，以增加条件计算容量。

这不是“所有专家都在同一个 Transformer 里竞争”的普通 MoE。它还包含三个设计约束：

- 输入专家应并行直读原始外部信息，不把一个专家变成另一个专家的唯一输入；
- 专家必须先有一等语义、监督和单独门禁，router accuracy 不能替代最终任务 exact 或 no-expert 消融；
- Boundary-MoE 与 FFN-MoE 必须先拆开验证，再做 2×2 组合，否则无法判断收益来自接口、核心容量还是总参数增加。

当前项目状态必须单独标注：V2-A 只完成文本基座和 latent foundation 的机制/代理验证；V2-B 才计划实现 Boundary-MoE 和 FFN-MoE，而 V2-A 尚未通过，V2-B 尚未启动。因此当前仓库没有正式的 V2 MoE 质量或成本结果。

## 2. 与公开路线的对应关系

| 公开模型/方法 | 与本项目重合的层 | 他们实际做了什么 | 已报告成果 | 与本项目的关键差异 |
| --- | --- | --- | --- | --- |
| VLMo / MoME Transformer | 边界模态专家 + 共享核心 | 每个 block 有模态专家和共享 self-attention；同一 backbone 可作 dual encoder 或 fusion encoder | 在 VQA、NLVR2、图文检索等任务上报告 SOTA；消融显示 MoME 比标准 Transformer 更好 | 不是连续 latent recurrence，没有显式控制面，也不是输入/输出边界与内部 FFN-MoE 的双层系统。来源：[论文](https://arxiv.org/abs/2111.02358) |
| Uni-MoE | 模态专用 encoder/connector + LLM 内 MoE | 不同模态使用专用 encoder 和 connector，形成统一多模态表征；LLM 内使用稀疏 MoE；采用对齐→专家偏好→LoRA 混合调优的渐进训练 | 报告降低混合模态数据上的性能偏置，并改善专家协作与泛化 | 统一的是 connector/LLM 表征，不是本项目定义的 latent workspace；论文没有验证 READ/REASON/审计链。来源：[论文](https://arxiv.org/abs/2405.11273) |
| MoME | 最接近“边界视觉专家 + 核心语言专家”两层组装 | `MoVE` 动态聚合 CLIP、DINOv2、Pix2Struct 等视觉专家；`MoLE` 将稀疏专家/adapter 插入 LLM 的 FFN | 作者报告 MoVE 在全部 VL 任务平均提升 12.87 个百分点，Document 组提升超过 20 个百分点；路由分布出现任务族专长 | 仍是任务条件的视觉/语言路径选择，不是连续递归潜变量，也没有本项目的代码控制面。受限于算力，作者未扩展到更多数据和模态。来源：[论文](https://arxiv.org/abs/2407.12709) |
| Uni-Med | Boundary/connector MoE | 在视觉特征与 LLM 之间放置 connector-MoE，用投影专家解决六类医学任务间的梯度竞争 | 报告在不同配置下最多平均提升约 8%，覆盖 QA、VQA、报告生成、指代表达和分类等六类任务 | 这是 connector 级的边界路由，不是任意异构转换器接入统一 latent workspace。来源：[论文](https://arxiv.org/abs/2409.17508) |
| DeepSeek-VL2 | 模态输入 + 内部 DeepSeekMoE | 动态 tiling 视觉编码器接入 DeepSeekMoE 语言模型；MoE 主要位于语言核心，视觉侧仍通过视觉编码器/adapter 接入 | Tiny/Small/主模型分别约 1.0B/2.8B/4.5B active parameters；论文报告在 VQA、OCR、文档/表格/图表理解和 grounding 上达到有竞争力或 SOTA | 是“视觉 encoder + adaptor + MoE language model”，不是 Boundary-MoE 与 latent FFN-MoE 的完整对偶结构。来源：[论文](https://arxiv.org/abs/2412.10302) |
| MoE-LLaVA | 内部稀疏专家 | 用 router 只激活 top-k 专家，训练阶段采用分阶段 MoE-Tuning | 约 3B active parameters 时，在多项视觉理解任务上接近 LLaVA-1.5-7B，并在 object hallucination 基准上超过 LLaVA-1.5-13B | 主要是 LVLM 内部稀疏化，没有本项目的异构边界专家或连续 latent recurrence。来源：[论文](https://arxiv.org/abs/2401.15947) |
| DeepSeekMoE / DeepSeek-V2/V3 | 内部 FFN-MoE、共享专家 + 路由专家 | 细粒度拆分 routed experts，同时保留 shared experts；DeepSeek-V3 为 671B 总参数、37B active/token | DeepSeekMoE 2B 报告接近 GShard 2.9B，16B 以约 40% 计算量接近 LLaMA2 7B；DeepSeek-V3 官方报告 14.8T tokens、671B/37B，并在多项数学、代码和通用基准上达到强结果 | 这是本项目 FFN-MoE 的强先例，但它不证明 Boundary-MoE、连续 latent recurrence 或显式审计控制。来源：[DeepSeekMoE](https://arxiv.org/abs/2401.06066)、[DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) |
| Qwen3 / Qwen3-VL | 内部 MoE，以及多模态模型中的 MoE 变体 | Qwen3-30B-A3B 和 235B-A22B 使用 128 experts、每 token 激活 8 个；Qwen3-VL 也提供 dense 与 MoE 规模档位 | 官方报告 Qwen3-30B-A3B 以约 3B active parameters 超过 QwQ-32B；Qwen3-VL 把 dense/MoE 用作质量—延迟折中 | 仍主要是 MoE backbone 的规模化，不等于本项目的边界专家—latent workspace—内部 MoE 组装。来源：[Qwen3 官方说明](https://qwenlm.github.io/blog/qwen3/)、[Qwen3-VL 技术报告](https://arxiv.org/abs/2511.21631) |

## 3. 这些路线已经证明了什么

### 3.1 核心 FFN-MoE 已经是成熟的规模化技术

DeepSeekMoE、DeepSeek-V2/V3、Mixtral 和 Qwen3 共同说明：当目标是扩大总参数容量、保持较低每 token active compute、提升训练/推理性价比时，内部 FFN-MoE 可以取得实用成果。DeepSeek-V3 官方表格给出 671B 总参数、37B active/token，并报告其在 MMLU-Pro、DROP、代码、数学等项目上的强结果；这类结果支持“条件计算可以扩容”，但不支持“专家自然形成可靠任务语义”或“MoE 会自动修复推理状态”。

DeepSeekMoE 的 shared experts + routed experts 也与本项目的“共享语义骨架 + 条件专家”直觉相近。它验证的是专家分解、共享知识和负载/计算效率，不是把共享核心换成连续 latent recurrence。

### 3.2 输入/边界专家组装确实能降低任务干扰

VLMo、Uni-MoE、MoME 和 Uni-Med 的共同结论是：视觉、文本、文档、grounding、报告生成等任务对输入表征和优化方向的需求并不相同，把所有任务强行压进一条完全共享路径会产生干扰；在视觉侧、connector 侧或模态侧保留专家，可以改善任务覆盖、表征对齐或平均性能。

其中 MoME 与本项目最接近，因为它同时处理视觉专家和语言专家，并且将视觉专家输出变换到统一长度后再动态聚合。但它仍然是“任务条件的多模态表征/语言路径选择”，不是“把不同模态接入连续工作区后，让 latent reasoner 递归更新状态”。

### 3.3 路由优化本身正在成为独立问题

截至 2026 年，SMoES 这类工作已经不再只问“有没有 MoE”，而是问模态信号如何影响不同层的专家专长、通信和负载。该方法在 4 个 MoE-VLM、16 个基准上报告多模态任务平均提升 0.9%、语言任务平均提升 4.2%、专家并行通信开销下降 56.1%、吞吐提高 12.3%。

这与本项目先用显式 modality/type routing、分别记录 Boundary router 与 FFN router 的使用率/熵/collapse，再考虑更复杂语义路由的顺序一致。不过，SMoES 是对已有 MoE-VLM 的路由优化，不是 Project-Yggdrasil 完整架构的验证。来源：[SMoES](https://arxiv.org/abs/2604.23996)。

## 4. 与本项目已有实验的关系

旧路线的 MoE 结果不能直接升级为 V2 MoE 结果。旧实验已经给出一个重要的负结果：router 可以训练到较高准确率，但最终任务不一定改善；无 replay 的 staged MoE 还出现灾难性遗忘，staged+replay 只能缓解，未超过 mixed 或更结构化的 unified bus。由此形成的规则是：先让专家和 slot 语义成型，再让 router 选择；路由指标必须与最终 exact、no-expert/no-modality 消融一起报告。

这条规则与公开路线的经验相容，但证据性质不同：

- 公开模型证明某些专家组装方式在特定数据和规模上能工作；
- 本项目旧实验证明在当前代理任务和训练合同下，MoE 不是自动修复 latent 对齐和 query policy 的工具；
- 当前 V2-A1.5 只证明结构化 recurrent core 在 ordinary split 可拟合，同时 composition-heldout 失败；它还没有给 Boundary-MoE 或 FFN-MoE 提供正式结果。

因此，不能用 DeepSeek/Qwen 的 FFN-MoE 成功反推 V2 的 latent MoE 已经成立，也不能用旧代理 MoE 的失败反推目标架构不可行。两者都只约束下一轮实验如何设计。

## 5. 对 V2-B 的直接启示

当前测试计划中的顺序是合理的，且应保持：

1. 先做 `Boundary-MoE + Dense core`，用普通统一接口/早期拼接作对照；第一版使用显式 modality routing，不要一开始就训练语义 router。
2. 先比较 raw expert tokens、pump-only、pump+residual/bypass，并同时看重建、最终任务、no/shuffled modality 消融和成本。
3. 再做普通接口/Dense、Boundary/Dense、普通接口/FFN-MoE、Boundary/FFN-MoE 的 2×2；Boundary router 和 FFN router 分开记录，不能只给一个总 router accuracy。
4. 必须用多 seed、active compute、通信/吞吐、expert collapse、任务族分布和专家去除实验判断净收益；如果只是总参数增加而质量或 active-compute 没有收益，应回退 Dense。

这意味着 V2-B 的可证伪点不是“能不能把代码拼起来”，而是：在相同任务质量和公平计算预算下，边界异构性是否降低模态/任务干扰，内部 FFN-MoE 是否提供额外容量，以及二者组合是否有可重复的非加和收益。

## 6. 最终判断与未完成事项

### 已完成

- 核对当前 V2 白皮书、测试计划和旧路线经验，确认本项目的 MoE 不是普通单层 FFN-MoE。
- 检索并比较 VLMo、Uni-MoE、MoME、Uni-Med、DeepSeek-VL2、MoE-LLaVA、DeepSeekMoE/DeepSeek-V3、Qwen3 和 SMoES。
- 确认相近思想已经在其他模型/研究原型中分别取得了质量、容量、active-compute、任务干扰或通信效率方面的成果。
- 明确没有证据表明公开模型已经完整实现并验证 Project-Yggdrasil 的连续 latent workspace + 双层 MoE + 显式控制闭环。

### 未完成

- 本项目没有完成 V2-B1 Boundary-MoE formal；没有 V2-B3 FFN-MoE 2×2 结果。
- 没有用统一数据、统一 active compute、统一输出质量和多 seed 对本项目方案与 DeepSeek/Qwen 类 MoE backbone 做可比实验。
- 没有证据证明本项目的连续 latent recurrence 会给 MoE 带来超出标准 token-space Transformer 的独立收益。

上述未完成项不是文档缺口，而是当前架构尚未获得外部可比证据的真实边界。
