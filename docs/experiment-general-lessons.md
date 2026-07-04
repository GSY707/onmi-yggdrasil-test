# 当前实验通用经验

本文按“任务线索”记录这几天真正发生过的变化：先遇到什么失败，后来怎样修，最后沉淀成什么后续规则。这里不按 Stage 编号聚合；Stage 编号只作为证据来源，详细指标仍回到各实验报告。

## 总体变化

这几天路线从“证明某条 latent 链路能跑通”，逐步转成“失败后拆瓶颈、修 latent 空间、修专家分工、修输出头和修长训基础设施”。

最重要的变化是：我们一开始把 text codec、evidence codec、reasoner、answer writer 分开训练，再用局部 loss 把它们接起来。Stage AC-AH 暴露出这种方式会让潜变量空间没有统一语义：单独看 Q/A codec 和 evidence codec 都很高保真，但 reasoner 读不动、query 不准、compare 不稳，最终答案挂。Stage AK 之后改成先训练统一 object/pair latent bus，让证据 slot、文本 query、compare 使用同一套对象语言，relation 立刻闭合。这说明之前很多失败不是“任务学不了”，而是潜空间对齐和结构化对象语言没有提前建好。

## 任务线经验

### 1. 从“互译成功”改成“统一潜空间先行”

失败表现：Q/A token latent 和 evidence latent 可以各自高保真还原，但普通 latent reasoner 仍然不能稳定输出答案。color-only 这种简化任务里，前两步都能到 100%，reasoner 仍在低位徘徊。这说明“两个空间都能被 decoder 读懂”不等于“reasoner 能在两个空间之间操作”。

中间修复：我们尝试只修 reasoner，加入 explicit readout、trace、reader supervision、active read、teacher-forced query、process supervision 和 query alignment。这些都给出局部正信号：readout 能增强证据使用，active read 让 no-evidence 更明显下降，process supervision 能修正确读取后的 compare 上限。

修复失败点：这些补丁没有解决根因。query alignment 会破坏 reader，detached-key 能保住 reader 但 query 仍弱；process supervision 能修 teacher-forced 上限但修不了自由 query；多个 loss 合在一起会互相牵制。

最终变化：Stage AK 把统一 object/pair latent bus 前置，不急着训练最终 answer-token reasoner，而是先让 evidence slot、question query、relation op 和 compare 在同一对象语言里闭合。这个变化把问题从“后期补 loss”切到“先建统一潜空间”。

后续规则：

1. 新任务先问 latent 空间有没有统一语义，再问 reasoner 强不强。
2. 不要把多个局部训练出的 latent 空间硬接起来后再靠 trace/readout/query loss 补。
3. 先训练可读、可检索、可比较的 slot/bus，再接 reasoner、MoE 或 answer writer。
4. 任何“潜空间推理成功”都必须有 no-evidence、shuffled-evidence、slot readout、query retrieval 和 final exact 同时支持。

### 2. 从“加更多 reasoner 训练”改成“拆 query、reader、compare、writer”

失败表现：reasoner full answer 低的时候，直觉上容易继续加步数、加层数、加 MoE 或加 trace loss。但实验显示失败不是单一训练不足：reader 有时已经能读出对象 row/col，query 却找错对象；teacher-forced query 后 compare 仍可能不够；writer 也可能没有把正确中间状态写成答案 token。

中间修复：我们把任务拆成 query policy、queryable reader、relation compare、answer-token writer、process trace。拆开后能看到具体短板：relation 中经常不是 reader 没信息，而是 left/right object query 没学会；有时是正确读取后 compare/truth-table 不够硬。

最终变化：后续不再把 reasoner 当一个黑盒优化目标，而是把中间操作显式化：target cell、selected object、count accumulator、pair row/col、delta/truth-table、answer token 都要能被 probe 或监督。

后续规则：

1. 失败时先定位 query、reader、compare、writer 哪个环节挂，不要直接归因到整体架构。
2. relation/count/cell lookup 不能只靠最终 answer loss。
3. teacher-forced 上限、自由 query、no-evidence gap 要分开报告。
4. 如果 teacher-forced 高而自由 query 低，下一步是 query policy 或 scheduled sampling，不是继续加 answer loss。

### 3. 从“Attention Pump/Resampler 压缩”改成“保留原始证据和 residual 通道”

失败表现：早期用 Attention Pump 或固定数量 latent slots 作为唯一信息通道，很多视觉和多物体任务会丢信息。最终 scorer 换了也修不好，因为信息在 pump/thought latent 阶段已经丢了。

中间修复：我们做 reconstruction probe 和 no-pump 对照，确认信息主要还在 raw expert tokens、patch/wide tokens 或 expert concat 中。Stage T 的 wide residual、raw token、route-weighted token 和 summary token 组合明显恢复，纯 32-slot resampler 仍失败。

最终变化：第一层默认不再强压缩。raw evidence tokens 或 residual passthrough 要保留，summary token 只能追加，不能覆盖原始证据。

后续规则：

1. 入口处不要为了 token 数好看而过早丢证据。
2. resampler/Attention Pump 可以是视图之一，不能作为唯一证据路径。
3. token 成本应交给后续记忆树/工作树分层摘要，不在输入第一层硬压缩。
4. 新压缩器必须先过 reconstruction probe、no-image/no-evidence 消融和最终任务 exact。

### 4. 从“专家会自然分工”改成“专家必须有一等语义和监督”

失败表现：只加 object/spatial/count 专家并不会自然分工。很多时候去掉这些专家不掉点，说明模型主要还在靠 patch/wide/raw tokens；object slots 没有对象化，跨专家互读也弱。

中间修复：我们加了 shared decoder、slot targets、contrastive alignment、training-time common bus、teacher distillation、token-level alignment。它们证明监督方向有用：功能专家开始承载信息，token 对齐降低信息丢失。

修复边界：这些方法没有自动解决最终任务。teacher 很强不代表 student latent 继承完整语义；common bus 能做训练期 teacher 或诊断器，但会和主任务优化竞争；最终 scorer 可能仍不使用对齐后的专家。

最终变化：专家分工从“希望自然出现”改为“明确训练对象语言”。object/cell-level set matching、objectness、cell/box、color、shape、count、同对象 token contrastive 等应成为专家训练目标。

后续规则：

1. 每个新增专家都要有自己的语义门禁，而不是只看最终任务准确率。
2. 去掉专家不掉分时，不能宣称专家有效。
3. common semantic bus 只作为训练期 teacher、诊断器或蒸馏目标，不默认作为推理主路径。
4. 最终 scorer/answer writer 必须显式读取 aligned object/cell/count tokens，否则对齐成果可能被浪费。

### 5. 从“结构化 count 能靠 attention 学出”改成“count slot 一等化”

失败表现：在四任务统一 bus 里，color/shape/relation 可以推进，但 count 会挂。把 count head 接在 pair slots 上不够，softmax attention 读 objects 不够，给 selected count 加 loss 也不稳定，非归一化聚合也没闭合。

修复：切到一等 count 输入专家，让 count slots 直接承载每个 color-shape pair 的计数。count-only 和 all-task 的 count 才闭合。

最终变化：count 不再被当作普通 attention 能顺手学出的副产物。它需要 count 输入专家、count slots 或等价的计数归纳偏置。

后续规则：

1. cell/count/object/pair slots 都要成为一等 latent bus 成员。
2. 真实视觉计数不能让 reasoner 自己从 patch 里“悟出来”，应先有 detector/segmentation/counting expert 产出可验证 count slots。
3. count answer、count table exact、selected count value、no-evidence count 都要同时报告。

### 6. 从“输入专家串联加工”改成“外部信息并行直读”

失败表现：串联输入专家让 object/spatial/count 等模块吃前一级变换后的 patch tokens，结果路径更长、信息更混，功能专家没有自然分工。

修复：改成输入专家并行直读外部信息，latent reasoner 和 answer scorer 再读取各专家输出。这个拓扑比旧串联更合理，也更符合后续真实专家接入。

后续规则：

1. 输入专家负责读外部信息，reasoner 负责推理，不要把两者串成一条越来越长的变换链。
2. 专家之间可以共享监督目标，但不要让一个专家成为另一个专家唯一输入。
3. 真实 OCR/layout/DOM/detector/counting expert 接入时，也应按并行直读口径设计。

### 7. 从“MoE 可能自动修复”改成“MoE 只在专家已成型后使用”

失败表现：router 可以训到很高，但任务不一定变好。朴素 MoE reasoner、balance loss 或无 replay 分阶段训练没有自动修复 query policy；naive staged 还出现灾难性遗忘。

修复：staged+replay 能缓解遗忘，但没有超过 mixed 或更结构化的统一 bus 路线。强专家能带来收益，但这是专家本身强，不是 MoE 自动解决问题。

后续规则：

1. MoE 不是修复未对齐 latent 空间的工具。
2. 没有任务族、证据类型、专家使用监督、replay、蒸馏或冻结策略时，不再做无 replay staged MoE。
3. 先让专家和 slot 语义成型，再让 router 选择专家。
4. router accuracy 要和最终 exact、no-expert/no-modality 消融一起看。

### 8. 从“生成链路能画图”改成“必须过对象保真门禁”

失败表现：图像输出中，全图 MSE 会被背景主导。模型可以学到背景纹理，却完全丢对象；copy/edit 看起来 loss 不差，但 scene exact、foreground MSE 和视觉样例都显示失败。

修复：加入对象属性和 mask 辅助监督后，copy-only 正式门禁闭合；edit/generation 在 probe 规模出现强正信号。这个变化说明 patch decoder 不是完全不会画对象，真正瓶颈是 latent 没有可操作对象约束。

后续规则：

1. 图像输出必须同时报告 foreground/object/scene 指标，不能只看全图 MSE。
2. copy、edit、generation 要分开；copy 正式闭合不能替代 edit/generation 正式闭合。
3. no-source、source-no-edit、对象属性、mask IoU、样例 PNG 都是必需证据。
4. 无辅助监督 memory-tree copy/edit 已经是负对照，不应继续拉长旧配置。

### 9. 从“长训靠一次跑完”改成“恢复链是前置条件”

失败表现：正式 edit/generation 长训超过工具超时没有写出 JSON，说明没有恢复链时长训结果容易丢失，也无法判断是训练失败、工具中断还是环境问题。

修复：Stage AJ P0 加入 latest/best checkpoint、resume、训练中样例、schema_version=2、设备记录、stop-after-steps 恢复链 smoke，并验证 resume 与不中断对照关键指标一致。

后续规则：

1. 超过 30 分钟的正式训练必须先有 checkpoint/resume。
2. latest 用于继续训练，best 用于最终评估，不能混用。
3. JSON 必须记录 config、seed、设备、checkpoint 目录、resume 状态、best checkpoint、训练步数和停止原因。
4. 长训中途必须落样例图或 trace，避免最后才发现模型只学了背景或走偏。

### 10. 从“训练慢就是模型问题”改成“先查数据和评估路径”

失败表现：Stage AJ 训练曾看起来 GPU 低功率、像是模型训练慢；后来发现旧评估路径逐样本构造大量候选图，还在 batch 内重复解析，CPU/Python 调度拖住 GPU。

修复：改成批量 GPU template parser、训练期评估子集、GPU 常驻数据、AMP、合适 batch size 和 `--torch-num-threads 1`。

后续规则：

1. 遇到训练慢，先查数据加载、评估循环、CPU/Python 串行逻辑和 GPU 利用率。
2. 性能修复要落成脚本参数或默认路径，不要只靠手动记忆。
3. 速度问题没归因前，不把失败解释成架构失败。

### 11. 从“真实任务失败归因给 latent”改成“拆专家质量和 bus 质量”

失败表现：DocVQA、LLaVA/COCO、真实预训练 CLIP 接入都显示真实任务上 tiny 从零或不匹配专家会很弱。这个失败不能直接说明 latent bus 不行，也不能说明真实任务已经不可做。

修复方向：把感知专家质量单独测出来。DocVQA 应用 OCR/layout/document experts；UI 应用 DOM/screenshot/layout experts；计数/空间应用 detector/segmentation/spatial relation/counting experts。

后续规则：

1. 真实任务失败时，先报告专家输出质量，再报告 latent bus/reasoner。
2. 不要让 tiny latent reasoner 同时承担感知、结构化、推理和输出所有压力。
3. 真实专家必须按任务选，不要在几何计数/空间任务上硬压不匹配的 CLIP。

### 12. 从“正结果写进主线”改成“失败也要收口成路线规则”

失败表现：很多旧路线有弱正信号，但继续推进价值变低：低熵图像任务饱和、纯 resampler 丢信息、无 replay staged MoE 遗忘、全图 MSE 误导、无辅助监督 image IO copy 失败。

修复：把这些路线明确降级为负对照或历史证据，不再作为主线继续加步数。

后续规则：

1. 失败实验必须给出决策：继续、换结构、降级为负对照、删除旧测试。
2. 废旧测试如果锁旧契约，应删除或改成当前契约。
3. 不为了兼容旧设计保留过渡代码、过渡断言或旧命令清单。
4. 每次路线收口都要同步 `docs/DIRECTORY_REFERENCE.md`。

## 当前仍未解决

1. 统一 bus 已修复 relation-only 和结构化四任务的核心路径，但还没有完整证明 answer-token latent writer 在四任务上稳定闭合。

2. count slots 当前来自结构化 evidence，不证明真实图像里的 detector/segmentation/counting expert 已经能产出同质量 slot。

3. edit/generation 仍有 probe 和正式长训等级差异；copy-only 正式成功不能替代正式 edit/generation 成功。

4. query policy 的最终形态还没完全确定。AK 路线说明统一空间能大幅缓解，但未来 agent 式 active read 仍需要 scheduled sampling、hard negative query retrieval 或过程监督。

5. 真实文件、真实 UI、OCR/layout、长程记忆污染、工具失败恢复还没进入综合挑战；后续不能用 synthetic 闭环直接宣称真实产品能力。

# 我的经验

1 出问题无非就3点，潜空间没建起来，训练没练好，或者这个架构是错的。潜空间方面，需要一开始就统一，在统一的基础上才能训练。在训练方面，有可能是指导不足，或者指导是偏的。指导偏了还有一种特殊情况，看起来都是对的，但偏离了模型架构设计。先考虑如何让潜空间成型。

2 建潜空间是第一要务，稳定，好用的潜空间是一切的基础。在大型训练中，一般推荐使用文本建立潜空间，因为文本浓缩了大量信息。如果需要其他模态建潜空间，一般使用互译法，如果 外部模态->潜空间->外部模态 这个循环多跑几次，信息仍然不丢，或者最终外部模态与初始外部模态相同，那么可以认为潜空间建立成功。

3 训练启动后看一下显卡功率，60w以上为正常，20w左右训练会很慢。