# 从 0 训练架构验证的显存与量化口径

## 结论

如果目标是“完整从 0 训练，但只验证架构闭环，不要求完成真实任务质量”，量化可以再省一些，但省法和推理量化不一样。

2026-07-04 新增本机实测：RTX 4070 Laptop 8GB 可以运行 Stage AV 的 70,994,707 参数从零集成 Micro-Omni probe，配置为 `d_model=768`、`layers=10`、`heads=12`、`latent_tokens=8`、`batch_size=32`、AMP，1000 step 约 92 秒，峰值 CUDA allocated 约 1,810.62 MB。这个结果证明“70M 级架构验证模型”在本机可训练，但 probe 未通过正式能力门槛：overall 64.32%，视觉子任务 23.44%，说明显存可承受不等于架构闭合。

同日 Stage AV-B 已切到分阶段潜空间训练方案。70M batch 256 smoke 参数量为 71,025,455，峰值 CUDA allocated 约 3,094.87 MB；因此 8GB 本机长训建议从 batch 256 启动，若功耗长期低于 60W，再试 batch 384/512。功耗以 `nvidia-smi -l 2` 实测为准。

推荐口径：

| 显存 | 量化/省显存后能做什么 | 判断 |
| --- | --- | --- |
| 8GB | 只能做拆分式 tiny proof、单模块或短链路验证；可以跑极小多模态输入和极短 agent loop | 不建议称为完整正式验证 |
| 12GB | 可以做 micro complete：小模型、小图像、小音频 token、极低分辨率生成头、micro-batch 1 | 勉强验证整体数据流，不适合多 seed/ablation |
| 16GB | 使用 BF16、8-bit optimizer、activation checkpointing、梯度累积和低分辨率离散生成后，可以做一套压缩版完整闭环 | 量化后的硬最低 |
| 24GB | 可做较完整从 0 训练架构验证，保留多模态输入、latent reasoner、agent loop 和简化生成专家 | 推荐最低 |
| 48GB | 可减少实验形状妥协，做更长 rollout、多 seed、关键消融和更宽 latent bus | 正式验证更稳 |

因此，量化后可以把“硬最低”从 24GB 压到约 16GB；但 24GB 仍是更合理的正式下限，48GB 仍是更稳的准备量。

## 按整机价格的购卡建议

新增前提：本架构可以每次只训练一部分权重，例如只训练当前输入专家、latent reasoner、输出专家或少数 router/readout 组件。这个前提会明显降低单次训练的显存压力，所以最低价优先时不必直接追 24GB/32GB 顶卡。

优先级：

| 优先级 | 显卡/整机 | 适合情况 | 判断 |
| --- | --- | --- | --- |
| 1 | RTX 5060 Ti 16GB 整机 | 买新整机、希望保修、只训部分专家权重 | 当前最低价优先的首选 |
| 2 | RTX 4060 Ti 16GB 整机 | 找到明显低于 5060 Ti 16GB 的新机或二手机 | 可买，但要确认不是 8GB 版 |
| 3 | 二手 RTX 3090 24GB 整机 | 价格接近 16GB 新整机，且能现场/到手压力测试 | 同价优先 3090，风险高于新机 |
| 4 | RTX 4090 24GB / RTX 5090 32GB 整机 | 不以最低整机价为目标，或需要更快迭代 | 不符合“整机最低价优先” |
| 5 | Intel Arc Pro B70 32GB 等非 CUDA 方案 | 只看显存/价格，且能接受 oneAPI/OpenVINO/框架适配 | 不建议作为本项目第一台训练机 |

实际采购口径：

- 如果买新整机：优先找 RTX 5060 Ti 16GB、32GB 系统内存、1TB NVMe、650W 以上电源的机器；到手后建议补到 64GB 内存。
- 如果买二手整机：只有在 RTX 3090 24GB 整机价格接近 RTX 5060 Ti 16GB 新整机时才值得冒险；必须要求 30 分钟以上压力测试、显存压力测试、温度截图和电源线/接口照片。
- 如果只买显卡自组：RTX 3090 需要 850W 级别电源和较强散热；便宜整机里常见的小电源、小机箱不适合直接塞 3090。
- 如果预算只能压到最低：16GB 卡足够做“只训练当前专家”的架构闭环；实验设计上把专家训练拆开，冻结其它专家，只在最终做短轮联合校准。

不建议为了本项目优先买 12GB 或 8GB 卡。它们可以跑局部实验，但会迫使架构验证变成“为了显存而设计”，后续结论不够硬。

## 哪些量化有用

1. 8-bit optimizer 有用。它主要压缩 Adam optimizer states，而从 0 训练时 optimizer states 往往是显存大头之一。
2. FP8 / mixed precision training 有用，但依赖硬件和训练栈。H100/H200 这类服务端 GPU 上工具链更成熟；消费级 GPU 是否值得依赖，要以当前 PyTorch、Transformer Engine、CUDA 和具体算子支持实测为准。
3. Activation checkpointing 非量化但很关键。它通过反向传播时重算中间激活来换显存，适合长序列、多模态 token 和 agent rollout。
4. 生成侧离散化更关键。不要从 0 训练高分辨率像素/视频生成；先用 VQ/VAE token、低分辨率图像、短帧视频或 spectrogram token 验证“latent -> output expert”的路径。

## 哪些量化帮助有限

1. 4-bit / QLoRA 不适合作为从 0 全参训练的主省显存方案。QLoRA 的典型用途是冻结一个预训练大模型，再训练少量 LoRA adapter；这不等于从 0 训练完整架构。
2. 推理权重量化不能直接减少训练显存。训练还需要 gradients、optimizer states、activations 和临时 workspace。
3. 只量化最终评估阶段没有意义。它能降低跑验证样本的显存，但不能降低训练闭环的门槛。
4. 强行 int4 全参训练风险很高。对本项目这种自定义多模态专家、latent reasoner、agent loop 和生成头混合结构，低 bit 全参训练的数值稳定性本身会变成新的研究变量，不适合做“架构是否闭合”的基础门禁。

## 建议的验证配置

16GB 压缩版：

- BF16 或 FP16 mixed precision。
- 8-bit Adam / paged optimizer。
- activation checkpointing。
- micro-batch 1，梯度累积。
- 图像 64-128 分辨率或 patch/token 输入。
- 音频用 mel/spectrogram token，不直接生成高保真波形。
- 视频只做 8-32 帧低分辨率 token，优先验证调度与生成专家接口。
- 模型总参数控制在几十 M 到一两百 M 量级。

24GB 推荐版：

- 保留同样省显存手段。
- 可以扩大 latent bus、expert token 数、context length 和 rollout 步数。
- 可以做至少 3 seed、关键 ablation、no-image/no-text/no-history/no-latent 门禁。
- 生成专家仍建议低分辨率或离散 token，不要把验证目标变成高质量生成模型训练。

## 需要报告的担忧

- 如果“多模态生成”要求视频质量，而不是架构链路，显存不是主要瓶颈，数据量和训练算力会先成为硬阻塞。
- 如果用 FP8 或 int4 全参训练，实验结论会混入数值格式风险；失败时很难区分是架构失败还是量化训练不稳定。
- 如果为了塞进 8GB/12GB 过度缩短 rollout、降低分辨率、减少专家和消融，验证会退化成接口 smoke，不足以支撑“正式完整架构验证”的结论。

## 外部口径来源

- [Hugging Face Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)：4-bit/8-bit quantization 主要围绕加载/微调预训练模型。
- [Hugging Face bitsandbytes 8-bit optimizers](https://huggingface.co/docs/bitsandbytes/en/optimizers)：8-bit optimizer 面向训练/微调时的 optimizer state 显存压缩。
- [PyTorch activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/)：checkpointing 用重算换显存。
- [NVIDIA mixed precision / FP8 training](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/mixed-precision.html)：FP8 training 依赖相应 GPU 与 Transformer Engine 等训练栈支持。
- [Best Buy RTX 5060 Ti listing](https://www.bestbuy.com/site/searchpage.jsp?id=pcat17071&st=rtx+5060+ti)：2026-07 查询时可见 RTX 5060 Ti 16GB 新卡和整机价位。
- [NVIDIA RTX 5090 official page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)：RTX 5090 为 32GB GDDR7，官方起价高，不适合最低整机价优先。
- [BestValueGPU RTX 3090 tracker](https://bestvaluegpu.com/history/new-and-used-rtx-3090-price-history-and-specs/) 与 [RTX 4090 tracker](https://bestvaluegpu.com/history/new-and-used-rtx-4090-price-history-and-specs/)：用于判断 24GB 二手卡和高端卡当前价位。
