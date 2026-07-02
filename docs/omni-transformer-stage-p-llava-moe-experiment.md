# Stage P：LLaVA-Instruct-150K MoE/Latent 实验

## 目的

Stage P 把前面 Stage M/O 的 MoE + Attention Pump + latent thought 骨架接到真实 LLaVA-Instruct-150K 数据上，验证：

- 真实 COCO 图像、真实自然语言 prompt / dialogue turn 能否进入 frozen CLIP 专家。
- 变长视觉/text expert tokens 能否经 Attention Pump 压入固定数量 latent tokens。
- 输出专家只读 latent 时，是否还能在纯文本候选答案中选出正确答案。
- tiny 从零字符级输出专家是否足够支撑 LLaVA 长文本回答。

这轮不是训练 7B LLaVA，也不是证明 Yggdrasil 架构优于现有 VLM；它是本机 8GB VRAM 上的架构链路验证。

## 代码与产物

- `experiments/omni_transformer_stage_p_llava_moe.py`
- `artifacts/omni_transformer_stage_p_llava_moe/sweep_results.json`
- `artifacts/omni_transformer_stage_p_llava_moe/sweep_runs/`
- `artifacts/omni_transformer_stage_p_llava_moe/sweep_runs/*/samples/*/llava_coco_grid.jpg`
- `artifacts/omni_transformer_stage_p_llava_moe/sweep_runs/*/samples/*/samples.json`

## 数据

使用官方 `liuhaotian/LLaVA-Instruct-150K` 三个 JSON：

- `detail_23k.json`
- `conversation_58k.json`
- `complex_reasoning_77k.json`

脚本解析 human -> gpt turn，并按 image name 分组切分，避免同图跨 train / val / test 泄漏。图像从 COCO URL 按需下载到本地缓存。

本次 3 seed 聚合：

| 项 | 数值 |
| --- | ---: |
| 解析出的 turn examples | 356,753 |
| unique images | 81,398 |
| 每 seed train / val / test | 384 / 96 / 128 |
| 每 seed 接受图像数 | 135.67 |
| 图像下载失败 | 0 |

## 架构

```text
CLIP Vision Token Expert
CLIP Prompt/Text Token Expert
Dialogue-History Text Expert
Image-Text Fusion Expert
        -> Router
        -> Attention Pump / mean-pool baseline / direct concat baseline
        -> ThoughtExpert latent update
        -> Text output head
```

本轮有两类输出头：

1. 字符级 TextOutputExpert：从零生成 LLaVA 英文长答案，用候选答案 NLL 做 ranking。
2. Candidate Scorer TextOutputExpert：候选答案仍是纯文本，但先由 frozen CLIP text expert 编码；训练小头从 context/latent 中选择正确答案。

第二类不是开放式生成，但更适合验证“真实多模态专家输出 -> latent -> 文本输出选择”这条数据流。

## 正式命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_p_llava_moe.py --sweep --seeds 20260701,20260702,20260703 --annotation-files detail_23k.json,conversation_58k.json,complex_reasoning_77k.json --train-size 384 --val-size 96 --test-size 128 --batch-size 16 --d-model 128 --layers 2 --heads 4 --answer-len 96 --latent-tokens 12 --candidate-count 8 --text-only-steps 500 --direct-steps 800 --mean-steps 800 --moe-steps 1000 --scorer-steps 800 --output-dir artifacts\omni_transformer_stage_p_llava_moe\sweep_runs --aggregate artifacts\omni_transformer_stage_p_llava_moe\sweep_results.json
```

## 主结果

指标是 8 个候选答案中的 top-1 ranking accuracy；随机期望为 12.50%。

| 模型 | Top-1 | MRR | 结论 |
| --- | ---: | ---: | --- |
| CLIP zero-shot candidate rank | 86.98% | 91.35% | frozen CLIP 文本/图像对齐本身很强 |
| text-only char decoder | 15.63% | 38.86% | 从零字符生成基本不成立 |
| direct early-concat char decoder | 18.23% | 38.88% | 略高于随机，但远弱于 CLIP |
| mean-pool latent char decoder | 9.38% | 30.84% | 低于随机 |
| MoE Attention-Pump latent char decoder | 9.38% | 30.78% | 低于随机 |
| scorer text-only | 49.48% | 69.94% | prompt 文本信号很强 |
| scorer direct | 57.03% | 74.09% | 小训练头能用 direct expert tokens |
| scorer mean-pool latent | 44.27% | 63.57% | 均值池化 latent 可用但弱 |
| scorer MoE Attention-Pump latent | 42.45% | 66.63% | latent path 明显高于随机，但低于 direct |

## 消融

以下以 `scorer MoE Attention-Pump latent` 为对象：

| 消融 | Top-1 | MRR | 解释 |
| --- | ---: | ---: | --- |
| full | 42.45% | 66.63% | latent path 可用 |
| no latent access | 13.28% | 31.92% | 接近随机，说明输出确实依赖 latent |
| no image modality | 42.97% | 65.83% | 图像贡献不稳定，当前任务主要被 prompt/answer 语义支撑 |
| no text modality | 21.88% | 45.26% | 文本 prompt 是主要控制信号 |

latent probe 对 route 的 exact 为 78.39%，说明 latent 中保留了可读的专家路由信息。

## 成本

| 模型 | trainable params | 训练秒数 | cached ms/example | with CLIP ms/example |
| --- | ---: | ---: | ---: | ---: |
| scorer direct | 1,282,821 | 15.03 | 0.421 | 12.287 |
| scorer MoE Attention-Pump latent | 1,282,821 | 24.70 | 0.617 | 12.483 |
| char direct decoder | 1,653,311 | 17.65 | 2.439 | 14.304 |
| char MoE Attention-Pump latent decoder | 1,626,687 | 34.28 | 1.759 | 13.625 |

frozen CLIP expert 参数量约 151M；真实在线成本主要来自 CLIP 编码，约 11.87 ms/example。

## 判断

Stage P 是部分正结果：

1. 真实 LLaVA 数据链路已经闭合：官方 JSON、COCO 图像、CLIP vision/text tokens、router、Attention Pump、latent thought、文本候选输出都能在本机 GPU 上训练和评估。
2. `scorer MoE Attention-Pump latent` 明显高于随机，并且 `no latent access` 接近随机，证明 latent 不是空壳。
3. direct scorer 仍高于 MoE latent scorer，CLIP zero-shot 又远高于所有小训练头，所以本轮不能证明架构更优。
4. 从零字符级长答案生成失败，说明本机 tiny 模型不适合直接学习 LLaVA 风格开放式长文本输出；后续应接入预训练小语言模型或把输出专家也做成 frozen/LoRA 文本专家。
5. `no image modality` 没有明显掉点，说明这个候选答案评估会被 prompt 与候选答案语义支撑；它不是强视觉 grounding 测试。

## 下一步

- 若继续 LLaVA 路线，应把 TextOutputExpert 换成小型预训练 LM 的 LoRA，而不是从零字符头。
- 若要验证视觉依赖，应构造 hard negative：候选答案来自同 prompt 类型、相近语言风格、但图像事实不同的样本。
- 若要证明 Attention Pump 比 direct 更有价值，需要增加上下文长度、专家数量或高分辨率/多图输入，让 direct concat 的成本劣势真正出现。
