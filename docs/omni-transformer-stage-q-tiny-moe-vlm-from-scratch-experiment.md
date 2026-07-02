# Stage Q：从零训练 Tiny MoE-VLM 实验

## 目的

Stage P 使用 frozen CLIP 作为真实预训练视觉/文本专家。Stage Q 刻意移除所有预训练专家，从零训练一个 tiny MoE-VLM，测试：

- 仅靠本机小模型和少量 LLaVA-Instruct-150K 样本，是否能从真实 COCO 像素 + 自然语言 prompt 中学到多模态对齐。
- 从零视觉专家、文本专家、router、Attention Pump、latent thought 和候选答案 scorer 能否闭合。
- 和 text-only / direct / mean-pool latent baseline 相比，MoE Attention-Pump latent 是否有优势。

本轮不是为了得到可用 VLM，而是校准“从零训练 tiny 多模态模型”在真实指令数据上的下限。

## 代码与产物

- `experiments/omni_transformer_stage_q_tiny_moe_vlm_from_scratch.py`
- `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_results.json`
- `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_runs/`
- `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_runs/*/samples/*/llava_coco_grid.jpg`
- `artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_runs/*/samples/*/samples.json`

## 架构

```text
COCO pixels 64x64 -> tiny CNN VisionExpert
prompt chars -> character TextExpert
history chars -> HistoryExpert
vision + prompt -> FusionExpert
        -> Router
        -> direct concat / mean-pool latent / Attention Pump latent
        -> ThoughtExpert
        -> from-scratch answer candidate encoder + scorer
```

关键限制：

- 不加载 CLIP/SigLIP/VLM。
- 不使用 frozen text embedding。
- prompt 和候选答案都用字符级 encoder 从零训练。
- 输出任务仍是 8 候选答案 ranking；随机期望为 12.50%。

## CPU 瓶颈修正

初版训练时 CPU 满载、GPU 不满。原因是每一步都在 Python 中重新拼候选答案文本、重新字符编码、并把 batch/candidate tensor 从 CPU 拷到 GPU。

已修正：

- train / val / test 像素和 prompt tensor 在 run 开始时常驻 GPU。
- answer pool 预编码为 GPU tensor。
- 训练时只采样候选答案整数索引，再用 GPU gather 取出候选 token。
- smoke 时间从约 49 秒降到约 7 秒。
- 同一正式配置 3 seed 总耗时约 153 秒。

## 正式命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_q_tiny_moe_vlm_from_scratch.py --sweep --seeds 20260701,20260702,20260703 --annotation-files detail_23k.json,conversation_58k.json,complex_reasoning_77k.json --train-size 384 --val-size 96 --test-size 128 --batch-size 16 --image-size 64 --prompt-len 128 --answer-len 128 --d-model 96 --layers 2 --heads 4 --latent-tokens 12 --candidate-count 8 --text-only-steps 400 --direct-steps 400 --mean-steps 400 --moe-steps 600 --output-dir artifacts\omni_transformer_stage_q_tiny_moe_vlm_from_scratch\sweep_runs --aggregate artifacts\omni_transformer_stage_q_tiny_moe_vlm_from_scratch\sweep_results.json
```

## 结果

3 seeds，8 候选答案 ranking，随机期望 12.50%：

| 模型 | Top-1 | MRR |
| --- | ---: | ---: |
| scratch text-only | 13.80% | 35.10% |
| scratch direct | 14.58% | 34.27% |
| scratch mean-pool latent | 14.58% | 34.95% |
| scratch MoE Attention-Pump latent | 14.58% | 36.57% |

消融：

| 消融 | Top-1 | 解释 |
| --- | ---: | --- |
| full scratch MoE latent | 14.58% | 仅略高于随机 |
| no latent access | 8.85% | 断 latent 会掉点，但 full 本身很弱 |
| no image modality | 15.63% | 不看图像反而略高，说明视觉 grounding 没学出来 |
| no text modality | 10.16% | 文本 prompt 仍是主要信号 |

latent probe 对 route 的 exact 为 99.74%，但这只说明 latent 中能读出路由标签，不代表它学到了图像语义。

## 成本

| 模型 | trainable params | 训练秒数 | 预测 ms/example |
| --- | ---: | ---: | ---: |
| scratch text-only | 1,233,509 | 7.03 | 0.436 |
| scratch direct | 1,233,509 | 8.49 | 0.440 |
| scratch mean-pool latent | 1,233,509 | 10.37 | 0.497 |
| scratch MoE Attention-Pump latent | 1,233,509 | 16.22 | 0.530 |

和 Stage P 对照：

| 路线 | Top-1 |
| --- | ---: |
| Stage P CLIP zero-shot | 86.98% |
| Stage P scorer direct | 57.03% |
| Stage P scorer MoE Attention-Pump latent | 42.45% |
| Stage Q scratch MoE Attention-Pump latent | 14.58% |

## 判断

Stage Q 是负结果，但很有价值：

1. 从零 tiny MoE-VLM 的训练链路可以闭合，且优化后本机运行成本很低。
2. 在真实 LLaVA/COCO 数据上，1.23M 参数级别、几百样本、字符级从零编码器几乎学不出有效视觉语言对齐。
3. MoE Attention-Pump latent 只略高于随机，且不优于 direct / mean-pool / text-only。
4. `no image modality` 不掉点，说明从零视觉专家没有形成可用 grounding。
5. 这个结果支持 Stage P 的判断：真实多模态路线需要预训练视觉/文本/语言专家；本机 tiny from-scratch 更适合作为架构接口验证，不适合作为真实 AGI/VLM 能力验证。

## 下一步

如果继续真实 VLM：

- 不要继续放大从零 tiny 字符模型；收益很低。
- 改成 tiny pretrained LM + LoRA 输出专家，视觉侧用 frozen/LoRA vision expert。
- 若坚持 from-scratch，需要换成更低熵、更强监督的数据，例如 caption matching、对象类别/属性、多选 VQA 或合成到真实的 curriculum，再逐步升到 LLaVA。
- 对 LLaVA ranking 应构造 hard negative，避免 prompt/answer 语义本身主导评估。
