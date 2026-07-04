# Stage AR：视觉中心跨专家不可单解任务

## 目标

Stage AR 用来验证一个比 Stage AO 更接近“多专家共同工作”的问题：单个专家即使能独立读出自己的信息，也不能单独解题。任务主线按用户要求偏视觉；agent 状态路线暂不并入本轮，保留给 Stage AT 或 AR 加强版。

本阶段的四路输入是：

1. 图像：64x64、4x4 cell 场景，每个 cell 有一个颜色/标记值。
2. 文本 rule：受控 DSL token，指定读取哪个 cell，以及使用哪条组合规则。
3. telemetry：4 类结构化状态。
4. memory：4 类短记忆偏移。

答案是 4 类动作之一。公式故意设计成四路信息共同决定：缺图像、缺文本 rule、缺 telemetry 或缺 memory，理论上都会接近 4 类随机。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_ar_cross_expert_visual.py
```

模型结构：

- `CellVisionExpert` 从像素图中抽取 16 个 cell token。
- text-rule expert 解析受控 DSL 的 rule/cell token，用来选择目标视觉 cell。
- telemetry 与 memory 各自进入独立 embedding。
- latent 模型把 selected visual token、rule token、telemetry token、memory token 写入 latent tokens，再由 answer head 输出动作。
- 同时训练 visual probe，确保视觉 cell token 真能读出目标 cell 的视觉值。

对照：

- `DirectAllInputBaseline` 记录 direct all-input baseline 的成本和准确率。
- full 模型评估后再分别做 `no_image`、`no_text_rule`、`no_telemetry`、`no_memory` 消融。

## Smoke

smoke 结果：

- 聚合结果：`artifacts/omni_transformer_stage_ar_cross_expert_visual/smoke_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_ar_cross_expert_visual/smoke_runs/seed20260701.json`

smoke 只证明脚本、训练循环、样例 PNG、消融评估和聚合 JSON 能跑通，不代表训练效果。

## Probe 修正

第一版 probe 暴露了一个结构问题：视觉 probe 已经能到 100%，但模型没有稳定学会从文本 token 里选择目标 cell；direct baseline 也出现训练集过拟合而非泛化。这说明问题不在视觉专家，而在 text-rule 到 cell selection 的结构没有固定住。

已修正为受控 DSL parser：text-rule expert 显式把 rule/cell token 解析成选择信号，再让 latent 模型学习四路组合。修正后的单 seed probe：

- 聚合结果：`artifacts/omni_transformer_stage_ar_cross_expert_visual/probe_results_v2.json`
- 结果：latent full 100%，四个缺模态消融均下降 66 个百分点以上。

## 正式 3 seed 结果

正式结果：

- 聚合结果：`artifacts/omni_transformer_stage_ar_cross_expert_visual/formal_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_ar_cross_expert_visual/formal_runs/seed20260701.json`、`seed20260702.json`、`seed20260703.json`
- 样例图像与样例说明：`artifacts/omni_transformer_stage_ar_cross_expert_visual/formal_runs/*/samples/`

| 指标 | 3 seed 平均 | 结论 |
| --- | ---: | --- |
| latent full answer accuracy | 100.00% | 通过 |
| latent visual probe accuracy | 100.00% | 通过 |
| no-image answer accuracy | 23.96% | 明显下降 |
| no-text-rule answer accuracy | 25.07% | 明显下降 |
| no-telemetry answer accuracy | 32.10% | 明显下降 |
| no-memory answer accuracy | 30.21% | 明显下降 |
| direct full answer accuracy | 51.04% | 未同等解决 |

按门槛看：

- full >= 90%：通过。
- 每个 no-modality ablation 至少下降 30 个百分点：通过。
- direct all-input baseline 记录成本：已记录；本轮 direct baseline 未达到 latent full。

## 结论

Stage AR 已经证明：在一个视觉中心、受控合成的跨专家任务里，image、text rule、telemetry、memory 四路输入都具有因果作用。full latent 模型稳定闭合，任意缺一路都会掉到接近随机或弱先验水平。

这比单纯“某个视觉专家能 100%”更强，因为答案不是视觉专家单独能解，也不是 telemetry/memory/text 任一路单独能解。

## 边界和担忧

1. 视觉仍是低熵合成 cell 图，不证明真实图像、真实 OCR 或真实 VLM grounding。
2. text-rule 是受控 DSL parser，不证明自然语言 instruction following。
3. memory 是短离散状态，不是 Stage AT 要挑战的长程 memory tree / work tree。
4. 本轮 direct baseline 没有同等解决，但 direct baseline 结构仍比较轻；如果后续给 direct baseline 同样的强结构先验，它可能追上，这需要单独作为负结果或成本对照记录。
5. agent 状态路线更难，适合作为 Stage AT 的长程探索/记忆成本任务，不应塞回 AR 本轮结论里。

## 正式复现命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ar_cross_expert_visual.py --sweep --seeds 20260701,20260702,20260703 --train-size 2048 --val-size 512 --test-size 512 --batch-size 128 --d-model 96 --heads 4 --layers 2 --latent-tokens 6 --steps 800 --direct-steps 500 --eval-every 200 --output-dir artifacts\omni_transformer_stage_ar_cross_expert_visual\formal_runs --aggregate artifacts\omni_transformer_stage_ar_cross_expert_visual\formal_results.json
```
