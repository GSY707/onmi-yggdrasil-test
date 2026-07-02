# Stage E+F：多模态融合与潜空间数据流动实验

## 目的

Stage D 之前的走地图实验已经证明：主动试探、短期 legend 记忆和潜变量承载结构化语义是可行的。Stage E+F 换成新的任务族，专门验证两个问题：

1. 图像、文本目标、遥测时序三路输入进入共享 latent bus 后，能否融合成稳定的纯文本输出。
2. 潜空间里的数据是否真的按模态流动，而不是模型碰巧从单一路径拟合答案。

本实验不证明 latent fusion 优于所有非 latent 方案；它只验证多模态融合链路和潜空间数据流动机制是否闭合。

## 任务设计

脚本：

- `experiments/multimodal_fusion_latent_flow.py`

结果：

- `artifacts/multimodal_fusion_latent_flow/sweep_results.json`
- `artifacts/multimodal_fusion_latent_flow/sweep_runs/`
- `artifacts/multimodal_fusion_latent_flow/sweep_runs/*/samples/*/diagnostic_panel_grid.png`

每个样本由三路输入组成：

| 输入 | 形式 | 含义 |
| --- | --- | --- |
| 图像 | `48x48 RGB` 设备面板 | 4 种视觉状态：正常锁、联锁打开、热标记、流路阻塞 |
| 文本 | 5 个 token | 4 种目标：安全优先、吞吐优先、节能、维护检查 |
| 遥测 | `12x4` 连续时序 | 4 种状态：稳定、升温、低电压、错误尖峰 |

最终动作不是由任意单模态决定，而是由固定的 `visual x goal x telemetry` 三维策略表决定。数据集均匀覆盖 64 种组合，使单模态 baseline 无法完整解题。

模型输出不是只给动作分类，还要输出可读文本所需的 4 个字段：

```text
action=...; visual=...; goal=...; telemetry=...
```

因此 `text_exact` 要求动作、视觉因素、文本目标、遥测因素全部预测正确。

## 对照组

| 对照 | 说明 |
| --- | --- |
| `oracle_structured_policy` | 直接读取结构化三因素并查策略表 |
| `image_only` | 只看图像 |
| `text_only` | 只看文本目标 |
| `telemetry_only` | 只看遥测时序 |
| `late_fusion_no_cross_modal_interaction` | 三个单模态专家各自出 action logits，再加权求和；没有跨模态交互 |
| `early_concat_fusion` | 三个 encoder 输出直接拼接后 MLP 融合 |
| `shared_latent_fusion` | 三个 encoder 输出进入带 type embedding 的 latent slots，再经 attention latent bus 聚合 |

## 正式结果

配置：

- seeds：`20260701,20260702,20260703`
- train / val / test：`4096 / 1024 / 2048`
- batch：`256`
- shared steps：`600`
- baseline steps：`600`
- probe steps：`250`
- GPU：`NVIDIA GeForce RTX 4070 Laptop GPU`

聚合准确率：

| 方法 | action exact | text exact |
| --- | ---: | ---: |
| oracle structured policy | 100.00% | 100.00% |
| image only | 26.56% | - |
| text only | 28.12% | - |
| telemetry only | 31.25% | - |
| late fusion, no cross-modal interaction | 38.02% | - |
| early concat fusion | 100.00% | 100.00% |
| shared latent fusion | 100.00% | 100.00% |

解释：

- 单模态都明显低，因为策略表需要三路信息。
- late fusion 只能合并单模态 action 分布，不能表达三路交互，因此只到约 38%。
- early concat 和 shared latent fusion 都能达到 oracle，说明在该合成任务上，跨模态融合本身可闭合。
- shared latent 没有超过 early concat，所以这里仍不能说 latent 更优。

## 潜空间数据流动验证

对训练好的 `shared_latent_fusion` 做 latent ablation：

| 干预 | action exact | text exact |
| --- | ---: | ---: |
| full | 100.00% | 100.00% |
| zero image latent | 36.98% | 10.42% |
| zero text latent | 26.56% | 9.38% |
| zero telemetry latent | 30.73% | 9.90% |
| shuffle image latent | 38.41% | 约 24% |
| shuffle text latent | 34.26% | 约 25% |
| shuffle telemetry latent | 34.23% | 约 26% |

解释：

- 清零任一路 latent 后，动作准确率从 100% 掉到 26% 到 37%。
- `text_exact` 掉到约 9% 到 10%，因为文本输出要求所有字段都正确；任一因素缺失都会破坏最终可读答案。
- 打乱任一路 latent 后，模型仍看得到另外两路，所以比清零略高，但无法维持完整答案。

这说明 shared latent bus 的最终答案确实依赖三路 latent 数据，而不是某一路单独决定。

## Slot probe

冻结 shared model 的三个 pre-fusion slots，只训练线性 probe，测每个 slot 能解出什么因素：

| source slot -> target | 准确率 |
| --- | ---: |
| image -> visual | 100.00% |
| image -> goal | 25.00% |
| image -> telemetry | 25.00% |
| text -> visual | 25.00% |
| text -> goal | 100.00% |
| text -> telemetry | 25.00% |
| telemetry -> visual | 25.00% |
| telemetry -> goal | 25.00% |
| telemetry -> telemetry | 100.00% |

4 类因素的随机水平是 25%。这个矩阵说明：

- 图像 slot 携带视觉状态。
- 文本 slot 携带目标。
- 遥测 slot 携带遥测状态。
- 其他交叉方向基本停留在随机水平。

因此，pre-fusion latent slots 有清晰的模态边界；shared latent bus 再把它们合并成最终答案。

## 结论

Stage E+F 支持以下判断：

1. 多模态融合链路可行：图像、文本、遥测三路可以进入共享 latent bus，并输出完整纯文本答案。
2. 潜空间数据流动可验证：slot probe 和 ablation 都显示三路信息在进入融合前有清晰边界，融合后共同决定答案。
3. 跨模态交互是必要的：单模态和无交互 late fusion 都明显低于 oracle。
4. latent fusion 不是唯一方案：early concat 同样达到 100%，所以本实验不证明 latent 更好，只证明它可行且可诊断。

## 未证明边界

- 任务仍是低熵合成诊断表，不是真实工业图像、自然语言日志或复杂时序。
- 图像是程序生成面板，不是照片级输入。
- 策略表固定，未验证分布外新规则泛化。
- factor head 参与了 supervised 训练，虽然 slot probe 是冻结后线性 probe，但仍不能等同于无监督语义形成。
- 还没有专家聚合、弱专家、错误专家、专家置信度或路由器实验。

## 后续路线调整

原先可做 Stage G 专家聚合，但这条路线距离最终 omni Transformer 目标较远。当前已转向 Stage H：统一 token stream 的 tiny omni Transformer，并用 latent bottleneck 验证特殊 latent token 是否能承载多模态信息。
