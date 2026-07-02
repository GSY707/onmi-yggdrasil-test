# Stage H5/H6：视觉/组合泛化与多输出格式实验

## 目的

H3/H4 暴露了两个关键负点：

1. 单一视觉风格训练后，未见视觉风格几乎全崩。
2. 随机 lookup 策略表没有组合规律，held-out 三因素组合不能自然泛化。

H5 先修这两个问题：

- 视觉增强：训练时使用多 palette、多位置、多形状偏移，测试第 8 种 held-out 风格。
- 可组合规则：把随机策略表换成可组合规则，再测试 held-out `(visual, goal, telemetry)`。

H5 成功后，H6 加入多输出格式：

- `FULL`
- `ACTION_ONLY`
- `READ_VISUAL`
- `READ_GOAL`
- `READ_TELEMETRY`

## 实验脚本与结果

脚本：

- `experiments/omni_transformer_stage_h5.py`
- `experiments/omni_transformer_stage_h6.py`

结果：

- `artifacts/omni_transformer_stage_h5/sweep_results.json`
- `artifacts/omni_transformer_stage_h6/sweep_results.json`

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_h5.py --sweep --seeds 20260701,20260702,20260703 --train-size 4096 --val-size 1024 --test-size 2048 --batch-size 128 --d-model 128 --layers 3 --heads 4 --visual-steps 950 --compositional-steps 950 --latent-tokens 8

.\.venv\Scripts\python.exe experiments\omni_transformer_stage_h6.py --sweep --seeds 20260701,20260702,20260703 --train-size 8192 --val-size 1536 --test-size 2048 --batch-size 128 --d-model 128 --layers 3 --heads 4 --train-steps 1200 --latent-tokens 8
```

## H5：视觉增强泛化

H5 视觉增强训练使用 7 种训练风格，测试第 8 种 held-out 风格。另保留 H4 的远距离 OOD 风格作为边界测试。

结果：

| 测试 | answer exact | visual exact |
| --- | ---: | ---: |
| augmented ID style | 100.00% | 100.00% |
| held-out visual style | 98.70% | 100.00% |
| far unseen visual style | 54.69% | 56.25% |

解释：

- 多风格增强基本修复了同一增强族内的 held-out style 泛化。
- 远距离 OOD 仍然只有约 55%，说明这不是通用视觉鲁棒性，只是增强分布内泛化。
- 下一步若进入真实图像，仍需要更宽视觉分布、预训练视觉编码或真实图像增强。

## H5：可组合规则泛化

H5 把随机 lookup action 改成可组合规则：

```text
error_spike -> restart_controller
heat_ramp or thermal_marker -> cool_down
low_voltage -> reroute_power
open_interlock or safety_first -> isolate_system
throughput_first and not blocked_flow -> increase_throughput
otherwise -> hold_state
```

结果：

| 测试 | answer exact | action exact |
| --- | ---: | ---: |
| seen target triples | 100.00% | 100.00% |
| held-out target triples | 100.00% | 100.00% |

解释：

- 和 H4 的随机 lookup 不同，组合规则可学习，因此 held-out 三因素组合可以泛化。
- 这说明 H4 的 held-out 失败不是 latent bottleneck 本身失败，而是任务没有组合规律可学。

## H6：多输出格式

H6 在 H5 的成功前提上，把 query 扩展为 5 种输出格式：

| Query | 输出 |
| --- | --- |
| `FULL` | `ACTION VISUAL GOAL TELEMETRY EOS` |
| `ACTION_ONLY` | `ACTION EOS EOS EOS EOS` |
| `READ_VISUAL` | `VISUAL EOS EOS EOS EOS` |
| `READ_GOAL` | `GOAL EOS EOS EOS EOS` |
| `READ_TELEMETRY` | `TELEMETRY EOS EOS EOS EOS` |

聚合结果：

| 测试 | answer exact |
| --- | ---: |
| seen + augmented style | 100.00% |
| seen + held-out style | 98.01% |
| held-out triples + augmented style | 99.02% |
| held-out triples + held-out style | 96.61% |

最难组合：`held-out triples + held-out style`

| 格式 | exact |
| --- | ---: |
| FULL | 87.16% |
| ACTION_ONLY | 95.82% |
| READ_VISUAL | 100.00% |
| READ_GOAL | 100.00% |
| READ_TELEMETRY | 100.00% |

样例：

```text
query:      READ_TELEMETRY
target:     TELEMETRY_heat_ramp EOS EOS EOS EOS
prediction: TELEMETRY_heat_ramp EOS EOS EOS EOS

query:      FULL
target:     ACTION_reroute_power VISUAL_green_lock GOAL_power_saving TELEMETRY_low_voltage EOS
prediction: ACTION_reroute_power VISUAL_green_lock GOAL_power_saving TELEMETRY_low_voltage EOS
```

解释：

- 单字段读取格式全部稳定到 100%。
- `FULL` 格式最难，因为它同时要求 action、visual、goal、telemetry 全部正确，且 action 在第一个 autoregressive token 上出错会直接破坏 exact match。
- 在同时叠加 held-out triples 和 held-out style 时，整体仍有 96.61%，说明多输出格式可行，但 full 格式仍是下一步优化重点。

## 结论

H5/H6 支持以下判断：

1. 视觉增强可以显著改善近分布未见风格泛化：H4 unseen style 0.52%，H5 held-out style 98.70%。
2. 可组合规则可以修复 held-out 组合泛化：H4 随机 lookup held-out action 4.17%，H5 可组合规则 held-out action 100%。
3. 多输出格式可以和 latent bottleneck、视觉增强、组合规则同时工作。
4. 远距离视觉 OOD 仍未解决，H5 far OOD 只有 54.69%。

## 下一步

下一阶段建议继续强化两点：

1. 视觉端：把远距离 OOD 纳入更宽增强族，或接入预训练视觉 encoder，再测真实图像。
2. 输出端：优化 `FULL` 格式在复合泛化下的错误，尝试字段重排、非自回归字段头或 answer-side parallel heads。
