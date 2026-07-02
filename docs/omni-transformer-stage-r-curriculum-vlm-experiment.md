# Stage R：低熵强监督 Curriculum VLM 实验

## 目的

Stage Q 在真实 LLaVA/COCO 上失败后，Stage R 把数据换成更低熵、更强监督的合成视觉问答阶梯，目标不是证明真实 VLM 能力，而是校准当前 from-scratch tiny MoE-VLM 的能力上限：

- 从单一颜色识别开始。
- 逐步增加颜色、形状、位置、计数、二物体关系和三物体 caption 熵。
- 保持 Stage Q 的 tiny CNN vision expert、字符级 prompt/answer encoder、router、Attention Pump、latent thought 和 candidate scorer。
- 对比 `scratch_direct` 与 `scratch_moe_attention_pump_latent`，并用 `no image` / `no latent` 消融确认是否真正依赖视觉和 latent。

## 代码与产物

- `experiments/omni_transformer_stage_r_curriculum_vlm.py`
- `artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_results.json`
- `artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_runs/`
- `artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_runs/*/*/samples/`

## 难度阶梯

| Level | 任务 | 随机 Top-1 |
| --- | --- | ---: |
| L1 color | 单个中心物体，回答 8 色之一 | 12.50% |
| L2 object | 单个中心物体，回答颜色 + 形状 | 12.50% |
| L3 position | 单个物体，回答颜色 + 形状 + 位置 | 12.50% |
| L4 count | 多物体，按目标颜色/形状计数 | 20.00% |
| L5 relation | 两个物体，回答 queried object 左/右侧物体颜色 | 12.50% |
| L6 scene caption | 三物体场景，选择从左到右完整 caption | 12.50% |

判定一个 level 被 MoE latent 解决：`MoE latent Top-1 >= 90%` 且 `MoE latent - no image >= 20pp`。

## 正式命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_r_curriculum_vlm.py --sweep --seeds 20260701,20260702,20260703 --levels l1_color,l2_object,l3_position,l4_count,l5_relation,l6_scene_caption --models direct,moe_latent --train-size 1536 --val-size 384 --test-size 512 --batch-size 128 --image-size 64 --prompt-len 96 --answer-len 96 --d-model 96 --layers 2 --heads 4 --latent-tokens 8 --candidate-count 8 --direct-steps 200 --moe-steps 600 --probe-steps 80 --eval-every 200 --output-dir artifacts\omni_transformer_stage_r_curriculum_vlm\sweep_runs --aggregate artifacts\omni_transformer_stage_r_curriculum_vlm\sweep_results.json
```

正式 3 seed 平均每 seed 约 352.82 秒；MoE latent 每 level 约 42.4-42.7 秒训练，direct 每 level 约 13.3 秒训练。MoE latent trainable params 为 1,225,061。

## 结果

3 seeds，Top-1：

| Level | Random | Direct | MoE latent | MoE no image | Visual gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 color | 12.50% | 100.00% | 100.00% | 12.17% | +87.83pp |
| L2 object | 12.50% | 100.00% | 46.03% | 13.15% | +32.88pp |
| L3 position | 12.50% | 99.22% | 16.60% | 11.78% | +4.82pp |
| L4 count | 20.00% | 19.86% | 19.40% | 19.40% | +0.00pp |
| L5 relation | 12.50% | 94.14% | 25.20% | 11.26% | +13.93pp |
| L6 scene caption | 12.50% | 79.30% | 15.82% | 11.39% | +4.43pp |

能力上限判定：

- solved levels：`L1 color`
- highest solved level：`L1 color`
- first failed level：`L2 object`

## 判断

Stage R 说明 Stage Q 的失败不是单纯因为像素输入不可学。`scratch_direct` 在 L1/L2/L3/L5 上达到 94%-100%，L6 也有 79.30%，证明同一个 tiny CNN + 字符 answer scorer 可以从强监督低熵视觉数据中学到相当多的图像-文本匹配。

但当前 `MoE Attention-Pump latent` 的稳定能力上限很低：它只稳定解决 L1 单颜色识别。L2 虽然高于随机且有明显 visual gap，但 46.03% 距离解决还很远；L3/L5/L6 基本无法稳定把组合视觉属性压进 latent 后再用于 answer scorer。这里的问题集中在 latent bottleneck / Attention Pump / scorer 训练，而不是图片生成或 direct 路径。

L4 counting 是共同短板：direct 19.86%、MoE 19.40%，都等于随机附近。这说明当前 tiny CNN token + 字符 scorer 不具备对象级计数归纳偏置；即使不经过 latent，也不能可靠计数。

## 对架构的含义

1. 从零 tiny VLM 路线并非完全不可训练；低熵单属性视觉 grounding 可以成立。
2. 组合属性一进入 latent bottleneck 就明显掉点，当前 Attention Pump latent 还不能替代 direct token access。
3. direct baseline 已经能解决较高熵的 synthetic matching，所以后续要证明 MoE/latent 更好，必须修 latent 训练路径，而不是继续换更容易的数据。
4. counting 需要对象级专家、slot attention、detector-like inductive bias 或显式辅助监督；纯 CNN + pooled token scorer 不够。

## 下一步

- 对 MoE latent 做局部修复实验：latent answer scorer 加 cross-attention，而不是只 mean-pool latent；或加入 latent-to-answer contrastive auxiliary loss。
- 对 counting 单独换对象 slot/patch-level expert，验证 direct 与 MoE 是否能超过随机。
- 如果继续真实路线，保留 Stage R 的 curriculum 作为从零训练 sanity check：模型必须先稳定过 L1/L2/L3，再回到 LLaVA/DocVQA。
