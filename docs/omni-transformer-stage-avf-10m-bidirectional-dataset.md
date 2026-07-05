# Stage AV-F：10M 级双向互译数据集准备

## 目标

Stage AV-F 把 AV-D 的 1B-first 数据设计落成当前可训练的 10M 级数据集。它不再沿用 AV-C 的识别/分类式任务分布，而是直接服务 AV-E 的核心链路：

```text
source external -> source latent -> source external
source external -> source latent -> target latent -> target external
target external -> target latent -> target external
target external -> target latent -> source latent -> source external
target latent -> answer
```

因此每条样本都同时存：

- `source_tokens`
- `target_tokens`
- `answers`
- 辅助统计用的 `ops`、`target_zones`

每条样本按 `source_tokens + target_tokens + answer` 计为 55 pair tokens。

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_avf_10m_bidirectional_dataset.py
```

生成器复用 AV-E 的 `make_example` 与 `encode_external`，但输出为 tensor shard：

- token 用 `int16` 存储，训练时再转为 `long`。
- manifest 中的 shard path 相对 manifest 目录，方便训练脚本直接解析。
- scale 支持 `smoke`、`10m`、`100m`、`1b`；当前只实际生成 `smoke` 和 `10m`。

## 已生成数据

Smoke 数据集：

- manifest：`artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/smoke_dataset/manifest.json`
- examples：896
- total pair tokens：49,280
- unique train pair tokens：28,160

10M 级数据集：

- manifest：`artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m/manifest.json`
- examples：184,000
- train examples：160,000
- val/test/heldout examples：各 8,000
- total pair tokens：10,120,000
- unique train pair tokens：8,800,000
- 当前落盘大小：约 21.0 MB

## 训练入口改动

AV-E 训练脚本已支持：

- `--dataset-manifest`
- `--gpu-resident-data`
- `--checkpoint-dir`
- `--resume`
- `--save-every`
- `--eval-batch-size`
- `--train-limit` / `--val-limit` / `--test-limit` / `--heldout-limit`

训练脚本现在按 batch 做 eval，避免 8k 验证集一次性前向；结果 JSON 会记录 dataset source、manifest scale、unique train tokens、processed pair tokens、checkpoint 路径和 peak CUDA allocated。

## 已验证 smoke

CUDA 环境：

- 默认 `python` 是 CPU-only Torch 2.9.1。
- `.venv\Scripts\python.exe` 是 Torch 2.11.0+cu128，可以看到 `NVIDIA GeForce RTX 4070 Laptop GPU`。

已通过的验证：

| 验证 | 结果 |
| --- | --- |
| `py_compile` | AV-E 与 AV-F 脚本通过 |
| AV-F smoke manifest | 896 examples、49,280 pair tokens |
| AV-E 读取 AV-F smoke manifest | CUDA smoke 20 step 通过，peak CUDA allocated 74.35 MB |
| AV-E 读取完整 10M train manifest | CUDA loader smoke 2 step 通过，peak CUDA allocated 31.35 MB |
| AV-E 70M batch 256 capacity smoke | 参数量 75,970,770，peak CUDA allocated 4,753.32 MB |
| AV-E 70M batch 512 capacity smoke | 参数量 75,970,770，peak CUDA allocated 8,923.00 MB |

Smoke 准确率不作为能力结论；这些验证只说明 10M 数据、manifest IO、CUDA 前向/反向、checkpoint 和 samples 输出链路可执行。

## 建议长训命令

AV-E joint baseline 已经可以用 batch 512 吃满 GPU；这条路线不是严格分阶段训练，只适合作为 joint 对照：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_ave_bidirectional_latent_external.py `
  --dataset-manifest artifacts\omni_transformer_stage_avf_10m_bidirectional_dataset\dataset_10m\manifest.json `
  --output artifacts\omni_transformer_stage_ave_bidirectional_latent_external\train_10m_70m_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_ave_bidirectional_latent_external\train_10m_70m_checkpoints `
  --resume `
  --steps 6000 `
  --eval-every 250 `
  --save-every 250 `
  --batch-size 512 `
  --d-model 768 `
  --heads 12 `
  --layers 10 `
  --latent-tokens 8 `
  --eval-batch-size 128 `
  --gpu-resident-data `
  --sample-count 8
```

如果显存抖动或其它程序占用导致 OOM，把 `--batch-size 512` 改为 `384` 或 `256`。batch 256 已验证峰值约 4.75 GB，稳定余量更大。

功耗/利用率监控：

```powershell
nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv -l 2
```

严格架构路线应使用 AV-G：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avg_strict_staged_10m_training.py `
  --dataset-manifest artifacts\omni_transformer_stage_avf_10m_bidirectional_dataset\dataset_10m\manifest.json `
  --output artifacts\omni_transformer_stage_avg_strict_staged_10m_training\train_10m_70m_strict_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avg_strict_staged_10m_training\train_10m_70m_strict_checkpoints `
  --resume `
  --batch-size 512 `
  --d-model 768 `
  --heads 12 `
  --layers 10 `
  --latent-tokens 8 `
  --text-steps 400 `
  --codec-steps 1200 `
  --translate-steps 2200 `
  --reason-steps 1400 `
  --joint-steps 800 `
  --eval-every 250 `
  --save-every 250 `
  --eval-batch-size 128 `
  --sample-count 8
```

如果 AV-E joint baseline 正在跑，不要同时启动 AV-G；等当前 run 结束或明确释放 GPU 后再跑，避免互相抢显存和功耗。

## 10M 训练结果

2026-07-05 用户本地完成两批 10M 长训：

| 路线 | 结果文件 | test answer | test source->target token | test source->target exact | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| AV-E joint baseline | `artifacts/omni_transformer_stage_ave_bidirectional_latent_external/train_10m_70m_result.json` | 97.25% | 73.48% | 0.00% | answer shortcut，不通过 |
| AV-G strict staged | `artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_result.json` | 77.85% | 72.49% | 0.00% | strict schedule 未修复 codec，不通过 |

两条路线都证明：AV-F 10M 数据集和训练入口可用，但当前 decoder/codec 结构无法完整重建 external sequence。下一步不应直接扩大到 100M，而应先做 decoder/slot binding 修复实验。

## 边界和担忧

1. 10M 数据集现在是 compact token external，不是真实像素、真实文件或真实 action 空间。
2. 数据生成很快，说明当前分布仍是低成本合成分布；它解决的是“小数据重复训练”的问题，不等价于 1B 高熵母分布已经完成。
3. batch 512 已能启动，但峰值显存接近上限；长训时如果 Windows 桌面或其它进程抢显存，可能需要降到 batch 384/256。
4. 长训通过标准不能只看 answer accuracy；必须同时看 source recon、target recon、source->target、target->source、sequence exact、heldout 和消融。
