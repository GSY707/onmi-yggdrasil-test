# Stage AV-J-B：长链 trace / verifier 潜空间推理准备

日期：2026-07-06

## 目标

AV-J 70M 暴露的关键问题不是 source codec，而是长链潜空间推理和 target record 写入：

- answer sequence exact 接近 99%，但 target record exact 只有约 68%；
- no-source/no-process 下 answer 仍很高；
- `same_row_move` 和 `count_delete_or_add` 明显低于 `conditional_recolor`。

AV-J-B 的目标是把“oracle solver 生成的粗 trace”变成更细的结构化过程监督，用它引导模型产生自己的 latent chain，而不是继续奖励短答案模板。

## 新增入口

```powershell
experiments\omni_transformer_stage_avjb_trace_verifier.py
```

AV-J-B 复用 AV-J 的 record/text 数据分布和三类任务，但增加：

- same-row candidate mask；
- count value；
- copy-vs-update gate；
- target record 字段级 accuracy；
- record verifier；
- answer loss 后移并默认降权到 `0.05`。

## 设计区别

| 项 | AV-J | AV-J-B |
| --- | --- | --- |
| trace | `read_a/read_b/edit_slot/condition/action` | 加入 candidate mask、count value、copy/update gate |
| target decoder | 每个 target slot 从全局 process state 自己猜 | source copy logits + update logits 通过 copy-vs-update gate 混合 |
| answer | joint 中权重 0.5 | joint 中默认 0.05，避免先背答案 |
| 诊断 | record exact 为主 | 增加字段级 target accuracy、candidate/count/copy gate、verifier |
| 训练阶段 | codec -> operation -> process -> joint | codec -> trace_sft -> target_verifier -> joint |

这里的 trace 仍是 oracle solver 生成的结构化过程，不等价于真正高质量自然语言 CoT。它的定位是“引导真实 latent chain 产生的脚手架”：先让小模型知道应该保留哪些中间态，再看它能不能在不输入 trace 的情况下生成 target record。

## 已验证

单元测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_stage_avjb_trace_verifier.py
```

结果：2 passed。

CPU smoke：

```powershell
artifacts\omni_transformer_stage_avjb_trace_verifier\cpu_smoke_result.json
```

验证 dataset、rich trace loss、copy gate、verifier、四阶段 schedule、checkpoint 和 eval variants 可跑。

CPU resume smoke：

```powershell
artifacts\omni_transformer_stage_avjb_trace_verifier\cpu_resume_smoke_result.json
```

验证从 `latest.pt` 恢复训练。

CUDA smoke：

```powershell
artifacts\omni_transformer_stage_avjb_trace_verifier\cuda_capacity_smoke_result.json
```

验证 AMP、GPU resident data 和 eval variants。

70M capacity：

```powershell
artifacts\omni_transformer_stage_avjb_trace_verifier\capacity_70m_batch256_2step_result.json
```

| 项 | 数值 |
| --- | ---: |
| d_model / layers / heads | 480 / 8 / 8 |
| 参数量 | 72,161,403 |
| batch size | 256 |
| peak CUDA allocated | 4,628.25 MB |
| elapsed | 4.41 sec |

这些 smoke 不提供能力结论。短 smoke 中 verifier 可能因为 target record 全错而显得高或低，都不应作为能力指标。

## 建议长训命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avjb_trace_verifier.py `
  --output artifacts\omni_transformer_stage_avjb_trace_verifier\train_70m_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avjb_trace_verifier\train_70m_checkpoints `
  --resume `
  --train-size 200000 `
  --val-size 4096 `
  --test-size 4096 `
  --heldout-size 4096 `
  --batch-size 256 `
  --eval-batch-size 512 `
  --d-model 480 `
  --layers 8 `
  --heads 8 `
  --codec-steps 2000 `
  --trace-steps 5000 `
  --target-steps 10000 `
  --joint-steps 3000 `
  --eval-every 1000 `
  --save-every 1000 `
  --answer-token-weight 0.05
```

如果训练中 answer 仍过快升高而 target record 不动，可以把 `--answer-token-weight` 降到 `0.0` 跑纯 target/trace 版。

## 通过标准

必须同时看：

- `full_target_record_exact`；
- `full_answer_sequence_exact`；
- `full_candidate_mask_exact`；
- `full_count_value_accuracy`；
- `full_copy_gate_accuracy`；
- target active/color/shape/row/col field accuracy；
- no-source/no-operation/no-process gap；
- heldout 是否跟 test 接近。

必须判失败：

- answer 高但 target record 低；
- copy gate 高但 changed slot / target record 没上去；
- candidate/count 指标高但 same_row_move/count_delete_or_add 仍低；
- verifier 高但 target record 低。

## 担忧

1. oracle trace 仍可能不是小模型最容易学的 trace；它只是比直接让模型自己产生 CoT 更好的引导。
2. copy-vs-update gate 使用 source one-hot logits 作为结构偏置，可能让模型过度 copy，需要重点看 changed slot 和 target record exact。
3. verifier 当前是诊断/辅助 loss，还不是 RL verifier；如果 SFT trace 仍不能闭合，下一步再引入 target-record reward / GRPO-like 训练更合理。
