# Stage AV-J-C：RL / verifier reward 潜空间推理修复

日期：2026-07-07

## 背景

AV-J 和 AV-J-B 的共同失败点是：answer 可以接近 99%，但 target record 只在 68%-70% 左右；AV-J-B 的 count/copy 辅助头能学，candidate mask 和 record 级状态改写没有闭合。继续小修 CE 权重没有意义，下一步必须让模型直接为“写对 target record”承担采样级 reward。

## 新增入口

```powershell
experiments\omni_transformer_stage_avjc_rl_verifier_reward.py
```

AV-J-C 复用 AV-J-B 模型结构，但训练语义直接切换：

- SFT 阶段只负责 `codec -> trace_sft -> target_sft`，把模型拉到可采样状态；
- RL 阶段采样 target record、candidate mask 和 count value；
- reward 使用 oracle verifier：record exact、字段正确率、changed-slot 正确性、same-row candidate exact、count exact；
- baseline 使用 greedy record reward，loss 为 self-critical policy gradient；
- answer loss 默认 0，不再让 answer shortcut 主导方向；
- learned verifier 保留为辅助/诊断，不把当前 83% 左右的 learned verifier 当最终裁判。

这不是把 eval 指标写进 history，而是让 `advantage * sampled_log_prob` 对 target/candidate/count logits 反传。

## 已验证

单元测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_stage_avjc_rl_verifier_reward.py tests\test_stage_avjb_trace_verifier.py
```

结果：5 passed。

CPU smoke：

```powershell
artifacts\omni_transformer_stage_avjc_rl_verifier_reward\cpu_smoke_result.json
```

验证 `rl_verifier` 阶段、policy-gradient stats、eval variants、JSON 输出和 checkpoint 写入。

CPU resume smoke：

```powershell
artifacts\omni_transformer_stage_avjc_rl_verifier_reward\cpu_resume_smoke_result.json
```

验证从 `latest.pt` 继续训练。

70M RL capacity：

```powershell
artifacts\omni_transformer_stage_avjc_rl_verifier_reward\capacity_70m_rl_2step_result.json
```

| 项 | 数值 |
| --- | ---: |
| d_model / layers / heads | 480 / 8 / 8 |
| 参数量 | 72,161,403 |
| batch size | 256 |
| peak CUDA allocated | 3,279.69 MB |
| elapsed | 3.57 sec |

capacity 只证明 70M batch256 的 RL loss 前后向可跑，不提供能力结论。

## 建议长训命令

优先从 AV-J-B 长训 checkpoint 继续做 RL，避免重复 SFT：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avjc_rl_verifier_reward.py `
  --output artifacts\omni_transformer_stage_avjc_rl_verifier_reward\rl_from_avjb_70m_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avjc_rl_verifier_reward\rl_from_avjb_70m_checkpoints `
  --init-from artifacts\omni_transformer_stage_avjb_trace_verifier\train_70m_checkpoints\latest.pt `
  --train-size 200000 `
  --val-size 4096 `
  --test-size 4096 `
  --heldout-size 4096 `
  --batch-size 256 `
  --eval-batch-size 512 `
  --d-model 480 `
  --layers 8 `
  --heads 8 `
  --codec-steps 0 `
  --trace-steps 0 `
  --target-sft-steps 0 `
  --rl-steps 8000 `
  --eval-every 500 `
  --save-every 500 `
  --lr 2e-4 `
  --rl-entropy-weight 0.01 `
  --rl-supervised-anchor-weight 0.10
```

如果怀疑 AV-J-B checkpoint 的策略分布已经过窄，可以从头跑 SFT+RL：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_avjc_rl_verifier_reward.py `
  --output artifacts\omni_transformer_stage_avjc_rl_verifier_reward\train_70m_result.json `
  --checkpoint-dir artifacts\omni_transformer_stage_avjc_rl_verifier_reward\train_70m_checkpoints `
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
  --target-sft-steps 8000 `
  --rl-steps 8000 `
  --eval-every 500 `
  --save-every 500 `
  --lr 5e-4
```

## 通过标准

必须优先看：

- `full_target_record_exact` 是否显著超过 AV-J-B 的 69.48%；
- `same_row_move_target_record_exact` 和 `count_delete_or_add_target_record_exact` 是否同步提升；
- `full_candidate_mask_exact` 是否脱离 41%-44% 平台；
- `no_source` / `no_process` target record 是否仍低，避免奖励被 shortcut 吃掉；
- history 中 `rl_sample_reward`、`rl_greedy_reward`、`rl_advantage`、`rl_entropy` 是否没有塌缩。

answer 不作为本轮通过门槛。target record 闭合后再恢复 answer writer。

## 70M 长训结果

用户本地已完成两条 AV-J-C 70M 长训：

```powershell
artifacts\omni_transformer_stage_avjc_rl_verifier_reward\rl_from_avjb_70m_result.json
artifacts\omni_transformer_stage_avjc_rl_verifier_reward\rl_from_avjb_70m_summary.json
artifacts\omni_transformer_stage_avjc_rl_verifier_reward\train_70m_result.json
artifacts\omni_transformer_stage_avjc_rl_verifier_reward\train_70m_summary.json
```

| 路线 | test target record exact | heldout target record exact | test candidate mask exact | test count value accuracy | test no-source target exact | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AV-J-B checkpoint -> RL | 75.24% | 77.25% | 43.44% | 98.56% | 31.47% | 部分提升，但不通过 |
| 从头 SFT+RL | 65.87% | 67.58% | 43.44% | 97.68% | 40.28% | 低于 AV-J-B，不通过 |

AV-J-B 对照的 test target record exact 是 69.48%，candidate mask exact 是 41.76%，no-source target exact 是 23.71%。因此：

- 从 AV-J-B checkpoint 继续 RL，把 test target record exact 提高约 5.76 个百分点，说明 record-level reward 有真实信号；
- candidate mask 基本没有突破，仍在 42%-44% 平台；
- no-source target exact 从 23.71% 升到 31.47%，从头训练甚至升到 40.28%，说明 reward 被部分 shortcut 吃掉；
- 从头 SFT+RL 没有比 AV-J-B 更好，说明当前 RL 不能替代较充分的 SFT warm start；
- answer 指标在 checkpoint->RL 路线仍高，是旧 answer head 保留下来的结果，不作为本轮能力证据；从头路线 answer 为 0，符合 answer loss 关闭预期。

结论：AV-J-C 当前实现不通过。它证明“RL/verifier reward 比纯 SFT 更接近正确方向”，但也证明当前 reward 太粗：它能推高一部分 target record exact，却没有强迫模型学会 candidate selection，也没有压住 no-source shortcut。

下一步不应继续延长同一 reward。应改成 AV-J-D：把 reward 从整条 record 分解到任务族和步骤级 edit rollout，尤其是 same-row candidate selection、count delete/insert、changed-slot 写入；同时把 no-source/no-process penalty 纳入 reward，不能只在 eval 里看。

## 担忧

1. Self-critical baseline 可能导致 advantage 很小；如果 reward 无明显变化，需要提高 entropy 或改成 group-relative 多样本 advantage。
2. 当前 reward 使用 oracle target record，不是 learned verifier。它是正确的第一步，因为 learned verifier 本身还不够可靠。
3. 如果 RL 只提高字段均值但 record exact 不动，说明 credit assignment 仍太散，下一步需要按任务族拆 reward 或引入逐步 edit action rollout。
