# V2-R1R P0-M v2 训练前复核

## 判决

v2 cache qualification 已在 `artifacts/v2-r1r/p0m-v2-cache-qualification-20260810-1` sealed PASS，但随后非正式真实 GPU 单步诊断发现训练实现尚未冻结，因此 v2 标记为 `SUPERSEDED_PRE_TRAINING`。没有创建 ERE/CPS/joint/baseline/throughput/assessment v2 roots，也没有形成 P0-M 结论。

## 发现与修复

latent 路径首次 BF16 forward 在 choice mask 处报 overflow。根因是 BF16 logits 使用 FP32 pooled tensor 的 `finfo.min`；修复为按 logits dtype 取 mask fill，并新增 BF16 回归测试。修后真实 ERE forward/backward finite，约 `0.82s/step`、峰值约 `0.69 GiB`。

baseline 路径原实现为整个 400–700 token prompt 计算 248,320 维词表 logits，尽管 loss 只覆盖尾部 assistant target；同时 LoRA 投影没有 autocast。直接单步约 `3.11s`、峰值约 `5.24 GiB`。修复后使用 left padding 保证 target 尾部对齐，通过 `logits_to_keep=maximum_target+1` 只生成精确 loss window，并在 FP16 autocast 下保留 FP32 LoRA master parameters。等价 direct loss 从 `4.77734` 变为 `4.77696`（仅低精度舍入差），logits 由整段缩至 `[1,6,248320]`，单步约 `2.37s`、峰值约 `4.22 GiB`，greedy generation 也成功执行。

因为上述修改发生在 v2 qualification snapshot 之后，后续训练不混入 v2；新建完整 P0-M v3 合同并重新资格化 cache。
