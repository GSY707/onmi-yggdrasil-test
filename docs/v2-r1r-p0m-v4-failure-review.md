# V2-R1R P0-M v4 M04 失败复核

## 判决

v4 cache、ERE M02、CPS M03 均 sealed PASS。ERE 单任务为 answer `1.000`、claim `.988`，但 owner-shuffle `.982`（drop `.0065`）；CPS 为 answer/claim `1.000`、shuffle `.735`（drop `.265`）。这证明 interaction probe 已解决“不可训练”，同时暴露 ERE 固定集上的 claim-only 记忆捷径。

joint 在 update 1800 达到本轮最佳附近：ERE/CPS `1.000/1.000`、claim约 `.86`、owner-shuffle约 `.71`、drop约 `.15`。原 schedule 在 1800 后将 claim weight 置零；到 2400 终值为 claim `.834677`、shuffle `.710349`、drop `.124328`，M04 FAIL，baseline 与后续未运行。

## 根因与 v5 修复

v4 的事后 owner-shuffle Gate 能发现捷径，却没有在训练中提供“同一正 claim 必须更匹配正确 owner 状态”的梯度；同时监督在指标尚未过线时提前关闭。v5 保留 v4 probe 与所有答案路径，新增同 family mismatch-owner 对比：仅对正 claim，要求 correct-state truth score 高于另一个 episode 同 prefix 的 truth score；joint batch 的 owner 置换严格限制在各自 8 条 ERE/CPS 半批内，避免用任务族差异取巧。

claim loss 变为 CE + positive/negative pair ranking + owner contrast。claim 权重在 400 后保持 `.5` 直到 smoke 通过或 2400，不再制造无梯度尾段；probe 在 checkpoint 中仍物理删除。Gate 不变且额外机制约束未降低。
