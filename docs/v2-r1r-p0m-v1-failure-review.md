# V2-R1R P0-M v1 缓存 Gate 失败复核

## 判决

P0-M v1 在唯一 cache root `artifacts/v2-r1r/p0m-v1-cache-20260810-1` 停止，训练未启动。该 root 保持封存，不能改写。

失败不是 cache 内容、Qwen revision、截断、数值或显存问题。cache audit 为 PASS，四个 hidden bank 的 schema、hash、shape、finite、mask 与 forbidden-field 检查均通过；batch 4/8/12 也都成功完成，峰值 allocation 分别约 3.81/4.10/4.39 GiB。

唯一根因是 v1 用单次 batch 的绝对 wall latency 选 batch：4/8/12 分别为 `1.079/1.204/1.852s`，该指标结构性偏向较小 batch。换算吞吐后分别为 `3.71/6.64/6.48 examples/s`，冻结的 batch 8 实际最优。因此 v1 Gate 测量对象错误，不能把 aggregate FAIL 解释为 cache 不可用或 P0-M 模型失败。

## 修复边界

P0-M v2 不修改 v1 root，不重复提取相同 hidden。它新建资格 root，重新验证 v1 evidence seal、全部 cache 内容与固定输入，并把 benchmark 判据改为：4/8/12 完整且 finite、峰值不超过 6 GiB、batch 8 吞吐高于 batch 4、正式 batch 仍固定为 8。该修复改变错误的测量语义，不降低缓存正确性或资源 Gate。
