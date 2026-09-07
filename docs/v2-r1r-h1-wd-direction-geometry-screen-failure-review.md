# V2-R1R H1-WD direction-geometry 非正式 screen 失败复盘

日期：2026-08-23  
机器终态：`CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY`  
授权：`authorizes=nothing`

## 核心判决

唯一真实运行在完成 train 4096 与 heldout 1024 条 target-before trajectory 重放后，
被 replay identity Gate 拒绝。它没有进入 R2 fit、R3 Jacobian/Fisher 或 R4 null，
也没有写出 `result.json`。因此本次运行既不能支持也不能反驳“transfer target 存在
input-predictable 或 projection-parameter-reachable 稳定方向”；它只判定当前 v1
执行合同无法用不同 batch geometry 复现旧 target bank 到冻结的绝对容差。

输出 root
`artifacts/v2-r1r/h1-wd-direction-geometry-screen-20260823-1/` 已消耗，保留
`contract-manifest.json`、`preflight.json`、`events.jsonl`、`run-state.json` 与
`crash.json`。`result.json` 不存在；禁止删除 root、放宽容差或重跑同一 identity。

## 终态证据

运行耗时 `36.58s`，进程已退出。全量 replay audit 得到：

| 字段 | 数值 |
| --- | ---: |
| maximum common absolute error | `7.9691410e-05` |
| maximum projection absolute error | `9.5039606e-05` |
| frozen absolute tolerance | `2.5e-05` |
| direction-geometry capture batch | `128` |
| causal target materialization microbatch | `4` |

旧 bank 中 common/projection 的 RMS 约为 train `0.709/0.750`、heldout
`0.598/0.636`。因此最大 replay 差异相对 RMS 约为 `1.1e-4–1.6e-4`，数值上很小，
但仍严格超过本合同的绝对 Gate。fail-closed 的 crash 是正确执行冻结规则，不得把
“误差很小”改写为 PASS。

终态只读复核确认：

- source checkpoint SHA-256 仍为
  `112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D`；
- causal target bank SHA-256 仍为
  `DFC8F08776CE56EFB8022BF35B7C3DD72F8F1C53C95DBF0DD28F593DC2BFD503`；
- 旧 causal/overlap root 未修改，没有 optimizer step、模型写入或 checkpoint；
- sibling stderr 保存完整 traceback，stdout 为空；`rerun_authorized=false`。

## 失败原因

直接原因不是 checkpoint、record order、route schedule 或 target hash 漂移，而是 v1
把旧 bank 的 microbatch `4` 输出与新 replay batch `128` 输出进行过严的逐元素绝对
等价检查。CUDA attention/matmul 在不同 batch shape 下可以选择不同 reduction/kernel
geometry；每条记录在数学上独立，有限精度路径仍会不同，误差又沿八步 recurrent
trajectory 累积到 common/projection。

启动前的 batch-128 spot check 只覆盖首批 128 条，最大 projection 误差约
`1.58e-05`，低于 `2.5e-05`；显式 integration smoke 又按 batch `4` 运行。两者都没
覆盖全 split 的最坏记录，因而没发现生产 batch 的尾部最大误差。这是 preflight
覆盖不足与 contract geometry 不匹配，不是 R0–R4 统计假设失败。

## 没有得到的结论

- R0/R1 centroid 虽在内存中构造，但未形成终态 artifact，不能升级为本 screen 结果；
- R2 的 attention state、projection feature trunk、common-private hidden 与两个
  output-proxy ridge 均未拟合；
- R3 的 full-bank exact final-head 解与 sampled matrix-free trunk+head 解均未运行；
- R4 的 unrestricted/within-route permutation、route permutation、六项 Bonferroni
  residual null 与 split-half Gate 均未运行；
- 不得用本次 crash 解释旧 W `1.00450/0.99247`，也不得恢复训练、H1 formal、F1 或 P2。

## 若未来另立后继合同

本次运行不自动授权 successor。若用户另行授权，新 identity 至少必须在启动前冻结：

1. target-before replay 与 bank 使用相同 microbatch/kernel geometry，或把这些 state
   在新只读 bank 中一次性物化，避免运行时重构身份；
2. 对全部 train/heldout 做 batch-invariance preflight，而不是首批 spot check；
3. 将 bitwise identity 与数值等价分开：先冻结相对/绝对混合误差、functional error
   和 downstream feature stability，再创建 single-use root；
4. 保留本次已实现的 record-coordinate R0/R1、target-before R2、exact-head + local
   matrix-free R3，以及 unrestricted/conditional permutation 与 split-half R4；
5. 新 root、阈值、代码与报告身份必须与本次 crash 完全隔离，不得覆盖或续跑。

