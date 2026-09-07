# V2-R1R P0-D v15 G09 qualification 主设计层验收

日期：2026-08-10

正式 root：`artifacts/v2-r1r/p0d-v15-g09-qualification-20260810-1/`

正式状态：`FAIL_G09_QUALIFICATION`

## 1. 核心判决

v15 的唯一 Q formal 必须判为 **machine-fail、main-review rejected**。Q01–Q07 与 Q09 全 true，只有 Q08 false；冻结 conjunction 不允许用“只差一点”覆盖失败。assessment 明确 `fresh_production_authorized=false`，因此固定 fresh-seed production root 没有创建，P0-M、模型、cache、GPU 和训练继续禁止。

失败不在用户要求的 decision-power 本体。100,000-trial 合成 null/local/diffuse qualification、14 项 fail-closed fault、3,264 个 exact-equivalence prediction、single-bank/batch/progress 以及完整 replay 都通过。唯一失败是一次性 wall-clock 相对速度：fast batch `0.1261895s`，legacy exact `0.6008768s`，speedup `4.761702x < 5.0x`；prediction identity 仍为 100%，该 batch exact fallback 为 0。

## 2. 已获得但不能越级的窄证据

### 2.1 G09 decision power 已被有效测量

冻结规则在 98-cell/14-aggregate 完整矩阵上与 scalar production evaluator 56/56 一致。各场景的 desired rate 与 one-sided 95% Wilson lower 为：

| 场景 | rate | lower | Gate |
| --- | ---: | ---: | --- |
| global null 接受 | 0.99998 | 0.999940 | 通过 |
| ERE local `+0.04` 接受 | 0.95832 | 0.957268 | 通过 |
| CPS local `+0.04` 接受 | 0.98550 | 0.984865 | 通过 |
| ERE local `+0.10` 拒绝 | 0.99946 | 0.999325 | 通过 |
| CPS local `+0.10` 拒绝 | 0.99956 | 0.999437 | 通过 |
| ERE diffuse `+0.05` 拒绝 | 0.99673 | 0.996419 | 通过 |
| CPS diffuse `+0.05` 拒绝 | 0.99673 | 0.996419 | 通过 |

这支持 train 4,096、heldout 1,536 的样本量选择：它能高概率放行 v14 量级的单格弱信号，同时拒绝 cell/family ceiling 边界的禁止级 profile。它不证明真实 production 数据没有 shortcut，也不能反向接受 v14。

### 2.2 scorer correctness 与内存编排通过

- v14 sealed ERE/CPS 三种 NB 共 1,152 个 accepted dense + compact-exact 对照，0 mismatch；普通 exact fallback 5/1,152，即 0.434%。
- 2,048 个合成 sparse model 对照 0 mismatch；64 个 exact tie 全部触发 fallback 并保持 label-order tie-break。
- streaming probe 同时只持有一个 analyzer bank，33 个 batch，预测完整，六个 prepare/done 进度事件齐全。
- Q replay 的 decision bytes、fault bytes、scorer deterministic projection、v14 只读输入与 accepted source 均一致。

因此 scorer 的语义修复有效，Q08 不能解释为答案错误、fallback 失控或内存编排失败。

## 3. Q08 为什么仍必须失败

合同把两种不同职责合并进 Q08：执行路径正确性与一次性相对计时。前者通过，后者以 `4.7617x` 未达 `5x`。正式运行前的开发观测曾为约 `6.79x/8.20x`，而 formal 同一 192-prediction batch 降至 `4.76x`；这说明单次 `legacy_seconds / fast_seconds` 对 factorization cache、调度和短 benchmark 噪声敏感。相对 legacy 的瞬时速度不是 G09 统计判别正确性的系统不变量。

但该设计问题只能指导下一合同，不能修改 v15。formal 没有预注册 warmup、重复次数、paired median、置信区间或绝对时限；事后把 5x 降到 4.5x、改用开发结果或只看 fast `0.126s` 都会破坏 single-use 证据纪律。

## 4. Artifact 验收与停止纪律

Q root 的 evidence seal 可完整复算，根 `evidence-seal.json` SHA-256 为：

`997166EECC490F7B6AF6E80C01E28E187539BEDA351764597F0D79DE42C1EE38`

48 个 source snapshot 文件被封存；decision/fault/scorer projection replay 全一致，v14 sealed input 保持只读。固定 F root `artifacts/v2-r1r/p0d-v15-full-production-20260810-1/` 不存在，证明执行层遵守 Q FAIL 后的硬停止。

## 5. 后继边界

不能重跑或修补 v15 Q，也不能继续使用 v15 的 F root/seed。若继续，应另立 v16：把 sealed v15 Q01–Q07/Q09 作为只读 component evidence，只重做独立的 operational runtime entry；性能 Gate 应使用预热后的多次 paired median 与绝对 batch/full-audit 预算，并把 exact identity、fallback、single-bank 继续设为 correctness Gate。只有新的 operational qualification PASS，才可再冻结另一个从未生成的新 seed production formal。

本轮没有完成 fresh-seed production、完整 P0-D、P0-M 或任何架构验证。
