# V2-R1R P0-D v14 完整 production 主设计层验收

日期：2026-08-10

验收对象：`r1r-p0d-v14-full-production`

## 1. 核心判定

v14 的正式判定是 **`P0-D machine-fail、main-review rejected`**。唯一 formal assessment 为 `FAIL_P0D_PRODUCTION`；G01–G08、G10、G11 全 true，只有 G09 false。按冻结 conjunction，完整 production P0-D 没有关闭，不能进入 P0-M、Qwen cache、模型、GPU 或训练。

这次失败不能简化成“generator 仍有明显捷径”。两个失败 cell 的 point accuracy 都低于原始 `mean_random + 0.10` ceiling，且 14 个 family-level aggregate 全通过；失败来自 98-way Bonferroni-Wilson simultaneous upper bound 略高于 ceiling。现有证据同时支持两个判断：数据没有出现 v1/v2 那类大幅、系统性的 surface leak；但在两个 OOD cell 上仍存在约 4 个百分点的局部 char-NB 信号，而 512 条不足以按冻结规则证明其真实 effect 小于 0.10。机器 FAIL 必须保留，不能事后改统计口径放行。

## 2. 正式 artifact 与机器事实

唯一 formal root：

`artifacts/v2-r1r/p0d-v14-full-production-20260809-1/`

固定 seed 为 `2026081402`。数据精确包含 ERE/CPS 各 train 4,096、六个 heldout/pair split 各 512，共 14,336 records、512 causal pairs、167,580 claims。14,336/14,336 semantic 与 surface fingerprint 均唯一；完整 provenance、source roundtrip、fresh simulator、answer/label binding、teacher trace、claim replay、单叶因果翻转、结构配额、语言模板、token budget、标签配平和 permutation audit 全部通过。

最大 source 为 993 pinned Qwen tokens；ERE train/validation p99 均 559，CPS 均 662，没有截断。所有 split 的生成最大重采样 attempt 为 11，overflow rejection 为 0；v13 的 ERE 容量耗尽已经由 3-attribute/6-value 状态域修复，而不是放松 overlap Gate。

正式开始于 2026-08-09 23:31:26，完成于 2026-08-10 00:11:01，约 39 分 35 秒。外层 30 分钟等待先返回 124，但 Windows child worker 持续运行并最终正常退出；这不是第二次 formal，也没有创建替代 root。正式 runner 随后完成 full regeneration、fixed artifact 只读重审、snapshot 与 seal。

replay ledger 的 `dataset_bytes_equal/artifact_read_only_replay_equal/accepted_inputs_read_only/passed` 均为 true。root evidence-seal SHA-256 为：

`6FD9AB9CD3F7CB1793F959762BA07AF0A0931052E7A3D05E3DD2C35455D19F52`

outer seal 直接登记 61 个非 `evidence-seal.json` 文件并复算 61/61 一致；source snapshot 共 44 文件，包含六个上游同名 seal，其完整 hash map 由 sealed run-metadata 的 `D075BF948ACAFB3668B3F2E797CC8B30BEAD1A77CD0BA9CA1724FAAA4E594FA6` 传递绑定，44/44 复算一致。artifact 总计约 235.4 MB。assessment 明确 `p0m_or_training_authorization_created=false`。

## 3. G09 失败的精确含义

两个且仅两个逐 cell 判定失败：

| cell | correct / N | accuracy | mean random | point excess | simultaneous upper | ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ERE composition OOD / full-text char 3–5 NB | 146 / 512 | 0.285156 | 0.250000 | +0.035156 | 0.354605 | 0.350000 |
| CPS horizon OOD / full-text char 3–5 NB | 107 / 512 | 0.208984 | 0.166667 | +0.042318 | 0.273711 | 0.266667 |

两格的 point accuracy 分别比原 ceiling 低 0.064844 和 0.057682；失败只发生在 simultaneous upper，越线分别为 0.004605 和 0.007044。其余 96 个 source cell 全通过，所有 prediction 完整，sparse scorer 对 accepted dense/exact scorer 的正控等价通过。

14 个 `family × baseline` pooled-heldout aggregate 全通过。最窄 aggregate 仍有约 0.02 upper margin；ERE/CPS 的 full-text char family aggregate 分别约为 accuracy 0.26/0.16、upper 0.29/0.18、ceiling 0.31/0.20。这反对“整个 family 存在扩散式 char shortcut”，但不能排除 composition/horizon 的局部弱信号。

在保持当前观察率不变时，98-way Wilson rule 至少需要约 584 个 ERE composition 样本和 635 个 CPS horizon 样本才会刚好通过；512 的设计把名义 `+0.10` effect ceiling 实际收紧到约 `+0.03–0.04`。因此根因包含一个明确的合同设计问题：样本规模与 simultaneous non-inferiority 证明强度没有先做 power matching。不能据此把两格直接判为纯噪声，因为它们确实是局部正 excess；也不能把它们写成稳定大捷径，因为 point ceiling 和 family aggregate 都通过。

G10 的 24 个 claim cell 全通过，最窄 upper 约 0.52、ceiling 0.60；claim 局部交叉配平在完整规模继续成立。

## 4. 对架构路线的影响

v14 没有训练模型，因此对 mixed core、通用 recurrent reasoner 或白皮书架构成功概率不形成正负行为证据。它主要更新数据层判断：

1. production generator 的容量、结构、因果、provenance、语言、claim 和复现机制已在完整规模成立；继续重做这些部分的边际价值低。
2. 当前真正未关闭的是 G09 的决策设计与两个局部 OOD char 信号，不是 ERE/CPS 任务整体失效。
3. 即便 v14 的 14,336 条数据结构质量很高，它仍是 failed formal diagnostic，不能被训练阶段当作已资格化 P0-D 数据来绕过 Gate。
4. 39 分钟 formal 暴露 exact char-NB 的 CPU/内存与可观测性成本；后继合同必须先做等价的批量 scorer 性能资格和阶段进度记录，不能再把外部等待上限设得低于已测 wall time。

## 5. 后继建议

立即进入 P0-M 是错误方向；直接修两条正式数据或换 seed 重跑同一合同同样违反证据纪律。合理后继是另立 v15 **G09 decision/power qualification + fresh formal**，而不是再重做 G01–G08/G10：

- 把 v14 sealed dataset 只作为开发期 power/故障注入材料，不用它事后接受自身；
- 在新合同冻结前，用合成 null、已知 3%/5%/10% leak、cell-local 与 family-wide fault 验证统计规则的 false-pass/false-fail 行为；
- 明确 primary family aggregate 与 secondary cell-local Gate 的职责，或保留 98-way upper rule但把每个 heldout 提高到至少 1,024；两者只能在新鲜 seed 上验收，不能用 v14 结果挑规则；
- 对 ERE composition 与 CPS horizon 分别加入与 char NB 预测无关的局部反例分析，判断是随机 label schedule 波动、choice-map 表面关系还是结构模板真的可被 bag-of-char 利用；
- 先资格化更快但与 exact scorer 逐 prediction 等价的 batch path，再生成新的唯一 root。

在新合同获批前，路线停在 P0-D；P0-M、cache、模型、GPU、训练和架构结论继续禁止。
