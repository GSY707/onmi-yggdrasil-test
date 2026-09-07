# V2-A Closure C1U S2 多 Bank、单 Seed、Matched K1/K8 结果复盘

## 1. 终局

C1U S2 已按冻结身份各启动一次 preflight 与 formal。preflight sealed PASS；formal 完成全部六个固定端点和 24,000 个 optimizer updates，随后在 R203、R204、R205 失败并封存为：

`FAIL_V2_A_C1U_S2_QUALIFICATION`

这是完整科学 FAIL，不是运行故障。R201、R202、R206、R207 通过，证明前驱、cache、split、schedule、matched parity、fixed-endpoint 会计、K1 owner-swap null、K8 多地址敏感性和证据完整性均有效；但 K8 没有在 fresh bank 上形成合格行为，也没有稳定优于 K1，其 heldout 因果效果同样没有通过。

formal 终态为 `authorizes=nothing`。同一 identity 不得重跑、续训、换 seed、选择 checkpoint、降低 Gate 或进入 S3。multi-seed、S3 training、single-seed formal、C2、V2-A PASS、V2-B 与 V2-C 均为 `NOT_RUN`。

## 2. 身份、封印与会计

preflight：

- identity：`V2-A-CLOSURE-C1U-PGF-S2-MULTIBANK-MATCHED-K1-K8-PREFLIGHT-20260902-1`
- result SHA-256：`494A72AD163286FE02B7C6FDDDD382A2B1B8B43177D1939D6DBA9781D545C23B`
- evidence-seal SHA-256：`EBB51DDE0C9642F66133E75CC6A7D850E7DD3C97CD57E60E5F23AD02FF502421`
- P201–P207：全 PASS
- disposable optimizer steps：`32`
- formal optimizer steps / model writes：`0 / 0`
- seal replay：`60/60`

formal：

- identity：`V2-A-CLOSURE-C1U-PGF-S2-MULTIBANK-MATCHED-K1-K8-20260902-1`
- result SHA-256：`F0EABDE1B17FD87077EDBC7A75208B38469DEC344ED820F454E1C3C2DB46EAA4`
- evidence-seal SHA-256：`00B7824AD4A0F964CDD1A89CF12FC2A5D334D37FE6923F9859C34B9B48125A4B`
- qualification SHA-256：`273CBC1FDA48427B6CFAD9F24D5DAE586E60B099515E56EC57D453610983996D`
- endpoints manifest SHA-256：`0ED704B3B2790E2F3468D1CAE741A92F4CFA2D2793BF09ED2E67814D36689406`
- formal optimizer steps / model writes：`24,000 / 6`
- disposable optimizer steps：`0`
- checkpoint selection / intermediate checkpoints：`false / 0`
- seal replay：`76/76`

六个端点均由相同 tensor initialization `DED9C6…D517` 开始，每个恰好完成 4,000 updates 并只写一次 `fixed_4000.pt`：

| Fold / arm | Endpoint SHA-256 |
| --- | --- |
| F0 / K1 | `FAC5D3BF…B6776` |
| F0 / K8 | `B8212229…9DFDB` |
| F1 / K1 | `22067AF6…56228` |
| F1 / K8 | `77DD94E3…72CCD` |
| F2 / K1 | `ACDCBBD8…7EF79` |
| F2 / K8 | `309D8433…F8359` |

## 3. 正式 Gate

| Gate | 结果 | 含义 |
| --- | --- | --- |
| R201 predecessor/cache/split/schedule/source | PASS | S1 前驱、六-bank cache、fold 与源码身份一致 |
| R202 fixed endpoints/accounting/matched parity | PASS | 六端点、24,000 steps、六写入及 K1/K8 初始化/参数匹配成立 |
| R203 K8 fresh-bank absolute behavior | FAIL | heldout answer、逐 bank/family 和 factorial 均远低于门槛 |
| R204 paired K8 beats K1 | FAIL | overall 与 CPS gain 不合格，所有 bootstrap lower 均不大于 0 |
| R205 K8 heldout causal behavior | FAIL | support flip、margin 与 two-contributor 大面积失败 |
| R206 functional K8 and measured K1 null | PASS | K8 对地址交换敏感、K1 交换严格为零，但只证明敏感性 |
| R207 endpoint/source/predecessor/cache/seal integrity | PASS | endpoint 与完整 evidence seal 可复放 |

## 4. Heldout 行为与 paired gain

| Arm | Overall | CPS | ERE | Factorial exact |
| --- | ---: | ---: | ---: | ---: |
| K1 | `50/192`（26.04%） | `39/96`（40.63%） | `11/96`（11.46%） | `0/48` |
| K8 | `52/192`（27.08%） | `33/96`（34.38%） | `19/96`（19.79%） | `3/48` |

K8−K1 的 bank-cluster paired gain 为：

| Slice | Point | 95% bootstrap interval |
| --- | ---: | ---: |
| Overall | `+1.04pp` | `[-6.77pp, +9.38pp]` |
| CPS | `-6.25pp` | `[-14.58pp, +5.21pp]` |
| ERE | `+8.33pp` | `[-6.25pp, +21.88pp]` |

ERE 只有 point gain 越过 `+5pp`，但置信下界仍小于 0；overall 与 CPS 连 point gate 都未通过。六个 bank/family 的 K8 accuracy 为 6.25%–50%，没有一个证据面支持“只差统计功效”。

## 5. Heldout 因果与 functional-K 边界

K8 no-core accuracy 数值较低：CPS `14/96`，ERE `20/96`。但 no-core margin-drop 的 lower 95% 分别为 `-2.398` 与 `-3.810`，没有证明正确答案 margin 依赖 recurrence。两个 support 的 answer flip 仅为 CPS `33/96, 33/96`、ERE `19/96, 19/96`；two-contributor 仅 CPS `22/96`、ERE `12/96`。

R206 通过不能覆盖这些失败。K8 的两个 owner-swap logit-L2 lower 95% 在 CPS 为 `11.90/11.57`、ERE 为 `8.07/8.03`，而 K1 全为 0；这证明八地址状态真实存在并会影响 logits，却没有证明影响方向正确。正式结果因此把“functional address sensitivity”和“correct causal computation”分开判决。

## 6. 证据边界

本 formal 只使用一个 scientific model seed；它没有估计训练随机性的稳定性。但本次 FAIL 不可解释为“还没做多 seed”：同一个 seed 内 K8 对训练 bank 几乎完全拟合、对 fresh bank 系统性崩溃，并且 post-stop 确定性反事实又分别复现 label-permutation 与 opaque-renaming 失败。增加 seed 不能修复结构不变性。

完整 post-stop 实验与根因见 `docs/v2-a-closure-c1u-s2-failure-attribution.md`。
