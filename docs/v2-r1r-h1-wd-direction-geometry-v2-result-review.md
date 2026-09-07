# V2-R1R H1-WD direction-geometry v2 结果复盘

日期：2026-08-23  
身份：`H1-WD-DG-V2-20260823-1`  
机器终态：`COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`  
科学终态：`NO_QUALIFIED_R2_R3_COMPONENT`  
授权：`authorizes=nothing`

## 1. 核心判断

本轮已经取得完整、身份有效的 R1–R4 测量。结果不是“方向全为随机”，也不是
“projection 已拿到可转移能力”，而是三个层次同时成立：

1. route 与 target-before input state 中存在跨记录关联；
2. 这些关联在当前按 record ID 排列的 heldout 上存在严重幅度/稀疏度 shift，projection
   trunk 的小增益不稳定，shared projection head 与 sampled local-J 都不能泛化；
3. R3 后 residual 明显不与当前 registered null 相容，但当前 screen 不能把它合格地归因于
   更细 subclass、完整 projection 参数可达方向或纯随机散步。

因此，用户先前“公共专家自带分类会把增删方向变成随机散步”的猜想不是本轮的主解释。
更直接的证据是：方向本身相当一致，但 target 是否需要非零 transfer 在 train/heldout 间
发生了巨大的 prevalence shift；静态 global/route 写入在大量 heldout 零 target 记录上变成
系统性误写。当前瓶颈首先是 target-before write gate 与 split nonstationarity，其次才是
projection 内部如何承接方向。

## 2. 执行与完整性

唯一命令创建 exclusive preflight lease 后运行一次。root 前以原生 microbatch `4` 完整重放
train 4096 与 heldout 1024，common/projection 最大绝对误差均为 `0.0`；正式计算耗时
`3090.02s`。stdout 给出 COMPLETE，stderr 为零字节，launcher 与 worker 均已退出。

`result.json` 明确报告：source checkpoint、causal target bank、内存模型参数、旧 crash
archive 与所有固定输入均未改变；optimizer step 为 0，没有写模型或 checkpoint。顶层
`valid_complete_measurement=true`，但 `h1_qualified=false`、`p1_completed=false`，不授权
calibration、formal、F1、P1、P2 或训练 successor。

主要终局文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `result.json` | `0E482421FEA8C251F42D1014941F81891E86E0C41A35EBF3407D073CB605F3A0` |
| `null-distributions.json` | `FFBA108B17D50AF1BFA1AA5E05615F388224D43D1990631C486DFDD1E31523E7` |
| `run-state.json` | `37E97D1A548C24208740871CD7A3D35B67174ECCE4CE70158700723EE9FDF550` |
| `preflight.json` | `C0DF270F0DE509CD0C6823B3705D7FA9D1B8D16CEA29F738C8B6A8E14493B7AF` |
| sibling lease | `02CD445D630A918B1F18A3F01E724C10DD97D5174D96BC791D8AD4387A2853EB` |

## 3. R1–R4 结果

### R1：route 有语义，但无条件 route centroid 在 heldout 仍然误写

train 上 global centroid 解释 `4.394%`，route 相对 R0 再解释 `5.388%`；heldout 上两者分别
变成 `-0.792%` 与 `-0.586%`，route centroid 相对零基线的总 gain 为 `-1.383%`。然而正确
route ID 的结果显著优于保持计数的 route permutation：observed `-0.586%`，null mean
`-11.258%`，`p=9.999e-05`。

这不是“route 有正 heldout 收益”，而是“正确 route 比错误 route 少错很多”。train 与
heldout centroid 的 cosine 仍高：global `0.8944`、route-0 `0.8728`、route-1 `0.9338`；
但 norm 从 `2.458/2.764/4.314` 降到 `1.262/0.803/2.563`。方向大体一致而幅度变化，已经
足以解释为何 route permutation 显著、实际 energy gain 却为负。

### R2：上游输入态可预测，projection trunk 的可用增益未复现

主对象 `projection_feature_trunk` 在 train 解释 `66.301%`，heldout 只剩 `3.189%`
（nMSE `0.968114`，cosine `0.5494`，heldout/train nMSE ratio `2.873`）。exact record
bootstrap 95% CI 为 `[-10.921%, 15.566%]`，site-macro observed `2.291%`、CI
`[-12.896%, 15.298%]`，四个 lambda 只有 `2/4` 为正，因此 `r2_replicated=false`。

旁证并不等于“输入无信息”：attention state 的 heldout gain 为 `6.429%` 且 `4/4` lambda
为正；common-private hidden 为 `4.672%`、`2/4` 为正。两个近 target output proxy 反而为
`-5.587%/-9.075%`。这提示可预测信息更像存在于上游 input-conditioned state，而不是已经
稳定编码在 projection 输出附近。attention 不是本轮 primary、也没有独立 feature-specific
qualification，故只能作为机制线索，不能改写 R2 FAIL。

### R3：shared final head 不可达；sampled local-J 也未泛化

exact full-bank shared final-head 解在 train 仅解释 `0.070%`，heldout 为 `-0.334%`
（nMSE `1.003340`、cosine `-0.3633`）；record bootstrap CI
`[-0.372%, -0.295%]`，四个 lambda 为正比例 `0/4`，两套 sketch 的 R3 score/fit null
均为 `p=1.0`。这是当前最强的 shared-head 不可达证据。

sampled trunk+heads local-J 在两层 train 分别解释 `5.712%/7.614%`，heldout 却为
`-4.692%/-7.120%`，aggregate heldout gain `-5.922%`。同时 CG 16 步未收敛，relative
residual 为 `0.820/0.852`，finite-difference relative error 为 `0.268/0.299`。因此它只
支持“当前 sampled solve 没找到可泛化方向”，不能升级成完整 recurrent projection
Jacobian 不可达证明。结合 exact-head 结果，简单加强 final projection head 的训练没有
证据基础；若再研究参数可达性，先要修复 local-J solver fidelity 并加入 write gate。

### R4：residual 有稳定结构，但当前 screen 未完成结构归因

R3 后 heldout residual 的 raw micro energy 仍为原 target 的 `98.478%`。两套独立 sketch
都得到 `residual_random_compatible=false`：global resultant 的 `p=9.999e-05`，两个 route
conditional resultant 均为 `p=0.0004880`，通过注册的 `0.01/12=0.0008333` 阈值。
所以“大量局部方向只是独立随机散步”与当前数据不相容。

本结果中的 heldout record-macro residual fraction 达到约 `2.83e31`，不是可解释的效应量：
433 条 target-energy 为零的记录被 `1e-30` denominator floor 放大。该字段只能标作 undefined-
denominator artifact；本复盘只使用 aggregate/micro SSE、active-site macro 与 record bootstrap。

centered spectrum、train–heldout alignment 与 heldout split-half 也远高于 matched diagonal
null；例如 primary 的 top-eigen fraction `0.2368` 对 null mean `0.03384`，cross alignment
`0.5735` 对 `0.2235`，split-half `0.6857` 对 `0.1642`。但这三类 null 只有 1024 次，采用
`+1` empirical p 后最小可能值是 `1/1025=0.0009756`，高于 Bonferroni 阈值
`0.0008333`。因此冻结判决只能写 `unexplained_under_this_screen`，不能事后把它提升为
`R5_latent_or_unmodeled_structure`。这是合同功效边界，不改变 resultant 已经拒绝
random-compatible 的结论；未来同类合同应至少使用 1199 次，实际宜统一为 4096 或 10000。

## 4. 主失败原因：target prevalence shift，而不是方向消失

对 immutable target bank 的只读分层显示，transfer energy 为零的记录与
`requested_margin=positive_common_margin_drop=0` 完全一致：

| split | 零 target | 非零 target | route-0 零 target | route-1 零 target |
|---|---:|---:|---:|---:|
| train | 86 / 4096（2.10%） | 4010 | 24 / 2048（1.17%） | 62 / 2048（3.03%） |
| heldout | 433 / 1024（42.29%） | 591 | 268 / 512（52.34%） | 165 / 512（32.23%） |

这使 train 学到的 nonzero centroid/predictor 在 433 条 heldout 零 target 上必然产生系统性
残差。只读、事后 diagnostic 进一步验证：在 591 条 target-positive heldout 上，global
gain 是 `+3.298%`，route absolute gain 是 `+6.619%`，route 相对 global 再 gain
`+3.434%`；全体记录合并后才变成 `-0.792%/-1.383%`。该分层使用 target-derived mask，
只用于解释结果，绝不能进入 X 或冒充可部署 router。

shift 不只发生在零/非零比例：positive records 的 requested-margin mean 也从 train
`0.5761` 升到 heldout `0.7657`。所以当前 prepared split 同时改变了 write prevalence 与
positive-request 幅度，不能把 R2/R3 heldout 失败单独归因于 projection 表达能力。

所以本轮真正暴露的是“何时写”的问题：方向在 positive records 上有用，但当前二元 route
不能可靠判断是否应写；projection trunk 虽含少量 input-conditioned gating 信息，却没有
跨 ordered split 稳定泛化。R4 的强 resultant 很可能主要记录了这种系统性过写和 split
shift，而不是新的随机方向鬼故事。由于 bank 没有 family/class/site-time OOD，且 1536 条
ID gap 未使用，这一归因仍是证据支持的推断，不是已隔离因果结论。

## 5. 对“先写后删”的影响

当前结果不支持立即启动“加强 projection 后再删 common”的训练。若 projection 先学到的是
train 中 97.9% 的 nonzero-write 先验，它会在 heldout 42.3% 的 no-write 记录上系统性误写，
这不是增大 projection update norm 能解决的。

下一份独立诊断合同应先把 transfer 分解为 `write gate × direction`：gate 只能读取 target
构造前的 input/attention/projection state，target-positive mask 只作为离线评价标签；随后在
prevalence-stratified random split、原 ordered split 与 1536 gap discovery split 上分别测试。
只有 gate 跨 split 成立，才值得重做 shared-parameter Jacobian，并要求 CG convergence、
finite-difference fidelity 和足够分辨率的 residual null。该建议不等于本 root 授权 successor；
本轮机器授权仍严格为 `nothing`。
