# V2-R1R P0-D / P0-M 结果

日期：2026-08-23
证据等级：P0-D v1–v16 的分层 qualification/失败证据、v17 accepted full production formal、P0-M v1–v4 失败诊断与 v5 accepted training-path smoke、P1 v1–v5 的数据/训练失败定位、P1 v6/v6R accepted causal-temporal mechanism、P1 v7/v8/v8R/v8D/v8L rejected answer/state-closure experiments、P1-NR1 accepted numeric/relation measurement qualification，以及 H1 factorized、H1-WD overlap-residual 与 H1-WD decision-causal 三个非正式失败 screen。v8L 关闭匿名 K=8 + lexical-anchor 主线；NR1 只关闭 measurement 前置缺口；decision-causal screen 进一步拒绝“raw per-record output VJP 可以直接由当前共享 projection 参数泛化写入”的命题。2026-08-23 的 direction-geometry 尝试只形成 replay identity crash 证据，没有 R0–R4 统计结果。P1 尚未完成 shared-vs-typed、K 对照或 matched baselines，不得升级为完整架构或 mixed core 的成败结论。

## 当前前向状态：v8L sealed FAIL，NR1 sealed PASS，三个 H1/H1-WD screens FAIL，direction-geometry identity CRASH

P0-D v17 另立 repair qualification 与 fresh production roots，修复 v16 暴露的 ERE visible-domain 后置不变量和 path-insensitive alpha-renamer。qualification R01–R07 全 true；fresh seed `2026081702` 生成 `26,624` records，G01–G11 全 true，regeneration、artifact replay 与 watched-input immutability 全部通过。正式 root seal SHA-256 为 `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`，P0-D 正式 accepted。

P0-M v5 在固定 P0-D v17 数据上完成 cache、ERE/CPS 单任务 K=8、joint shared K=8、direct/text-CoT、100-step CUDA throughput 与总 assessment。M01–M08 全 true，最终 `PASS_P0M`、`p1_eligible=true`、`p1_started=false`。joint ERE/CPS answer 均为 `1.0`，claim `0.90659`，owner-shuffle drop `0.37231`；direct/text-CoT greedy answer 均为 `1.0`；throughput 为 `71.706 examples/s`。assessment seal SHA-256 为 `17F45EAC8A3DF129B236688D3D1E2B22BD9B634D4098F903142CA42E7FEEF09E`。完整边界见 `docs/v2-r1r-p0m-v5-main-review.md`。

P1 v1 随后以 fresh generator seed `2026081901` 正式启动。preflight 通过；唯一 data root 生成 28,672 records，G01–G08/G10/G11 与全量 regeneration/replay 通过，但 G09 因 `ERE/validation/full_text_char_3_5_nb` 的 simultaneous upper `0.38190468 > 0.37099609` 失败，最终 `FAIL_P1_DATA`。`cache_authorized=false`，没有运行任何 P1 模型。完整复核见 `docs/v2-r1r-p1-v1-data-failure-review.md`。

P1 v2 以 4,096 heldout 重新资格化完整 G09 decision power，Q01–Q09 全 true；fresh seed `2026082002` 的 65,536-record data formal 为 G01–G11 全 true，模型 heldout 仍固定为 12×1,024。随后 cache 完成 132 个 source 与 26 个 claim shards，但外层四小时等待退出后，存活子进程在内容审计 progress callback 中遇到 `OSError [Errno 22]`，正式 cache root sealed FAIL。post-stop 同一 full audit 在关闭 progress 输出后对 219,816 entries 全通过，故障属于 execution telemetry；K=8 与后序均未运行。完整复核见 `docs/v2-r1r-p1-v2-cache-infrastructure-failure-review.md`。

P1 v3 随后以新 roots 资格化 console fail-open/ledger fail-closed 语义，并对 v2 immutable cache 完成两次全量只读审计；T01–T06、R01–R09 全 true，旧 cache 仍保持 formal FAIL，只新增训练复用授权。唯一 K=8 formal 完成 2,400 updates 后为 `FAIL_P1_K8`：ERE/CPS validation `0.24902/0.19043`，八个 OOD cells 全低于 `0.30`，causal flip `0.00391/0`，middle zero/shuffle/permutation drop 近零，claim accuracy `0.5`。K01–K06 false、K07/K09 true；K08 因 hook audit 写死 batch `2` 产生非决定性误报。K=1、baselines、assessment 与 P2 roots 均未创建。完整复核见 `docs/v2-r1r-p1-v3-k8-failure-review.md`。

P1 v4-LQ 的 preflight 与 B128 sealed PASS；S512 在 75 次平均 episode 暴露后训练 answer `1.0/1.0`、claim `0.9629`，但 validation `0.2344/0.1885`，最终 `FAIL_P1_LQ_SCALE512`，后序 roots 未创建。父任务只读诊断确认未见 train episode 的 claim 回落到 `0.498/0.506`，定位到任意跨 episode owner negative 奖励联合身份记忆。

P1 v5 preflight sealed PASS；唯一 Q7168 跑满 14,336 updates 后训练 answer 为 `1.0/1.0`，但隔离 audit answer `0.4746/0.1846`、claim `0.5/0.5`、state drop `0/-0.0034`，最终 sealed `FAIL_P1_V5_QUALIFICATION7168`。P1 v6 随后以同-query 相邻状态真值翻转对照 static signal：510,401 witnesses 全重放，18,899-query cache PASS；temporal audit ERE/CPS `0.7058/0.8053` 且两类因果 drop 通过，static 约随机。原 assessment 因 JSON key normalization 仅 W604 false，保持 sealed FAIL。v6R 以 pinned seals、canonical/hash replay、三类 mutation 负控和 full query content reaudit 得到 R601–R610 全 true、`PASS_P1_V6R_SELECTION_RECOVERY`；recovery assessment seal SHA-256 为 `B191B4058B014F912710D325C676D62C134851524832502CA4DACAB3092E5F98`。

P1 v7 的 preflight、86,016-query cache 均 sealed PASS；K=8 跑满 20,480 updates 后 validation ERE/CPS `0.999023/0.851562`，temporal accuracy `0.766357/0.938965`、zero/swap drop 与 batch-shuffle state dependence 通过。CPS 主 OOD 为 `0.587891/0.754883/0.472656/0.214844`。正式 K03 因 evaluator 要求错误 role 名而报告 `0/0`；父任务从 sealed predictions 修正为 ERE/CPS pair flip `0.705078/0.0078125`，仍不通过。v7 保持 `FAIL_P1_V7_INTEGRATED_K8`，三根不可重跑。

P1 v8 的 preflight/query-cache sealed PASS，qualification sealed `FAIL_P1_V8_CAUSAL_BRIDGE`。三臂 temporal 与 integrity 成立，但 ordinary validation 只有 `0.25/0.1758`，所有臂 B02/B03/B05/B06 失败。停止后的只读训练集复评显示 causal-paired 的 CPS pair flip 为 `0.5807`，显著高于 unpaired `0.1458`，而 audit 仍为 `0.0078`：pair objective 可优化，但 scratch 小覆盖只形成记忆。causal 臂没有 ordinary pretrain/rehearsal，故 v8 不能判定机制上限。

P1 v8R 随后从同一个 v7 competent checkpoint、完整 ordinary rehearsal 与相同 mixed schedule 比较 `replay_ce/replay_pair`。preflight sealed PASS，qualification sealed `FAIL_P1_V8R_CAUSAL_CURRICULUM`：paired optimization ERE/CPS pair 为 `0.9948/0.3281`，audit 为 `0.8047/0.0078`，validation 为 `0.9990/0.7480`；R01/R02 true，R03–R06 false。CPS audit 的正确切换只有 1/128，训练改善没有迁移。R06 source gradient false 被 FP32 重测定位为 BF16 饱和下溢，但不改变真实 causal failure。

P1 v8D 的 preflight/query-cache sealed PASS，qualification sealed `FAIL_P1_V8D_CAUSAL_DECISION_WITNESS`。joint audit decision ERE/CPS 为 `0.8887/0.5143`，CPS zero/swapped drop 只有 `0.0143/0.0286`；answer audit ERE raw/pair `0.8984/0.7969`，CPS `0.375/0`，ordinary validation `0.9990/0.8193`。只读诊断显示 CPS source perturbation 经 Boundary 保留并在 `H_T` 放大，但只形成 pair/surface-specific delta；正式根因是 `causal_perturbation_detected_but_semantic_reduction_not_formed`。

P1 v8L 的 preflight 与 anchor-cache sealed PASS，qualification 在 384-update bootstrap sealed `FAIL_P1_V8L_BOOTSTRAP`。ERE optimization/audit fixed-anchor direction 为 `0.7726/0.7109`，CPS 只有 `0.5682/0.5281`；CPS optimization 五层为 `0.6042/0.6042/0.5547/0.5521/0.5260`，未达到两族各 `0.65` 的 bootstrap Gate，故 3,072-update joint、答案迁移和 retention 均未运行。三个 seals 为 `06BA4E18...5115`、`5F1AABBF...3C14`、`3332CD3D...FE75`。

post-stop 只读复核发现 global anchor effective rank `25.61` 主要由 ERE 支撑；CPS cost-trace/final-cost 的 effective rank 只有 `3.47/3.84`，按数值大小重定向后的 shared-axis alignment 也只有 `0.340/0.380` 且存在反向方向。合同没有在训练前资格化单调性、加法一致性和 oracle-state decision power，因此失败不能唯一归因于 recurrent core。正式边界是关闭“匿名 shared K=8 + lexical metric teacher”的 v8 主线，不降低 Gate、不补跑 joint、不授权 v9/P2；完整复核见 `docs/v2-r1r-p1-v8l-causal-state-ladder-failure-review.md`。

P1-NR1 的唯一 formal 已完成，preflight/qualification 均 PASS，seals 为 `1825282B…95DAF`、`DADBDDBF…F6FB3`，N01–N07 全 true。两阶段 source start/after/snapshot identity 同为 `1D837A0F…B78657E`，20 项预测试、process/transport/Git/post-action 与 successor absence 均闭合。numeric/relation 两域注册 exact 与 metamorphic 指标全为 `1.0`，11 类 fault detection 全为 `1.0`；结果只设置 `p1_h1_design_authorized=true`，仍保持 `p1_completed=false`、`p2_eligible=false`。

H1 获准后的五个结构方向均只在非正式开发层评估。最后的 factorized routed-projection screen 使用预登记 `2026081761/2026081762`，两臂各完成 4,000 updates；fresh cache、架构、初始化、matched active FLOPs、shared no-op、strip/reload、source 与 recurrence 因果检查均通过，但 heldout mixed-shared overall gain 为 `-0.01074`，numeric 回退 `-0.03516`，最大 wrong-route/conditional-write effect 只有 `0.00281/0.00781`。H05 与 H06 同时失败，机器结果 `authorizes=nothing`。独立 calibration、H1 formal roots/transport、F1 与 P2 均未创建；该结果不改写 NR1 PASS，也不能提升为完整 mixed-core 反证。

H1-WD 随后以该失败 mixed checkpoint 为只读 predecessor，按固定 W800/D800/J1200 预算单次验证“先写后删”。W/D 成功把约 `25.07%` 的 common 输出能量对应分量搬入 projection 并保持 predecessor 函数，W/D heldout nMSE 为 `0.00196/0.01393`，D free-rollout agreement 为 `0.99121`；但最终 common-off/projection-off 都只掉 `0.01172`，projection effect 仅比 source 增加 `0.00391`。WD/control heldout 为 `0.54785/0.54297`，净增益 `+0.00488`、95% CI `[-0.00195,0.01172]`。机器终态 `FAIL_H1_WD_NONFORMAL_MECHANISM`、`authorizes=nothing`，说明输出空间重参数化成立而任务关键能力分工未形成。完整复盘见 `docs/v2-r1r-p1-h1-wd-failure-review.md`。

在该失败身份保持封存的前提下，另立的 decision-causal screen 只从原 immutable mixed deployment 构造新数据：以真实 common-off answer-margin drop 的一半确定请求量，以 selected projection output 的 scalar VJP/Fisher 确定方向，并以负 VJP 同范数臂检验方向性。W/D 的资格指标改为转移增量 nMSE，J 不使用 teacher logits、route supervision 或 family target。该 screen 是 `authorizes=nothing` 的机制研究，不恢复 factorized H1 的 calibration/formal，也不把负 VJP control 冒充 shared-only 架构基线。

decision-causal 唯一运行在 target Gate PASS 后，以 `FAIL_WD_DECISION_CAUSAL_WRITE_FIT` 正常停止。train/heldout target realized/request 为 `0.85606/0.93106`，CUDA peak `0.32335 GB`；W causal/control heldout transfer-nMSE 为 `1.00450/0.99247`，几乎等于完全不写入 `ΔP` 的基线。完整 target nMSE 约 `0.0047` 只是 transfer energy 仅占约 `0.468%` 的稀释结果。D/J、干预、Shapley 与方向收益未执行；该结果定位到 raw output-space VJP 缺少跨记录 projection-parameter reachability 资格，而不是 J 奖励不足。完整复盘见 `docs/v2-r1r-p1-h1-wd-decision-causal-failure-review.md`。

### P0-M v1–v5 闭环

v1 修正错误的 cache batch 选择标准；v2 在训练前修复 BF16 mask overflow 与 baseline 全序列 logits 浪费；v3 定位 claim probe 缺少 state–query interaction；v4 加入 interaction/ranking 后仍暴露 claim 监督提前关闭和 owner shortcut；v5 通过同族 owner contrast、贯穿训练窗口的 claim 梯度与 probe 剥离复载闭合机制。这个闭环证明当前实现可训练，但只有单一 fixed selection 的 overfit 证据，不代表架构泛化成立。

## v17 P0-D accepted 状态

repair qualification root `artifacts/v2-r1r/p0d-v17-repair-qualification-20260810-1/` 的 R01–R07 全 true，seal SHA-256 为 `6A31B0218F2B46979714A1D8F296BBE085E8FE9F0FF6E9B10476904C3885D103`。它包含 50,000-seed `if_copy` sweep、13,312-fingerprint capacity、v16 307 条失败记录回归、58 项 schema/reserved collision、旧 alpha transform fault kill 和 v16 runtime/G09/G10 保护 hash。

fresh production root `artifacts/v2-r1r/p0d-v17-full-production-20260810-1/` 使用 seed `2026081702`，含 26,624 records、1,536 causal pairs；G01–G11 全 true，总耗时 `1958.12s`，seal SHA-256 为 `453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D`。接受范围是 production data + verifier，不是模型或架构证据。

## v16 历史状态：runtime-Q accepted、fresh production rejected

v16 唯一 Q 的 Q01–Q09 全 true：192 条 × 15 轮 prediction identity，paired median `6.7282x`、15/15 fast wins、p95 `0.1119s`；完整 v14 G09/G10 投影一致，总耗时 `170.81s`、峰值工作集约 `3.46GB`，18/18 runtime fault 通过。Q root seal SHA-256 为 `EFBF833D5E55321012C98E6FF67C6B19E509687360E61588D4E34063B958513E`，有限判定为 runtime measurement component accepted。

Q PASS 后唯一 fresh F 使用 seed `2026081602` 生成 26,624 records、1,536 causal pairs、311,636 claims。G01–G04、G06、G08–G11 true，G05/G07 false，最终 `FAIL_P0D_PRODUCTION`。G05 全量复算为 307 条 `if_copy` ERE 在 pattern-specific control overwrite 后只保留五个而非声明六个 visible state values；G07 是资源名 `rules` 触发 path-insensitive alpha-renamer 的单条误报。regeneration 与 artifact replay 一致，F root seal SHA-256 为 `2CE3CBCDB720FA777C1E357026209108D3B57C444E751FD2689ED9C6B6AB74E7`。即使剔除 G07 误报，G05 仍使 P0-D rejected；不得修补或重跑，P0-M 未授权。

## v15 G09 qualification 历史状态：machine-fail、fresh formal 未运行

v15 唯一 Q formal root `artifacts/v2-r1r/p0d-v15-g09-qualification-20260810-1/` 的 Q01–Q07/Q09 true、Q08 false，最终 `FAIL_G09_QUALIFICATION`。合成 decision-power 本体通过：100,000 trials 下 null accept lower `0.999940`，ERE/CPS 单格 `+0.04` accept lower `0.957268/0.984865`，单格 `+0.10` 和六 heldout diffuse `+0.05` reject lower 均高于 `0.996`。14 项 fault、1,152 registry + 2,048 synthetic + 64 tie exact 对照、single-bank/batch/progress 与 replay 也全部通过。

唯一失败是 Q08 的一次性相对性能：192 个 char-NB prediction 的 fast `0.1261895s`、legacy exact `0.6008768s`，speedup `4.761702x < 5.0x`；prediction identity 仍为 100%、该 batch fallback 为 0。机器失败不可事后降 Gate。根 seal SHA-256 为 `997166EECC490F7B6AF6E80C01E28E187539BEDA351764597F0D79DE42C1EE38`；固定 v15 fresh production root 未创建，P0-D/P0-M/模型/训练仍未授权。完整判决见 `docs/v2-r1r-p0d-v15-g09-qualified-production-main-review.md`。

### v15 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| G09 decision power | 窄组件通过 | null/local/diffuse 七场景及 98/14 scalar parity 通过；不能接受 v14 或替代真实数据 |
| scorer correctness | 窄组件通过 | 3,264 prediction 0 mismatch；tie fallback、single-bank 与进度通过 |
| scorer operational Gate | 正式失败 | 单次 speedup 4.7617x，低于冻结 5x；Q08 false |
| fresh-seed production | 未运行 | `fresh_production_authorized=false`，固定 F root 不存在 |
| P0-M/model/cache/train/GPU | 未授权且未完成 | v15 仅为 Q qualification，且 conjunction 失败 |

## v14 历史状态：full production machine-fail、main-review rejected

v14 在固定 seed `2026081402` 上完成唯一 4096/512 formal。root `artifacts/v2-r1r/p0d-v14-full-production-20260809-1/` 包含 ERE/CPS 各 7,168、总 14,336 records、512 causal pairs 与 167,580 claims。机器 G01–G08、G10、G11 true，G09 false，最终 assessment 为 `FAIL_P0D_PRODUCTION`；完整 P0-D 没有关闭。

两个失败 cell 都是 full-text char 3–5 NB：ERE composition OOD accuracy `0.285156`、simultaneous upper `0.354605 > 0.350000`；CPS horizon OOD accuracy `0.208984`、upper `0.273711 > 0.266667`。两者 point excess 仅约 `+0.035/+0.042`，原始 point ceiling 与全部 14 个 family aggregate 均通过；因此这是局部弱信号叠加 512-sample、98-way upper-bound power 不匹配，不是已证实的大幅扩散捷径，也不能事后放行。

结构/因果/provenance/语言/claim 与复现部分均在完整规模通过：14,336 semantic/surface fingerprint 全局唯一，最大 993 tokens，full regeneration 与 artifact replay 一致。根 seal SHA-256 为 `6FD9AB9CD3F7CB1793F959762BA07AF0A0931052E7A3D05E3DD2C35455D19F52`。完整判决见 `docs/v2-r1r-p0d-v14-full-production-main-review.md`。

### v14 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| 完整 production dataset | 已生成、formal rejected | 14,336 records；G01–G08/G10/G11 true，但 conjunction 因 G09 false |
| renderer/simulator/teacher/claim/counterfactual | 完整规模通过 | 167,580 claims fresh replay；512 对单叶答案翻转 |
| fingerprint/overlap/permutation/provenance | 完整规模通过 | semantic/surface 14,336/14,336 唯一；root-to-record seed 可逆算 |
| source shortcut | 正式失败 | 两个 char-NB cell 的 simultaneous upper 越线；14 个 aggregate 全过 |
| claim shortcut | 完整规模通过 | 24 个 simultaneous claim cell 全过，最窄 upper 约 0.52 < 0.60 |
| P0-M/model/cache/train/GPU | 未授权且未完成 | v14 assessment 明确不创建权限，也不构成架构证据 |

## v13 历史状态：fixed-seed production generator smoke accepted

v13 唯一 formal root `artifacts/v2-r1r/r1g-v13-generator-smoke-20260809-1/` 生成 ERE/CPS 各 720、总 1,440 records，并得到 G01–G11 全 true。父任务复算 62/62 个 seal entry，根 seal SHA-256 为 `A9CF968F15634E7F1F9471CDE5383736C6D0E17DCC8208E871B6C779062AE455`；400 个新 seed 结构探针通过。接受范围只到 `R1G fixed-seed production generator smoke accepted`，不被 v14 失败改写，也不能升级为完整 P0-D。

## v12 历史状态：production entry accepted

v12 唯一 formal root `artifacts/v2-r1r/r1e-v12-entry-qualification-20260809-1/` 对完整 ERE/CPS 可逆 controlled-natural-language renderer、精确 model view、Qwen pinned token 长度和 compact exact-rational NB 排序完成资格化。G01–G08、15 metrics、29 adversary、20 metamorphic 和 replay 全部通过；57 个非 root-seal 文件封存，evidence-seal SHA-256 为 `86B49779D4E5A7EF496EE6FC0C0C722071983E5B3C6FFB9F8D90D92FBB9E78CA`。

父任务以 480 个 registry 外 compact-vs-v10 预测和三项深层嵌套 renderer roundtrip 接受有限 production entry，只授权另立 generator smoke；该授权已由 v13 使用并终止。v12 活动 tests/CLI 已由 v13 直接替换，历史 source snapshot 保留。

## v11 历史状态：R0D-integrated accepted

v11 直接切换 active CLI/tests，只新增 `integration/` 编排层并保持 v7 `common/`、v9 `audit/`、v10 `learner/` 逐字节只读。固定 12-case pack 包含三组 ERE open/closed 单叶 counterfactual 与 CPS 三候选六排列；同一 case identity 经 simulator、answer-label binding、typed invariant、toy parser、word/char grouped CV 与 train-heldout。规范化 source 固定不超过 400 characters，qualification surface 不冒充 production renderer。

唯一 formal root 为 `artifacts/v2-r1r/p0d-v11-r0d-integrated-20260802-1/`。机器结果为 G01–G08 `8/8`、F401–F420 `20/20`、metric kill `19/19`、positive metamorphic `4/4`、artifact replay `1/1`；十项顶层 entry 精确，bundle input seal 为 135/135，root seal 为 142 个直接文件加 2 个 nested seal 的传递覆盖，共 144/144。evidence-seal SHA-256 为 `1C4AD436CECA3F96F4286A2E5A62D272782E7EF02DF5A52BBD5C6493A43408CF`，`authorization_created=false`。

主设计层另用 unknown `Answer:` line、hidden model field 和 over-400 source 三项 registry 外临时副本探针，全部被目标边界拒绝；公开 API 正控仍为 8 Gate 全真。正式判定为 **有限 R0D integrated measurement-system accepted**。完整边界见 `docs/v2-r1r-p0-v11-r0d-integrated-main-review.md`。

### v11 R0D-integrated 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| semantic/invariant/statistical 三层组合 | 已完成 | 12 cases、8/8 Gate；不重新实现 accepted 算法 |
| fault/metamorphic/import/replay | 已完成 | 20/20、19/19、4/4、replay 1/1 |
| v11 唯一 formal 与主审 | 已完成并接受 | 144 文件直接/传递覆盖；三项 registry 外探针通过 |
| production renderer / learner scaling | 未完成 | qualification surface 不充分；char exact score 只验证到 source `<=400` |
| R1/P0-M/model/cache/train/GPU | 未授权且未完成 | 必须另立任务、数据、长度、Gate 与停止合同 |

## v10 历史状态：R0C-statistical accepted

v10 直接切换当前 CLI/tests，新增与 v7 simulator、v9 invariant audit 解耦的 `learner/`：精确四参数 source-only parser、条件多数、word/character multinomial NB、确定性 grouped five-fold、split-local CV 与 train-fit-heldout。资格输入固定为 parser 14、categorical 1、analyzer 3、NB 2、fold/CV/heldout 各 1、negative 10；34 项 raw metric 之外另设输入只读、novel group 原子性和固定语义摘要，阻止“改输入、重算 golden、重新 seal”自证。

唯一 formal root 为 `artifacts/v2-r1r/p0d-v10-r0c-statistical-20260802-1/`。机器结果为 G07/G08 `2/2`、adversary `39/39`、六 family holdout `6/6`、声明 metric kill `34/34`、positive metamorphic `5/5`；九项顶层 entry 精确，52 个递归 evidence hash 全部复算一致，evidence-seal SHA-256 为 `3F003C8BED40F54F010F4C7A96207D703ABA59F1088F189C1265D0F9E1C79BA1`，`authorization_created=false`。

主设计层以未登记的新输入检查 parser 隐藏字段、duplicate-token multiplicity、Unicode/OOV mask tie、character whitespace、group atomicity/row order 和 heldout isolation/overlap，六组均通过；公开 API 对 formal bundle 的只读复审也为 G07/G08/semantic/overall 全真、0 failures。因此正式判定为 **R0C-statistical accepted**。完整边界见 `docs/v2-r1r-p0-v10-r0c-statistical-main-review.md`。

### v10 R0C-statistical 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| source-only parser / categorical / NB / fold / CV / heldout | 已完成 | 只资格化有限 source/numeric profile，不是 production renderer/data |
| targeted registry / holdout / metamorphic / semantic pin | 已完成 | 34/34、6/6、5/5；自洽重算攻击被 overall 拒绝 |
| prior 与 accepted runtime 不变 | 已完成 | v7/v9 seal 全树复算；`common/`、`audit/` 等于 v9 source snapshot |
| v10 唯一 formal 与主审 | 已完成并接受 | 2/2、39/39、34/34、5/5；52 文件根 seal 与六组外部探针通过 |
| 后继阶段 | 未授权且未完成 | R0D/R1–R3/P0-M/model/cache/train/GPU 必须另立合同 |

## v9 历史状态：R0B-invariant accepted

v9 没有在 v8 formal 上补丁式重跑，而是直接替换 active audit/CLI/tests：qualification profile 固定为 11 records、18 claims、2 causal pairs 和 2 reversible language pairs；输入 certificate 中的 ERE/CPS witness path 被删除，query provenance、IF/relation control 与 CPS composition 改由 AST、fresh trace 和 candidate outcome 独立推导。自然语言 Gate 明确收缩为四种 full-match 可逆 grammar，不宣称理解自由释义。

唯一 formal root 为 `artifacts/v2-r1r/p0d-v9-r0b-invariant-20260802-1/`。机器结果为 G02–G06 `5/5`、adversary `48/48`、四类未映射 holdout 各 `1/1`、metric kill `45/45`、positive metamorphic `6/6`；九项顶层 entry 精确，45 个递归 evidence hash 全部复算一致，evidence-seal SHA-256 为 `94B80FBD2BB38BC46E95F88E0747D9DB1D8CB279E689F68BB4A43AE7B76D8209`。

主设计层又运行六项 registry 外公共 API 探针，包括重算 teacher/ablation/fingerprint 后的 ERE 初态答案捷径，以及保持 CPS 答案、成本、失败原因和 rich/NONE budget-only pairing 的 fact shortcut。前者在 G03/G04 仍真时被 G05 拒绝，后者在 G03/G04 仍真时被 G06 拒绝；其余 extra claim、role swap、自报 origin path 和 nonsense language 也分别被 G02/G03/G04 拒绝。没有发现 v8 式 false negative，因此正式判定为 **R0B-invariant accepted**。完整边界见 `docs/v2-r1r-p0-v9-r0b-invariant-main-review.md`。

### v9 R0B-invariant 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| fixed profile / exact schema / reversible grammar | 已完成 | 输入不能通过自改 manifest 自我扩容；自由自然语言不在本 Gate |
| typed ERE provenance / derived CPS composition | 已完成 | 不消费输入 witness path；fresh main-review shortcut probes 被拒绝 |
| v9 audit/CLI/tests 直接切换 | 已完成 | v8 active source/tests/alias 删除；v7 simulator 三文件逐字节不变 |
| v9 唯一 formal | 已完成、机器通过 | 5/5、48/48、45/45、6/6；九项 root 与 45 文件 seal 完整 |
| v9 主设计层验收 | 已通过 | 接受有限 R0B measurement-system qualification，不升级为数据/模型/架构证据 |
| 后继阶段 | 未授权且未完成 | R0D/R1–R3/P0-M/模型/训练必须另立合同 |

## v8 历史状态：R0B-structural machine-pass、main-review rejected

v7 已取得主设计层最终接受：唯一 formal 的 canonical/prefix/corruption/legacy 为 14/4/12/245，operand lattice invalid/acceptance/coverage 为 508/508、350/350、858/858；父任务复算七文件 artifact、19 个 frozen input 与 evidence seal，并用 27 项 registry 外公共 API 探针验证 query token、嵌套 `$neighbor`、event/state/CPS literal 边界，全部通过。R0A 因此关闭，accepted simulator 在 v8 保持逐字节只读。

v8 已直接切换到下一测量层，没有恢复 v4 的大型自洽 reference generator。唯一 formal 在固定 root `artifacts/v2-r1r/p0d-v8-r0b-structural-20260801-1/` 得到机器 `PASS_R0B_STRUCTURAL`：正控 G02–G06 为 5/5，41/41 fault 的 false Gate set 精确，metamorphic 4/4，metric kill 37/37。八个顶层 entry、43 个递归 evidence hash、24 个 frozen input、attempt-before-evaluation 和 `authorization_created=false` 均经父任务独立复算成立。

这些机器事实只证明当前正控与登记 fault 矩阵自洽。父任务在 sealed control bundle 的临时副本上另行运行五个 registry 外反例：把 18 条 claim 增为 20 条、互换一对 claim 的 positive/negative role、把 initial-copy origin 指到无关 `/query`、把 relation neighbor 改成不存在名称、把 language pair 正文换成与 AST 无关的香蕉/石头语句。每个副本都独立重算 manifest/hash/seal 后走同一个公共 API，结果仍为 G02–G06 全真。

这些反例直接违反 v8 冻结合同的精确 claim 基数/极性、provenance AST/trace witness 与 language AST identity，不是 R0C/R0D 才处理的统计问题。因此正式判定为 **R0B-structural machine-pass、main-review rejected**。根因是 metric kill coverage 没有覆盖遗漏的不变量，certificate verifier 又把“存在路径/字符串类型”误当语义证据，language audit 只复核可自报的表面调度。完整判决见 `docs/v2-r1r-p0-v8-r0b-structural-main-review.md`。

### v8 R0B-structural 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| v7 R0A-lattice | 已完成并经主审接受 | machine 14/4/12/245、508/350/858 全通过；父任务 27/27 新鲜探针通过 |
| v8 control/fault/metric/metamorphic | 已冻结并执行 | 11 records、18 claims、2 causal、2 language、41 faults、4 metamorphic、37 metrics |
| v8 audit/CLI/tests 直接切换 | 已完成 | accepted simulator 未改；v7 tests/alias 已删除；无 generator/model/train |
| v8 唯一 formal | 已完成、机器通过 | 正控 5/5、fault 41/41、metamorphic 4/4、metric kill 37/37；八项顶层 artifact 已封存 |
| v8 主设计层验收 | 未通过 | 五项 registry 外合同反例全部被错误接受；machine result 不升级为 R0B 资格 |
| 后继阶段 | 未授权且未完成 | v8 不修补、不重跑；下一步只能另立 R0B 合同，R0C/R0D/R1/P0-M/训练继续停止 |

## v7 历史状态：R0A-lattice accepted

v7 的 fixed root `artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1/` 精确七文件。父任务复算的 evidence-seal SHA-256 为 `794CB9518F094EE6CF99800AF0B5848D22A7746DAA5E76A84489F3F8201987F0`，coverage required/observed set hash 均为 `DBB047B92CDD4523F7881CFA78440BA06D7A29630F57BD0DF391ADC2717C33B5`；guard、pytest `32 passed`、compileall/help/diff-check 全部通过。新鲜探针没有复用 builder/fixture，并覆盖 20 个 query `$` 拒绝组合、eager query-before-prefix、普通 literal 正控、嵌套 neighbor 遮蔽、动态 relation、event/state/CPS `$` literal。

因此正式判定是 **R0A-lattice accepted**。接受范围只限当前有限 AST schema 的 semantic-oracle/API boundary qualification；它不构成 G02–G08、生产数据、模型、训练或架构证据。完整判决见 `docs/v2-r1r-p0-v7-r0a-lattice-main-review.md`。

## v6 历史状态：R0A-strict machine-pass、main-review rejected

v6 已按冻结合同直接切换四个实现文件、删除 `tests/v2_r1r_v5/`，并执行唯一一次固定 `seal-strict`。`artifacts/v2-r1r/p0d-v6-r0a-strict-20260801-1/` 精确包含六个 sealed JSON；machine assessment 为 canonical `14/14`、prefix `4/4`、corruption `12/12`、strict matrix `245/245`，12 项 conjunction 全部为 `true`。父任务独立复算 frozen/source/artifact/evidence hash，重跑 guard、pytest `18 passed`、compileall/help/diff-check，确认 formal 纪律、机器数值和封存完整性成立。

但主设计层的新鲜黑盒探针发现冻结矩阵没有完整实现书面 placeholder 作用域：attribute query 的 `entity/attribute` 与 relation query 的 `relation/source/target` 分别注入 `$neighbor` 和 `$arg:*`，十个组合全部被公共 API 静默接受并返回 `None/False`。书面合同规定 `$arg:name` 必须绑定当前 rule 参数、`$neighbor` 只允许位于 `FOREACH_LINKED.effect` 子树；query 不满足任一条件。冻结矩阵只有四个 rule-primitive placeholder 探针，没有 query placeholder 探针，粗粒度 `covers` 标签不能证明位置笛卡尔积覆盖。

因此正式判定是 **R0A-strict machine-pass、main-review rejected**。v6 的 14/4/12/245 与 sealed protocol 只保留为窄机器证据，不能升级为阶段通过。v6 实现、冻结输入和 artifact 不修补、不覆盖、不重跑；其后已另立并冻结 v7 R0A-lattice，但这不改变 v6 的历史判决。R0B/R0C/R0D、R1、P0-M、模型、cache、训练/GPU 继续禁止。完整判决见 `docs/v2-r1r-p0-v6-r0a-strict-main-review.md`。

### v6 R0A-strict 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| v6 独立语义 fixture 与 strict matrix | 已冻结并执行 | machine canonical/prefix/corruption/strict 为 `14/14`、`4/4`、`12/12`、`245/245` |
| 四文件直接切换与 v5 活跃测试删除 | 已完成 | 最小 public surface；无 compatibility、audit/generator/model/train 路径 |
| 唯一 R0A-strict formal attempt | 已完成、机器通过 | 固定 root 六文件、12 项 conjunction 与 evidence seal 全通过；不可重跑 |
| 书面 placeholder 作用域 | 主审失败 | query 五个 operand × 两类 reserved token 共十项全部静默接受 |
| R0A-strict 主设计层验收 | 未通过 | 接受窄机器事实，拒绝矩阵完备性与阶段通过 |
| R0B/R0C/R0D、R1、P0-M、模型与训练 | 未授权且未完成 | 失败后停止；后继 v7 仅获 R0A-lattice 单次执行权 |

## v5 历史状态：R0A machine-pass、main-review rejected

主设计层已拒绝 v4 R0 实现作为有效审计器：除了首条 F401 编排错误，v4 reference expected values 实际由第二套语义引擎计算，三个预测试没有执行真实 fault 路径，G07 读取 source 中显式 structured fields，G08 没有拟合 Naive Bayes 或运行 grouped CV。完整判决见 `docs/v2-r1r-p0-v4-main-review.md`。v4 partial artifact 继续保留，但其 G02–G08 正向值没有正式 Gate 地位。

v5 把测量系统拆为 R0A semantic oracle、R0B structural audit、R0C shortcut audit 和 R0D integrated reference。R0A 已直接删除 v4 implementation，收缩到三个公共 simulator API 与唯一 `verify-oracle` CLI，并生成 `artifacts/v2-r1r/p0d-v5-r0a-20260801-1/`。机器 assessment 为 14/14 canonical、4/4 prefix、12/12 negative，coverage、determinism、import boundary、frozen validation 与 evidence-seal hash 均通过；主设计层独立重跑 `contract_guard`、`pytest`、compileall/help/diff-check 并复算 artifact/source hash，确认这些机器事实成立。

但同一冻结合同还要求未知 primitive/condition/effect、参数缺失和格式错误必须明确抛异常。主审六项黑盒 malformed-AST 探针全部未抛异常：已执行 ERE/CPS attribute condition 缺 `value` 被当成普通 false，未执行 ERE rule 中的未知 primitive/缺值 SET 和未使用 CPS action 中的缺参 condition/effect 都被静默接受。根因是实现只按执行路径延迟校验，冻结 10 项 pytest 与 12 个 corruption negative controls 又没有测量这一要求。执行层还在正确 formal 调用前发生一次错误 execution-hash CLI 预调用；runner 在建 artifact 前阻断，未污染结果，但字面上的单次调用纪律不完整。

因此项目不接受机器 `PASS_R0A` 为阶段通过，正式状态是 **R0A machine-pass、main-review rejected**。当前 artifact 只证明冻结样例语义和 validator corruption sensitivity，不构成 G02–G08、P0-D、数据、模型或架构通过。v5 不修补、不覆盖、不换 suffix 重跑；R0B/R0C/R0D、R1 generator、P0-M、模型、cache、训练/GPU 全部未授权。完整判决见 `docs/v2-r1r-p0-v5-r0a-main-review.md`。

### v5 R0A 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| v4 source/tests 直接删除与 v5 最小 surface | 已完成 | 当前仅 `common/__init__.py + simulator.py` 和唯一 `verify-oracle` CLI；无 compatibility/audit/generator |
| 冻结样例与 sealed conjunction | 机器通过、主审接受其窄证据 | 14/14 canonical、4/4 prefix、12/12 negative；五文件与三项 evidence hash 完整 |
| 公共 API fail-closed 语法合同 | 主审失败 | 六项未知/缺参黑盒探针均静默返回；invalid AST matrix 未冻结、未进入 assessment |
| 单次 formal 调用纪律 | 有偏差 | 错误 hash 调用在 output 创建前被拒绝，随后正确调用产生 artifact；内容未污染，但不是字面无偏差单调用 |
| R0A 主设计层验收 | 未通过 | machine result 不升级为 stage pass；保留为 diagnostic |
| R0B/R0C/R0D、R1、P0-M、模型与训练 | 未授权且未完成 | 失败后停止；下一步只能先另立并冻结新的 R0A-strict 合同 |

## v4 历史状态：R0 `FAIL_R0`

主设计层已把 `docs/v2-r1-revalidation-task-design.md` 第 17 节直接切换为 P0-D v4 reference-first 合同，并新增唯一允许的 `docs/v2-r1r-p0d-v4-r0-execution-command.md`。v4 把旧 15 Gate 收敛为八个职责明确的 Gate；R0 不生成训练候选数据，只用手写 expected values 的 reference pack 验证 known-good 8/8、F401–F420 精确失败集合、required metric 全覆盖、四个 metamorphic test、import boundary 和 artifact-internal snapshot replay。

预测试全部通过（pytest `3 passed`、compileall、CLI help、`git diff --check`）；唯一 sealed root `artifacts/v2-r1r/p0d-v4-r0-20260801-1/` 的 build-reference 与 audit-reference 均返回 0，G02–G08 known-good audit 通过。第三条正式命令在 `F401/matrix_row_missing` 处因 fault harness `mutate()` 缺少该故障映射而返回非零，故 R0 立即判定为 **`FAIL_R0`**。partial root 保留 manifest、reference data、language pairs、input-seal、source snapshot 和 audit-core；没有 fault-matrix、assessment、evidence-seal 或 snapshot replay。R1 generator、R2 preflight、R3 formal、P0-M、模型、cache 与训练全部未授权，不能把 known-good audit 通过写成 P0-D 通过。

### v4 R0 完成与未完成

| 项目 | 状态 | 证据/边界 |
| --- | --- | --- |
| v3 直接删除并切换 v4 common/audit/CLI/tests | 已完成 | 当前 source tree 与 sealed source snapshot 均为 v4 layout；无 production generator、ERE/CPS renderer、model、cache 或 train |
| 预测试 | 机器运行完成、主审拒绝 | `pytest 3 passed` 只检查 shape/命令名/case 数，没有执行 semantic world、公开 fault、metamorphic 或 replay |
| 设计/命令 hash 与 sealed build | 已完成 | 目标 hash 正确；build-reference exit `0` |
| known-good G02–G08 audit | 机器返回全真、主审拒绝 | reference 不独立，G07/G08 未实现合同算法；只能作为自洽实现诊断 |
| F401–F420 fault matrix | 未完成 | 在首个 `F401/matrix_row_missing` 进入 mutate 时 `ValueError: unknown fault`；未运行后续 case |
| metamorphic、import boundary、assess-r0、snapshot replay | 未完成 | 因 fault 命令失败按合同停止；没有推断这些检查通过 |
| R1/R2/R3、P0-M、模型、cache、训练/GPU | 未授权且未完成 | R0 `FAIL_R0` 后停止 |

精确失败证据为 `tests/v2_r1r_v4/fault_harness.py:241` 的 `ValueError(f"unknown fault {group}/{case}")`，调用 case 为 `F401/matrix_row_missing`。sealed root 不修复、不覆盖、不重跑；本次没有执行第四条 `assess-r0`。

## v3 历史执行结论

2026-08-01 的 v3 最终历史状态为 **`FAIL_D2`**。本轮曾把 `r1_revalidation` package 直接切换为 v3，完成 D0 合同/审计器自验证和 D1 功能 smoke；唯一一次 D2 使用完整规模做 formal-threshold preflight，结果为 `G01–G04、G09、G10、G12、G14、G15=true`，`G05、G06、G07、G08、G11、G13=false`。D2 失败后按合同停止，没有修改实现、没有重生数据、没有运行 D3 formal，也没有创建 formal authorization 文件。

| 阶段 | 实际命令/结果 | artifact 与证据边界 |
| --- | --- | --- |
| D0 | `self-test-contract` exit `0`；目标 contract/audit/data 测试 exit `0`；`compileall` exit `0`；`git diff --check` exit `0` | machine 通过，但主设计层已否决：缺 known-good 全 conjunction，G02/G04 无 fault，未捕获 G07/G08 恒假聚合 |
| D1 | 最后一次生成 exit `0`，audit exit `0`，`smoke_passed=true` | `artifacts/v2-r1r/p0-smoke-v3-20260801-10/`；D1 distribution 只作诊断，不作 formal conjunction |
| D2 | 唯一 preflight 生成 exit `0`，audit exit `0`，`formal_conjunction_passed=false` | `artifacts/v2-r1r/p0-preflight-v3-20260801-1/`；这是 preflight 失败证据，不是 formal authorization |

D2 失败诊断的直接证据如下：G05 的 CPS 正确候选位置和 causal label balance 未达到阈值，ERE train choice deviation 也超过 train 限值；G06 在多个 CPS action/candidate 条件分组出现 conditional-majority 超阈值；G07/G08 的 gate 聚合读取不存在的 family-level `passed`，因此对任何数据都会为 false，不能把 G07 机器红灯解释成 ERE 主任务结构失败；G11 的 CPS `goal_only` 在 causal/distractor/horizon 超过上限，language OOD source parser coverage 低于 `0.98`；G13 的 CPS composition OOD `prefix_legality` 仅占 `0.064516...`。主设计层另发现 G06/G08/G13/G15 低于冻结合同、CPS per-record hard-negative 与较长 valid-suboptimal 大面积缺失。上述内容只作失败归因，未在 D2 后修复。

### v3 D2 记录的 15-Gate 结果

| Gate | 结果 |
| --- | --- |
| G01 contract self-validation | `true` |
| G02 simulator/teacher/replay | `true` |
| G03 schema/source/token provenance | `true` |
| G04 overlap/declared pairs | `true` |
| G05 answer choice/surface position balance | `false` |
| G06 conditional nuisance independence | `false` |
| G07 ERE structure/necessity/provenance | `false` |
| G08 CPS structure/hard-negative complexity | `false` |
| G09 causal single-leaf diff/flip | `true` |
| G10 paired language syntax distribution | `true` |
| G11 structured surface heuristics | `false` |
| G12 statistical train-heldout heuristics | `true` |
| G13 claim truth/coverage/shortcuts | `false` |
| G14 model-view forbidden fields | `true` |
| G15 scale/source snapshot | `true` |

preflight 规模为 ERE/CPS 各 `train=1024`、`validation=256`、四个 OOD split 各 `256`、`causal_pairs=256` 条记录；manifest 保存设计 hash `E36D77BD4C5555FF4FEDE4FCC4B1EAFB9B0C11C44185064FB3F1DD42ADB890F4`、执行命令 hash `63411667CD0C08FDB090430CA5829855EC77C85611CEDB7FF870762B3893C583`、dirty diff hash、source snapshot 和 Qwen tokenizer provenance。主设计层复核确认 snapshot 文件本身完整，但 verifier 把运行后必须更新的状态文档也绑定为当前 hash，且未完整检查冻结合同要求的依赖/named-stream provenance。所有 v3 smoke/preflight 失败与中间目录均保留，未覆盖历史 v1/v2 artifact。

本轮明确没有 formal、P0-M、model/cache/train、Boundary/core、baseline、GPU 或训练 artifact；因此 `FAIL_D2` 不能外推为架构结论。

## v3 主设计层验收

主设计层独立重跑 28 项目标测试、`compileall`、`git diff --check` 和 5,120 条记录的只读 audit。测试均通过，但只证明当前负向 fixture 与实现自洽；fresh audit 除复现六个业务 Gate 外，还因结果文档按合同同步而新增 G03/G15 current-source mismatch，证明 provenance 流程会自失效。验收开始时当前 Python 代码与 D2 source snapshot hash 一致，漂移仅发生在 README、目录索引、阶段计划和本结果文档；主判决落盘后，v3 合同被 v4 覆盖，当前 v3 执行命令被删除，精确历史副本仍在 D2 source snapshot。

更严格的逐行审计表明，CPS non-NONE 记录满足 per-record failure-type 要求的比例仅为 train `0.2075`、validation `0.1878`，distractor 的四类要求为 `0`；所有非 causal split 的较长 valid-suboptimal 数为 `0`；同时具备四类 claim 的比例为 train `0.6602`、validation `0.6758`、composition `0.1953`。相对地，ERE 的 train/validation/entity/language/length structure report 均通过，G07 为 false 由聚合 bug 造成。完整验收见 `docs/v2-r1r-p0-v3-main-review.md`。

最终判决是：接受 `FAIL_D2` artifact 和停机纪律，拒绝 v3 作为合格合同实现。旧第 17 节与 v3 执行命令只保留在 D2 source snapshot；当前文档已由 v4 覆盖。下一步不是修 v3 或重跑 D2；v4 R0 已执行并因 fault harness 首案失败而停止，不能据此进入 generator 或 preflight。

## 结论

generator v1–v4 的生成/审计路线均已被拒绝；v5/v6 的 R0A machine PASS 也被主审反例推翻。后续 v7 关闭有限 simulator/API R0A，v9 关闭有限 G02–G06 R0B，v10 关闭有限 G07/G08 statistical component，v11 又关闭规范化 source `<=400` 的有限 integrated measurement system。当前不能再笼统写成“测量链全部失败”，但也不能写成 P0-D production data 或完整系统通过：语义充分 renderer、可扩展 learner、R1 generator、P0-M 和模型训练仍不存在。完整逐轮复核见各版本 main-review 文档。

formal v2 任一 Gate 失败后按合同停止；未创建 P0-M/model/cache/train 代码，未启动 GPU 训练，也未启动 OPS。machine result 不能升级为架构结论，P0-M 仍等待主设计层独立批准。

## v2 formal machine result

正式生成使用 seed `20260801`、generator `r1r-p0-generator-v2` 和设计 hash `df4f72d74de4b50c05553932d4cf4314a7525b79e16d982c89eaaad5dee2c004`。独立进程 audit 读取完整 JSONL 后得到 13 项 conjunction `10/13`，assessment `passed=false`；失败证据保留在 `artifacts/v2-r1r/p0-v2-failed-audit-20260801-1/`。

| v2 Gate | formal 结果 | 关键证据 |
| --- | --- | --- |
| simulator/teacher/replay | `true` | fresh replay 与 teacher 校验通过 |
| semantic/surface overlap | `true` | 未声明跨 split overlap 为 0 |
| candidate/action/answer/NONE position | `false` | CPS horizon grouped action-definition deviation `1.0` |
| ERE ablation/CPS necessity | `true` | 审计通过 |
| causal single-diff/certificate | `true` | 单点 diff、端点与答案翻转通过 |
| split/composition contract | `true` | ERE 三类、CPS 四类保留组合覆盖且无 train/validation overlap |
| source completeness | `true` | source 定义、状态、候选和 choices 完整 |
| language syntax/distribution | `true` | 三套 feature 可检测，distribution matching 通过 |
| surface-only/statistical heuristic | `false` | CPS causal/horizon `longest_plan` 为 `0.310546875/0.2734375`，超过允许上界 |
| claim truth/kind/balance | `false` | CPS distractor/horizon 缺少 required claim kind；其余 claims truth accuracy `1.0` |
| exact Qwen token/schema/hash/provenance | `true` | 实际最大 `qwen_token_count=1011`，无 >1024 |
| model view forbidden fields | `true` | forbidden-field audit 通过 |
| formal scale | `true` | 总计 `14,336` records，split 规模完整 |

三项 false 已触发停止规则；没有为了使 assessment 变绿而降低阈值、减少 split、修改 model view 或重跑 formal。

## v1 历史 provenance

- 唯一设计规范：`docs/v2-r1-revalidation-task-design.md`
- 设计规范 SHA-256：`8e90c4ac9d3413cb959904f43a8a5618c940b8c66a24a804fa5d5609a378c796`
- generator version：`r1r-p0-generator-v1`
- formal seed：`20260801`
- formal manifest：`artifacts/v2-r1r/p0-v1/manifest.json`
- formal assessment：`artifacts/v2-r1r/p0-v1/p0-assessment.json`
- formal audit SHA-256（assessment 中）：`57d0759b3938ea036f7c4b0a768d9daf6e82fb059773fa50118c434dea73403c`

v2 provenance：

- generator version：`r1r-p0-generator-v2`
- formal manifest：`artifacts/v2-r1r/p0-v2-failed-audit-20260801-1/manifest.json`
- formal audit：`artifacts/v2-r1r/p0-v2-failed-audit-20260801-1/audit.json`
- formal assessment：`artifacts/v2-r1r/p0-v2-failed-audit-20260801-1/p0-assessment.json`
- tokenizer：`Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，`use_fast=true`，`add_special_tokens=false` 在 tokenizer call 固定；manifest 含 class/version/config。

每条 JSONL 记录保留 generator version、episode seed、输入/设计文档 hash、环境、命令、开始/结束时间；manifest 另外记录所有数据文件 SHA-256。正式生成和审计均使用项目 `.venv\Scripts\python.exe`。

v2 records and manifest use only the current design hash shown in the v2 result section; the v1 hash above is historical evidence and is not an input to v2.

## 数据规模

| family | train | validation | OOD splits | causal_pairs |
| --- | ---: | ---: | --- | ---: |
| ERE | 4096 | 512 | composition/length/entity/language 各 512 | 512 条 = 256 对 |
| CPS | 4096 | 512 | composition/horizon/distractor/language 各 512 | 512 条 = 256 对 |

smoke 也按合同完成：每族 train 64、validation 与各 OOD 32、causal_pairs 32 条；smoke 只作 generator/audit 连通性证据，formal Gate 以完整规模结果为准。

## v1 rejected diagnostic 的十项旧 Gate

formal `audit.json` 的十项 conjunction 全部为 `true`：

| # | Gate | formal 值 |
| ---: | --- | --- |
| 1 | simulator 重放、answer verifier、teacher trace verifier 100% | `true` |
| 2 | 未声明跨 split semantic/surface overlap 为 0 | `true` |
| 3 | train 与 512 split 的 label/choice balance | `true` |
| 4 | 必要事件、条件、资源或 action 至少 95%，且因果 certificate 有效 | `true` |
| 5 | causal pair 单点修改与答案翻转 100% | `true` |
| 6 | ERE dependency depth、CPS hard-negative/valid-suboptimal coverage | `true` |
| 7 | majority、position、length、last mention、question/full-text unigram NB | `true` |
| 8 | claim positive/negative balance 与 claim-only heuristic 上限 | `true` |
| 9 | token limit、schema、hash、seed、version 与 provenance integrity | `true` |
| 10 | `model_view` forbidden-field audit | `true` |

旧机器总判定：`artifacts/v2-r1r/p0-v1/p0-assessment.json::passed = true`。主设计层判定：`rejected`。这里保留旧布尔值是为了准确记录审计器当时的行为，不代表当前路线 Gate 通过；v2 结果见上节。

## smoke 与 formal 命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_v2_r1r_audit.py tests\test_v2_r1r_data.py
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py generate-p0 --seed 20260801 --output artifacts\v2-r1r\p0-smoke-v2 --smoke
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py audit-p0 --input artifacts\v2-r1r\p0-smoke-v2
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py generate-p0 --seed 20260801 --output artifacts\v2-r1r\p0-v2
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py audit-p0 --input artifacts\v2-r1r\p0-v2
.\.venv\Scripts\python.exe -m compileall -q src\yggdrasil_v2\r1_revalidation experiments\v2_r1_revalidation.py
git diff --check
```

结果：v2 目标 pytest 16 passed；smoke audit passed=true；formal audit passed=false（13 项中 10 项为真）；compileall 与 git diff --check 均通过。formal 生成、审计、失败证据保留和停止均完成。

## v2 历史实现与修复记录

本轮实现是独立的 `src/yggdrasil_v2/r1_revalidation/` package，包含 v2 schema/view、独立 RNG streams、typed simulator、ERE/CPS causal generator、自然语言 renderer 和 13-Gate audit；CLI 只有 `generate-p0` 与 `audit-p0`。没有触碰旧 `reasoning_medium` 代码。

执行中完成了已知捷径故障注入测试，并修复了 ERE `SWAP` 路径、episode seed/RNG 分离、label schedule、source renderer、fresh replay、真实 Qwen token count、causal certificate 和跨 split overlap 等 P0-D 范围问题。随后正式 audit 暴露出尚未完成的 CPS horizon position/claim-kind/heuristic 缺口；按停止规则没有继续修复或重跑 formal。

未决合规担忧：当前 necessity audit 对 `causal_pairs` 的 `pair_role=flip` 行跳过了逐事件 ablation，以避免部分 relation mutation 在翻转世界中变成观测冗余；这不应被解释为“每个 flip spine event 都已通过 fresh ablation”。该边界与上述三项正式 Gate 失败一起保留为 P0-D 未完成项。

失败运行没有覆盖或删除，已保留为可追溯诊断目录：v1 的 `p0-v1-failed-heuristics/`、`p0-v1-failed-label-balance/`、`p0-v1-failed-answer-balance/`；v2 的 `p0-v2-failed-generation-20260801-1/`、`p0-v2-failed-generation-20260801-2/` 和 `p0-v2-failed-audit-20260801-1/`。

## 当前未完成项与路线判决

已经完成 P0-M 的 frozen-Qwen hidden cache、tokenwise Boundary、K=8 shared latent core、direct/text-CoT baseline、overfit 与 100-step throughput；这些只形成 v5 的训练通路 smoke。P1 v3 已关闭 cache infrastructure，并完成唯一 K=8 fresh-seed validation/OOD/causal intervention；该模型接近选择先验、对 recurrent middle 无因果依赖，正式被拒绝。尚未完成 K=1 对照、matched direct/text-CoT、P2 Pareto、P3 多 seed和完整成本账本，不能把 K=8 失败升级为完整架构反证。

当前路线判决是：v1–v16 保持各自历史 accepted/rejected 身份；v17 正式关闭 P0-D；P0-M v5 正式关闭 training-path smoke；P1 v3–v5 分别拒绝低曝光、owner-contrast 与 static-pair 路径，P1 v6/v6R 接受 causal-temporal mechanism，P1 v7 接受其可扩展状态机制但拒绝 integrated causal/OOD answer，P1 v8/v8R 依次排除 scratch/coverage，v8D 再排除 source blindness并拒绝 final-state probe 路径；v8L 最终拒绝匿名 K=8 + lexical-anchor 的 CPS state-closure 路径。状态仍为 `p1_started=true`、`p1_model_started=true`、`p1_passed=false`、`fresh_p1_v9_authorized=false`、`p2_started=false`。

当前 H1 factorized routed-projection、H1-WD overlap-residual 与 decision-causal 方向均已按各自 screen FAIL 停止：NR1 roots/transport、factorized screen root 与两个 WD roots 均已消耗且禁止重跑，H1 calibration/formal 未授权，F1 也未获设计授权。若未来继续，只能先另立新的 H1 架构开发合同；不能重跑任一 screen、降低 Gate，或把 raw output VJP/非正式输出空间搬运包装为 parameter-reachable 能力转移。只有新的 H1 PASS 才可能授权 F1，F1 才补 K=8/K=1/direct/text-CoT 完整 P1。不得修补或重跑历史 formal，也不得在 F1 完整 conjunction 前进入 P2、A1.22A 或 V2-B。历史失败 artifact 只作诊断，accepted artifact 只在其声明层级有效，本轮仍没有架构泛化或 Pareto 通过结论。

后续唯一启动的 direction-geometry 诊断没有改变上述机制判决。它在全量重放 train 4096 与 heldout 1024 后被 replay identity Gate 拒绝：common/projection 最大绝对误差 `7.96914e-05/9.50396e-05`，高于冻结容差 `2.5e-05`。旧 target bank 的 microbatch `4` 与新 capture batch `128` 导致 CUDA batch geometry 的有限精度差异，且启动前首批 spot check 没有覆盖尾部最坏记录。机器终态为 `CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY`；没有 `result.json`，R2 input-predictable、R3 parameter-reachable 与 R4 null 均未运行，R0/R1 也未形成可引用 artifact。因此本运行既不验证也不反驳 parameter-Jacobian/Fisher 猜想，只暴露执行身份合同缺陷；root 已消耗、`rerun_authorized=false`、`authorizes=nothing`。完整复盘见 `docs/v2-r1r-h1-wd-direction-geometry-screen-failure-review.md`。
