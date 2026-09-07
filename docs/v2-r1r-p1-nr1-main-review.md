# V2-R1R P1-NR1 正式结果与主审

日期：2026-08-17  
正式判决：`PASS_P1_NR1_MEASUREMENT_QUALIFICATION`  
授权边界：只授权 `P1-H1` 设计与实现；`p1_completed=false`、`p2_eligible=false`

## 1. 核心判决

P1-NR1 的唯一正式命令已执行一次并正常退出。preflight 与 qualification 均 PASS，N01–N07 全 true，两根 evidence seal 在进程内即时验证并由父任务在进程退出后独立复算有效。该结果关闭了 V8L 后的 numeric/relation measurement 前置缺口：当前 typed measurement、独立 reference、手工预期、复杂关系拓扑与 fault registry 足以作为 H1 的评测面。

这不是模型学习结果。NR1 没有加载 Qwen、没有训练 shared 或 mixed core，也没有运行 K=1、direct/text-CoT、fresh-seed full P1 或 Pareto。因此不能把本次 PASS 写成 mixed core、P1、V2-A 或 P2 通过。

## 2. 唯一启动与证据完整性

正式入口为 `.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-nr1`。fixed transport 的 `launch.json` 记录 attempt `p1-nr1-20260817-1`、PID/PPID、cwd、argv 与 stdout/stderr 绑定；`completion.json` 记录 exit code `0`，正式 stderr 为空。进程审计识别到 pwsh → venv python → Python 3.11 的单一祖先链，没有外部同名 formal 或 H1/F1/v9/P2 进程。

正式根与 seal SHA-256 为：

- `artifacts/v2-r1r/p1-nr1-preflight-20260817-1`：`PASS_P1_NR1_PREFLIGHT`，seal `1825282BF46C821BFD89636B58DCCDAD53CAAE1C9F226BA87B3CE8387B095DAF`；
- `artifacts/v2-r1r/p1-nr1-qualification-20260817-1`：`PASS_P1_NR1_MEASUREMENT_QUALIFICATION`，seal `DADBDDBF3B672702A4FFADE74504EAD2F941628D729B5A29B0B7D4DA18EF6FB3`。

两阶段 source start/after/snapshot identity 均为 `1D837A0F04BD39B609DC70DE3F187E108E90944B771C2CA8615485A82B78657E`。Git HEAD、branch、dirty-entry fingerprint 在各阶段前后相同；post-action process/transport/successor audit 通过。preflight 为 `20 passed`，formal 前 V8L 三根 status、精确 seal SHA 与 tree seal 保持不变。

## 3. Measurement 结果

Numeric qualification/heldout 分别为 384/256 cases。两个域的 state、step、final、comparison、winner 与 audit 指标全部为 `1.0`；共同平移、正比例缩放、候选置换与 handle rename 均为 `1.0`。qualification 最大 `|delta|=32`，heldout 最小 `|delta|=64`；candidate、horizon 与 fingerprint 域严格分离，全部 case 同时含正负 update。

Relation qualification/heldout 分别为 384/256 cases。closure、query、direct edge、composition、antisymmetry 与 audit 指标全部为 `1.0`；handle/edge/query permutation 与 transitive redundancy 均为 `1.0`。两个域都有 branch/merge；multi-component 分别为 96/64 cases，redundant-direct 为 301/245 cases；query 数覆盖 4–8，truth pattern 为 139/113 种，positive rate 为 `0.4996/0.4993`。

手工 fixture SHA-256 `C94D372B7A06448F2031ABCA03C2EEE2538B3C627E4A5CD00E522C3F18EFB57F` 与冻结值一致；6 numeric、6 relation 对 measurement/reference 均 exact，11 类固定 fault target registry 完整。正式大样本的 11 类 fault state detection 全为 `1.0`；所有 decision-affecting fault 的 decision kill 为 `1.0`，required metric Gate 全通过。numeric off-by-one 的 decision kill 为 `0.09375`，但它按预注册只承担 final-exact fault；unknown handle alias 按预注册由 schema rejection 关闭，二者都未被事后升级为 decision fault。

qualification/heldout 的 numeric 与 relation fingerprint overlap 均为 0；`training_performed=false`、`audit_statistics_fit=false`。

## 4. 对 V8L 的解释更新

NR1 PASS 不改写 `FAIL_P1_V8L_BOOTSTRAP`。它证明现在已经具备一个能独立识别累计、比较、选择、可达性与组合错误的 measurement surface；它不证明 V8L 当时的 lexical anchors 已经具备这些性质，更不证明旧匿名 K=8 失败只由 teacher 导致。

因此，历史结论仍是关闭“匿名 shared K=8 + lexical metric teacher”组合。新的因果问题是：在 fresh data、同一合格 teacher 与 matched compute 下，task-independent mixed/typed inductive bias 是否相对 anonymous shared core 形成稳定的 numeric/relation mechanism gain，同时不依赖 task id、oracle span、candidate index、simulator forward 或答案旁路。

## 5. H1 授权与停止线

NR1 只设置 `p1_h1_design_authorized=true`。H1 必须另立冻结合同、fresh roots 与独立预测试，直接比较 anonymous shared core 和 content-routed mixed/typed core；两臂使用同 fresh data、同 frozen-Qwen/Boundary 输入、同 teacher、matched active parameters/compute/token/FLOPs 与同答案读取约束。NR1 measurement 只作为 teacher/audit，不能进入部署 forward 或提供答案。

H1 PASS 也只授权 F1；H1 FAIL 必须停止，不得以 F1、调阈值、改 seed 或增加 exposure 补救。只有后续 F1 的 fresh-seed K=8/K=1/direct/text-CoT、OOD、causal、strip 与成本 conjunction 全部通过，才可能设置 `p1_completed=true`。
