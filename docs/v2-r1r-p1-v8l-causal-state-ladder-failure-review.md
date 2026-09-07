# V2-R1R P1 v8L causal-state ladder 失败复核

日期：2026-08-12  
正式状态：`FAIL_P1_V8L_BOOTSTRAP`  
证据等级：sealed development-mechanism failure；关闭当前匿名 K=8 + lexical-anchor 路线，不构成白皮书或 mixed core 的总否证

## 1. 核心判决

v8L 的 single-use formal 严格停在 384-update bootstrap。ERE 已形成可迁移的 fixed-anchor state direction，CPS 则只产生很弱的局部改善，未达到预注册的 optimization fit `0.65`，因此 joint closure、答案因果迁移和 ordinary retention 均未运行。

机器结论是：

> `cps_fixed_anchor_bootstrap_not_qualified`

进一步只读复核又发现，v8L 的全局 anchor geometry Gate 没有单独资格化 CPS 数值语义。冻结 Qwen + V7 Boundary 的 exact-cost lexical contrasts 不形成稳定、单调的共享数值轴。因此本轮无法把失败唯一归因于 recurrent core；更准确的机制结论是：

> 当前 lexical metric teacher 与匿名 shared K=8 core 的组合没有形成共享 cost algebra。

这足以执行预注册停止线：不降低 Gate、不延长 bootstrap、不补跑 joint、不建立 v8 后继，也不授权 v9/P2。但它不支持“整个 V2 架构已经失败”的表述。

## 2. 正式执行与封存

唯一 formal 于 `2026-08-12T01:53:16.7785475+08:00` 启动，命令为：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-causal-state-ladder
```

venv launcher PID 为 `44116`，实际 Python 子进程 PID 为 `50076`；父子链一致，未重启或分段补跑，最终 exit code 为 `1`。三个 root 均复算 `verify_seal=true`，source identity 都是 `59211BAC6D358D9825A98E9108CBCA36BB1DE31798410833B33382D21C6DB5CA`：

- preflight：`PASS_P1_V8L_CAUSAL_STATE_LADDER_PREFLIGHT`，seal `06BA4E180AED6EB2FC5BBFF3DEECC85D8FF74E958B60106E2494E4F48B685115`；
- anchor cache：`PASS_P1_V8L_CAUSAL_STATE_LADDER_ANCHOR_CACHE`，seal `5F1AABBF3D601559BE3CEC1DEDC0390B4C84780962947768B089DBC3D3AF3C14`；
- qualification：`FAIL_P1_V8L_BOOTSTRAP`，seal `3332CD3D24DCE85BEB3D8ECA2A7D1D669E97CAF40924CABED88AC07F2C44FE75`。

preflight 的 11 项 checks、12 个预测试、六个 partition/catalog/query/contrast/schedule identity 与 CUDA objective smoke 全通过。没有 `p1-v9*`、`p2*` 或其他 successor root；formal 退出后无同名 Python 进程。

## 3. Bootstrap 的真实结果

bootstrap 按合同完成 `384` updates、`12,288` episode exposures，只训练 Boundary/core；optimizer groups 为 `boundary/core`，answer readout 与 `final_norm` 均冻结。训练 finite，wall time `80.469 s`，吞吐 `152.704 examples/s`。GPU 采样 util 为 `29–41%`，功率 `27.62–34.17 W`，显存约 `5.4 GiB`。

zero-update 到 update 384 的 fixed-anchor direction accuracy 为：

| split | ERE | CPS |
|---|---:|---:|
| optimization zero-update | 0.5503 | 0.4719 |
| optimization after 384 | **0.7726** | **0.5682** |
| audit zero-update | 0.5807 | 0.4781 |
| audit after 384 | **0.7109** | **0.5281** |

ERE 同时通过 optimization bootstrap threshold，并在未进梯度的 audit 上超过 `0.70`。CPS 有非零学习，但 improvement 只有 optimization `+0.0964`、audit `+0.0500`；optimization mean signed margin 仅 `0.00175`，audit 仅 `0.00052`。

CPS optimization 分层 accuracy 为 `cost_trace 0.6042`、`final_cost 0.6042`、`cost_order 0.5547`、`unique_optimum 0.5521`、`choice 0.5260`；audit 分别为 `0.5391/0.5547/0.5000/0.5391/0.5078`。训练日志的当前 minibatch accuracy 为 `0.656–0.688`，但全量 optimization 只有 `0.568`，说明模型能追随局部 batch 梯度，却没有形成跨 pair 的稳定共享坐标。

由于 bootstrap Gate 失败，正式 Gate 只有 `L01=true`、`L02_bootstrap_fit=false`、`L07=true`。L03–L06 没有被评估，不能写成答案迁移、retention 或最终架构完整性失败。

## 4. Anchor measurement 的漏检

正式 cache 本身完整：2,121 个 raw queries、1,448 个 unique anchors、4,096 个 pair-level contrasts、30,369 tokens，1,448/1,448 content audit 通过；global effective rank 为 `25.605`，contrast norm 为 `0.4766–5.4483`。

但 post-stop 只读分层复核显示，global rank 主要由 ERE semantic anchors 支撑：

| anchor kind | unique | effective rank | audit anchor 与 optimization 重合 |
|---|---:|---:|---:|
| ERE semantic transition | 466 | 39.89 | 0.86% |
| ERE semantic final | 466 | 33.76 | 0.86% |
| CPS cost trace | 230 | 3.47 | 75.49% |
| CPS final cost | 230 | 3.84 | 75.49% |
| CPS cost order | 10 | 4.30 | 100% |
| CPS unique optimum | 10 | 3.32 | 100% |

对 `cost_trace/final_cost` 按“低成本 query 减高成本 query”重新定向后，mean shared-axis alignment 只有 `0.340/0.380`，最小值为 `-0.245/-0.178`；部分 lexical contrasts 与数值顺序反向。低秩本身不必然错误，因为标量代数可以是低维的；真正的问题是合同没有在训练前要求单调性、加法一致性、比较反对称性或已知 oracle-state decision power。global effective-rank Gate 因而不足以证明这个 teacher 能测量 CPS cost algebra。

这解释了为什么 ERE 在大量 fresh anchors 上迁移，而 CPS 即使 audit anchors 大量复用仍接近随机：监督方向在 lexical embedding 中不是稳定的算术坐标，shared core 接收到的是彼此干扰的局部 metric constraints。

## 5. 架构层结论与后继边界

本轮确认三件事：

1. parameter-free scorer、冻结坐标系、mutation-aligned state supervision 和训练通路在 ERE 上可工作；失败不是通用实现崩溃。
2. 当前匿名 shared core 没能在这个 teacher 下形成 CPS cost/order/choice closure；继续同类 exposure、loss weight、prompt wording 或阈值调整没有正式依据。
3. teacher 的 CPS decision power 未被预先资格化，所以不能从这次失败区分“纯 shared core 缺少算术归约能力”和“lexical teacher 给出了不一致坐标”。

因此当前 v8 主线正式结束，`fresh_p1_v9_authorized=false`、`p1_completed=false`、`p2_eligible=false`、`p2_started=false`。若项目继续，下一份合同必须是新的架构路线，而不是 v8L 修补：先独立资格化数值/关系 measurement system，再比较匿名 shared core 与 mixed/typed core。一个与白皮书兼容的候选是让通用 recurrent workspace 保留控制与组合，数值累计、比较和关系归约由任务无关、可路由的 typed expert 承担；它必须使用 learned/content routing、matched compute 与新数据，不能退化为 oracle task id、显式候选寄存器或 simulator 旁路。

在这个新合同建立前，不运行 K=1、direct/text-CoT、P1 v9、P2 或 V2-B，也不删除或改写三个 sealed v8L roots。
