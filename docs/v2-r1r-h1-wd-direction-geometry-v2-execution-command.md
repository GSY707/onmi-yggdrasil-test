# V2-R1R H1-WD direction-geometry v2 执行合同

日期：2026-08-23  
身份：`H1-WD-DG-V2-20260823-1`

## 1. 唯一入口

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1r_h1_wd_direction_geometry.py run-h1-wd-direction-geometry-v2
```

当前 identity 的唯一输出 root：

```text
artifacts/v2-r1r/h1-wd-direction-geometry-screen-v2-20260823-1/
```

单次 attempt 的 sibling lease：

```text
artifacts/v2-r1r/h1-wd-direction-geometry-screen-v2-20260823-1.preflight-lease.jsonl
```

旧 `run-h1-wd-direction-geometry` 命令与 v1 root 已消费，不是兼容入口。新命令不得调用
decision-causal 或 overlap-residual runner。

## 2. Root 创建前

程序先以 exclusive create 写 lease；成功即消费 identity。随后必须完成且不写 output root：

1. source checkpoint、token cache、causal target bank/manifest hashes；
2. v1 crash root 五文件与两份 sibling log hashes；
3. v2 root absence 与 CUDA availability；
4. 冻结模型 eval/no-grad/parameter digest；
5. microbatch `4` 全量 train 4096 + heldout 1024 replay identity；
6. common/projection max `<=2.5e-05`、RMS/relative RMS/percentiles、finite 与完整 coverage。

任一项失败时不得创建 v2 root，也不得自动切换 batch 或放宽容差；程序向 lease 追加
`CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_PRE_ROOT`，同 identity 不得重跑。

## 3. Root 创建后顺序

```text
contract/preflight
-> R0/R1 centroids
-> batch-4 target-before capture + replay audit
-> R2 five feature surfaces + lambda stability
-> R3 exact final-head + sampled matrix-free local-J/Fisher
-> dual-sketch R4 nulls + site-macro + negative-sign audit
-> null-distributions.json
-> result.json / run-state complete
```

R1–R4 无论是否显著都必须继续到 result；只有 hard identity/integrity/finite failure才停止。
不得保存 optimizer、模型、checkpoint 或重新生成 target。

site-macro 只注册全 heldout aggregate baseline energy `>1e-30` 的 active sites。单条记录的
site energy 可以为零；若某次 bootstrap 丢失任一 active-site 支持，则显式标为 undefined
并令 signal Gate 为 false，而不是抛异常消费运行。

## 4. 终态

完整测量顶层状态必须为 `COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`，科学含义由
`scientific_status`、R1、R2、R3、R4 字段表达。异常则写
`CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2`，该 root 原样封存；root 前异常写入 sibling
lease。lease 或 root 任一存在都表示同 identity 永不重跑。

本次始终 `authorizes=nothing`。只有基础设施/实现 crash 才可在用户已给的三 identity
预算内另立 successor；no-signal 或 residual structure 是有效结果，不触发调参重试。
