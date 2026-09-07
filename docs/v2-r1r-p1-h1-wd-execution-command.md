# V2-R1R P1-H1-WD 非正式执行命令

本命令只执行 2026-08-21 冻结的 overlap-write/common-residual screen。它不调用旧 `run-p1-h1`，不创建 calibration/formal/F1/P2 root。

## 启动前条件

- source mixed checkpoint SHA-256 必须为 `1126228617…745D`；
- screen package identity 必须为 `A52C5222…486DA4`；
- 旧 token cache 全量 audit 必须通过；
- 新 root `artifacts/v2-r1r/p1-h1-wd-overlap-residual-screen-20260821-1` 必须不存在；
- CUDA 可用，旧 H1 artifact 只读。

## 唯一命令

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1r_h1_write_delete.py run-h1-wd
```

runner 内部固定执行 predecessor baseline → W 800 → W Gate → D 800 → D Gate → J/matched-control 1,200 → interventions/result。W/D 任一 Gate 失败即停止；不得删除 root 后重跑、改 coefficient、换 schedule 或降低 Gate。

训练进度写入 `run-state.json` 与 `training-events.jsonl`；最终机器结果为 `result.json`。无论结果如何，`authorizes=nothing`。

## 执行状态

该唯一命令已于 2026-08-21 正常退出，固定 root 已消耗。`result.json` 状态为 `FAIL_H1_WD_NONFORMAL_MECHANISM`，无 `crash.json`；禁止删除 root 后重跑、换 schedule、调 coefficient 或降低 Gate。结果复盘见 `docs/v2-r1r-p1-h1-wd-failure-review.md`。
