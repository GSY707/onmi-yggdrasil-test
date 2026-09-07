# V2-R1R P1 v8L fixed-anchor causal-state ladder 执行合同

日期：2026-08-12

## 唯一正式入口

工作目录固定为仓库根。先运行目标预测试；三根 fixed root 与 transport 必须不存在，且没有同名 formal 进程。随后只能启动一次：

```powershell
.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p1-causal-state-ladder
```

固定 root：

- `artifacts/v2-r1r/p1-v8l-causal-state-ladder-preflight-20260812-1`
- `artifacts/v2-r1r/p1-v8l-causal-state-ladder-anchor-cache-20260812-1`
- `artifacts/v2-r1r/p1-v8l-causal-state-ladder-qualification-20260812-1`

transport：`tmp/p1-v8l-causal-state-ladder-transport/`。

formal 必须由单一 `Start-Process -WindowStyle Hidden -PassThru` 进程树运行，stdout/stderr 重定向到 transport。不得调用分段 CLI 拼接 formal，不得重启、补跑、换 root、换 seed、换 checkpoint 或修改 Gate。任一 stage 非 PASS 立即停止，后序 root 不得创建。

## 结束回传

确认进程退出、无残留并只读复算每个已创建 root 的 seal。回传格式：

```text
[P1_V8L_CAUSAL_STATE_LADDER_WAKE] status=<PASS|FAIL_PREFLIGHT_SETUP|FAIL_PREFLIGHT|FAIL_ANCHOR_CACHE|FAIL_BOOTSTRAP|FAIL_QUALIFICATION|CRASH>; failed_stage=<...>; artifact=<last-root-or-none>; later_roots_absent=<true|false>; summary=<高密度验收>
```

摘要必须包含：唯一启动证据、预测试、三根 status/seal/source identity、partition/catalog/query/contrast/schedule hash、anchor-cache raw-query/contrast count、geometry/content、bootstrap/joint fixed-anchor 分层指标、答案 causal/validation、compute/GPU、FP32 gradient、deployment key identity、successor absence 与 git fingerprint diff。

无论 PASS/FAIL，都不得启动 v9、K=1、direct、text-CoT、assessment 或 P2。
