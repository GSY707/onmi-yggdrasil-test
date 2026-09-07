# V2-R1R P0-M v1 执行命令

## 1. Preflight

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_p0m_v1 -q
.\.venv\Scripts\python.exe -m py_compile experiments\v2_r1_revalidation.py src\yggdrasil_v2\r1_revalidation\p0m\*.py tests\v2_r1r_p0m_v1\*.py
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
nvidia-smi
```

必须使用 `.venv`、CUDA、bf16-capable GPU 与本地 exact Qwen revision。preflight 通过后删除旧 `tests/v2_r1r_v17/*.py`，CLI 直接切换到 P0-M；v17 source snapshot 保留 P0-D 追溯面。

## 2. 固定顺序

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p0m
```

runner 必须依次执行并在首个失败处停止：cache → ERE K=8 → CPS K=8 → shared K=8 → direct SFT → text-CoT SFT → 100-step throughput → assessment。已 sealed PASS 的前序 root 可只读复用，不能覆盖。

## 3. 结束边界

assessment 只有在 M01–M08 全 true 时写 `PASS_P0M` 与 `p1_eligible=true`。任何失败写 `FAIL_P0M`、保留 root，父任务分析后另立新版本。即使 PASS，也必须停在 P1 之前，不生成 P1 数据或启动 P1 训练。
