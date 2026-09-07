# V2-R1R P0-M v2 执行命令

## Preflight

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_p0m_v2 -q
.\.venv\Scripts\python.exe -m py_compile experiments\v2_r1_revalidation.py src\yggdrasil_v2\r1_revalidation\p0m\*.py tests\v2_r1r_p0m_v2\*.py
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
nvidia-smi
```

使用 `.venv`、CUDA、bf16-capable GPU 与本地 exact Qwen revision。preflight 通过后删除旧 `tests/v2_r1r_p0m_v1/*.py`；v1 cache root 仅作为 v2 明确引用的 immutable component，不是兼容入口。

## 固定顺序

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p0m
```

runner 顺序为 cache qualification → ERE K=8 → CPS K=8 → shared K=8 → direct SFT → text-CoT SFT → 100-step throughput → assessment。首个失败立即停止；已 sealed PASS 的同版本前序 root 只读复用。

assessment 只有 M01–M08 全 true 才写 `PASS_P0M`、`p1_eligible=true` 与 `p1_started=false`。无论成功与否，本轮都不得创建或运行 P1。
