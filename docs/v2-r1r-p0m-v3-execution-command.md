# V2-R1R P0-M v3 执行命令

## Preflight

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_p0m_v3 -q
.\.venv\Scripts\python.exe -m compileall -q experiments\v2_r1_revalidation.py src\yggdrasil_v2\r1_revalidation\p0m tests\v2_r1r_p0m_v3
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
nvidia-smi
```

使用 `.venv`、CUDA、bf16-capable GPU 与本地 exact Qwen revision。preflight 后删除旧 v2 测试源；v1 cache 仅通过 v3 明确引用作为 immutable component。

## 唯一正式顺序

```powershell
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p0m
```

runner 依次执行 cache qualification → ERE → CPS → shared → direct → text-CoT → 100-step throughput → assessment。首个失败即停；同版本 sealed PASS 前序 root 只读复用。任何成功都不得自动进入 P1。
