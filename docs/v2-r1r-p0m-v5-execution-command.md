# V2-R1R P0-M v5 执行命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_p0m_v5 -q
.\.venv\Scripts\python.exe -m compileall -q experiments\v2_r1_revalidation.py src\yggdrasil_v2\r1_revalidation\p0m tests\v2_r1r_p0m_v5
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
nvidia-smi
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p0m
```

固定顺序：cache qualification → ERE → CPS → joint → direct → text-CoT → 100-step throughput → assessment。首个失败即停；同版本 sealed PASS 只读复用。无论结果如何不得进入 P1。
