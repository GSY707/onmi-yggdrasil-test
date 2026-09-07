# V2-R1R P0-M v4 执行命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests\v2_r1r_p0m_v4 -q
.\.venv\Scripts\python.exe -m compileall -q experiments\v2_r1_revalidation.py src\yggdrasil_v2\r1_revalidation\p0m tests\v2_r1r_p0m_v4
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py --help
nvidia-smi
.\.venv\Scripts\python.exe experiments\v2_r1_revalidation.py run-p0m
```

唯一顺序：cache qualification → ERE → CPS → joint → direct → text-CoT → 100-step throughput → assessment。首个失败停止；同版本 sealed PASS root 只读复用。成功后也必须停在 P1 前。
