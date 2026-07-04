# Stage AS：真实文件证据审计

## 目标

Stage AS 验证一个比 synthetic DSL 更接近文件审计的边界：模型不能直接从内置标签回答，必须先通过受控工具读取真实落盘文件，再输出带引用的 audit 结论。

本阶段不让 tiny 模型从零承担 OCR/layout。文件边界先做小而硬：

1. `dashboard.html`：HTML DOM 中的 reported status。
2. `metrics.csv`：CSV 表中的 actual status。
3. `policy.pdf`：真实 PDF 文件中写入 policy threshold。
4. `screenshot.png` + `screenshot.json`：截图 PNG 和对应受控 metadata。

结论类别：

- `html_csv_mismatch`
- `policy_threshold_violation`
- `screenshot_html_conflict`
- `clean`

每个结论都有对应引用组合：

- `html+csv`
- `csv+pdf`
- `html+screenshot`
- `all-clear`

## 实现

新增脚本：

```powershell
experiments\omni_transformer_stage_as_file_audit.py
```

脚本会生成真实样例文件，并通过受控工具读取：

- HTML：`HTMLParser` 提取 `data-status`。
- CSV：`csv.DictReader` 提取 `actual_status`。
- PDF：读取 PDF bytes 中的 `threshold=N`。
- Screenshot：PNG 作为视觉文件落盘，受控 metadata 提供截图状态。

模型输入不是原始标签，而是工具历史 token：

```text
READ_HTML -> html status
READ_CSV -> csv actual status
READ_PDF -> policy threshold
READ_SCREENSHOT -> screenshot status
```

模型输出三项：

1. audit conclusion。
2. citation pattern。
3. final/tool legality。

同时评估 `no_tool_history` 消融，确认没有工具读取历史时模型不能完成审计。

## Smoke

smoke 结果：

- 聚合结果：`artifacts/omni_transformer_stage_as_file_audit/smoke_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_as_file_audit/smoke_runs/seed20260701.json`
- 样例文件：`artifacts/omni_transformer_stage_as_file_audit/smoke_runs/sample_files/`

smoke 只证明脚本、文件生成、工具读取、训练循环、JSON 和 PNG 输出链路能跑通。

## Probe

单 seed probe：

- 聚合结果：`artifacts/omni_transformer_stage_as_file_audit/probe_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_as_file_audit/probe_runs/seed20260701.json`

结果：

| 指标 | probe |
| --- | ---: |
| conclusion accuracy | 100.00% |
| citation accuracy | 100.00% |
| report exact | 100.00% |
| tool step legality | 100.00% |
| no-tool-history conclusion accuracy | 25.00% |

probe 说明任务结构可学，且结论依赖工具历史。

## 正式 3 seed 结果

正式结果：

- 聚合结果：`artifacts/omni_transformer_stage_as_file_audit/formal_results.json`
- 每 seed 明细：`artifacts/omni_transformer_stage_as_file_audit/formal_runs/seed20260701.json`、`seed20260702.json`、`seed20260703.json`
- 样例文件：`artifacts/omni_transformer_stage_as_file_audit/formal_runs/*/sample_files/`

| 指标 | 3 seed 平均 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| conclusion accuracy | 100.00% | >= 85% | 通过 |
| citation accuracy | 100.00% | >= 85% | 通过 |
| report exact | 100.00% | 诊断项 | 通过 |
| tool step legality | 100.00% | >= 85% | 通过 |
| no-tool-history conclusion accuracy | 25.00% | 明显低于 full | 通过 |
| no-tool-history gap | 75.00 pp | >= 30 pp | 通过 |

## 结论

Stage AS 已经证明：在一个小型真实文件边界任务里，HTML、CSV、PDF、截图文件可以通过受控工具进入 latent 审计链路，并产出正确结论和引用组合。没有工具历史时，结论回到 25% 随机水平，说明模型不是只靠类别先验完成任务。

这比 Stage J 的 synthetic audit 更进一步，因为本轮样例确实落成了本地文件，并通过文件解析工具抽取证据。

## 边界和担忧

1. PDF 是本脚本生成的低熵 PDF，不证明真实复杂 PDF layout。
2. 截图状态来自受控 metadata，不证明 OCR 或视觉 layout 解析能力。
3. HTML/CSV schema 是固定的，不证明开放网页审计。
4. 当前动作顺序是受控工具链，不证明自由规划型 tool-use agent。
5. 若后续要挑战真实文件审计，应接 OCR/layout/html table parser 或预训练文档专家，不让 tiny latent 从零承担感知。

## 正式复现命令

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_as_file_audit.py --sweep --seeds 20260701,20260702,20260703 --train-size 2048 --val-size 512 --test-size 512 --batch-size 128 --d-model 96 --heads 4 --layers 2 --latent-tokens 6 --steps 650 --eval-every 200 --sample-count 12 --output-dir artifacts\omni_transformer_stage_as_file_audit\formal_runs --aggregate artifacts\omni_transformer_stage_as_file_audit\formal_results.json
```
