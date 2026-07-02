# Stage J/K：多模态证据审计与 UI+DOM Agent 实验

## 目的

Stage J/K 继续沿用 tiny omni Transformer，但把任务从维修工单扩展成两个更接近 agent 使用目标的闭环任务：

```text
像素观察 / 文本目标 / 结构化证据或 DOM / memory / 工具历史
-> latent scratchpad
-> 工具动作
-> 工具结果进入历史
-> 下一步动作
-> memory write
-> final report
```

Stage J 验证“多模态证据审计”：模型必须读取图像、表/日志、policy 文本、memory 和工具历史，输出根因与证据码。

Stage K 验证“UI+DOM 操作”：模型必须结合截图、DOM 结构、目标文本和工具历史，执行 click/type/validate/final 这样的受控 UI 操作 DSL。

## 实验脚本与结果

脚本：

- `experiments/omni_transformer_stage_jk_audit_ui.py`

结果：

- `artifacts/omni_transformer_stage_jk_audit_ui/sweep_results.json`
- `artifacts/omni_transformer_stage_jk_audit_ui/sweep_runs/`
- `artifacts/omni_transformer_stage_jk_audit_ui/sweep_runs/*/samples/*/agent_observation_grid.png`

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_jk_audit_ui.py --sweep --task both --seeds 20260701,20260702,20260703 --train-size 4096 --val-size 1024 --test-size 1024 --batch-size 128 --d-model 128 --layers 3 --heads 4 --direct-steps 650 --bottleneck-steps 750 --probe-steps 160
```

## Stage J 任务设计

四类 evidence audit episode：

| 类型 | 要求 |
| --- | --- |
| `chart_table_mismatch` | 先看图像，再查询表；根因和证据码来自不同模态 |
| `screenshot_log_contradiction` | 先看截图，再读日志；日志/截图共同决定 final |
| `policy_case_data` | 读取 policy，再查询表；policy 与表共同决定根因和证据码 |
| `memory_aided_incident` | 当前线索误导，必须使用 memory cue |

动作空间：

```text
TOOL_INSPECT_IMAGE
TOOL_QUERY_TABLE
TOOL_READ_POLICY_OR_LOG
RECORD_FINDING_<target>
MEMORY_WRITE_<target>
FINAL_<target>_EVIDENCE_<code>
```

成功条件：

1. `FINAL` 的 target 正确。
2. `FINAL` 的 evidence code 正确。
3. 已记录正确 finding。
4. 已写入正确 memory。

## Stage K 任务设计

四类 UI+DOM episode：

| 类型 | 要求 |
| --- | --- |
| `button_by_screenshot` | 结合截图和 DOM 点击正确按钮 |
| `form_by_dom_label` | 查询 DOM 后输入字段，再点击提交 |
| `toggle_by_state` | 结合截图和 DOM 状态切换正确控件 |
| `disabled_conflict` | 截图提示与 DOM enabled state 冲突，必须读 DOM state 后点击可用目标 |

动作空间：

```text
TOOL_INSPECT_SCREEN
TOOL_QUERY_DOM
TOOL_READ_DOM_STATE
CLICK_ELEMENT_<target>
TYPE_FIELD_<target>
TOOL_VALIDATE_UI
MEMORY_WRITE_<target>
FINAL_<target>_<operation>
```

成功条件：

1. 最终 target 和 operation 正确。
2. click/type/validate 状态正确。
3. 已写入正确 memory。

## 模型结构

输入 token stream：

```text
image patches
text policy/goal tokens
structured evidence / DOM telemetry tokens
memory tokens
tool/action history tokens
latent scratchpad tokens
action query token
```

H2 latent bottleneck 约束继续保留：

- latent tokens 可以 attend 全部输入与工具历史。
- action query token 不能直接 attend 原始输入，只能通过 latent scratchpad 决策。

同时训练 direct agent baseline：

- direct agent 的 action query 可以直接 attend 原始输入和历史。

## 正式结果

3 seed 聚合：

| 任务 | 模型 | step action exact | closed-loop episode success | invalid action rate |
| --- | --- | ---: | ---: | ---: |
| Stage J audit | direct agent | 100.00% | 100.00% | 0.00% |
| Stage J audit | latent bottleneck agent | 100.00% | 100.00% | 0.00% |
| Stage K UI+DOM | direct agent | 100.00% | 100.00% | 0.00% |
| Stage K UI+DOM | latent bottleneck agent | 100.00% | 100.00% | 0.00% |

latent bottleneck 按类型：

| 任务 | 类型 | success |
| --- | --- | ---: |
| audit | `chart_table_mismatch` | 100.00% |
| audit | `screenshot_log_contradiction` | 100.00% |
| audit | `policy_case_data` | 100.00% |
| audit | `memory_aided_incident` | 100.00% |
| UI+DOM | `button_by_screenshot` | 100.00% |
| UI+DOM | `form_by_dom_label` | 100.00% |
| UI+DOM | `toggle_by_state` | 100.00% |
| UI+DOM | `disabled_conflict` | 100.00% |

## 消融结果

Stage J audit：

| 消融 | episode success |
| --- | ---: |
| full latent bottleneck agent | 100.00% |
| no tool history | 0.00% |
| no memory | 15.89% |
| no image | 66.21% |
| no structured table/log | 66.24% |
| no policy text | 34.70% |
| no latent access | 0.00% |

Stage K UI+DOM：

| 消融 | episode success |
| --- | ---: |
| full latent bottleneck agent | 100.00% |
| no tool history | 0.00% |
| no DOM/structured channel | 0.00% |
| no image | 52.15% |
| no goal text | 34.08% |
| no latent access | 0.00% |

解释：

- `no tool history` 为 0%，说明模型必须读取工具结果和前序动作，不能只靠初始观察一次性猜完。
- `no latent access` 为 0%，说明 action query 没有绕过 latent scratchpad。
- `no DOM/structured` 在 UI+DOM 中为 0%，说明 DOM 查询结果是操作闭环的硬依赖。
- `no image` 在 audit 和 UI+DOM 中均明显掉点，说明截图/图像不是装饰输入。
- `no policy text` / `no goal text` 明显掉点，说明文本目标或规则参与了路由与输出格式选择。

## Latent Probe

冻结 latent bottleneck agent，从 latent tokens 的 hidden state 线性 probe：

| 任务 | target | phase | final kind |
| --- | ---: | ---: | ---: |
| Stage J audit | 72.64% | 96.33% | 73.13% |
| Stage K UI+DOM | 89.02% | 91.37% | 74.33% |

probe 没有达到 100%，说明目标和证据码不是完全线性暴露；但 phase 高，说明 latent scratchpad 中有稳定的 agent 阶段状态。

## 结论

Stage J/K 支持以下判断：

1. tiny omni Transformer 可以在同一个 token stream 中融合像素观察、文本规则/目标、结构化证据/DOM、memory 和工具历史。
2. latent bottleneck agent 可以闭环完成 evidence audit 和 UI+DOM 操作任务。
3. 工具历史、DOM/结构化证据、图像、文本目标和 latent scratchpad 都是有效因子；关键消融会明显破坏成功率。
4. 这比 Stage I 更接近实际 agent 使用场景，但仍是受控 synthetic DSL，不是真实浏览器或开放网页环境。

## 未证明边界

- UI+DOM 任务使用合成截图和 DOM token，不是真实浏览器 DOM、CSS layout、异步事件或登录态页面。
- Evidence audit 使用合成图表/日志/policy，不是真实 PDF、网页、表格截图或自然语言审计材料。
- 训练仍是 expert trajectory teacher forcing，没有强化学习、自我探索或错误恢复训练。
- Episode 最长 7 步，未验证长程网页导航、多窗口、多文件或跨会话任务。
- memory 是结构化 token，不是真实 `世界树计划` 记忆树持久化读写。
