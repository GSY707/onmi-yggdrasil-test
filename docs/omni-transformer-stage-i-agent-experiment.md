# Stage I：Tiny Omni Tool-Using Agent 实验

## 目的

Stage I 从单轮回答转向更接近 agent 的闭环任务：

```text
多模态观察 -> latent scratchpad -> 工具动作 -> 工具结果进入历史 -> 下一步动作 -> 写记忆 -> 最终报告
```

本阶段不再只看 answer exact，而是看 episode 是否真正完成。

## 实验脚本与结果

脚本：

- `experiments/omni_transformer_stage_i_agent.py`

结果：

- `artifacts/omni_transformer_stage_i_agent/sweep_results.json`
- `artifacts/omni_transformer_stage_i_agent/sweep_runs/`
- `artifacts/omni_transformer_stage_i_agent/sweep_runs/*/samples/*/agent_panel_grid.png`

正式命令：

```powershell
.\.venv\Scripts\python.exe experiments\omni_transformer_stage_i_agent.py --sweep --seeds 20260701,20260702,20260703 --train-size 4096 --val-size 1024 --test-size 1024 --batch-size 128 --d-model 128 --layers 3 --heads 4 --direct-steps 750 --bottleneck-steps 850 --probe-steps 200
```

## 任务设计

每个 episode 是一个合成维修工单，有四类：

| 类型 | 要求 |
| --- | --- |
| `diagnose_and_fix` | 图像/遥测足够，直接应用正确修复 |
| `missing_info` | 初始信息不足，必须先 `TOOL_QUERY_LOG` |
| `conflicting_signals` | 图像和遥测冲突，必须先 `TOOL_RUN_TEST` |
| `memory_aided` | 当前线索误导，必须依赖记忆提示 |

可输出动作：

```text
TOOL_QUERY_LOG
TOOL_RUN_TEST
TOOL_APPLY_FIX_<fault>
MEMORY_WRITE_<fault>
FINAL_REPORT_<fault>
```

成功条件：

1. 最终 `FINAL_REPORT` 的 fault 正确。
2. 已经应用正确 fix。
3. 已经写入正确 memory summary。

## 模型结构

输入 token stream：

```text
image patches
text work-order tokens
telemetry tokens
memory tokens
tool/action history tokens
latent scratchpad tokens
action query token
```

H2 的 latent bottleneck 约束继续保留：

- latent tokens 可以 attend 全部输入和工具历史。
- action query token 不能直接 attend 原始输入，只能通过 latent scratchpad 做决策。

同时训练 direct agent baseline：

- direct agent 的 action query 可以 attend 原始输入和历史。

## 正式结果

3 seed 聚合：

| 模型 | step action exact | closed-loop episode success |
| --- | ---: | ---: |
| direct agent | 100.00% | 100.00% |
| latent bottleneck agent | 100.00% | 100.00% |

latent bottleneck 按 episode 类型：

| 类型 | success |
| --- | ---: |
| diagnose_and_fix | 100.00% |
| missing_info | 100.00% |
| conflicting_signals | 100.00% |
| memory_aided | 100.00% |

样例 rollout：

```text
diagnose_and_fix:
TOOL_APPLY_FIX_1 -> MEMORY_WRITE_1 -> FINAL_REPORT_1

conflicting_signals:
TOOL_RUN_TEST -> TOOL_APPLY_FIX_2 -> MEMORY_WRITE_2 -> FINAL_REPORT_2

memory_aided:
TOOL_APPLY_FIX_1 -> MEMORY_WRITE_1 -> FINAL_REPORT_1
```

## 消融结果

| 消融 | episode success |
| --- | ---: |
| full latent bottleneck agent | 100.00% |
| no tool history | 0.00% |
| no memory channel | 14.52% |
| no image | 79.95% |
| no telemetry | 73.44% |
| no latent access | 0.00% |

解释：

- `no tool history` 为 0%，说明模型必须读取工具结果和前序动作，不能只靠初始观察一次性猜完。
- `no latent access` 为 0%，说明 action query 不是绕过 latent 直接决策。
- `no memory channel` 消融的是整个 memory channel，包括初始维修记忆和写入后的 memory state，因此掉点很大；它不是只测“过去记忆”。
- `no image` 和 `no telemetry` 都明显掉点，但不是 0%，因为部分 episode 可以通过工具或记忆恢复。

## Latent Probe

冻结 latent bottleneck agent，从 latent tokens 的 hidden state 线性 probe：

| probe target | 准确率 |
| --- | ---: |
| current fault | 98.34% |
| phase | 100.00% |

这说明 latent scratchpad 中包含当前故障假设和 agent 阶段状态。

## 结论

Stage I 支持以下判断：

1. Tiny omni Transformer 可以从单轮问答扩展到多步 tool-using agent loop。
2. Latent bottleneck agent 能在闭环 rollout 中完成工具调用、使用工具结果、写记忆和最终报告。
3. 工具历史、记忆、多模态输入和 latent scratchpad 都是有效因子；消融会明显破坏成功率。
4. 该实验比 H 阶段更接近实际 agent，但仍是 synthetic expert-trajectory imitation，不是开放环境 AGI。

## 未证明边界

- 训练仍是 expert trajectory teacher forcing，没有强化学习或自我探索。
- 工具是合成 DSL，不是真实浏览器、文件系统或代码执行环境。
- Episode 只有 3 到 4 步，未验证长程规划。
- 错误恢复只通过消融间接观察，未训练模型从自己错误动作后恢复。
- 记忆是结构化 token，不是真实 `世界树计划` 记忆树持久化读写。

## 下一步

建议下一阶段做 Stage J：

1. 增加错误恢复：让 rollout 中出现错误工具结果或错误动作后仍可修正。
2. 增加 longer horizon：episode 扩到 8 到 12 步。
3. 把 memory channel 替换为更接近世界树计划的记忆树节点读写：摘要、关联边、证据 URI、按需展开。
4. 接入一个真实但受控的本地工具，例如文件检索/小型代码运行/网页 DOM 查询。
