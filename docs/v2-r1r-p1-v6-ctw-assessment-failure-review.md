# V2-R1R P1 v6 CTW assessment 失败复核

日期：2026-08-11

## 1. 判决

P1 v6 的五个 fixed roots 保持原判：前四阶段 sealed PASS，assessment sealed `FAIL_P1_V6_CTW`，不得修改、覆盖、重封或重跑。失败只发生在 `W604_query_cache`；这不是训练机制失败，而是 persisted JSON 与内存对象使用了不一致的等价关系。

formal mechanism compare 已预注册选择 `temporal_witness`。static arm 的未见 episode audit 为 ERE/CPS `0.480469/0.486842`，state drop 近零；temporal arm 的 audit 为 `0.705830/0.805267`，state drop 为 `0.205830/0.305267`，swapped-state drop 为 `0.411659/0.610534`，其 seen、audit、state dependency、swap intervention 和相对优势 Gate 全部通过。这一证据只说明 causal-temporal target 解决了当前静态信用对称，不代表完整 P1 或架构成立。

## 2. 精确根因

query-cache 中的 `selection-witnesses.json` 是合法 canonical JSON。写入前，partition 统计里的六个 `reasoning_budgets` 键是 Python `int`；JSON object 只能使用字符串键，因此读取后对应键为 `str`。assessment 使用：

```python
recorded_selection == replayed_selection
```

于是同一选择在 Python 原始对象层不相等。父任务的只读字段级复现得到：

- 18,899 条 selection records 完全相等；
- partition 的 canonical bytes 完全相等；
- 整体 canonical bytes 与 JSON round-trip 完全相等；
- recorded/replayed 的 selection preimage SHA-256 均为 `3FFED81487CEBF35735E7733C4B4909C53934A8B9357791FD64E68F7E1BFC97E`；
- 唯一差异是 ERE optimization/audit 的 budget `5/6` 和 CPS optimization/audit 的 budget `5` 共六个整数键，经 JSON round-trip 转为字符串；
- query content 重审仍为 `18,899/18,899`、零 failure。

因此 `selection_replays=false` 是验收器的表示层 false negative。它没有改变任何 example、query、label、split、pair、count、hidden content 或模型结果。

## 3. 影响边界

不能把原 assessment 改判成 PASS，因为其代码、结果和 seal 都属于已消耗合同。也不能直接进入 integrated P1，因为 v6 书面合同要求 assessment PASS 才授权后继。正确处理是另立 v6R recovery：只读固定原五个 evidence-seal，使用 canonical JSON 与已嵌入的 selection preimage hash 作为持久证据等价关系，并证明真正的 query、label 与 count 篡改仍会被拒绝。

若 v6R PASS，只恢复“temporal witness 机制资格”，授权另立 fresh-seed integrated P1；P1 仍未完成，P2、Pareto、跨 seed 与架构成功仍不成立。若 v6R FAIL，应停在恢复层分析，不得重训或进入后序。
