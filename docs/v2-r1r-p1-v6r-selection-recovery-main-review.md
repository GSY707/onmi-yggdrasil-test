# V2-R1R P1 v6R selection-recovery 主审

日期：2026-08-11

## 结论

正式判定为 `PASS_P1_V6R_SELECTION_RECOVERY`。原 P1 v6 assessment 继续保持 sealed `FAIL_P1_V6_CTW`；v6R 没有修改或重封任何 v6 artifact，而是以两个新 roots 证明 W604 的唯一失败来自 JSON object key normalization。由此恢复的是 `temporal_witness` 训练机制资格，不是完整 P1。

preflight 与 recovery assessment 的 evidence-seal 文件 SHA-256 分别为 `F0002059099D55CB393254AD42933FDA03CC1638058BEF3870147A8E713619AD` 和 `B191B4058B014F912710D325C676D62C134851524832502CA4DACAB3092E5F98`；两者均已独立复算 `verify_seal=true`。

## 机器事实

v6 五个输入 roots 的原 seal 与本合同 pinned seal hash 全部成立，原 assessment 仍只有 `W604_query_cache=false`。sealed source snapshot 中 42 个 Python files 与用于 replay 的活动旧源逐文件一致。

independent selection replay 得到：18,899 条 records 完全一致；canonical bytes、JSON round-trip、partition canonical 与 selection preimage hash 全部一致，recorded/replayed hash 都是 `3FFED81487CEBF35735E7733C4B4909C53934A8B9357791FD64E68F7E1BFC97E`。raw Python equality 唯一差异是六个 `reasoning_budgets` integer keys 在 JSON 中转为 string。query ID 改写、label 顺序翻转和 count 改写三类负控均同时触发 canonical 与 hash 差异；query mmap 内容重审为 18,899/18,899、209,391 tokens、零 failure。

mechanism compare 仍满足 matched compute 和禁止项约束；static arm FAIL，temporal arm 的 seen/audit/state-dependency/swapped-state Gate 全 true，并按预注册规则选中 temporal。R601–R610 全 true，P1 v7/P2 roots 在验收时均不存在。

## 证据边界与后继

本结果证明 causal-temporal target 在当前 ERE/CPS、K=8、frozen-Qwen proxy 上打破了 static predicate-state 信用对称，并证明恢复验收器仍有 selection mutation decision power。它没有验证 answer objective 与 temporal objective 联训、full 8,192/族、K=1、direct/text-CoT、完整 K01–K09、fresh integrated seed、跨 seed 或 Pareto。

因此只授权另立 integrated P1 合同。integrated P1 必须复用 temporal mechanism，但重新冻结 seed、完整训练路径与公平基线；失败时仍停在 P1，不得进入 P2。
