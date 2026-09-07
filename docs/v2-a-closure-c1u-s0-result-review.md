# V2-A Closure C1U S0 结果复核

## 1. 终局判断

C1U Publicly-Grounded Gate-Free Workspace 的唯一 S0 preflight 与唯一正式 S0 均已在 2026-09-01 单次执行并 sealed PASS。该结论只证明：公开卡片足以唯一重放答案、gate-free target overwrite 的结构与梯度路径成立、真实 Qwen 逐卡缓存可复放，且整个 S0 没有 optimizer step 或模型写入。它不是学习能力、泛化能力或 V2-A 阶段成功证据。

正式 S0 的授权逐字为 `C1U_S1_CONTRACT_AND_IMPLEMENTATION_ONLY`。该授权已用于冻结新的 fresh Overfit32 S1 合同；它从未授权沿用 C1T checkpoint、直接进入 S2、single-seed formal、C2、V2-A PASS、V2-B 或 V2-C。

## 2. 单次执行与封印身份

S0 preflight：

- identity：`V2-A-CLOSURE-C1U-PGF-S0-PREFLIGHT-20260901-1`
- root：`tmp/v2-a-closure-c1u-pgf-s0-preflight-20260901-1`
- result SHA-256：`4D575562F89F58CA8C3AB9B5DA7DDBC5CD5C65B0711976D36DD58A310064BBB2`
- evidence-seal SHA-256：`63646AE12CFBE70598E9EB13455EE054ACB61D717224D3C2B83ABD36A85E3516`
- P001–P006：全部 PASS；48 cards；`optimizer_steps=0`、`model_writes=0`

正式 S0：

- identity：`V2-A-CLOSURE-C1U-PGF-S0-20260901-1`
- root：`artifacts/v2-a/closure-c1u-pgf-s0-20260901-1`
- result SHA-256：`39070BCD12AA15B5EBA6F55E84A55CE348A69992BBD5087BB6DCE526CABA714A`
- evidence-seal SHA-256：`F224DB2101897442742FD621BAA838A78D42ACAD5B6DAE68597E7B968D8A9ADE`
- source identity：`2068B1E54F18C82AC3984A1B01C483E4A3B2297C191861DD962DB925A883572E`
- U001–U010：全部 PASS；32 records / 192 cards；38 个封印条目完整复放

## 3. 资格证据

### 3.1 公开可识别性

ERE query card 明示 opaque value symbol 的 bit 0/1 legend；CPS operation card 明示 candidate index。public-only executor 对 32/32 records 可在不读取 family、AST、factor cell、support ledger 或答案字段的前提下重放 raw A–I 答案。六个注册 bridge faults——ERE missing/conflicting/swapped legend 与 CPS missing/duplicate/swapped candidate index——全部被杀死。

### 3.2 独立卡片缓存

真实 source 为固定 revision `Qwen/Qwen3.5-2B@15852e8c...`，每张 public card 单独 forward，禁止 whole-record hidden reuse。正式缓存含 192 cards，且 `contains_targets=false`：

- `card-cache/cards.pt`：`7CC6E33B73DFFF057C77489676B0A3AFB904507254FC55328242F79B6F82CAAD`
- `card-cache/ledger.json`：`74DFB049B56E4D013C150732E11BF11EB65988078ECF6D7DFAB728541A784929`

### 3.3 Gate-free 机制

正式配置为 source width 2048、payload width 512、address width 64、FFN width 2048。transition 的最后投影严格输出 512 维 proposal，不含额外 gate logit；有效 operation 直接覆盖注册 target，inactive step 为精确 identity，非 target slots 不变。模型总参数、可训练参数与 deployment 参数均为 `13,393,417`。

operation-2 对两个 support 方向均有有限非零梯度：source norm `0.0940`、proposal norm `0.3902`、shared transition norm `10.6841`、readout norm `26.5583`。注册的 `detach_source` 与 `detach_proposal` 两个机制 faults 均导致资格失败并被杀死。

### 3.4 运行与会计

设备为 RTX 4070 Laptop GPU，compute capability 8.9，CUDA 13.0，BF16 可用。source encoder 共 192 次独立调用、11,072 tokens，source 参数全部冻结。S0 全程 `training_started=false`、`optimizer_steps=0`、`model_writes=0`。

## 4. 后继边界

S0 PASS 仅允许在不修改 S0 scientific core 的条件下新增 S1 合同、训练器、evaluator、runner、测试与文档。S1 必须重新核对本页固定的 result/seal/source/cache hashes，并把当前 scientific core 与 S0 source snapshot 逐文件对齐。任何漂移、seal replay 失败或固定路径冲突都在消费 S1 root 前停止。
