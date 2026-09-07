# V2-A Closure C1U S2 多 Bank、单 Seed、Matched K1/K8 冻结执行合同

冻结日期：2026-09-02

## 1. 研究问题与授权边界

S1 已证明 C1U gate-free public-grounded workspace 能在一个固定 32-record bank 上形成完整行为与双 support 因果闭环。S2 不再重复 Overfit32，而回答两个更窄、更严格的问题：同一机制能否从多个训练 bank 迁移到从未参与更新的 heldout bank；保留八个分地址工作状态的 K8 是否稳定优于把全部公开 object cards 压入一个工作状态的 learned K1。

用户授权原文为：`没问题，那只做多bank，多seed先不做。那接下来就应该继续做S2了，开始吧`。因此本合同只允许一个固定 scientific model seed。三个 bank folds 与 K1/K8 两臂不是多 seed；bootstrap seed 只用于对已冻结的 paired bank 结果重采样，不产生训练端点。S2 PASS 最多授权 S3 合同设计，不授权 S3 训练、重试、换 seed、调阈值、checkpoint selection、C2、V2-A PASS、V2-B 或 V2-C。

## 2. 固定身份、路径与前驱

S2 preflight：

- identity：`V2-A-CLOSURE-C1U-PGF-S2-MULTIBANK-MATCHED-K1-K8-PREFLIGHT-20260902-1`
- root：`tmp/v2-a-closure-c1u-pgf-s2-multibank-matched-k1-k8-preflight-20260902-1`
- sibling lease：同名 `.preflight-lease.jsonl`

正式 S2：

- identity：`V2-A-CLOSURE-C1U-PGF-S2-MULTIBANK-MATCHED-K1-K8-20260902-1`
- root：`artifacts/v2-a/closure-c1u-pgf-s2-multibank-matched-k1-k8-20260902-1`
- sibling lease：同名 `.s2-lease.jsonl`

唯一前驱是 sealed PASS 的 C1U S1。启动前必须复放 S1 result、evidence seal、endpoint、source identity、完整 seal tree 与固定训练会计；当前源码不得改写 S1 的 sealed source closure。

## 3. 多 Bank 数据合同

S2 固定生成六个新 bank，每个 bank 仍为 CPS/ERE 各四个 2×2 factorial groups，共 32 records。每个 bank 使用独立 task seed、opaque value/address identities 与 bank-specific A–I label permutation；所有 example/group id 加 bank 前缀。六个 bank 与 S1 之间不得共享 opaque token/address，六个 label permutations 必须彼此不同且都不是 S1 identity permutation。

三个预注册 folds 为：

| Fold | Train banks | Heldout banks |
| --- | --- | --- |
| F0 | B2, B3, B4, B5 | B0, B1 |
| F1 | B0, B1, B4, B5 | B2, B3 |
| F2 | B0, B1, B2, B3 | B4, B5 |

每个 fold 因而有 128 train records 与 64 zero-finetune heldout records。每个 bank 恰好 heldout 一次；heldout cards、targets 与 answers 不得进入对应 endpoint 的 optimizer、loss 或 schedule。preflight 物化一个 target-free、one-card-per-forward 的 Qwen cache，正式 S2 只能读取其封存 tensors 和 ledger。

## 4. Matched learned K1/K8

两臂接收完全相同的八槽 padded public inputs、独立 card encodings、operation/query cards、训练 records、caller order、optimizer、LR schedule、更新数和初始参数。

- K8 保留每个公开 object card 的独立 payload，并按公开地址执行 hard route 与 target-only gate-free overwrite。
- K1 先用同一 shared card encoder 编码全部公开 object cards，再以 permutation-invariant present-object mean 压入唯一工作状态；所有 active operation 与 query 只读写这个状态。

`workspace_slots` 不是 parameter 或 learned embedding。K1/K8 state-dict names、shapes、trainable parameter count 与同 seed 初始 tensor hash 必须逐字一致。K1 必须真实训练到固定 endpoint；其多地址 owner swap 对工作状态严格为 permutation null，而不是硬编码行为结果。K8 必须以 heldout support deletion/counterpart、owner swap contrast 与 distinct-owner effects 建立 functional-K，不能用 route hit、cosine、energy 或参数量替代。

## 5. 固定优化与端点

scientific model seed 固定为 `2026090211`，三个 folds 和两臂都重置到这一个 seed。caller order seed 固定为 `2026090212`。每个 endpoint 使用 AdamW、batch 8（一个完整 CPS group 加一个完整 ERE group）、boundary/transition/head LR `1e-4/2e-4/3e-4`、weight decay `0.01`、clip `1.0`、256-step warmup 后 cosine，并恰好完成 4,000 optimizer updates。

正式 S2 共六个固定端点：`3 folds × 2 arms`，总计 24,000 formal optimizer steps、六次 model writes；每个 endpoint 只写一次 `fixed_4000.pt`，不含 optimizer state。禁止中间、best、resume、early-stop 或 endpoint 替换。

训练目标保持 C1U 的 raw A–I CE、no-core uniform KL 与 exact factorial counterpart answer-margin hinge。K1 的 counterpart intervention 发生在独立 public object payload 上，然后重新压缩到唯一 workspace state；不得把 support ledger 送入模型 forward。

## 6. Preflight Gate

preflight 在领取 root 前运行注册测试、S1 seal replay、source/assets/device audit、六-bank 生成与 disjointness/owner-only shortcut audit、三-fold split/schedule audit和未消费路径检查。领取后物化并 seal 六-bank target-free cache，随后只在 F0 train 上分别执行 K1/K8 各 16 个 disposable BF16 optimizer steps；权重不保存，formal steps/model writes 必须为零。

P201–P207 必须全部 PASS：

1. S1 predecessor、用户授权与固定路径；
2. source closure、离线资产、RTX 4070 CUDA BF16 与注册测试；
3. 六-bank factorial/public replay、S1/new-bank opaque disjointness、label permutation 与 owner-only shortcut；
4. target-free independent-card cache、ledger 与重载 hash；
5. K1/K8 参数及初始化 parity、K1 owner-swap null、K8 addressed workspace structural controls；
6. 两臂 disposable training finite、固定正式 schedules、性能预算与会计；
7. source stability、result 与 evidence seal replay。

任一 preflight FAIL/CRASH 后正式 S2 为 `NOT_RUN`，不得使用同 identity 修复或重跑。

## 7. 正式 S2 Gate

正式训练完成全部六个固定 endpoint 后，heldout records 每条只由未训练过该 bank 的 fold endpoint评估。判决顺序如下：

1. **R201 predecessor/cache/split/source identity**：preflight、S1、cache、bank payload、folds、schedules 和 source 全部复放一致。
2. **R202 fixed endpoints/accounting/parity**：六个 4,000-update endpoint、24,000 total steps、六次 model writes；每个 fold 的 K1/K8 初始化 hash、schedule 和参数完全 matched。
3. **R203 K8 fresh-bank absolute behavior**：ERE/CPS heldout point accuracy 各 `>=0.75`、Wilson lower 各 `>=0.70`；每个 bank/family point `>=0.50`，factorial exact point `>=0.75`。
4. **R204 paired K8>K1**：overall、ERE、CPS paired accuracy gain 均 `>=+0.05`，以 bank 为 cluster 的 2,000-replicate bootstrap lower 95% 必须 `>0`。
5. **R205 heldout causal behavior**：K8 两族 no-core necessity、两个 support counterpart/deletion、support answer flip 与 two-contributor 均通过预注册 point/Wilson/margin 门。
6. **R206 functional-K and K1-null**：K8 两族 owner swap 改变 logits、每条记录两个 distinct support owners、双 owner causal effects 达标；K1 为真实 one-state endpoint且 owner permutation null，不把 multi-address 指标伪造为 FAIL。
7. **R207 evidence integrity**：source/predecessor/cache/endpoint hashes 无漂移，结果与完整 evidence seal replay。

只有 R201–R207 全部 PASS 才返回 `PASS_V2_A_C1U_S2_MULTIBANK_MATCHED_K1_K8`，且只授权 `C1U_S3_CONTRACT_DESIGN_ONLY`。任一 Gate FAIL 返回 sealed terminal FAIL 并原样停止。

## 8. 唯一命令

```powershell
uv run python experiments/v2_a_closure_c1u_s2.py inspect
uv run python experiments/v2_a_closure_c1u_s2.py run-preflight
uv run python experiments/v2_a_closure_c1u_s2.py audit-preflight
uv run python experiments/v2_a_closure_c1u_s2.py run-s2
uv run python experiments/v2_a_closure_c1u_s2.py audit-s2
```

`inspect`、`audit-preflight` 与 `audit-s2` 只读。`run-preflight` 和 `run-s2` 各只能启动一次；正式命令仅在 preflight sealed PASS 后允许启动。
