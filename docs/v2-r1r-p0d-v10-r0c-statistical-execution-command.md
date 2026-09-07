# V2-R1R P0-D v10 R0C-statistical 执行合同

日期：2026-08-02
对应设计：`docs/v2-r1r-p0d-v10-r0c-statistical-design.md`
唯一 formal root：`artifacts/v2-r1r/p0d-v10-r0c-statistical-20260802-1/`

## 1. 唯一授权

本合同只授权实现并资格化 R0C statistical measurement component。允许新增 `learner/`、v10 tests/fixtures、三条 v10 CLI 命令与本阶段文档；允许删除 v9 active test source 和 v9 CLI alias。v7 `common/`、v9 `audit/`、所有既有 sealed artifact 和历史设计/主审文档只读。

禁止 R0D、production generator、R1–R3、P0-M、model/cache/train/GPU、第三方统计依赖、网络访问和 authorization 创建。

## 2. 直接切换范围

目标运行面：

```text
src/yggdrasil_v2/r1_revalidation/
  common/                  # v7 accepted，只读
  audit/                   # v9 accepted，只读
  learner/                 # v10 当前实现
experiments/v2_r1_revalidation.py
tests/v2_r1r_v10/
```

CLI 精确只暴露：

```powershell
python experiments/v2_r1_revalidation.py audit-statistical --bundle <path>
python experiments/v2_r1_revalidation.py preflight-statistical
python experiments/v2_r1_revalidation.py seal-statistical
```

不得保留 `audit-invariant`、`preflight-invariant`、`seal-invariant` alias。删除 `tests/v2_r1r_v9/` 的 Python/JSON active source；v9 artifact/source snapshot 保留完整历史。

## 3. 实现顺序

1. 写入独立 parser、analyzer、categorical learner、NB、fold/evaluation 与 qualification audit；parser 公共入口精确四参数并拒绝未声明 source line；
2. 写入冻结 inputs/expected/metric registry、independent materializer/sealer、adversary、metamorphic 与 runner；
3. 复算 v7/v9 prior evidence seal 全树，核对当前 `common/`、`audit/` 与 v9 source snapshot 字节一致；
4. 生成 frozen-input SHA-256 manifest，随后不得再改 frozen files；
5. 运行 contract guard、compileall、CLI contract、pytest；
6. 运行可重复 preflight，要求 positive 2/2、adversary 39/39、metamorphic 5/5、声明 metric kill 34/34；
7. 由主设计层运行 registry 外公共函数探针；
8. 只有上述全部通过才允许调用一次 `seal-statistical`；
9. formal 后只读复算 artifact、写 main-review 和同步 repo truth，不再运行 formal。

## 4. 预测试与 formal

预测试命令：

```powershell
python tests/v2_r1r_v10/contract_guard.py
python -m compileall -q src/yggdrasil_v2/r1_revalidation/learner experiments/v2_r1_revalidation.py tests/v2_r1r_v10
python -m pytest -q tests/v2_r1r_v10
python experiments/v2_r1_revalidation.py --help
python experiments/v2_r1_revalidation.py preflight-statistical
git diff --check
```

唯一 formal 命令：

```powershell
python experiments/v2_r1_revalidation.py seal-statistical
```

formal root 已存在时任何 preflight/formal 命令都必须返回 `BLOCKED`。formal 返回非零或 assessment false 时立即停止；不得修复、重跑或进入后继阶段。

## 5. 回传格式

```text
[P0D_V10_R0C_STATISTICAL]
status=PASS_R0C_STATISTICAL|FAIL_R0C_STATISTICAL|BLOCKED
artifact=artifacts/v2-r1r/p0d-v10-r0c-statistical-20260802-1
positive=<passed>/<total>
adversary=<passed>/<total>
metric_kill=<killed>/<required>
metamorphic=<passed>/<total>
failed_gates=<none|G07,G08>
authorization_created=false
```

此回传不授权 R0D。主设计层必须独立复算 seal、运行未登记探针并给出 accepted/rejected 判决。
