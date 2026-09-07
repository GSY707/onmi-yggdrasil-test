# V2-R1R P0-D v10 R0C-statistical 主设计层验收

日期：2026-08-02  
判决：**accepted**  
证据等级：有限 statistical measurement-component qualification；非 production data、非模型、非训练、非架构证据

## 1. 核心判决

v10 R0C-statistical 可以接受。接受的对象不是某个训练结果，而是 G07/G08 后续测量所需的一组有限算法原语：source-only heuristic parser、条件多数、word/character multinomial Naive Bayes、确定性 grouped five-fold、split-local cross-validation 与 train-fit-heldout。正式 artifact、独立复算和 registry 外公共函数探针没有暴露已知书面违约被错误接受的情况。

这个判决关闭 R0C，但不关闭 P0-D。v10 没有生成 ERE/CPS production records，没有证明 toy grammar 覆盖真实 renderer，没有运行模型、cache、GPU 或训练，也没有授权 R0D、R1–R3 或 P0-M。

## 2. 正式证据

唯一 formal command 为 `python experiments/v2_r1_revalidation.py seal-statistical`，只调用一次，返回 `PASS_R0C_STATISTICAL`。fixed root 为 `artifacts/v2-r1r/p0d-v10-r0c-statistical-20260802-1/`，顶层精确包含九项 entry，不存在 `failure.json` 或 authorization artifact。

机器结果为：

- G07/G08 `2/2`，34/34 raw metrics 通过；
- 39/39 adversary 通过，其中六个 transform family 各有一个未映射 holdout；
- 声明且实际命中的 metric kill 为 34/34，不使用级联失败补足覆盖；
- positive metamorphic 5/5，输入只读；
- novel grouped-pair 原子性与 row-order invariance 通过；
- qualification profile 为 parser 14、categorical 1、analyzer 3、NB 2、fold 1、CV 1、heldout 1、negative 10；
- `authorization_created=false`。

formal root 的 evidence seal 经逐路径重算为 52/52 一致，seal 文件自身 SHA-256 为 `3F003C8BED40F54F010F4C7A96207D703ABA59F1088F189C1265D0F9E1C79BA1`。qualification bundle 的 manifest 与 input seal 也分别按各自排除规则复算一致。随后再次通过公开 `audit-statistical` 读取 formal bundle，G07/G08、固定语义摘要和 overall 均为真，failures 为 0。

v10 guard 还重新验算了 v7、v9 prior artifact 的 evidence seal 全树，并确认当前 `common/`、`audit/` 十个 Python 文件与 v9 formal source snapshot 逐字节一致。因此 R0C 没有静默改写已接受的 R0A/R0B 运行面。

## 3. 主设计层独立探针

formal 前以未写入 qualification fixture 的新输入直接调用公共函数，六组性质全部通过：

1. `heuristic_prediction` 只有 `source, family, heuristic, valid_choice_mask` 四个参数，未知 `Answer:` source line 以 `malformed_source` 拒绝；
2. duplicate token 改变 exact NB score，feature multiplicity 保留，未退化为 Bernoulli NB；
3. Unicode word token、OOV prior tie 和 masked tie-break 均按合同工作；
4. character 3–5 gram 对可折叠 whitespace 不变；
5. novel multi-row group 不被拆分，row permutation 不改变 fold map；
6. heldout-only token 不进入 train vocabulary，train/heldout group overlap 以精确错误拒绝。

测试还构造了更强的自证攻击：修改 NB fit input，使用同一实现重算全部 golden，再重建 manifest/input seal。此时两个 raw Gate 可以保持为真，但固定语义摘要失败，overall 必须为假。因此 v10 不再允许“改题、重算答案、重新封印”获得 PASS。

## 4. 冻结前收紧及其意义

正式冻结前完成了四项设计收紧。第一，coverage ledger 只统计 adversary 声明且实际命中的 metric，不统计该输入异常顺带击杀的其他 metric。第二，输入只读与 group integrity 从 raw metric 中分离为 runner 独立不变量，避免 expected file 自证。第三，parser 改为精确四参数接口并拒绝未声明行；`World:`、`Planning:` 是 grammar 中唯一合法家族头。第四，v7/v9 prior artifact 全树和 accepted runtime 字节身份进入 contract guard。

这些修改都发生在 42 文件 frozen manifest 形成之前。冻结后 contract guard、compileall、21 项 pytest、CLI 精确入口、preflight、registry 外探针与唯一 formal 依次通过，冻结文件没有再改写。

独立复核过程有两项非产品异常。首次 standalone probe 因命令环境未设置 `PYTHONPATH=src` 而在 import 前退出；补齐运行环境后原探针原样通过，没有修改实现或输入。首次根 seal 检查器因 PowerShell helper 的对象返回语义误报 `false`；改为直接构造路径—hash map 后得到 52/52 一致，并由公开 re-audit 交叉确认。两者均未触发 formal 重跑。

## 5. 证据边界

R0C 现在证明的是：在固定有限 profile 上，测量算法的数学实现、split isolation、错误拒绝和证据封印足以支撑下一层集成测试。它不证明：

- v10 toy source grammar 能解析未来 ERE/CPS production renderer；
- production distribution 不含表面捷径或 train/heldout 泄漏；
- v7 simulator、v9 invariant audit 与 v10 learner 组合后接口自然兼容；
- shared Boundary/core、K=1/K=8 latent 或任何白皮书架构可训练；
- latent 路线相对 direct SFT/text-CoT 形成质量—成本优势。

因此 `R0C-statistical accepted` 不能升级为 P0-D accepted，更不能升级为 V2-A 或完整白皮书通过。

## 6. 完成、未完成与下一阶段

已完成：v10 直接切换、独立 learner package、固定 numeric/source fixtures、34-metric registry、39-case adversary lattice、5 个 metamorphic、prior/runtime guard、42 文件冻结、唯一 formal、52 文件根 seal 复算和主设计层外部探针。

未完成：R0D integration、production generator、R1–R3、P0-M、四条 matched 模型路线、cache、GPU 和训练。

下一阶段应另立 R0D 新合同，只组合而不重新实现 R0A–R0C：用固定 hand-authored reference pack 驱动既有 simulator、invariant audit 与 statistical learner，要求 G01–G08 全 conjunction、F401–F420 跨组件 faults、跨组件 metamorphic、import boundary、artifact-internal replay 和单次 sealed formal。R0D 必须显式处理 toy parser 与 reference surface 的接口，不能以兼容 wrapper、隐藏 label adapter 或 production-generator 旁路解决。新合同冻结并单独授权前，不执行 R0D。
