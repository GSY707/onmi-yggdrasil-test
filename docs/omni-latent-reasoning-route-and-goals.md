# 多模态潜空间路线回收：先打通潜空间思考

日期：2026-07-06

## 当前总目标

当前主目标不是继续把 AV-H/AV-I 的图像编辑或整图重建指标抠到更高，而是回到 Project-Yggdrasil 的核心命题：

```text
外部表征 <-> 统一潜空间
统一潜空间内部读/写/推理
统一潜空间 -> 外部输出专家
```

必须先证明一个学生模型能在潜空间里完成可验证的中间思考。图像、文本、record、工具状态都只是外设和监督来源，不应该让高熵图像重建任务吞掉主线。

## 已经找回的判断

1. **互译是硬门槛。** AV-E/AV-G 的方向正确，因为它要求 external 和 latent 双向互译；但训练结果显示 answer 可以走捷径，四个 external sequence exact 仍为 0，所以不能只看 answer accuracy。
2. **图像编辑不是第一门。** AV-H 证明 teacher-forced record/image output 有上限信号，但非泄漏的 text/image -> target record 和 image grounding 没闭合。继续直接训练图像编辑会把问题混成感知、推理、输出三层失败。
3. **整图 latent bank 只是诊断。** AV-I 说明 token 容量、prefix 分工和 residual loss 可以被单独测，但它不是最终架构。继续追 2 维 token 的像素复原，不能直接证明潜空间推理。
4. **我们有更好的数据条件。** 图像由我们自己构造，record、renderer、parser teacher、过程 trace 都可控。这意味着下一阶段不该让模型自己发明一套不可读语言，而应先用可验证 teacher 把 latent workspace 的语义打稳。

## 正确路线

### 1. 先建 record/text 潜空间核心

先用低到中熵的 `scene_record` 和文本描述建立统一 latent workspace：

```text
source record/text -> latent workspace -> reconstructed record/text
source latent + operation -> target latent -> target record/text/answer
```

这一阶段不使用像素图作为主任务。它的目标是让 latent 能承载对象、颜色、形状、位置、计数、关系和编辑操作，并在不输出可读思考链的情况下完成多步推理。

训练期可以使用 teacher trace：

- object/cell/count/relation slots 的目标标签；
- 每步 read address；
- 每步 working state；
- edit 前后的 delta；
- 最终 answer / target record。

推理期不能依赖 teacher trace。teacher 只用于监督和诊断。

### 2. 再接自构造图像的双向翻译

图像阶段的任务应是外设翻译，而不是立刻做图像编辑：

```text
record -> rendered image
rendered image -> latent workspace -> record
record/text latent -> image output expert
```

因为图像是自己构造的，teacher 可以提供强监督：

- renderer 是 record -> image 的 oracle；
- parser teacher 是 image -> record 的训练期监督；
- scene exact、object exact、color/shape/position exact 是主指标；
- pixel MSE 只能作为辅助指标。

只有 `image -> latent -> record` 和 `record -> latent -> image` 都稳定后，才进入图像编辑。

### 3. 然后训练潜空间主动读和工作态

完整记忆树/工作树对 70M 太重，但可以先做轻量等价物：

```text
latent workspace
-> query/read address
-> selected slots
-> working state update
-> answer / target latent
```

关键不是把所有图像 token 喂给 reasoner，而是让 reasoner 学会按任务读取所需部分。teacher 可监督 read address 和 process state；最终必须在 teacher trace 不输入时通过。

### 4. 最后才做端到端图像编辑/生成

图像编辑应拆成两段门禁：

```text
source image/text -> source record/latent
source latent + edit -> target latent/record
target latent/record -> target image
```

如果 target record exact 不高，图像输出不算编辑通过；如果 image MSE 低但 parser 还原 record 错，也不算通过。

## 推荐 Stage AV-J 目标

下一阶段建议命名为 Stage AV-J：`latent reasoning core with teacher traces`。

它应该先实现一个不依赖真实像素的核心训练脚本和数据集：

| 组件 | 目标 |
| --- | --- |
| 数据 | 自构造 scene record、文本描述、operation、teacher trace、target record、answer |
| 模型 | record/text encoder、unified latent workspace、latent reasoner、record/text decoder、answer head |
| 训练 | codec -> latent operation -> active read/process trace -> short joint |
| 评估 | record exact、target record exact、answer exact、trace/read accuracy、no-trace/no-source/no-operation 消融 |
| 通过标准 | target record exact 和 answer exact 同时高，且 answer 不能在 record exact 为 0 时单独很高 |

AV-J 暂不训练像素图像。图像只在下一阶段作为同一 record 分布的外设接入。

## 通过门槛

后续任何阶段都必须同时报告这些门槛：

1. `source_recon_exact`：外部表征编码到 latent 后能否还原。
2. `target_translate_exact`：latent edit/reason 后能否生成目标外部表征。
3. `answer_exact`：最终答案是否正确。
4. `no_source` / `no_operation` / `no_trace` / `shuffled_external` 消融：确认不是读头捷径。
5. `cross_modal_transfer`：同一 latent 是否能被不同外设读写。
6. `teacher_free_eval`：推理期不输入 teacher trace。

必须判失败的情况：

- answer 很高，但 source/target exact 接近 0。
- pixel MSE 很低，但 parser/record exact 很低。
- latent cosine 很高，但 token/record exact 很低。
- 训练期 teacher 指标很好，但 teacher-free eval 崩掉。

## 暂停和降级的旧路线

这些路线保留为历史证据或诊断工具，不作为当前主线：

- AV-E/AV-G compact external 10M 训练：保留为“互译 exact 未闭合、answer 可走捷径”的负对照。
- AV-H 文本锚定图像编辑：保留 teacher-forced output 上限，但端到端非泄漏链路未通过。
- AV-I 整图 latent token 容量：继续作为容量/前缀分工诊断，不作为最终架构任务。
- 只按 pixel MSE、loss、latent cosine 或 answer accuracy 判定成功的训练。

## 当前担忧

1. 如果 AV-J 仍把任务设计得太低熵，模型可能再次走模板捷径。因此 record 分布要有组合泛化、heldout operation、heldout object/color/position 组合。
2. teacher trace 不能变成推理期输入，否则会重演 AV-H `target_record` 泄漏问题。
3. latent workspace 需要一等结构，例如 object/cell/count/relation/working slots；不能再用 mean-pool latent + position query 承担所有绑定。
4. 图像外设接入前必须先有 record parser/inverse 指标，否则图像输出会被背景 MSE 或视觉样例误导。
