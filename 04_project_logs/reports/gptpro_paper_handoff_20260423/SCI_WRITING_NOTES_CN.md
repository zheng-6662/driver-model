# SCI 级写作注意事项

## 1. 这篇论文最应该强调什么

- 结构设计与建模机制
- 风格感知 driver modeling
- training-time multimodal teacher guidance
- response-fidelity evaluation protocol

不应把核心卖点写成：

- 纯性能 SOTA
- 所有指标上全面最优
- 直接在线使用生理和 EEG 的部署模型

## 2. 最安全的关键词和措辞

建议多使用：

- driver-style prior
- training-time privileged information
- teacher-state supervision
- pooled post-trigger steering-response prediction
- response fidelity
- structure-preserving prediction
- conditional road-geometry preview

避免直接使用：

- online EEG-driven prediction
- cognitive decoding
- scene-specific expert model
- universally best model
- real-time physiological closed-loop deployment

## 3. 必须明确写清楚的边界

- `style_id` 是模型上下文的一部分，可以说模型考虑驾驶风格先验。
- physio 与 EEG 在当前主线里最安全的写法是：
  - training-only
  - teacher-side
  - privileged supervision
- future road preview 必须明确写成：
  - available future road geometry
  - conditional preview
- 当前任务是：
  - post-trigger response prediction
  - not full-scene planning

## 4. 最容易被问到的 reviewer 问题

### 4.1 关于创新性

- 你的模型与现有车辆历史预测模型相比，到底新在哪里。
- style prior 和 teacher-state supervision 分别起什么作用。
- 你的 contribution 是模型本身，还是 evaluation protocol。

### 4.2 关于实验完整性

- 有没有 no-style ablation。
- 有没有 no-state-distill ablation。
- 有没有 vehicle-only 或 vehicle-plus-road baseline。
- dual-keeper 是否意味着方法没有统一效果。

### 4.3 关于可部署性

- 既然用了生理和 EEG，推理阶段是否也要依赖这些信号。
- 如果不依赖，那这些多模态信息到底怎样产生价值。

### 4.4 关于任务定义

- 任务为什么是 post-trigger response prediction，而不是 trajectory planning。
- future road preview 是否带来了过强先验。
- pooled model 会不会掩盖 subgroup collapse。

## 5. 最低限度应该补的实验

如果只能补最少的关键证据，优先级建议如下：

1. `no_style`
   - 去掉 `style_id`
   - 证明驾驶风格先验是否真的有价值
2. `no_state_distill`
   - 关闭 teacher-state distillation
   - 证明 physio/EEG teacher supervision 是否提供增益
3. `vehicle_plus_road_only`
   - 不用 style，不用 teacher state
   - 给出最基础的结构基线

如果预算极紧，至少补前两个。

## 6. 图表建议

### 6.1 必备图

- 一张总方法图
  - vehicle history
  - anchor context
  - style prior
  - future road preview
  - teacher-state supervision
  - output and evaluation
- 一张定性 case figure
  - Run A
  - `baseline_fixed_input`
  - 至少一个 collapse branch

### 6.2 必备表

- Table 1: dataset / subjects / event statistics
- Table 2: model inputs and role taxonomy
- Table 3: main comparison table
  - stable baseline
  - Run A
  - `baseline_fixed_input`
  - one or two frontier branches
- Table 4: ablation table
  - no-style
  - no-state-distill
  - optional vehicle-plus-road-only
- Table 5: metric definition table
  - why RMSE-only is insufficient

## 7. 结果叙事建议

最推荐的结果叙事不是“我们的最终模型全面最好”，而是：

1. 先证明任务边界和输入边界是清楚的。
2. 再证明引入风格和训练期生理监督的结构是有研究动机的。
3. 再证明如果只看整体误差，会误判模型质量。
4. 最后给出 dual-keeper 结论：
   - Run A 更强于 response structure
   - `baseline_fixed_input` 更稳于 fit and tail

这个叙事虽然保守一些，但更像成熟论文，而不是实验流水账。

## 8. Limitation 最安全写法

建议主动承认：

- 当前尚未得到同时统一最优的单一 checkpoint。
- 生理和 EEG 在当前主线里主要用于训练期教师监督，而非推理期必需输入。
- pooled setting 仍需要更多 subgroup audit。
- 当前实现最成熟的任务切片是 steering-response prediction，而不是更广义的全行为生成。

主动写 limitation 的好处是：

- 避免 reviewer 觉得你在过度包装
- 反而强化方法边界和工程真实性

## 9. 最重要的一句提醒

这篇论文最好的样子不是“把现有结果硬包装成最优”，而是：

> 把当前项目沉淀成一篇边界清楚、方法结构明确、评价逻辑扎实、结果诚实但有价值的 SCI 风格小论文。
