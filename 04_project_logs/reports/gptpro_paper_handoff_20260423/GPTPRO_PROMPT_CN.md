# 给 GPT Pro 的直接提示词

你现在拿到的是一个已经完成过多轮迭代的极端工况驾驶员建模项目 handoff 包。你的目标不是复述仓库历史，而是基于现有证据，把它整理成一篇有 SCI 水平潜力的小论文完整思路。

## 你的任务

请把自己当成论文合作者，而不是保守 reviewer。你需要在不越过当前证据边界的前提下，最大化这篇小论文的学术表达质量、实验设计完整性和投稿可行性。

## 你必须先接受的事实

1. 当前最安全的论文方向不是“单一模型已经全局最优”，而是“风格感知 + 生理教师监督 + 响应结构评价”的多模态驾驶员建模框架。
2. 当前项目的活跃实现最适合表述为：
   - pooled post-trigger steering-response prediction
   - not scene-specific planning
   - not cognition decoding
3. `style_id` 是模型上下文的一部分，可以作为 driver-style prior 来写。
4. physio 和 EEG 最安全的写法是：
   - training-time only
   - teacher-side
   - privileged supervision
   - not required at inference
5. 当前结果是 dual-keeper：
   - Run A 是 response-structure keeper
   - `baseline_fixed_input` 是 fit / tail keeper
6. `H15`、`bridge_50_50` 等结果说明更低的误差不一定意味着更好的响应结构，RMSE-only judgment is unsafe.

## 你优先阅读的文件

1. `PAPER_CONTEXT_SUMMARY_CN.md`
2. `SCI_WRITING_NOTES_CN.md`
3. `proposal/OPENING_REPORT_PAPER_RELEVANT_SUMMARY_CN.md`
4. `proposal/OPENING_REPORT_KEY_SECTIONS_EXTRACT_CN.md`
5. `context/current-state.md`
6. `context/INPUT_ROLE_AUDIT.md`
7. `context/THESIS_DEFENSE_TABLES.md`
8. `evidence/input_ablation_summary.md`
9. `evidence/bridge_summary.md`
10. `evidence/effectiveness_summary.md`
11. `evidence/effectiveness_comparison_table.csv`
12. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
13. `code/v58_modular/README.md`
14. `tools/recalc_v58_checkpoint_with_current_metrics.py`

## 我希望你输出什么

请严格按下面顺序输出，不要只给高层空话。

### 1. 论文定位与题目

请给出：

- 1 个你最推荐的中文题目
- 1 个最推荐的英文题目
- 2 到 4 个备选题目
- 这篇论文最合理的 paper positioning
- 一句最核心的 claim sentence

### 2. 摘要级核心逻辑

请用接近正式论文摘要的方式，写出：

- 研究问题
- 现有不足
- 我们的方法核心
- 关键实验发现
- 最终结论

要求：

- 不要把话说过头
- 但也不要写得像项目汇报

### 3. 全文详细提纲

请按 SCI 小论文标准给出详细提纲，至少覆盖：

- Introduction
- Related Work
- Data Collection And Preprocessing
- Problem Formulation
- Model
- Training Strategy
- Evaluation Protocol
- Experiments And Ablations
- Results And Discussion
- Limitations
- Conclusion

每一节都请说明：

- 这一节要回答什么问题
- 这一节该放哪些内容
- 最好放哪些图和表

### 4. Introduction 应该怎么写

请给出一个高质量 introduction 逻辑链，说明：

- 如何从极端工况和人机协同驾驶切入
- 如何引出“只看车辆和道路信息不够”
- 如何自然引出驾驶风格和训练期生理教师监督
- 如何引出“RMSE-only is insufficient”
- 最终如何收束到本文贡献

请尽量写成可直接扩展成正式 introduction 的逻辑骨架。

### 5. 方法部分应该如何组织

请结合当前代码和开题报告，说明：

- 当前实现最安全的问题定义
- 输入组成应该怎样介绍
- `style_id` 应该怎么写
- physio/EEG teacher-state 应该怎么写
- Transformer 结构怎样讲才专业
- loss / auxiliary objectives / checkpoint selection 该如何组织描述

并请指出：

- 哪些开题报告里的大目标适合保留
- 哪些当前不应写太大

### 6. 实验设计与消融

请给出一套最像 SCI 论文的实验矩阵，并明确区分：

- 已经有的证据
- 最好补充的最小新增实验

至少要回答：

- 主比较表应该放哪些方法或分支
- `no_style` 是否必须
- `no_state_distill` 是否必须
- `vehicle_plus_road_only` 是否值得补
- dual-keeper 结果怎样组织才不显得混乱

### 7. 结果叙事

请说明：

- 结果部分应该先讲什么，后讲什么
- 如何把 Run A 与 `baseline_fixed_input` 讲成优势互补而不是“谁都不够好”
- 如何处理 `bridge_50_50`、`H15`、`H10` 这些“误差更好但结构更差”的分支
- 如何把这些 no-go 结果变成论文价值而不是负担

### 8. SCI 水平下最关键的风险点

请站在专业论文写作角度，指出：

- 最容易被 reviewer 质疑的地方
- 最容易写过头的表述
- 最需要补证据的地方
- 最建议主动写进 limitation 的内容

### 9. 如果现在只允许补 1 到 2 组实验

请只给最推荐的实验组合，并解释：

- 为什么性价比最高
- 为什么最能支撑这篇论文
- 做完之后论文会从“阶段性结果”提升到什么程度

## 重要约束

- 不要把 physio/EEG 写成当前推理阶段必需输入，除非你明确说明那只是另一种未来扩展版本。
- 不要把当前故事写成 SOTA 或单一全局最优。
- 不要把任务写成 scene-specific planning。
- 不要忽视 response-fidelity 指标的重要性。
- 不要只围绕 RMSE 展开论文叙事。
- 不要给泛泛的“再调一下参数”式建议。

## 我的真实需求

我现在不是要一个普通汇报提纲，而是要你基于这个 handoff 包，给出一套专业、可落地、符合 SCI 水平的小论文完整构思与写法建议。
