# manuscript claim boundary notes

## 可以写入正文的边界

- 可以写：formal headline 已锁定为 loose=avg_joint_focus 与 strict=peak_floor_090。
- 可以写：主结果数值与 v225/v226 完全复现，并附带 sample bootstrap 与 subject-block bootstrap 区间。
- 可以写：tail error 集中度和 underestimation 是 limitation，需要在论文中透明呈现。
- 可以写：v228 是最终论文产物冻结，不是新实验。

## 不可以写的边界

- 不可以声称 v228 训练了新模型或改进了模型。
- 不可以把 reporting-only 表格整理解释成新的 leaderboard 提升。
- 不可以把诊断行、人工标签、oracle 类标签或回退行写入 formal evidence。
- 不可以基于 test split 重新选择模型、阈值、样本或 claim。

## 本轮 GPTPro 指令状态

本轮通过本地 ChatGPT 软件端取得有效 GPTPro 回复。GPTPro 接受 v227 作为 reporting-only closeout，并要求 v228 只做论文产物冻结；任何锁定指标、模型名、CI 值、guardrail 状态或 claim 与 v225/v226/v227 冲突时，都应停止并输出 failure report，而不是修补模型或扩展实验边界。
