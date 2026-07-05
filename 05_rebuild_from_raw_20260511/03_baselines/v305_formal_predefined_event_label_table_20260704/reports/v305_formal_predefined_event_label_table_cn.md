# v305 formal predefined event label table

## 这一步做了什么

本轮把“事件可以提前定好”正式落成一张事件标签表。它不直接继续训练模型，而是先把主事件类型、辅助诊断标签和人工审核状态分开，避免把未来轨迹形状误当成预测前输入。

当前表由 v301 自动事件标签草稿生成，因此仍是人工审核 seed，不是最终人工标签。后续如果用户或实验条件确认每个事件的主类型，`formal_primary_type` 就可以作为 v304/v305 条件模型的正式输入。

## 主标签设计

- 可作为条件输入的主标签：`普通/轻微/不确定`、`急停/强减速`、`急左转`、`急右转`、`连续变道/横向避让`、`紧急避让/连续变道`、`复合制动转向`。
- 更依赖未来过程形状的内容，如 `晚响应`、`多段修正`、`快速转向`，放入 `formal_secondary_tags`，默认不作为直接输入。

## 标签分布

| formal_primary_type   |   total_n |     ratio |
|:----------------------|----------:|----------:|
| 普通/轻微/不确定      |       697 | 0.597258  |
| 急停/强减速           |        80 | 0.0685518 |
| 急左转                |        56 | 0.0479863 |
| 急右转                |        46 | 0.0394173 |
| 连续变道/横向避让     |       175 | 0.149957  |
| 紧急避让/连续变道     |        54 | 0.0462725 |
| 复合制动转向          |        59 | 0.050557  |

## 人工审核工作量

- high priority：`869` 个事件。
- medium priority：`161` 个事件。
- 审核优先级主要来自：原 v301 需人工复核、v249/v300 高误差、原标签为多段修正/晚响应这类更像未来形状的标签、或自动置信度不足。

## 当前判断

- 如果事件主类型确实能在预测前由人工、实验条件、感知/规划模块确定，那么它可以作为合法输入。
- 当前 v305 表把这个输入边界固定下来：主类型可输入，诊断标签默认不可直接输入。
- 下一步应让人工审核 `manual_review_seed_pack.csv`，确认或修改 `formal_primary_type` 和 `manual_review_status`。
- 人工确认后，再用这张表替换 v304 里的 v301 自动标签，重跑 fixed event-label conditioned 曲线模型。

## guardrail

```json
{
  "pass": true,
  "version": "v305_formal_predefined_event_label_table_20260704",
  "event_n": 1167,
  "formal_primary_class_n": 7,
  "formal_primary_order": [
    "普通/轻微/不确定",
    "急停/强减速",
    "急左转",
    "急右转",
    "连续变道/横向避让",
    "紧急避让/连续变道",
    "复合制动转向"
  ],
  "task_allows_predefined_event_label_input": true,
  "formal_primary_type_can_be_model_input_after_confirmation": true,
  "diagnostic_tags_as_direct_input_allowed": false,
  "current_seed_source": "v301_future_behavior_auto_draft",
  "current_seed_derived_from_future_behavior": true,
  "requires_manual_or_experiment_confirmation": true,
  "deployable_without_manual_or_experiment_confirmation": false,
  "label_available_before_prediction_assumption": true,
  "high_priority_review_n": 869,
  "medium_priority_review_n": 161,
  "figure_paths": [
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v305_formal_predefined_event_label_table_20260704\\figures\\v305_formal_primary_type_distribution.png"
  ]
}
```