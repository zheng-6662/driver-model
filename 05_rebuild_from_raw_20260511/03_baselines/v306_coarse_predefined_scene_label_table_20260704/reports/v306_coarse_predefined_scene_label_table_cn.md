# v306 coarse predefined scene label table

## 这一步做了什么

本轮按用户重新确认的粗场景体系，把当前 1167 个事件收敛为下坡过弯、平路过弯、连续变道/连续左右修正、紧急变道/猛打方向失稳、其他/不确定五类。

这张表的目的不是继续细分“急左转/急右转/多段修正”，而是把模型条件输入改成更接近实验条件本身的粗场景标签。

## 标签分布

| coarse_scene_label                | coarse_scene_label_cn   |   total_n |     ratio |
|:----------------------------------|:------------------------|----------:|----------:|
| curve_downhill                    | 下坡过弯                |       277 | 0.237361  |
| curve_flat                        | 平路过弯                |       142 | 0.12168   |
| continuous_lane_change            | 连续变道/连续左右修正   |       414 | 0.354756  |
| emergency_lane_change_instability | 紧急变道/猛打方向失稳   |       115 | 0.0985433 |
| other_or_uncertain                | 其他/不确定             |       219 | 0.187661  |

## 输入边界

- 过弯标签来自当前 rolling manifest 的 `scene_type`，共 `419` 个事件，可作为预测前场景条件 seed。
- 直道内连续/紧急子类仍部分使用 v305/v301 自动 seed，共 `529` 个事件，需要人工或实验条件确认后才能写成最终标签。
- `other_or_uncertain` 不强行解释为某种实验事件，只用于避免把普通或不清楚直道样本误贴成连续/紧急变道。

## 人工审核工作量

- high priority：`529` 个事件，主要是直道内连续/紧急子类。
- medium priority：`219` 个事件，主要是其他/不确定或关键误差样本。

## 下一步

- v307 可直接用 `coarse_scene_label` 替换 v304 的 `event_primary_type` 条件输入，先做一轮训练对比。
- 如果 v307 有收益，再优先人工复核 high priority 中的连续/紧急变道样本。
- 如果 v307 不如 v304，则说明粗场景标签可能丢失了有用细节，需要在粗标签内保留少量二级标签，而不是回到过细的未来形状标签。

## guardrail

```json
{
  "pass": true,
  "version": "v306_coarse_predefined_scene_label_table_20260704",
  "event_n": 1167,
  "coarse_scene_class_n": 5,
  "coarse_scene_order": [
    "curve_downhill",
    "curve_flat",
    "continuous_lane_change",
    "emergency_lane_change_instability",
    "other_or_uncertain"
  ],
  "task_allows_predefined_scene_label_input": true,
  "curve_scene_labels_from_current_scene_type": true,
  "noncurve_subtypes_require_manual_or_experiment_confirmation": true,
  "uses_future_behavior_seed_for_some_noncurve_subtypes": true,
  "deployable_without_noncurve_manual_confirmation": false,
  "label_available_before_prediction_assumption": true,
  "curve_event_n": 419,
  "noncurve_future_seed_n": 529,
  "high_priority_review_n": 529,
  "medium_priority_review_n": 219,
  "figure_paths": [
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v306_coarse_predefined_scene_label_table_20260704\\figures\\v306_coarse_scene_label_distribution.png"
  ]
}
```