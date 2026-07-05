# v238 任务构造与小型 rolling 模型重搭报告

## 本轮到底改了什么

- 本轮没有继续 v222a gate / 删除样本 / light residual 路线。
- 本轮没有重新扫描原始车辆 CSV，而是复用 v236 已保存的 rolling 输入，降低数据口径变化风险。
- 主任务从 `receding_2s` 改成 `original_remaining`：delay 后只预测原始事件 `anchor+2s` 以内还剩下的部分。
- 训练形式从“一条样本输出 21 点整曲线”改成 point-level masked target：无效的新阶段点不进入训练 loss。
- 模型仍是小模型：validation-only 在 point Ridge 与小 MLP 中选择；没有 gate/router/selector。

## 总体判断

- 接受 v238 的任务构造方向：`original_remaining` masked target 是对 v236 receding 目标混入新阶段问题的修正。
- 不接受当前 selected MLP 作为正式替代模型：它改善了部分难例，但普通样本 no-harm 和 1000ms late delay 没守住。
- 下一步不应扩大模型，而应加 validation no-harm 约束，并把 1000ms 延迟作为单独策略/诊断处理。

## 任务构造

- rolling 样本数：7002；唯一事件数：1167。
- 每个 delay 的 original_remaining 有效点数：
  - delay=0ms：每样本有效点 21.0，尾段点 11.0。
  - delay=200ms：每样本有效点 19.0，尾段点 11.0。
  - delay=400ms：每样本有效点 17.0，尾段点 11.0。
  - delay=600ms：每样本有效点 15.0，尾段点 11.0。
  - delay=800ms：每样本有效点 13.0，尾段点 11.0。
  - delay=1000ms：每样本有效点 11.0，尾段点 11.0。

## 模型选择

- selected model：`v238_point_mlp_96x48_alpha1e-4`；validation rank=1；selection score=1.290127。
- 选择只使用 validation original_remaining；`test_used_for_selection=False`。

## Test original_remaining 对照

### all
- delay=0ms：tail v236=0.777846，v238=0.677294，delta=-0.100552；sample v236=0.641212，v238=0.589768
- delay=200ms：tail v236=0.644885，v238=0.646794，delta=+0.001909；sample v236=0.549574，v238=0.574347
- delay=400ms：tail v236=0.603264，v238=0.584581，delta=-0.018683；sample v236=0.528290，v238=0.536131
- delay=600ms：tail v236=0.569821，v238=0.582153，delta=+0.012332；sample v236=0.507444，v238=0.533492
- delay=800ms：tail v236=0.483962，v238=0.511864，delta=+0.027903；sample v236=0.447104，v238=0.486802
- delay=1000ms：tail v236=0.400838，v238=0.537904，delta=+0.137066；sample v236=0.400838，v238=0.537904

### observe_later_like
- delay=0ms：tail v236=1.100397，v238=0.897843，delta=-0.202554；sample v236=0.891287，v238=0.747557
- delay=200ms：tail v236=0.997138，v238=0.828008，delta=-0.169130；sample v236=0.821942，v238=0.700720
- delay=400ms：tail v236=1.044708，v238=0.783294，delta=-0.261414；sample v236=0.884018，v238=0.684182
- delay=600ms：tail v236=0.960424，v238=0.853871，delta=-0.106553；sample v236=0.836583，v238=0.764950
- delay=800ms：tail v236=0.840703，v238=0.732710，delta=-0.107993；sample v236=0.775528，v238=0.696034
- delay=1000ms：tail v236=0.744582，v238=1.054059，delta=+0.309477；sample v236=0.744582，v238=1.054059

### strong_steer
- delay=0ms：tail v236=0.961224，v238=0.710411，delta=-0.250813；sample v236=0.783853，v238=0.623086
- delay=200ms：tail v236=0.824720，v238=0.687179，delta=-0.137541；sample v236=0.700115，v238=0.609041
- delay=400ms：tail v236=0.734935，v238=0.690442，delta=-0.044493；sample v236=0.648727，v238=0.620229
- delay=600ms：tail v236=0.691450，v238=0.684357，delta=-0.007093；sample v236=0.617853，v238=0.621267
- delay=800ms：tail v236=0.591223，v238=0.613846，delta=+0.022622；sample v236=0.546390，v238=0.577035
- delay=1000ms：tail v236=0.446245，v238=0.523496，delta=+0.077251；sample v236=0.446245，v238=0.523496

### normal_predictable
- delay=0ms：tail v236=0.638458，v238=0.641138，delta=+0.002680；sample v236=0.533926，v238=0.555958
- delay=200ms：tail v236=0.505590，v238=0.610721，delta=+0.105131；sample v236=0.433528，v238=0.544721
- delay=400ms：tail v236=0.462133，v238=0.493829，delta=+0.031696；sample v236=0.403554，v238=0.465104
- delay=600ms：tail v236=0.445911，v238=0.460920，delta=+0.015009；sample v236=0.396505，v238=0.429134
- delay=800ms：tail v236=0.389533，v238=0.409670，delta=+0.020137；sample v236=0.359694，v238=0.395018
- delay=1000ms：tail v236=0.317906，v238=0.420524，delta=+0.102618；sample v236=0.317906，v238=0.420524

## 边界与解释

- 如果 v238 在 observe_later_like 上改善，优先解释为“任务窗口修正 + 小非线性模型缓解 Ridge 收缩”，不是正式 headline。
- 如果 v238 在某些 delay 或 bucket 变差，说明 point-level 原事件剩余任务还需要继续调输入/目标，不代表 rolling 方向失败。
- 本轮仍不允许用 test 反调配置，也不允许把 response type 变成硬路由。类型信息只用于评估分桶。

## 下一步决策

- `accept_task_construction`: `True`；original_remaining masked point-level target passes guardrail and removes new-phase points from the main loss.
- `accept_selected_model_as_formal_replacement`: `False`；normal_predictable no-harm fails and delay=1000 observe_later_like degrades; v238 is a prototype, not a formal replacement.
- `observe_later_mid_delay_gain`: `True`；test observe_later_like tail delta is negative for delays 0-800ms.
- `strong_0_to_600_gain`: `True`；test strong_steer tail delta is non-positive for delays 0-600ms.
- `normal_noharm_pass`: `False`；normal_predictable sample RMSE is worse than v236 at one or more delays.
- `delay_1000_policy_pass`: `False`；delay=1000 remains unsafe for the same selected point model; keep it diagnostic or handle separately.
- `recommended_next_task`: `v239_noharm_constrained_original_remaining_model`；Keep the new original_remaining task, but add validation no-harm criteria for normal samples and an explicit late-delay policy before any formal use.

## Guardrail

- `v237_allowed_v238`: `True`
- `test_used_for_selection`: `False`
- `same_event_uid_cross_split_count`: `0`
- `observe_later_like_deleted`: `False`
- `gate_router_selector_created`: `False`
- `formal_headline_changed`: `False`
- `mixed_delay_metric_used_as_headline`: `False`
- `primary_target_mode`: `original_remaining_masked_point_level_steering_delta`
- `dropped_events`: `0`
- `pass`: `True`
- `required_guardrails_from_v237`: `仍禁止 v222a gate/router/selector；必须继续按 delay 和 bucket 分开报告；不得使用 test 选择模型配置；不得改变 formal headline。`

## 输出

- `tables/v238_task_construction_audit.csv`
- `tables/v238_point_training_rows_by_delay.csv`
- `tables/v238_model_selection_validation_only.csv`
- `tables/v238_metrics_by_delay_and_bucket.csv`
- `tables/v238_compare_v236_original_remaining.csv`
- `tables/v238_selected_per_sample_metrics.csv`
- `tables/v238_next_model_decision.csv`
- ZIP：`v238_task_model_redesign_pack.zip`
