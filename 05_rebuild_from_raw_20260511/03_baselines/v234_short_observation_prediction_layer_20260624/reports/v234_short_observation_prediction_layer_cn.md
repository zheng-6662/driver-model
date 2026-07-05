# v234 短观察后预测评估层构建包

## 目的

本包把 v233 的 `observe_later_review` 样本单独构造成“短观察后预测”评估层。
它不是统一后移事件锚点，也不是训练新模型，而是把任务可观测性分层：纯提前预测和短观察后预测分开报告。

## 方法边界

- 不训练模型、不改标签、不改 formal headline。
- 不把旧 formal prediction 硬评到新观察层；旧 prediction 是从旧锚点出发的，不适合直接评估后移观察点。
- 不重启硬响应类型分类；不把简单多候选轨迹输出作为主线。

## 输出文件

- 层定义表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_layer_definition.csv`
- 样本层分配表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_layer_assignments.csv`
- 真实目标曲线长表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_target_curves.csv`
- 人工审核模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\tables\v234_short_observation_manual_review_template.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\figures`
- 图拼接总览：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\figures\v234_short_observation_layer_contact_sheet.png`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v234_short_observation_prediction_layer_20260624\v234_short_observation_prediction_layer_pack.zip`

## 默认建议层摘要

|rank|sample_id|old_anchor_s|suggest_delay|visible_frac|remaining_frac|remaining_peak|zero_hold_rmse|
|---:|---|---:|---:|---:|---:|---:|---:|
|1|`rjy_Entity_Recording_2025_09_28_19_51_44_v108_014`|230.100|0.5|0.387|1.142|3.146|1.850|
|2|`rjy_Entity_Recording_2025_09_28_19_51_44_v108_039`|569.300|0.5|0.340|0.582|1.492|0.847|
|3|`lx_Entity_Recording_2025_09_26_08_58_43_v108_011`|250.200|0.5|0.412|1.459|3.685|2.468|
|4|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_029`|526.900|0.5|1.027|1.861|4.496|3.522|
|5|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_031`|549.400|0.5|0.522|1.594|3.812|2.465|
|6|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_014`|268.000|0.5|0.417|1.467|3.230|2.066|
|7|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_004`|120.900|0.5|0.587|1.597|3.390|2.580|
|8|`rjy_Entity_Recording_2025_09_28_20_15_42_v108_001`|54.700|0.5|0.547|1.229|2.372|1.829|
|9|`rjy_Entity_Recording_2025_09_28_19_51_44_v108_023`|357.800|0.5|0.315|1.203|2.107|1.161|
|10|`rjy_Entity_Recording_2025_09_28_20_15_42_v108_008`|210.000|0.5|0.573|1.642|2.599|1.512|

## 解释

`visible_frac` 表示到观察点时已经看见的方向盘证据占原后续峰值的比例。
`remaining_frac` 表示从观察点之后 2 秒内仍然剩余的目标峰值占原后续峰值的比例。
如果 `visible_frac` 上升而 `remaining_frac` 仍不低，说明短观察后预测既有可见证据，也仍有真实未来要预测，不是简单补全已经发生的轨迹。

## 下一步

人工先审核默认 `0.5s` 层是否合理；如果某些样本 0.5 秒仍太早或太晚，可在模板里填写 `human_selected_observe_delay_s=1.0/1.5`。
人工确认后，下一步才生成 v235 的短观察层数据清单或重新评估对应模型。