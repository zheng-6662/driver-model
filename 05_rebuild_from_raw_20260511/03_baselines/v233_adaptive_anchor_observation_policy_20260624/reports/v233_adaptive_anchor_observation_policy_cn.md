# v233 自适应锚点 / 观察时长策略审核包

## 目的

本包回应用户的新判断：有些大变化事件在事件前几秒确实看不出区别。
这类样本不应强行归为锚点晚，也不应要求模型在没有可见证据时预测完整大响应；可以单独审核是否后移观察点或延长观察时长。

## 方法边界

- 本轮不训练模型、不修改标签、不改 formal headline。
- 不重启硬响应类型分类。
- 不把简单多候选轨迹输出作为主线。
- 后移观察点不是为了刷分，而是把任务拆成不同可观测性层级：提前预测、短观察后预测、已响应补全。

## 输出文件

- 样本策略表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\tables\v233_anchor_observation_policy_table.csv`
- 人工审核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\tables\v233_anchor_observation_policy_review_table.csv`
- 观察延迟表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\tables\v233_observe_delay_grid.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\figures`
- 策略图拼接总览：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\figures\v233_adaptive_anchor_policy_contact_sheet.png`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v233_adaptive_anchor_observation_policy_20260624\v233_adaptive_anchor_observation_policy_pack.zip`

## 策略分布

|policy|count|
|---|---:|
|observe_later_review|10|
|standard_anchor_review|7|
|reanchor_earlier_or_ambiguous_review|6|
|reanchor_earlier_review|5|
|large_change_standard_or_ambiguous|1|

## 人工审核重点

|rank|policy|sample_id|old_anchor_s|suggest_delay|pre3/post03|post_peak|reason|
|---:|---|---|---:|---:|---:|---:|---|
|1|reanchor_earlier_review|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_010`|143.100||2.669|2.825|旧锚点前已有强证据或人工确认，优先审核提前重锚定|
|2|reanchor_earlier_review|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_033`|638.700||1.193|2.260|旧锚点前已有强证据或人工确认，优先审核提前重锚定|
|3|reanchor_earlier_review|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_041`|624.900||0.996|1.933|旧锚点前已有强证据或人工确认，优先审核提前重锚定|
|4|reanchor_earlier_review|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_040`|620.900||1.150|1.823|旧锚点前已有强证据或人工确认，优先审核提前重锚定|
|5|reanchor_earlier_review|`rjy_Entity_Recording_2025_09_28_19_33_26_v108_032`|598.900||0.971|1.711|旧锚点前已有强证据或人工确认，优先审核提前重锚定|
|6|reanchor_earlier_or_ambiguous_review|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_026`|507.200||0.419|2.147|存在中等晚锚点证据，需人工判定提前重锚定还是保留原锚点|
|7|reanchor_earlier_or_ambiguous_review|`rjy_Entity_Recording_2025_09_28_19_33_26_v108_014`|391.200||1.042|1.569|存在中等晚锚点证据，需人工判定提前重锚定还是保留原锚点|
|8|reanchor_earlier_or_ambiguous_review|`lx_Entity_Recording_2025_09_26_09_17_22_v108_034`|527.600||0.738|1.436|存在中等晚锚点证据，需人工判定提前重锚定还是保留原锚点|
|9|reanchor_earlier_or_ambiguous_review|`cwh_Entity_Recording_2025_09_26_20_06_19_v108_017`|252.200||0.528|1.315|存在中等晚锚点证据，需人工判定提前重锚定还是保留原锚点|
|10|reanchor_earlier_or_ambiguous_review|`rjy_Entity_Recording_2025_09_28_20_15_42_v108_006`|180.800||0.570|1.086|存在中等晚锚点证据，需人工判定提前重锚定还是保留原锚点|
|11|reanchor_earlier_or_ambiguous_review|`cwh_Entity_Recording_2025_09_26_19_56_16_v108_021`|390.700||0.980|0.965|存在中等晚锚点证据，需人工判定提前重锚定还是保留原锚点|
|12|observe_later_review|`rjy_Entity_Recording_2025_09_28_19_51_44_v108_014`|230.100|0.5|0.247|2.756|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|13|observe_later_review|`rjy_Entity_Recording_2025_09_28_19_51_44_v108_039`|569.300|0.5|0.340|2.562|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|14|observe_later_review|`lx_Entity_Recording_2025_09_26_08_58_43_v108_011`|250.200|0.5|0.177|2.526|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|15|observe_later_review|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_029`|526.900|0.5|0.319|2.416|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|16|observe_later_review|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_031`|549.400|0.5|0.237|2.392|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|17|observe_later_review|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_014`|268.000|0.5|0.186|2.202|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|18|observe_later_review|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_004`|120.900|0.5|0.291|2.123|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|19|observe_later_review|`rjy_Entity_Recording_2025_09_28_20_15_42_v108_001`|54.700|0.5|0.328|1.930|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|20|observe_later_review|`rjy_Entity_Recording_2025_09_28_19_51_44_v108_023`|357.800|0.5|0.176|1.752|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|
|21|observe_later_review|`rjy_Entity_Recording_2025_09_28_20_15_42_v108_008`|210.000|0.5|0.304|1.583|后续变化很大但旧锚点前证据弱，适合审核后移观察点/延长观察时长|

## 解释

如果样本属于 `reanchor_earlier_review`，说明旧锚点前已经能看到较强变化，应优先审核是否提前重锚定。
如果样本属于 `observe_later_review`，说明旧锚点前证据弱但后续变化大，适合审核是否把观察点后移 0.5-2 秒，再做预测。
这两类不能混在一起：前者是标注/事件起点问题，后者是任务可观测性问题。

## 下一步

人工先看 `observe_later_review` 的图：如果旧锚点前确实看不出区别，但后移 0.5-1.5 秒后可见响应证据，则应建立一个单独的“短观察后预测”评估层，而不是把它和纯提前预测混在同一指标里。