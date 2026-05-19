# 快速转向候选按车辆响应强弱拆分 v0.3

用户复核指出：方向盘快速打只说明操作急，不一定说明车辆进入极限/近极限。 因此本次把快速转向候选继续拆成“车辆响应可见”“车辆响应边界”“只有快速转向但车辆响应弱”。

## 数量

- `FAST_STEER_BODY_RESPONSE_BOUNDARY`：19 个。
- `FAST_STEER_ONLY_WEAK_VEHICLE_RESPONSE`：35 个。
- `FAST_STEER_WITH_VISIBLE_VEHICLE_RESPONSE`：24 个。

## 结论

- `FAST_STEER_WITH_VISIBLE_VEHICLE_RESPONSE` 可以作为快速转向训练候选的第一优先级。
- `FAST_STEER_BODY_RESPONSE_BOUNDARY` 先进入风险池，人工确认后再决定。
- `FAST_STEER_ONLY_WEAK_VEHICLE_RESPONSE` 不建议进入极限主训练，最多作为普通/快速微调对照。

## 文件

- 拆分表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\fast_steer_vehicle_response_split_v0_3.csv`
- 图片目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\fast_steer_vehicle_response_split_v0_3`
- 图片说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\fast_steer_vehicle_response_split_v0_3\00_先看这里_快速转向按车辆响应拆分说明.md`