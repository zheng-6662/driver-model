# Goal2 输入泄漏检查

- 已移除后验峰值输入：`condition_score_peak, vehicle_score_peak, driver_score_peak, peak_abs_roll, peak_abs_ay`。
- 当前特征总数：`182`。
- 禁止特征是否仍在输入中：`[]`。
- 当前输入仅来自 anchor 前/anchor 时刻车辆时序，以及由输入窗口计算的统计量。
