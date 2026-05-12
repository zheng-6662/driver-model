# 阶段 3 用户查看版：复发坏样本详细曲线复盘

## 为什么做

前面已经看到 RBF/KNN/template 的平均误差有改善，但一些事件在很多配置下都会失败。这个阶段把这些反复失败的事件画成曲线，方便直接看问题是不是来自事件锚点、窗口、原始车辆信号，还是车辆-only 模型确实表达不了复杂响应。

## 检查了什么

- 复发坏样本 Top 12。
- 每个事件的输入窗口、标签窗口和事件锚点。
- 原始车辆方向盘、横向加速度、横滚、横摆角速度、速度、曲率、横向位置等波形。
- GT 方向盘响应与 RBF/KNN/template 等车辆-only 候选预测。

## 目前发现

复发最高的事件仍是 `vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435`。在 Top 12 事件的 5 个候选预测里，严重幅值不足率=0.700，错侧率=0.233，反向修正计数完全匹配率=0.033。这说明平均 RMSE 变低以后，复杂物理响应仍然没有被稳定解决。

## 哪些结果可信

这一步只做复盘和绘图，没有引入生理、脑电、连续风格或驾驶员 ID。图里的原始车辆波形来自 `samples_master.csv` 指向的原始车辆 CSV，只读取不修改。

## 哪些结果还不能下结论

现在还不能说失败一定是模型结构造成的，也不能说生理数据会解决这些失败。必须先看曲线里是否有锚点偏差、窗口覆盖不足或原始数据异常。

## 下一阶段是否可以继续

可以继续阶段 3，但不是进入生理或风格；下一步应该先根据这些曲线决定结构化车辆模型怎么做。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/bad_event_curve_contact_sheet.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_figure_index.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_model_error_table.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event`
