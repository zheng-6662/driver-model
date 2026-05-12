# 阶段 4 用户查看版：连续风格路线收口决策 v0.1

## 这个阶段为什么做

前面已经完成连续风格来源协议、session-level 探索对照和 subject-level 跨被试复核。现在需要把结论收口：连续风格当前能不能升级为主线，生理/脑电能不能进入下一阶段。

## 这个阶段检查了什么

- 风格是否无泄漏：通过，风格窗口在事件前，不接触直接车辆输入和未来标签。
- 风格是否超过 RBF：没有稳定超过。
- 风格是否改善物理错误：没有稳定改善错侧、大幅响应召回、困难样本或反向修正。
- 风格是否只是驾驶员 ID 替代品：目前不是主要问题，因为风格本身也没有稳定增益。

## 目前发现了什么

```text
session-level: RBF=0.533667, RBF+style60=0.534559, delta=0.000892
subject-level: RBF=0.484847, RBF+style60=0.483510, delta=-0.001337
```

subject-level 有很小 RMSE 改善，但 session-level 没有，而且物理指标没有稳定改善。因此不能说连续风格有效。

## 哪些结果可信

可信的是：在当前“事件前统计风格特征 + RBF 残差 Ridge”表示下，连续风格没有形成可升级为主线的稳定证据。这个结论经过了固定 RBF 参照、驾驶员 ID 对照、置乱控制、session-level 和 subject-level 检查。

## 哪些结果还不能下结论

不能说“风格永远无效”。只能说当前表示方式和融合方式没有形成强证据。未来如果换成更好的时序风格表示或门控结构，可以重新作为后备路线验证。

## 下一阶段是否可以继续

可以继续，但不应进入生理/EEG 有效性验证。下一步应回到车辆-only 结构化轨迹建模，先解决错侧、幅值、尾段、反向修正、多段修正和困难样本。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/figures/style_route_rmse_delta_summary.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_decision_gate_table.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_evidence_summary.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_next_actions.csv`
