# 数据版本卡：vehicle_instability_all_raw_rescreen_v0_1

生成时间：2026-05-12 16:25:03

## 数据版本定位

这是从所有原始车辆 CSV 直接重筛得到的车辆失稳候选版本。它替代了只基于旧候选表的筛选方式，覆盖 `原始车辆数据/<被试名>/*.csv` 下 91 个原始车辆文件。

## 事件定义

车辆失稳候选事件由非方向盘车辆动态异常触发：

- `|ay| >= 1.3`
- `|roll_rate| >= 0.05`

相邻动态种子先按 0.35 秒合并，再按 2.5 秒合并为候选事件片段。

## 证据融合

每个候选事件会补充：

- 横向加速度、横滚速率、横摆角速度、横向偏移、车速、事件后方向盘响应；
- 旧 v400 事件上下文；
- 道路中心线模块和映射可靠度；
- 已有 31 条键盘标注的精确或近邻校准。

## 数量

- 原始车辆 CSV：91
- 可读取 CSV：91
- 候选总数：1991
- 自动/已确认采用：1348
- 高置信主清单：908
- 中间复核：269
- 低证据剔除：374

## 不能下的结论

- 不能把自动采用事件称为完全人工真值。
- 不能把道路模块本身称为失稳原因。
- 不能用本版本证明连续风格或生理有效。
- 不能继续使用旧 404 个弯道样本作为车辆失稳主样本。

## 推荐使用

建议下一步先用 `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv` 作为保守主清单生成车辆失稳样本 manifest；`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_auto_accepted_v0_1.csv` 可作为扩展清单。正式训练前必须重新生成处理后车辆窗口和 split 表。
