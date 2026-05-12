# 旧深度入口车辆 CSV 清洗 manifest v0.1

生成时间：2026-05-12

## 为什么补这一步

旧 `vehicle_direct` 深度入口直接读取 `vehicle_file`，并在旧代码里把 CSV 中的缺失值直接填成 0。当前原始车辆 CSV 存在大量交替缺失点，如果直接读原始 CSV，会把方向盘标签变成高频 0 跳变，固定图会出现不真实的黑色填充块。因此本步骤把原始车辆文件先插值成旧深度入口可读的 200Hz CSV，再生成新的 clean manifest。

## 输入

- 原始旧 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_*_split.csv`
- 原始车辆 CSV：manifest 中记录的 `vehicle_file`

## 输出

- 清洗车辆 CSV：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/oldcode_deep_vehicle_csv_v0_1`
- clean manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_*_clean_vehicle_v0_1.csv`
- 状态表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_deep_clean_vehicle_status_v0_1.csv`

## 清洗结果

- 原始车辆文件数：84
- 清洗成功文件数：84
- 清洗失败文件数：0
- session-level clean manifest 行数：906
- session-level split：{'train': 611, 'val': 156, 'test': 139}

## 与旧 `.npz` 标签一致性检查

{
  "manifest": "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_processed_datasets\\vehicle_instability_allraw_highconf_v0_1\\tables\\oldcode_manifest_session_level_split_clean_vehicle_v0_1.csv",
  "dropped": 0,
  "checked_samples": 50,
  "max_abs_diff_vs_npz_first50": 1.996755599975586e-06,
  "mean_abs_diff_vs_npz_first50": 4.798173904418946e-07
}

如果最大差异接近 0，说明旧深度入口读取 clean manifest 得到的 2 秒方向盘标签，已经和我们前面插值生成的 `pre2_label2_old_main.npz` 标签一致。

## 结论

后续旧 `vehicle_direct` 全量对照必须使用 clean manifest；此前直接用原始 CSV 的 full run 只能作为失败诊断，不能作为模型结果引用。
