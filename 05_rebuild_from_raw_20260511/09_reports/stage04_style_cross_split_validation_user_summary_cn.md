# 阶段 4 用户查看版：连续风格跨 split 复核 v0.1

## 这个阶段为什么做

上一轮 session-level 探索里，`RBF+last60 风格` 没有超过 RBF。为了确认这不是某一种划分下的偶然现象，这一轮补做 subject-level 复核：训练集和测试集按被试分开，测试被试是训练中没见过的人。

## 这个阶段检查了什么

- 样本仍然是 B 轨道 270 个严格核心失稳响应样本。
- 比较 `session_level_split` 和 `subject_level_split` 两类切分。
- 每个 split 都重新用对应 train split 拟合风格标准化参数，不沿用上一轮 session 标准化。
- 主参照仍是 `rbf_kernel_ridge_context_no_subject`。
- 风格只作为 RBF 残差模型输入，不使用生理、脑电、EMG、RESP，也不训练 Transformer。

## 目前发现了什么

```text
split_strategy                                       session_level_split  subject_level_split
model_name
rbf_kernel_ridge_context_no_subject                             0.533667             0.484847
rbf_plus_style_last60_guard3_residual_ridge                     0.534559             0.483510
rbf_plus_style_all_windows_residual_ridge                       0.564143             0.482109
rbf_plus_driver_id_residual_ridge                               0.533661             0.484992
rbf_plus_style_last60_with_driver_id_residual_ridge             0.534558             0.483511
```

session-level：RBF RMSE=0.533667，RBF+last60 风格 RMSE=0.534559。

subject-level：RBF RMSE=0.484847，RBF+last60 风格 RMSE=0.483510。

## 哪些结果可信

可信的是：连续风格在 session-level 和 subject-level 两类切分下，都没有形成稳定超过 RBF 的证据。subject-level 复核尤其重要，因为它把测试被试放到训练外，更接近“风格是否有跨人泛化信息”的问题。

## 哪些结果还不能下结论

还不能说“风格永远无效”。目前只能说：在当前事件前风格特征、RBF 残差 Ridge 融合方式、B 轨道 3 秒严格核心样本上，没有形成足够证据支持连续风格有效。未来如果换更强的风格表示或结构，可以重新验证，但不能直接升级为主线。

## 下一阶段是否可以继续

生理和 EEG 仍然不能进入有效性验证。下一步更合理的是先把阶段 4 暂时降级收口，回到车辆-only 结构化轨迹建模，优先解决错侧、反向修正、多段修正和困难样本。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_metric_summary_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_subject_bad_samples_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_gate_table.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_metrics.csv`
