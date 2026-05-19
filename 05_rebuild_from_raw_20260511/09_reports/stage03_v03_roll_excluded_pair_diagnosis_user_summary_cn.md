# 横滚/姿态 excluded 版本 paired 诊断（用户查看版）

## 这次为什么做

横滚/姿态 excluded 版本整体 RMSE 比参考版本差，但大响应错侧率和严重幅值不足率更好。本报告只比较两版共同测试样本，并单独检查新增横滚/姿态 excluded 样本，判断它到底是在改善关键极限样本，还是只是改变测试集组成。

## 共同测试样本结论

- 共同测试样本数：205。
- 横滚版本逐样本 RMSE 改善：102 个；恶化：103 个。
- 共同测试样本平均 ΔRMSE（横滚版本 - 参考版本）：0.0079。负数代表横滚版本更好。
- 大响应共同样本数：52；大响应平均 ΔRMSE：-0.0185。
- 错侧/严重幅值不足至少一项改善的样本：23 个；至少一项恶化的样本：6 个。

## 新增横滚/姿态 excluded 测试样本

- 横滚版本中新增 excluded 测试样本数：65。
- 其中大响应样本数：32。
- 新增 excluded 的平均逐样本 RMSE：0.6854。
- 新增 excluded 的大响应错侧率：0.0462。
- 新增 excluded 的严重幅值不足率：0.1538。

## 分被试结果

| subject_ref | n | mean_delta_roll_minus_ref | large_n | large_mean_delta | wrong_side_improved_n | severe_under_improved_n |
| --- | --- | --- | --- | --- | --- | --- |
| txj | 43.0000 | 0.0128 | 16.0000 | 0.0081 | 3.0000 | 9.0000 |
| zx | 42.0000 | 0.0216 | 15.0000 | -0.0784 | 3.0000 | 5.0000 |
| xst | 28.0000 | 0.0022 | 6.0000 | -0.0739 | 1.0000 | 0.0000 |
| gf | 24.0000 | 0.0175 | 4.0000 | 0.1237 | 1.0000 | 2.0000 |
| byx | 12.0000 | -0.0144 | 4.0000 | -0.0716 | 0.0000 | 1.0000 |
| lx | 26.0000 | 0.0128 | 3.0000 | 0.1500 | 0.0000 | 1.0000 |
| zdq | 20.0000 | -0.0266 | 3.0000 | -0.0808 | 1.0000 | 1.0000 |
| jy | 10.0000 | 0.0053 | 1.0000 | 0.1112 | 0.0000 | 0.0000 |

## 分工况来源结果

| condition_context_cn_ref | n | mean_delta_roll_minus_ref | large_n | large_mean_delta | wrong_side_improved_n | severe_under_improved_n |
| --- | --- | --- | --- | --- | --- | --- |
| 低附着 | 167.0000 | 0.0139 | 34.0000 | 0.0029 | 5.0000 | 12.0000 |
| 弯道/曲率 | 12.0000 | -0.0960 | 12.0000 | -0.0960 | 3.0000 | 4.0000 |
| 横滚/姿态 | 7.0000 | 0.0089 | 5.0000 | -0.0042 | 1.0000 | 3.0000 |
| 普通驾驶对照 | 18.0000 | 0.0002 | 1.0000 | 0.1112 | 0.0000 | 0.0000 |
| 横向动态 | 1.0000 | 0.3878 | 0.0000 | NA | 0.0000 | 0.0000 |

## 当前判断

- 如果只看整体 RMSE，横滚/姿态版本不能直接替代参考版本。
- 如果看大响应物理问题，横滚/姿态版本有继续研究价值，尤其要看错侧和严重幅值不足是否集中改善在强姿态/大响应样本。
- 更合理的后续方向不是把横滚/姿态 excluded 全部混入普通训练，而是把它作为极限姿态子集：单独复核、加权训练，或者做响应类型分支。

## 可查看文件

- paired 明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/tables/roll_vs_ref_common_test_paired_metrics.csv`
- 改善最多样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/tables/top_roll_improved_common_test.csv`
- 恶化最多样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/tables/top_roll_worsened_common_test.csv`
- 改善样本对比图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/figures/roll_vs_ref_top_improved_common_test.png`
- 恶化样本对比图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/figures/roll_vs_ref_top_worsened_common_test.png`
- 物理指标改善样本对比图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/figures/roll_vs_ref_physical_improved_common_test.png`