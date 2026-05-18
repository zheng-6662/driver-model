# v0.3 样本纳入范围消融（用户查看版）

## 这次为什么做

用户复核后认为待复核样本和部分排除样本也可能可以进入训练，因此本轮不改模型结构，只逐步放宽样本纳入范围，看车辆-only 基线是否变好或变差。

三档设置如下：

1. 当前干净集：只用当前已经纳入的四类样本。
2. 干净集 + 待复核：加入 `manual_review`。
3. 干净集 + 待复核 + 可成窗排除样本：再加入 `excluded` 中仍能构建完整窗口的样本。

切分尽量沿用当前干净集的 session 划分；同一原始记录中新加入的样本跟随原记录的 train/val/test，减少因为重新切分导致的误判。

## 总体结果

| variant_id | name_cn | sample_count | test_best_model | test_rmse_steer | test_wrong_side_rate_large | test_severe_amp_under_rate_large | test_large_response_recall | clean_subset_test_sample_rmse_aggregate | clean_subset_wrong_side_rate_large | clean_subset_severe_amp_under_rate_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v03_clean | 当前干净集 | 482 | rbf_kernel_vehicle_context_alpha0.1_g2 | 0.797252 | 0.314286 | 0.742857 | 0.428571 | 0.797252 | 0.314286 | 0.742857 |
| v03_plus_review | 干净集 + 待复核 | 793 | rbf_kernel_vehicle_context_alpha0.1_g1 | 0.592277 | 0.196429 | 0.678571 | 0.642857 | 0.789786 | 0.2 | 0.68 |
| v03_plus_review_excluded | 干净集 + 待复核 + 可成窗排除样本 | 1284 | rbf_kernel_vehicle_context_alpha1_g1 | 0.735415 | 0.205882 | 0.735294 | 0.617647 | 0.763284 | 0.235294 | 0.735294 |

## 当前读法

- 如果加入待复核后 test RMSE 和干净子集指标同时改善，说明待复核样本大概率有训练价值。
- 如果总 RMSE 改善但干净子集恶化，说明新增样本可能改变了测试分布，不能直接说更好。
- 如果加入排除样本后明显恶化，说明 excluded 里仍有大量语义或信号问题，只能筛选后使用。

## 阶段性判断

- `干净集 + 待复核` 是当前最值得继续的训练样本范围：总体 RMSE 明显下降，大响应错侧率和大响应召回也同步改善。
- `excluded` 不建议直接全量加入：虽然比当前干净集好，但比 `干净集 + 待复核` 差，说明里面还有一批语义混乱、信号异常或锚点不合适的样本。
- 下一步更合理的是把 `manual_review` 升级为可训练样本，同时对 `excluded` 再按排除原因分层，只逐步加入低风险子类。

## 可查看文件

- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_inclusion_ablation\tables\v03_vehicle_only_inclusion_ablation_summary.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_inclusion_ablation`